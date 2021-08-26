#include "cuda/algorithms/component_connection.hpp"

#include "edm/cell.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/vector.hpp"

#include "sparse_ccl.cuh"

#include <iostream>

namespace {
    using index_t = unsigned short;
}

namespace traccc { namespace cuda { namespace details {
    __device__
    bool is_adjacent(const cell & a, const cell & b) {
        return (
            (a.channel0 - b.channel0)*(a.channel0 - b.channel0) <= 1 &&
            (a.channel1 - b.channel1)*(a.channel1 - b.channel1) <= 1
        );
    }

    __device__
    bool is_adjacent(channel_id c0, channel_id c1, const cell & b) {
        unsigned int p0 = (c0 - b.channel0);
        unsigned int p1 = (c1 - b.channel1);

        return p0 * p0 <= 1 && p1 * p1 <= 1;
    }

    __device__
    bool is_adjacent(channel_id ac0, channel_id ac1, channel_id bc0, channel_id bc1) {
        unsigned int p0 = (ac0 - bc0);
        unsigned int p1 = (ac1 - bc1);

        return p0 * p0 <= 1 && p1 * p1 <= 1;
    }

    __device__
    void
    reduce_problem_cell(
        geometry_id * ch0,
        geometry_id * ch1,
        index_t tid,
        unsigned char & adjc,
        index_t adjv[],
        index_t size
    ) {
        /*
        * The number of adjacent cells for each cell must start at zero, to
        * avoid uninitialized memory. adjv does not need to be zeroed, as
        * we will only access those values if adjc indicates that the value
        * is set.
        */
        adjc = 0;

        channel_id c0 = ch0[tid];
        channel_id c1 = ch1[tid];

        /*
        * First, we traverse the cells backwards, starting from the current
        * cell and working back to the first, collecting adjacent cells
        * along the way.
        */
        for (index_t j = tid - 1; j < tid; --j) {
            /*
             * Since the data is sorted, we can assume that if we see a cell
             * sufficiently far away in both directions, it becomes
             * impossible for that cell to ever be adjacent to this one.
             * This is a small optimisation.
             */
            if (ch1[j] + 1 < c1) {
                break;
            }

            /*
             * If the cell examined is adjacent to the current cell, save it
             * in the current cell's adjacency set.
             */
            if (is_adjacent(c0, c1, ch0[j], ch1[j])) {
                adjv[adjc++] = j;
            }
        }

        /*
         * Now we examine all the cells past the current one, using almost
         * the same logic as in the backwards pass.
         */
        for (index_t j = tid + 1; j < size; ++j) {
            /*
             * Note that this check now looks in the opposite direction! An
             * important difference.
             */
            if (ch1[j] > c1 + 1) {
                break;
            }

            if (is_adjacent(c0, c1, ch0[j], ch1[j])) {
                adjv[adjc++] = j;
            }
        }
    }

    __device__
    void
    fast_sv_1(
        index_t * f,
        index_t * gf,
        unsigned char * adjc,
        index_t * adjv,
        unsigned int size
    ) {
        /*
         * The algorithm finishes if an iteration leaves the arrays unchanged.
         * This varible will be set if a change is made, and dictates if another
         * loop is necessary.
         */
        bool gfc;

        do {
            /*
             * Reset the end-parameter to false, so we can set it to true if we
             * make a change to our arrays.
             */
            gfc = false;

            /*
             * The algorithm executes in a loop of three distinct parallel
             * stages. In this first one, we examine adjacent cells and copy
             * their cluster ID if it is lower than our, essentially merging
             * the two together.
             */
            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
                __builtin_assume(adjc[tst] <= 8);

                for (unsigned char k = 0; k < adjc[tst]; ++k) {
                    index_t q = gf[adjv[8*tst+k]];

                    if (gf[tid] > q) {
                        f[f[tid]] = q;
                        f[tid] = q;
                    }
                }
            }

            /*
             * Each stage in this algorithm must be preceded by a
             * synchronization barrier!
             */
            __syncthreads();

            /*
             * The second stage is shortcutting, which is an optimisation that
             * allows us to look at any shortcuts in the cluster IDs that we
             * can merge without adjacency information.
             */
            for (index_t tid = threadIdx.x; tid < size; tid += blockDim.x) {
                if (f[tid] > gf[tid]) {
                    f[tid] = gf[tid];
                }
            }

            /*
             * Synchronize before the final stage.
             */
            __syncthreads();

            /*
             * Update the array for the next generation, keeping track of any
             * changes we make.
             */
            for (index_t tid = threadIdx.x; tid < size; tid += blockDim.x) {
                if (gf[tid] != f[f[tid]]) {
                    gf[tid] = f[f[tid]];
                    gfc = true;
                }
            }

            /*
             * To determine whether we need another iteration, we use block
             * voting mechanics. Each thread checks if it has made any changes
             * to the arrays, and votes. If any thread votes true, all threads
             * will return a true value and go to the next iteration. Only if
             * all threads return false will the loop exit.
             */
        } while (__syncthreads_or(gfc));
    }

    __device__
    void
    fast_sv_2(
        index_t * f,
        index_t * gf,
        unsigned char * adjc,
        index_t * adjv,
        unsigned int size
    ) {
        bool gfc;

        do {
            gfc = false;

            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
                for (unsigned char k = 0; k < adjc[tst]; ++k) {
                    index_t j = adjv[8*tst+k];

                    if (f[f[j]] < gf[f[tid]]) {
                        gf[f[tid]] = f[f[j]];
                        gfc = true;
                    }
                }
            }

            __syncthreads();

            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
                for (unsigned char k = 0; k < adjc[tst]; ++k) {
                    index_t j = adjv[8*tst+k];

                    if (f[f[j]] < gf[tid]) {
                        gf[tid] = f[f[j]];
                        gfc = true;
                    }
                }
            }

            __syncthreads();

            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
                if (f[f[tid]] < gf[tid]) {
                    gf[tid] = f[f[tid]];
                    gfc = true;
                }
            }

            __syncthreads();

            // if (gfc) {
            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
                f[tid] = gf[tid];
            }
            // }
        } while (__syncthreads_or(gfc));
    }

    __device__
    void
    aggregate_clusters_1(
        const cell_container & cells,
        measurement_container & out,
        index_t * f
    ) {
        __shared__ unsigned int outi;

        if (threadIdx.x == 0) {
            outi = 0;
        }

        __syncthreads();

        /*
         * This is the post-processing stage, where we merge the clusters into a
         * single measurement and write it to the output.
         */
        for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {

            /*
             * If and only if the value in the work arrays is equal to the index
             * of a cell, that cell is the "parent" of a cluster of cells. If
             * they are not, there is nothing for us to do. Easy!
             */
            if (f[tid] == tid) {
                /*
                 * If we are a cluster owner, atomically claim a position in the
                 * output array which we can write to.
                 */
                unsigned int id = atomicAdd(&outi, 1);

                /*
                 * These variables keep track of the sums of X and Y coordinates
                 * for the final coordinates, the total activation weight, as
                 * well as the sum of squares of positions, which we use to
                 * calculate the variance.
                 */
                float sx = 0.0, sy = 0.0, sw = 0.0;

                /*
                 * Now, we iterate over all other cells to check if they belong
                 * to our cluster. Note that we can start at the current index
                 * because no cell is every a child of a cluster owned by a cell
                 * with a higher ID.
                 */
                for (index_t j = tid; j < cells.size; j++) {
                    /*
                     * If the value of this cell is equal to our, that means it
                     * is part of our cluster. In that case, we take its values
                     * for position and add them to our accumulators.
                     */
                    if (f[j] == tid) {
                        sx += cells.activation[j] * cells.channel0[j];
                        sy += cells.activation[j] * cells.channel1[j];
                        sw += cells.activation[j];
                    }
                }

                /*
                 * Write the average weighted x and y coordinates, as well as
                 * the weighted average square position, to the output array.
                 */
                out.channel0[id] = sx / sw;
                out.channel1[id] = sy / sw;
                out.module_id[id] = cells.module_id[tid];
            }
        }
    }

    __device__
    void
    aggregate_clusters_2(
        const cell_container & cells,
        measurement_container & out,
        index_t * f
    ) {
        __shared__ unsigned int outi;
        __shared__ float w[MAX_CLUSTERS_PER_PARTITION];
        __shared__ float x[MAX_CLUSTERS_PER_PARTITION];
        __shared__ float y[MAX_CLUSTERS_PER_PARTITION];
        bool parent[MAX_ACTIVATIONS_PER_PARTITION / THREADS_PER_BLOCK];

        if (threadIdx.x == 0) {
            outi = 0;
        }

        __syncthreads();

        /*
         * This is the post-processing stage, where we merge the clusters into a
         * single measurement and write it to the output.
         */
         for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
            /*
             * If and only if the value in the work arrays is equal to the index
             * of a cell, that cell is the "parent" of a cluster of cells. If
             * they are not, there is nothing for us to do. Easy!
             */
            if (f[tid] == tid) {
                /*
                 * If we are a cluster owner, atomically claim a position in the
                 * output array which we can write to.
                 */
                index_t oid;
                parent[tst] = true;
                oid = atomicAdd(&outi, 1);
                f[tid] = oid;

                x[oid] = 0.0;
                y[oid] = 0.0;
                w[oid] = 0.0;
            } else {
                parent[tst] = false;
            }
        }

        __syncthreads();

        for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
            index_t oid;

            if (parent[tst]) {
                oid = f[tid];
            } else {
                oid = f[f[tid]];
            }

            atomicAdd(&w[oid], cells.activation[tid]);
            atomicAdd(&x[oid], cells.activation[tid] * cells.channel0[tid]);
            atomicAdd(&y[oid], cells.activation[tid] * cells.channel1[tid]);
        }

        __syncthreads();

        for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
            index_t oid;

            if (parent[tst]) {
                oid = f[tid];

                out.channel0[oid] = x[oid] / w[oid];
                out.channel1[oid] = y[oid] / w[oid];
                out.module_id[oid] = cells.module_id[tid];
            }
        }
    }

    __global__
    __launch_bounds__(THREADS_PER_BLOCK)
    void
    sparse_ccl_kernel(
        const cell_container container,
        const ccl_partition * partitions,
        measurement_container _out_ctnr,
        unsigned int * gouti
    ) {
        const ccl_partition & partition = partitions[blockIdx.x];
        // unsigned int c1, c2, c3, c4;

        // if (threadIdx.x == 0) asm("mov.u32 %0, %clock;" : "=r"(c1) );

        /*
         * Seek the correct cell region in the input data. Again, this is all a
         * contiguous block of memory for now, and we use the blocks array to
         * define the different ranges per block/module. At the end of this we
         * have the starting address of the block of cells dedicated to this
         * module, and we have its size.
         */
        cell_container cells;
        cells.size = partition.size;
        cells.channel0 = &container.channel0[partition.start];
        cells.channel1 = &container.channel1[partition.start];
        cells.activation = &container.activation[partition.start];
        cells.time = &container.time[partition.start];
        cells.module_id = &container.module_id[partition.start];

        if (cells.size > MAX_ACTIVATIONS_PER_PARTITION) {
            cells.size = MAX_ACTIVATIONS_PER_PARTITION;
        }

        /*
         * As an optimisation, we will keep track of which cells are adjacent to
         * each other cell. To do this, we define, in thread-local memory or
         * registers, up to eight adjacent cell indices and we keep track of how
         * many adjacent cells there are (i.e. adjc[i] determines how many of
         * the eight values in adjv[i] are actually meaningful).
         *
         * The implementation is such that a thread might need to process more
         * than one hit. As such, we keep one counter and eight indices _per_
         * hit the thread is processing. This number is never larger than
         * the max number of activations per module divided by the threads per
         * block.
         */
        index_t adjv[8 * MAX_ACTIVATIONS_PER_PARTITION / THREADS_PER_BLOCK];
        unsigned char adjc[MAX_ACTIVATIONS_PER_PARTITION / THREADS_PER_BLOCK];

        /*
         * After this is all done, we synchronise the block. I am not absolutely
         * certain that this is necessary here, but the overhead is not that big
         * and we might as well be safe rather than sorry.
         */
        __syncthreads();

        /*
         * This loop initializes the adjacency cache, which essentially
         * translates the sparse CCL problem into a graph CCL problem which we
         * can tackle with well-studied algorithms. This loop pattern is often
         * found throughout this code. We iterate over the number of activations
         * each thread must process. Sadly, the CUDA limit is 1024 threads per
         * block and we cannot guarantee that there will be fewer than 1024
         * activations in a module. So each thread must be able to do more than
         * one.
         */
        {
            __shared__ geometry_id ch0[MAX_ACTIVATIONS_PER_PARTITION];
            __shared__ geometry_id ch1[MAX_ACTIVATIONS_PER_PARTITION];

            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
                ch0[tid] = cells.channel0[tid];
                ch1[tid] = cells.channel1[tid];
            }

            __syncthreads();

            for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
                reduce_problem_cell(ch0, ch1, tid, adjc[tst], &adjv[8*tst], cells.size);
            }
        }

        // if (threadIdx.x == 0) asm("mov.u32 %0, %clock;" : "=r"(c2) );

        /*
         * These arrays are the meat of the pudding of this algorithm, and we
         * will constantly be writing and reading from them which is why we
         * declare them to be in the fast shared memory. Note that this places a
         * limit on the maximum activations per module, as the amount of shared
         * memory is limited. These could always be moved to global memory, but
         * the algorithm would be decidedly slower in that case.
         */
        __shared__ index_t f[MAX_ACTIVATIONS_PER_PARTITION], gf[MAX_ACTIVATIONS_PER_PARTITION];

        for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
            /*
             * At the start, the values of f and gf should be equal to the ID of
             * the cell.
             */
            f[tid] = tid;
            gf[tid] = tid;
        }

        /*
         * Now that the data has initialized, we synchronize again before we
         * move onto the actual processing part.
         */
        __syncthreads();

        fast_sv_1(f, gf, adjc, adjv, cells.size);

        // if (threadIdx.x == 0) asm("mov.u32 %0, %clock;" : "=r"(c3) );

        /*
         * This variable will be used to write to the output later.
         */
        __shared__ unsigned int outi;

        if (threadIdx.x == 0) {
            outi = 0;
        }

        __syncthreads();

        for (index_t tst = 0, tid; (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
            if (f[tid] == tid) {
                atomicAdd(&outi, 1);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            outi = atomicAdd(gouti, outi);
        }

        __syncthreads();

        measurement_container out;
        out.channel0 = &_out_ctnr.channel0[outi];
        out.channel1 = &_out_ctnr.channel1[outi];
        out.module_id = &_out_ctnr.module_id[outi];

        aggregate_clusters_1(cells, out, f);
        // aggregate_clusters_2(cells, out, f);

        __syncthreads();

        // if (threadIdx.x == 0) {
        //     asm("mov.u32 %0, %clock;" : "=r"(c4) );
        //     printf("%u %u %u\n", (c2 - c1), (c3 - c2), (c4 - c3));
        // }
    }

    __host__
    void
    sparse_ccl(
        const cell_container container,
        vecmem::vector<ccl_partition> && partitions,
        const measurement_container out_ctnr
    ) {
        for (const ccl_partition & i : partitions) {
            std::cout << "Partition " << i.start << " has size is " << i.size << std::endl;
        }

        unsigned int * outi;
        cudaMallocManaged(&outi, sizeof(unsigned int));
        *outi = 0;

        sparse_ccl_kernel<<<partitions.size(), THREADS_PER_BLOCK>>>(container, partitions.data(), out_ctnr, outi);

        cudaError_t r = cudaDeviceSynchronize();

        if (r != cudaSuccess) {
            fprintf(stderr,"assert: %s %s %d\n", cudaGetErrorString(r), __FILE__, __LINE__);
        }

        for (unsigned int i = 0; i < *outi; ++i) {
            printf("%05d = (% 8.2f, % 8.2f) on %lu\n", i, out_ctnr.channel0[i], out_ctnr.channel1[i], out_ctnr.module_id[i]);
        }

        printf("Out length = %u\n", *outi);
    }
}}}
