/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/details/seed_finding_alt.hpp"

// Project include(s).
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/edm/device/device_doublet.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/device/seeding_global_counter.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/device/count_doublets.hpp"
#include "traccc/seeding/device/count_triplets.hpp"
#include "traccc/seeding/device/find_doublets.hpp"
#include "traccc/seeding/device/find_triplets.hpp"
#include "traccc/seeding/device/reduce_triplet_counts.hpp"
#include "traccc/seeding/device/select_seeds.hpp"
#include "traccc/seeding/device/update_triplet_weights.hpp"
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/seeding/triplet_finding_helper.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>
#include <vector>

namespace {
using namespace traccc;

template <typename SP>
__host__ __device__ std::tuple<float, float> thething(const SP& sp1,
                                                      const SP& sp2,
                                                      float minHelixR) {
    // The middle point between sp1 and sp2
    float midX = 0.5 * (sp1.x() + sp2.x());
    float midY = 0.5 * (sp1.y() + sp2.y());
    // The distance between the spacepoints in 2D space
    float deltaX = sp2.x() - sp1.x();
    float deltaY = sp2.y() - sp1.y();
    float deltaXY = std::sqrt(deltaX * deltaX + deltaY * deltaY);

    float slope = (sp2.y() - sp1.y()) / (sp2.x() - sp1.x());

    float centralAngle = std::atan(-1.f / slope);

    float sagittaLength = minHelixR - std::sqrt(minHelixR * minHelixR -
                                                (deltaXY * deltaXY) / 4.f);

    float mpDeltaX = (minHelixR - sagittaLength) * std::cos(centralAngle);
    float mpDeltaY = (minHelixR - sagittaLength) * std::sin(centralAngle);

    float mp1X = midX + mpDeltaX;
    float mp2X = midX - mpDeltaX;
    float mp1Y = midY + mpDeltaY;
    float mp2Y = midY - mpDeltaY;

    float mp1R = std::sqrt(mp1X * mp1X + mp1Y * mp1Y);
    float mp2R = std::sqrt(mp2X * mp2X + mp2Y * mp2Y);

    return {mp1R, mp2R};
}

__host__ __device__ float square(float x) {
    return x * x;
}

__host__ __device__ float distance3d(float x1, float x2, float y1, float y2,
                                     float z1, float z2) {
    return std::sqrt(square(x2 - x1) + square(y2 - y1) + square(z2 - z1));
}

struct device_triplet_alt {
    unsigned int bot_idx;
    unsigned int mid_idx;
    unsigned int top_idx;

    /// curvature of circle estimated from triplet
    scalar curvature;
    /// weight of triplet
    scalar weight;
    /// z origin of triplet
    scalar z_vertex;
};

template <typename T>
__host__ __device__ T& strided_access(T* arr, std::size_t global_idx,
                                      std::size_t local_idx,
                                      std::size_t stride1,
                                      std::size_t stride2) {
    // return arr[global_idx * stride1 + local_idx];
    return arr[local_idx * stride2 + global_idx];
}

__global__ void find_doublets(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoint_view,
    traccc::details::spacepoint_grid_types::const_view sp_grid_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        sp_prefix_sum_view,
    long int* doublets, const unsigned int max_doublets_per_sp,
    unsigned long long* num_doublets) {
    std::size_t global_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Set up the device containers.
    const edm::spacepoint_collection::const_device spacepoints{spacepoint_view};
    const traccc::details::spacepoint_grid_types::const_device sp_grid(
        sp_grid_view);

    if (global_index >= spacepoints.size()) {
        return;
    }

    const unsigned int sp_idx = global_index;
    const unsigned int num_sp = spacepoints.size();

    // Set up some administrative data.
    unsigned int doublet_idx = 0;

    // for (unsigned int i = 0; i < max_doublets_per_sp; ++i) {
    //     strided_access(doublets, sp_idx, i, max_doublets_per_sp, num_sp) =
    //     -1;
    // }

    // Get the spacepoint that we're evaluating in this thread, and treat that
    // as the "middle" spacepoint.
    const edm::spacepoint_collection::const_device::const_proxy_type middle_sp =
        spacepoints.at(sp_idx);

    // The the IDs of the neighbouring bins along the phi and Z axes of the
    // grid.
    const detray::dindex_range phi_bins =
        sp_grid.axis_p0().range(middle_sp.phi(), config.neighbor_scope);
    const detray::dindex_range z_bins =
        sp_grid.axis_p1().range(middle_sp.z(), config.neighbor_scope);
    assert(z_bins[0] <= z_bins[1]);

    // Iterate over all of the neighboring phi bins, including the same bin that
    // the middle spacepoint is in. The loop over the phi bins needs to take
    // into account that we may iterate over the "wrap around point" of the
    // axis.
    for (detray::dindex phi_bin_iterator = phi_bins[0];
         phi_bin_iterator <=
         (phi_bins[1] +
          (phi_bins[0] > phi_bins[1] ? sp_grid.axis_p0().n_bins : 0));
         ++phi_bin_iterator) {

        // Set up the phi bin index that we are actually meant to use inside of
        // the loop. We could also use a modulo operation here, but that would
        // be slightly more expensive in this specific case.
        const detray::dindex phi_bin =
            (phi_bin_iterator >= sp_grid.axis_p0().n_bins
                 ? phi_bin_iterator - sp_grid.axis_p0().n_bins
                 : phi_bin_iterator);

        // Iterate over all of the neighboring Z bins, including the same bin
        // that the middle spacepoint is in. This is a much easier iteration, as
        // the Z axis does not "wrap around".
        for (detray::dindex z_bin = z_bins[0]; z_bin <= z_bins[1]; ++z_bin) {

            // Ask the grid for all of the spacepoints in this specific bin.
            typename traccc::details::spacepoint_grid_types::const_device::
                serialized_storage::const_reference spacepoint_indices =
                    sp_grid.bin(phi_bin, z_bin);

            // Loop over all of those spacepoint indices.
            for (unsigned int other_sp_idx : spacepoint_indices) {

                // Get the other spacepoint.
                const edm::spacepoint_collection::const_device::const_proxy_type
                    other_sp = spacepoints.at(other_sp_idx);

                float minHelixRadius =
                    std::sqrt(config.minHelixDiameter2) / 2.f;

                std::tuple<float, float> result =
                    thething(middle_sp, other_sp, minHelixRadius);

                float minthing =
                    std::min(std::get<0>(result), std::get<1>(result));

                if (minthing <= (minHelixRadius - config.impactMax)) {
                    continue;
                }

                // Check if this spacepoint is a compatible "top" spacepoint to
                // the thread's "middle" spacepoint.
                if (doublet_finding_helper::isCompatible<
                        details::spacepoint_type::top>(middle_sp, other_sp,
                                                       config)) {
                    // assert(doublet_idx < max_doublets_per_sp);
                    unsigned int idx = doublet_idx++;
                    if (idx < max_doublets_per_sp) {
                        strided_access(doublets, sp_idx, idx,
                                       max_doublets_per_sp, num_sp) =
                            other_sp_idx;
                        atomicAdd(num_doublets, 1ULL);
                    } else {
                        printf("We're at index %u\n", idx);
                    }
                }
            }
        }
    }

    for (std::size_t i = doublet_idx; i < max_doublets_per_sp; ++i) {
        strided_access(doublets, sp_idx, i, max_doublets_per_sp, num_sp) = -1;
    }
}

__global__ void prune_doublets(
    const long int* doublets, long int* new_doublets,
    const unsigned int max_doublets_per_sp,
    edm::spacepoint_collection::const_view spacepoint_view) {
    unsigned int spacepoint_idx = blockIdx.x;
    unsigned int num_sp = gridDim.x;
    extern __shared__ int doublet_liveness[];

    const edm::spacepoint_collection::const_device spacepoints{spacepoint_view};

    __shared__ unsigned int old_dubs, new_dubs;

    if (threadIdx.x == 0) {
        old_dubs = 0;
        new_dubs = 0;
    }

    bool seen_invalid = false;

    for (unsigned int i = threadIdx.x; i < max_doublets_per_sp;
         i += blockDim.x) {
        if (strided_access(doublets, spacepoint_idx, i, max_doublets_per_sp,
                           num_sp) < 0) {
            doublet_liveness[i] = 0u;
            seen_invalid = true;
        } else if (seen_invalid) {
            doublet_liveness[i] = 0u;
        } else {
            doublet_liveness[i] = 1u;
        }
    }

    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < max_doublets_per_sp; ++i) {
            if (strided_access(doublets, spacepoint_idx, i, max_doublets_per_sp,
                               num_sp) >= 0) {
                old_dubs++;
            } else {
                break;
            }
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < old_dubs * max_doublets_per_sp;
         i += blockDim.x) {
        const unsigned int mid_idx_idx = i / max_doublets_per_sp;
        const unsigned int top_idx_idx = i % max_doublets_per_sp;

        const long int mid_idx = strided_access(
            doublets, spacepoint_idx, mid_idx_idx, max_doublets_per_sp, num_sp);

        if (mid_idx < 0) {
            printf("This should never fire\n");
            break;
        }

        const long int top_idx = strided_access(doublets, mid_idx, top_idx_idx,
                                                max_doublets_per_sp, num_sp);

        if (top_idx < 0) {
            continue;
        }

        for (unsigned int j = 0;
             j < max_doublets_per_sp &&
             strided_access(doublets, spacepoint_idx, j, max_doublets_per_sp,
                            num_sp) >= 0;
             ++j) {
            if (strided_access(doublets, spacepoint_idx, j, max_doublets_per_sp,
                               num_sp) == top_idx) {
                const auto& sp_a = spacepoints.at(spacepoint_idx);
                const auto& sp_b = spacepoints.at(mid_idx);
                const auto& sp_c = spacepoints.at(top_idx);

                float direct_distance = distance3d(
                    sp_a.x(), sp_c.x(), sp_a.y(), sp_c.y(), sp_a.z(), sp_c.z());
                float indirect_distance =
                    distance3d(sp_a.x(), sp_b.x(), sp_a.y(), sp_b.y(), sp_a.z(),
                               sp_b.z()) +
                    distance3d(sp_b.x(), sp_c.x(), sp_b.y(), sp_c.y(), sp_b.z(),
                               sp_c.z());

                if (indirect_distance <= 1.01f * direct_distance) {
                    // doublet_liveness[j] = 0;
                    break;
                }
            }
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < max_doublets_per_sp;
         i += blockDim.x) {
        if (doublet_liveness[i] >= 1) {
            strided_access(new_doublets, spacepoint_idx, i, max_doublets_per_sp,
                           num_sp) =
                strided_access(doublets, spacepoint_idx, i, max_doublets_per_sp,
                               num_sp);
        } else {
            strided_access(new_doublets, spacepoint_idx, i, max_doublets_per_sp,
                           num_sp) = -1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < max_doublets_per_sp; ++i) {
            if (strided_access(new_doublets, spacepoint_idx, i,
                               max_doublets_per_sp, num_sp) >= 0) {
                new_dubs++;
            }
        }
        assert(new_dubs <= old_dubs);
        // printf("SP %u Went from %u to %u\n", spacepoint_idx, old_dubs,
        //        new_dubs);
    }
}

__global__ void find_triplets(
    seedfinder_config finder_config, seedfilter_config filter_config,
    edm::spacepoint_collection::const_view spacepoint_view,
    const long int* doublets, device_triplet_alt* triplets,
    unsigned int* num_triplets, unsigned int max_doublets_per_sp,
    unsigned int max_triplets_per_sp) {
    unsigned int spacepoint_idx = blockIdx.x;
    const edm::spacepoint_collection::const_device spacepoints{spacepoint_view};

    __shared__ unsigned int local_num_triplets;

    if (threadIdx.x == 0) {
        local_num_triplets = 0;
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x;
         i < max_doublets_per_sp * max_doublets_per_sp; i += blockDim.x) {
        const unsigned int mid_idx_idx = i / max_doublets_per_sp;
        const unsigned int top_idx_idx = i % max_doublets_per_sp;

        const long int mid_idx =
            strided_access(doublets, spacepoint_idx, mid_idx_idx,
                           max_doublets_per_sp, spacepoints.size());

        if (mid_idx < 0) {
            continue;
        }

        const long int top_idx =
            strided_access(doublets, mid_idx, top_idx_idx, max_doublets_per_sp,
                           spacepoints.size());

        if (top_idx < 0) {
            continue;
        }

        const edm::spacepoint_collection::const_device::const_proxy_type sp_b =
            spacepoints.at(spacepoint_idx);

        const edm::spacepoint_collection::const_device::const_proxy_type sp_m =
            spacepoints.at(mid_idx);

        const edm::spacepoint_collection::const_device::const_proxy_type sp_t =
            spacepoints.at(top_idx);

        // Apply the conformal transformation to the two doublets
        const traccc::lin_circle lb =
            doublet_finding_helper::transform_coordinates<
                details::spacepoint_type::bottom>(sp_m, sp_b);
        const traccc::lin_circle lt =
            doublet_finding_helper::transform_coordinates<
                details::spacepoint_type::top>(sp_m, sp_t);

        const scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
        const scalar scatteringInRegion2 =
            finder_config.maxScatteringAngle2 * iSinTheta2 *
            finder_config.sigmaScattering * finder_config.sigmaScattering;

        scalar curvature, impact_parameter;

        // Check if mid-bot and mid-top doublets can form a triplet
        if (triplet_finding_helper::isCompatible(
                sp_m, lb, lt, finder_config, iSinTheta2, scatteringInRegion2,
                curvature, impact_parameter) &&
            doublet_finding_helper::isCompatible<details::spacepoint_type::top>(
                sp_m, sp_t, finder_config) &&
            doublet_finding_helper::isCompatible<
                details::spacepoint_type::bottom>(sp_m, sp_b, finder_config)) {

            bool confirmed = true;

            for (std::size_t j = 0; j < max_doublets_per_sp; ++j) {
                auto confirm_idx =
                    strided_access(doublets, top_idx, j, max_doublets_per_sp,
                                   spacepoints.size());

                if (confirm_idx >= 0) {
                    const edm::spacepoint_collection::const_device::
                        const_proxy_type sp_c = spacepoints.at(confirm_idx);

                    const traccc::lin_circle ltc =
                        doublet_finding_helper::transform_coordinates<
                            details::spacepoint_type::bottom>(sp_t, sp_m);
                    const traccc::lin_circle lc =
                        doublet_finding_helper::transform_coordinates<
                            details::spacepoint_type::top>(sp_m, sp_c);
                    // const scalar iSinTheta2Conf =
                    //     1 + ltc.cotTheta() * ltc.cotTheta();
                    // const scalar scatteringInRegion2Conf =
                    //     finder_config.maxScatteringAngle2 * iSinTheta2Conf *
                    //     finder_config.sigmaScattering *
                    //     finder_config.sigmaScattering;

                    scalar curvatureConf, impact_parameterConf;
                    // if (triplet_finding_helper::isCompatible(
                    //        sp_t, ltc, lc, finder_config, iSinTheta2Conf,
                    //       scatteringInRegion2Conf, curvatureConf,
                    //      impact_parameterConf)) {

                    if (true || triplet_finding_helper::isCompatible(
                                    sp_m, lb, lc, finder_config, iSinTheta2,
                                    scatteringInRegion2, curvatureConf,
                                    impact_parameterConf)) {

                        // if (std::abs(curvature - curvatureConf) <=
                        //         2.f * std::abs(curvature) &&
                        //     std::abs(impact_parameter - impact_parameterConf)
                        //     <=
                        //         2.f * std::abs(impact_parameter)) {
                        confirmed = true;
                        break;
                        //}
                    }
                }
            }

            if (confirmed) {
                unsigned int local_triplet_index =
                    atomicAdd(&local_num_triplets, 1);

                if (local_triplet_index < max_triplets_per_sp) {
                    atomicAdd(num_triplets, 1);
                    // assert(local_triplet_index < max_triplets_per_sp);

                    // Add triplet to jagged vector
                    triplets[spacepoint_idx * max_triplets_per_sp +
                             local_triplet_index] = {
                        spacepoint_idx,
                        static_cast<unsigned int>(mid_idx),
                        static_cast<unsigned int>(top_idx),
                        curvature,
                        -impact_parameter * filter_config.impactWeightFactor,
                        lb.Zo()};
                } else {
                    printf("We're at triplet index %u\n", local_triplet_index);
                }
            }
        }

        __syncthreads();

        for (unsigned int i = local_num_triplets + threadIdx.x;
             i < max_triplets_per_sp; i += blockDim.x) {
            triplets[spacepoint_idx * max_triplets_per_sp + i] = {0,  0,  0,
                                                                  0., 0., 0.};
        }
    }
}

__global__ void select_seeds(
    seedfilter_config filter_config,
    edm::spacepoint_collection::const_view spacepoint_view,
    const device_triplet_alt* triplets, const unsigned int max_triplets_per_sp,
    edm::seed_collection::view seed_view) {

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const edm::spacepoint_collection::const_device spacepoints{spacepoint_view};

    if (global_idx >= spacepoints.size() * max_triplets_per_sp) {
        return;
    }

    unsigned int spacepoint_idx = global_idx / max_triplets_per_sp;
    unsigned int triplet_idx = global_idx % max_triplets_per_sp;

    edm::seed_collection::device seeds_device(seed_view);

    const device_triplet_alt& triplet =
        triplets[spacepoint_idx * max_triplets_per_sp + triplet_idx];

    if (triplet.bot_idx == 0 && triplet.mid_idx == 0) {
        return;
    }

    assert(triplet.bot_idx == spacepoint_idx);

    const edm::spacepoint_collection::const_device::const_proxy_type bot_sp =
        spacepoints.at(triplet.bot_idx);
    const edm::spacepoint_collection::const_device::const_proxy_type mid_sp =
        spacepoints.at(triplet.mid_idx);
    const edm::spacepoint_collection::const_device::const_proxy_type top_sp =
        spacepoints.at(triplet.top_idx);

    // check if it is a good triplet
    if (seed_selecting_helper::single_seed_cut(filter_config, mid_sp, bot_sp,
                                               top_sp, triplet.weight)) {
        // check if it is a good triplet
        const edm::seed_collection::device::size_type iseed =
            seeds_device.push_back_default();
        edm::seed_collection::device::proxy_type seed = seeds_device.at(iseed);
        seed.bottom_index() = triplet.bot_idx;
        seed.middle_index() = triplet.mid_idx;
        seed.top_index() = triplet.top_idx;
    }
}
}  // namespace

namespace traccc::cuda::details {

seed_finding_alt::seed_finding_alt(const seedfinder_config& config,
                                   const seedfilter_config& filter_config,
                                   const traccc::memory_resource& mr,
                                   vecmem::copy& copy, stream& str,
                                   std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_seedfinder_config(config),
      m_seedfilter_config(filter_config),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

edm::seed_collection::buffer seed_finding_alt::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const traccc::details::spacepoint_grid_types::const_view& g2_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the sizes from the grid view
    auto grid_sizes = m_copy.get_sizes(g2_view._data_view);

    const auto num_spacepoints = m_copy.get_size(spacepoints_view);

    TRACCC_INFO("Running seed finding on " << num_spacepoints
                                           << " spacepoints");

    if (num_spacepoints == 0) {
        return {0, m_mr.main};
    }

    const unsigned int max_doublets_per_sp = 5000u;
    const unsigned int max_triplets_per_sp = 5000u;

    // Set up memory for the doublets
    vecmem::unique_alloc_ptr<long int[]> device_doublets =
        vecmem::make_unique_alloc<long int[]>(
            m_mr.main, max_doublets_per_sp * num_spacepoints);

    // Doublet finding
    {
        // Create prefix sum buffer
        vecmem::data::vector_buffer sp_grid_prefix_sum_buff =
            make_prefix_sum_buff(grid_sizes, m_copy, m_mr, m_stream);

        vecmem::unique_alloc_ptr<unsigned long long> num_doublets =
            vecmem::make_unique_alloc<unsigned long long>(m_mr.main);

        TRACCC_CUDA_ERROR_CHECK(
            cudaMemset(num_doublets.get(), 0, sizeof(unsigned long long)));

        const unsigned int num_threads = 512;
        const unsigned int num_blocks =
            (num_spacepoints + num_threads - 1) / num_threads;

        find_doublets<<<num_blocks, num_threads>>>(
            m_seedfinder_config, spacepoints_view, g2_view,
            sp_grid_prefix_sum_buff, device_doublets.get(), max_doublets_per_sp,
            num_doublets.get());

        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        TRACCC_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        unsigned long long num_doublets_host;

        TRACCC_CUDA_ERROR_CHECK(
            cudaMemcpy(&num_doublets_host, num_doublets.get(),
                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        std::cout << "Num doublets is " << num_doublets_host << std::endl;
    }

    // Doublet pruning
    {
        vecmem::unique_alloc_ptr<long int[]> new_device_doublets =
            vecmem::make_unique_alloc<long int[]>(
                m_mr.main, max_doublets_per_sp * num_spacepoints);

        const unsigned int num_threads = 512;
        const unsigned int num_blocks = num_spacepoints;

        prune_doublets<<<num_blocks, num_threads,
                         max_doublets_per_sp * sizeof(unsigned int)>>>(
            device_doublets.get(), new_device_doublets.get(),
            max_doublets_per_sp, spacepoints_view);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        device_doublets = std::move(new_device_doublets);
    }

    // Set up the triplet buffer.
    vecmem::unique_alloc_ptr<device_triplet_alt[]> device_triplets =
        vecmem::make_unique_alloc<device_triplet_alt[]>(
            m_mr.main, max_doublets_per_sp * num_spacepoints);
    vecmem::unique_alloc_ptr<unsigned int> device_num_triplets =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemsetAsync(device_num_triplets.get(), 0, sizeof(unsigned int)));

    // Triplet finding
    {
        const unsigned int num_threads = 512;
        const unsigned int num_blocks = num_spacepoints;

        find_triplets<<<num_blocks, num_threads>>>(
            m_seedfinder_config, m_seedfilter_config, spacepoints_view,
            device_doublets.get(), device_triplets.get(),
            device_num_triplets.get(), max_doublets_per_sp,
            max_triplets_per_sp);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    unsigned int num_triplets = 0;

    TRACCC_CUDA_ERROR_CHECK(cudaMemcpy(&num_triplets, device_num_triplets.get(),
                                       sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost));

    edm::seed_collection::buffer seed_buffer(
        num_triplets, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.setup(seed_buffer)->ignore();

    // Seed selection
    {
        const unsigned int num_threads = 512;
        const unsigned int num_blocks =
            (num_spacepoints * max_triplets_per_sp - 1) / num_threads;

        select_seeds<<<num_blocks, num_threads>>>(
            m_seedfilter_config, spacepoints_view, device_triplets.get(),
            max_triplets_per_sp, seed_buffer);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    TRACCC_INFO("Produced a total of " << num_triplets << " triplets");

    return seed_buffer;
}
}  // namespace traccc::cuda::details
