/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/containers/vector.hpp"

#include "cuda/algorithms/component_connection.hpp"

#include "details/sparse_ccl.cuh"

#include <iostream>

namespace traccc::cuda {
    vecmem::vector<details::ccl_partition>
    partition1(
        const host_cell_container & data,
        vecmem::memory_resource & mem
    ) {
        vecmem::vector<details::ccl_partition> partitions(&mem);
        std::size_t index = 0;

        for (std::size_t i = 0; i < data.size(); ++i) {
            partitions.push_back(details::ccl_partition{
                .start = index,
                .size = data.at(i).items.size()
            });

            index += data.at(i).items.size();
        }

        return partitions;
    }

    vecmem::vector<details::ccl_partition>
    partition2(
        const host_cell_container & data,
        vecmem::memory_resource & mem
    ) {
        vecmem::vector<details::ccl_partition> partitions(&mem);
        std::size_t index = 0;

        for (std::size_t i = 0; i < data.size(); ++i) {
            std::size_t size = 0;
            bool first_cell = true;
            channel_id last_mid = 0;

            for (const cell & c : data.at(i).items) {
                if (!first_cell && c.channel1 > last_mid + 1 && size >= 2 * THREADS_PER_BLOCK) {
                    partitions.push_back(details::ccl_partition{
                        .start = index,
                        .size = size
                    });

                    index += size;
                    size = 0;
                }

                first_cell = false;
                last_mid = c.channel1;
                size += 1;
            }

            if (size > 0) {
                partitions.push_back(details::ccl_partition{
                    .start = index,
                    .size = size
                });

                index += size;
            }
        }

        return partitions;
    }

    host_measurement_collection
    component_connection::operator()(
        const host_cell_container & data
    ) const {
        vecmem::cuda::managed_memory_resource upstream;
        vecmem::binary_page_memory_resource mem(upstream);

        std::size_t total_cells = 0;

        for (std::size_t i = 0; i < data.size(); ++i) {
            total_cells += data.at(i).items.size();
        }

        cell_container container;

        vecmem::vector<channel_id> channel0(&mem);
        channel0.reserve(total_cells);
        vecmem::vector<channel_id> channel1(&mem);
        channel1.reserve(total_cells);
        vecmem::vector<scalar> activation(&mem);
        activation.reserve(total_cells);
        vecmem::vector<scalar> time(&mem);
        time.reserve(total_cells);
        vecmem::vector<geometry_id> module_id(&mem);
        module_id.reserve(total_cells);

        for (std::size_t i = 0; i < data.size(); ++i) {
            for (std::size_t j = 0; j < data.at(i).items.size(); ++j) {
                channel0.push_back(data.at(i).items.at(j).channel0);
                channel1.push_back(data.at(i).items.at(j).channel1);
                activation.push_back(data.at(i).items.at(j).activation);
                time.push_back(data.at(i).items.at(j).time);
                module_id.push_back(data.at(i).header.module);
            }
        }

        container.size = total_cells;
        container.channel0 = channel0.data();
        container.channel1 = channel1.data();
        container.activation = activation.data();
        container.time = time.data();
        container.module_id = module_id.data();

        vecmem::vector<details::ccl_partition> partitions = partition2(
            data,
            mem
        );

        std::cout << "We have " << partitions.size() << " partitions" << std::endl;

        measurement_container mctnr;

        vecmem::vector<scalar> mchannel0(&mem);
        mchannel0.reserve(MAX_CLUSTERS_PER_PARTITION * partitions.size());
        vecmem::vector<scalar> mchannel1(&mem);
        mchannel1.reserve(MAX_CLUSTERS_PER_PARTITION * partitions.size());
        vecmem::vector<geometry_id> mmodule_id(&mem);
        mmodule_id.reserve(MAX_CLUSTERS_PER_PARTITION * partitions.size());
        
        mctnr.channel0 = mchannel0.data();
        mctnr.channel1 = mchannel1.data();
        mctnr.module_id = mmodule_id.data();

        details::sparse_ccl(container, partitions, mctnr);

        return {};
    }
}
