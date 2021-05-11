#pragma once

#include "cuda/algorithms/component_connection.hpp"

#include "edm/cell.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/memory_resource.hpp"

#include <vector>

#define MAX_ACTIVATIONS_PER_PARTITION 2048
#define MAX_CLUSTERS_PER_PARTITION 128
#define THREADS_PER_BLOCK 256

namespace traccc::cuda::details {
    struct ccl_partition {
        std::size_t start;
        std::size_t size;
    };

    void
    sparse_ccl(
        const cell_container container,
        const vecmem::vector<ccl_partition> & partitions,
        const measurement_container out_ctnr
    );
}
