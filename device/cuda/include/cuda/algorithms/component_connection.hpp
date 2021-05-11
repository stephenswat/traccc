/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"

namespace traccc::cuda {
    struct cell_container {
        std::size_t size = 0;
        channel_id * channel0 = nullptr;
        channel_id * channel1 = nullptr;
        scalar * activation = nullptr;
        scalar * time = nullptr;
        geometry_id * module_id = nullptr;
    };

    struct measurement_container {
        std::size_t size = 0;
        scalar * channel0 = nullptr;
        scalar * channel1 = nullptr;
        geometry_id * module_id = nullptr;
    };

    struct component_connection {
        host_measurement_collection
        operator()(
            const host_cell_container & cells
        ) const;
    };
}
