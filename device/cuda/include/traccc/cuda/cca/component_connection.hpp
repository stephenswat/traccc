/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/utils/algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::cuda {
struct component_connection : algorithm<host_measurement_container(
          const host_cell_container& cells)> {
    host_measurement_container operator()(
        const host_cell_container& cells) const;
};
}  // namespace traccc::cuda
