/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/measurement.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"

namespace traccc::cuda::kernels {

__global__ void make_barcode_sequence(
    device::make_barcode_sequence_payload payload);
}