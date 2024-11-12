/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../../utils/barrier.hpp"
#include "traccc/cuda/utils/thread_id.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    cuda::thread_id1 thread_id;
    cuda::barrier barrier;

    __shared__ unsigned int queue_index, queue_size;
    extern __shared__ unsigned int original_param_ids[];

    device::propagate_to_next_surface<cuda::thread_id1, cuda::barrier,
                                      propagator_t, bfield_t, finding_config>(
        thread_id, barrier, cfg, payload,
        {&queue_index, &queue_size, original_param_ids});
}

}  // namespace traccc::cuda::kernels
