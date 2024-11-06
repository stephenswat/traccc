/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include <limits>

#include "detray/core/detail/tuple_container.hpp"
#include "detray/propagator/constrained_step.hpp"
#include "detray/utils/tuple.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/utils/particle.hpp"

namespace traccc::device {

template <device::concepts::thread_id1 thread_id_t,
          device::concepts::barrier barrier_t, typename propagator_t,
          typename bfield_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    const thread_id_t& thread_id, barrier_t& barrier, const config_t cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    unsigned int* queue_index, unsigned int* queue_size) {
    using actor_list_type =
        typename propagator_t::actor_chain_type::actor_list_type;

    struct actor_chain {
        typename detray::detail::tuple_element<0, actor_list_type>::type::state
            s0;
        typename detray::detail::tuple_element<1, actor_list_type>::type::state
            s1;
        typename detray::detail::tuple_element<2, actor_list_type>::type::state
            s2;
        typename detray::detail::tuple_element<3, actor_list_type>::type::state
            s3;
        typename detray::detail::tuple_element<4, actor_list_type>::type::state
            s4;

        TRACCC_DEVICE actor_chain(const config_t& _cfg)
            : s0{}, s1{}, s3{}, s2{s3}, s4{} {
            s4.min_step_length = _cfg.min_step_length_for_next_surface;
            s4.max_count = _cfg.max_step_counts_for_next_surface;

        }
    };

    // Create propagator
    propagator_t propagator(cfg.propagation);

    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);
    vecmem::device_vector<unsigned int> n_tracks_per_seed(
        payload.n_tracks_per_seed_view);
    vecmem::device_vector<const candidate_link> links(payload.links_view);
    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        payload.tips_view);
    typename propagator_t::detector_type det(payload.det_data);
    bound_track_parameters_collection_types::device params(payload.params_view);
    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);
    vecmem::device_vector<typename propagator_t::state> state_scratch(
        payload.state_scratch_view);

    const unsigned int block_begin =
        thread_id.getBlockIdX() * thread_id.getBlockDimX() * payload.coarsening;
    const unsigned int block_end =
        std::min(static_cast<unsigned int>((thread_id.getBlockIdX() + 1u) *
                                           thread_id.getBlockDimX() *
                                           payload.coarsening),
                 param_ids.size());
    const unsigned int block_size = block_end - block_begin;

    if (thread_id.getLocalThreadIdX() == 0) {
        *queue_index = 0;
        *queue_size = 0;
    }

    barrier.blockBarrier();

    vecmem::device_atomic_ref<unsigned int> queue_size_atomic(*queue_size);

    for (unsigned int i = thread_id.getLocalThreadIdX(); i < block_size;
         i += thread_id.getBlockDimX()) {
        unsigned int param_id = param_ids.at(block_begin + i);

        if (const unsigned int s_pos =
                vecmem::device_atomic_ref<unsigned int>(
                    n_tracks_per_seed.at(links.at(param_id).seed_idx))
                    .fetch_add(1);
            s_pos >= cfg.max_num_branches_per_seed) {
            params_liveness[param_id] = 0u;
            continue;
        }

        if (links.at(param_id).n_skipped > cfg.max_num_skipping_per_cand) {
            params_liveness[param_id] = 0u;
            tips.push_back({payload.step, param_id});
            continue;
        }

        if (params_liveness.at(param_id) == 0u) {
            continue;
        }

        const bound_track_parameters& par = params.at(param_id);

        unsigned int pos = queue_size_atomic.fetch_add(1);

        new (&state_scratch.at(pos))
            typename propagator_t::state(par, payload.field_data, det);

        // @TODO: Should be removed once detray is fixed to set the volume
        // in the constructor
        state_scratch.at(pos)._navigation.set_volume(
            par.surface_link().volume());
        state_scratch.at(pos).set_particle(
            detail::correct_particle_hypothesis(cfg.ptc_hypothesis, par));
        state_scratch.at(pos)
            ._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                cfg.propagation.stepping.step_constraint);
    }

    barrier.blockBarrier();

    vecmem::device_atomic_ref<unsigned int> queue_index_atomic(*queue_index);
    std::optional<actor_chain> actor_chain = std::nullopt;

    unsigned int param_id;
    bool is_init = false;
    unsigned int thread_curr_idx = std::numeric_limits<unsigned int>::max();

    while (barrier.blockOr(thread_curr_idx !=
                           std::numeric_limits<unsigned int>::max()) ||
           *queue_index < *queue_size) {

        if (thread_curr_idx == std::numeric_limits<unsigned int>::max()) {
            thread_curr_idx = queue_index_atomic.fetch_add(1);

            if (thread_curr_idx < block_size) {
                actor_chain.emplace(cfg);
                propagator.propagate_init(
                    state_scratch[block_begin + thread_curr_idx],
                    detray::tie(actor_chain->s0, actor_chain->s1,
                                actor_chain->s2, actor_chain->s3,
                                actor_chain->s4));
                is_init = true;
            } else {
                thread_curr_idx = std::numeric_limits<unsigned int>::max();
            }
        }

        barrier.blockBarrier();

        if (thread_curr_idx != std::numeric_limits<unsigned int>::max()) {
            typename propagator_t::state& state =
                state_scratch[block_begin + thread_curr_idx];

            if (state.is_alive()) {
                is_init = propagator.propagate_step(
                    state, is_init,
                    detray::tie(actor_chain->s0, actor_chain->s1,
                                actor_chain->s2, actor_chain->s3,
                                actor_chain->s4));
            }
        }

        barrier.blockBarrier();

        if (thread_curr_idx != std::numeric_limits<unsigned int>::max()) {
            typename propagator_t::state& state =
                state_scratch[block_begin + thread_curr_idx];

            if (!state.is_alive()) {
                // If a surface found, add the parameter for the next step
                if (actor_chain->s4.success) {
                    params[param_id] = state._stepping.bound_params();

                    if (payload.step ==
                        cfg.max_track_candidates_per_track - 1) {
                        tips.push_back({payload.step, param_id});
                        params_liveness[param_id] = 0u;
                    } else {
                        params_liveness[param_id] = 1u;
                    }
                } else {
                    params_liveness[param_id] = 0u;

                    if (payload.step >=
                        cfg.min_track_candidates_per_track - 1) {
                        tips.push_back({payload.step, param_id});
                    }
                }

                actor_chain.reset();
                thread_curr_idx = std::numeric_limits<unsigned int>::max();
            }
        }
    }
}

}  // namespace traccc::device
