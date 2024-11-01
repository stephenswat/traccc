/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
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
    unsigned int* queue_index) {
    using actor_list_type =
        typename propagator_t::actor_chain_type::actor_list_type;

    struct full_propagator_state {
        typename propagator_t::state state;

        typename detray::detail::tuple_element<0, actor_list_type>::type::state
            s0;
        typename detray::detail::tuple_element<1, actor_list_type>::type::state
            s1;
        typename detray::detail::tuple_element<3, actor_list_type>::type::state
            s3;
        typename detray::detail::tuple_element<2, actor_list_type>::type::state
            s2;
        typename detray::detail::tuple_element<4, actor_list_type>::type::state
            s4;

        TRACCC_DEVICE full_propagator_state(
            const config_t& _cfg, const bound_track_parameters& in_par,
            const bfield_t& field,
            const typename propagator_t::detector_type& det)
            : state(in_par, field, det), s0{}, s1{}, s3{}, s2{s3}, s4{} {
            s4.min_step_length = _cfg.min_step_length_for_next_surface;
            s4.max_count = _cfg.max_step_counts_for_next_surface;
            // @TODO: Should be removed once detray is fixed to set the volume
            // in the constructor
            state._navigation.set_volume(in_par.surface_link().volume());
            state.set_particle(detail::correct_particle_hypothesis(
                _cfg.ptc_hypothesis, in_par));
            state._stepping
                .template set_constraint<detray::step::constraint::e_accuracy>(
                    _cfg.propagation.stepping.step_constraint);
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
    }

    barrier.blockBarrier();

    vecmem::device_atomic_ref<unsigned int> queue_index_atomic(*queue_index);
    std::optional<full_propagator_state> prop_state = std::nullopt;

    unsigned int param_id;
    bool is_init = false;

    while (barrier.blockOr(prop_state.has_value()) ||
           *queue_index < block_size) {
        while (!prop_state && *queue_index < block_size) {
            unsigned int idx = queue_index_atomic.fetch_add(1);

            if (idx >= block_size) {
                break;
            }

            param_id = param_ids.at(block_begin + idx);

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

            prop_state.emplace(cfg, params.at(param_id), payload.field_data,
                               det);
            propagator.propagate_init(prop_state->state, detray::tie(prop_state->s0, prop_state->s1, prop_state->s2,
                            prop_state->s3, prop_state->s4));
            is_init = true;
        }

        __syncthreads();

        if (prop_state && prop_state->state.is_alive()) {
            is_init = propagator.propagate_step(prop_state->state, is_init,
                                        detray::tie(prop_state->s0, prop_state->s1, prop_state->s2,
                        prop_state->s3, prop_state->s4));
        }

        __syncthreads();

        if (prop_state && !prop_state->state.is_alive()) {
            // If a surface found, add the parameter for the next step
            if (prop_state->s4.success) {
                params[param_id] = prop_state->state._stepping.bound_params();

                if (payload.step == cfg.max_track_candidates_per_track - 1) {
                    tips.push_back({payload.step, param_id});
                    params_liveness[param_id] = 0u;
                } else {
                    params_liveness[param_id] = 1u;
                }
            } else {
                params_liveness[param_id] = 0u;

                if (payload.step >= cfg.min_track_candidates_per_track - 1) {
                    tips.push_back({payload.step, param_id});
                }
            }

            prop_state.reset();
        }
    }
}

}  // namespace traccc::device
