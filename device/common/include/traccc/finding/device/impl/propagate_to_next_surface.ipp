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
    const propagate_to_next_surface_shared_payload& shared) {
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
    vecmem::device_vector<actor_chain_state<propagator_t>> actor_state_scratch(
        payload.actor_state_scratch_view);

    const unsigned int block_begin =
        thread_id.getBlockIdX() * thread_id.getBlockDimX() * payload.coarsening;
    const unsigned int block_end =
        std::min(static_cast<unsigned int>((thread_id.getBlockIdX() + 1u) *
                                           thread_id.getBlockDimX() *
                                           payload.coarsening),
                 param_ids.size());
    const unsigned int block_size = block_end - block_begin;

    if (thread_id.getLocalThreadIdX() == 0) {
        *shared.queue_index = 0;
        *shared.queue_size = 0;
    }

    barrier.blockBarrier();

    vecmem::device_atomic_ref<unsigned int> queue_size_atomic(
        *shared.queue_size);

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

        new (&state_scratch.at(block_begin + pos))
            typename propagator_t::state(par, payload.field_data, det);
        new (&actor_state_scratch.at(block_begin + pos))
            actor_chain_state<propagator_t>();

        typename propagator_t::state& new_prop_state =
            state_scratch.at(block_begin + pos);
        actor_chain_state<propagator_t>& new_actor_state =
            actor_state_scratch.at(block_begin + pos);

        // @TODO: Should be removed once detray is fixed to set the volume
        // in the constructor
        new_prop_state._navigation.set_volume(par.surface_link().volume());
        new_prop_state.set_particle(
            detail::correct_particle_hypothesis(cfg.ptc_hypothesis, par));
        new_prop_state._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                cfg.propagation.stepping.step_constraint);

        new_actor_state.s4.min_step_length =
            cfg.min_step_length_for_next_surface;
        new_actor_state.s4.max_count = cfg.max_step_counts_for_next_surface;

        propagator.propagate_init(new_prop_state, new_actor_state.tie());

        shared.original_param_ids[pos] = param_id;
    }

    barrier.blockBarrier();

    vecmem::device_atomic_ref<unsigned int> queue_index_atomic(
        *shared.queue_index);

    unsigned int fixed_queue_size = *shared.queue_size;

    {
        actor_chain_state<propagator_t>* st_actor_chain = nullptr;
        typename propagator_t::state* st_prop = nullptr;
        bool is_init = false;

        while (st_actor_chain != nullptr ||
               *shared.queue_index < fixed_queue_size) {
            if (st_actor_chain == nullptr) {
                unsigned int thread_curr_idx = queue_index_atomic.fetch_add(1u);
                if (thread_curr_idx < fixed_queue_size) {
                    st_actor_chain =
                        &actor_state_scratch.at(block_begin + thread_curr_idx);
                    st_prop = &state_scratch.at(block_begin + thread_curr_idx);
                    is_init = true;
                }
            }

            if (st_actor_chain != nullptr) {
                if (st_prop->is_alive()) {
                    auto& navigation = st_prop->_navigation;
                    auto& stepping = st_prop->_stepping;
                    const auto& track = stepping();

                    // Set access to the volume material for the stepper
                    auto vol = navigation.get_volume();
                    stepping.set_volume_material(
                        vol.has_material()
                            ? vol.material_parameters(track.pos())
                            : nullptr);

                    // Break automatic step size scaling by the stepper when a
                    // surface was reached and whenever the navigation is
                    // (re-)initialized
                    const bool reset_stepsize{navigation.is_on_surface() ||
                                              is_init};
                    // Take the step
                    st_prop->_heartbeat &= propagator.m_stepper.step(
                        navigation(), stepping, propagator.m_cfg.stepping,
                        reset_stepsize);

                    // Reduce navigation trust level according to stepper update
                    typename propagator_t::stepper_type::policy_type{}(
                        stepping.policy_state(), *st_prop);

                    // Find next candidate
                    is_init = propagator.m_navigator.update(
                        track, navigation, propagator.m_cfg.navigation);
                    st_prop->_heartbeat &= navigation.is_alive();

                    typename propagator_t::actor_chain_type::state
                        actor_chain_states = st_actor_chain->tie();

                    // Run all registered actors/aborters after update
                    propagator.run_actors(actor_chain_states, *st_prop);

                    // And check the status
                    is_init |= propagator.m_navigator.update(
                        track, navigation, propagator.m_cfg.navigation);
                    st_prop->_heartbeat &= navigation.is_alive();
                }

                if (!st_prop->is_alive()) {
                    st_actor_chain = nullptr;
                }
            }
        }
    }

    barrier.blockBarrier();

    for (unsigned int i = thread_id.getLocalThreadIdX(); i < *shared.queue_size;
         i += thread_id.getBlockDimX()) {
        auto& prop_state = state_scratch.at(block_begin + i);
        auto& actor_state = actor_state_scratch.at(block_begin + i);
        auto param_id = shared.original_param_ids[i];

        assert(!prop_state.is_alive());

        // If a surface found, add the parameter for the next step
        if (actor_state.s4.success) {
            params[param_id] = prop_state._stepping.bound_params();

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
    }
}
}  // namespace traccc::device
