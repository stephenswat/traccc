/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename propagator_t, typename bfield_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    std::size_t globalIndex, const config_t cfg,
    typename propagator_t::detector_type::view_type det_data,
    bfield_t field_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    vecmem::data::vector_view<const typename propagator_t::state> in_prop_state_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int& n_in_params,
    vecmem::data::vector_view<typename propagator_t::state> out_prop_state_view,
    vecmem::data::vector_view<unsigned int> param_to_link_view,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view,
    unsigned int& n_out_params) {

    if (globalIndex >= n_in_params) {
        return;
    }

    // Number of tracks per seed
    vecmem::device_vector<unsigned int> n_tracks_per_seed(
        n_tracks_per_seed_view);

    // Links
    vecmem::device_vector<const candidate_link> links(links_view);

    // Seed id
    unsigned int orig_param_id = links.at(globalIndex).seed_idx;

    // Count the number of tracks per seed
    vecmem::device_atomic_ref<unsigned int> num_tracks_per_seed(
        n_tracks_per_seed.at(orig_param_id));

    const unsigned int s_pos = num_tracks_per_seed.fetch_add(1);

    // tips
    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        tips_view);

    if (links[globalIndex].n_skipped > cfg.max_num_skipping_per_cand) {
        tips.push_back({step, globalIndex});
        return;
    }

    // Detector
    typename propagator_t::detector_type det(det_data);

    // Input parameters
    vecmem::device_vector<const typename propagator_t::state> in_prop_states(
        in_prop_state_view);

    // Out parameters
    vecmem::device_vector<typename propagator_t::state> out_prop_states(out_prop_state_view);

    // Param to Link ID
    vecmem::device_vector<unsigned int> param_to_link(param_to_link_view);

    // Create propagator
    propagator_t propagator(cfg.propagation);

    // Create propagator state
    typename propagator_t::state propagation = in_prop_states.at(globalIndex);

    // Input bound track parameter
    const bound_track_parameters in_par = propagation._stepping._bound_params;

    // Actor state
    // @TODO: simplify the syntax here
    // @NOTE: Post material interaction might be required here
    using actor_list_type =
        typename propagator_t::actor_chain_type::actor_list_type;
    typename detray::detail::tuple_element<0, actor_list_type>::type::state
        s0{};
    typename detray::detail::tuple_element<1, actor_list_type>::type::state
        s1{};
    typename detray::detail::tuple_element<3, actor_list_type>::type::state
        s3{};
    typename detray::detail::tuple_element<2, actor_list_type>::type::state s2{
        s3};
    typename detray::detail::tuple_element<4, actor_list_type>::type::state s4;
    s4.min_step_length = cfg.min_step_length_for_next_surface;
    s4.max_count = cfg.max_step_counts_for_next_surface;

    // @TODO: Should be removed once detray is fixed to set the volume in the
    // constructor
    // propagation._navigation.set_volume(in_par.surface_link().volume());

    // Propagate to the next surface
    propagator.propagate_sync(propagation, std::tie(s0, s1, s2, s3, s4));

    // If a surface found, add the parameter for the next step
    if (s4.success) {
        vecmem::device_atomic_ref<unsigned int> num_out_params(n_out_params);
        const unsigned int out_param_id = num_out_params.fetch_add(1);

        memcpy(&out_prop_states[out_param_id], &propagation, sizeof(typename propagator_t::state));

        param_to_link[out_param_id] = static_cast<unsigned int>(globalIndex);
    }
    // Unless the track found a surface, it is considered a tip
    else if (!s4.success && step >= cfg.min_track_candidates_per_track - 1) {
        tips.push_back({step, globalIndex});
    }

    // If no more CKF step is expected, current candidate is
    // kept as a tip
    if (s4.success && step == cfg.max_track_candidates_per_track - 1) {
        tips.push_back({step, globalIndex});
    }
}

}  // namespace traccc::device
