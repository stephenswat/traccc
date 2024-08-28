/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template<typename propagator_t>
TRACCC_DEVICE inline void add_links_for_holes(
    std::size_t globalIndex,
    vecmem::data::vector_view<const unsigned int> n_candidates_view,
    vecmem::data::vector_view<const typename propagator_t::state> in_prop_state_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int& n_max_candidates,
    vecmem::data::vector_view<typename propagator_t::state> out_prop_state_view,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        out_nav_candidates_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_total_candidates) {

    // Number of candidates per parameter
    vecmem::device_vector<const unsigned int> n_candidates(n_candidates_view);

    if (globalIndex >= n_candidates.size()) {
        return;
    }

    vecmem::jagged_device_vector<typename propagator_t::intersection_type>
        out_nav_candidates(out_nav_candidates_view);

    // Input parameters
    vecmem::device_vector<const typename propagator_t::state> in_prop_states(
        in_prop_state_view);

    // Previous links
    vecmem::device_vector<const candidate_link> prev_links(prev_links_view);

    // Previous param_to_link
    vecmem::device_vector<const unsigned int> prev_param_to_link(
        prev_param_to_link_view);

    // Output parameters
    vecmem::device_vector<typename propagator_t::state> out_prop_states(out_prop_state_view);

    // Links
    vecmem::device_vector<candidate_link> links(links_view);

    // Last step ID
    const unsigned int previous_step =
        (step == 0) ? std::numeric_limits<unsigned int>::max() : step - 1;

    if (n_candidates[globalIndex] == 0u) {

        // Add measurement candidates to link
        vecmem::device_atomic_ref<unsigned int> num_total_candidates(
            n_total_candidates);

        const unsigned int l_pos = num_total_candidates.fetch_add(1);

        if (l_pos >= n_max_candidates) {

            n_total_candidates = n_max_candidates;
            return;
        }

        // Seed id
        unsigned int orig_param_id =
            (step == 0 ? globalIndex
                       : prev_links[prev_param_to_link[globalIndex]].seed_idx);
        // Skip counter
        unsigned int skip_counter =
            (step == 0 ? 0
                       : prev_links[prev_param_to_link[globalIndex]].n_skipped);

        // Add a dummy link
        links.at(l_pos) = {{previous_step, globalIndex},
                           std::numeric_limits<unsigned int>::max(),
                           orig_param_id,
                           skip_counter + 1};

        new (&out_prop_states.at(l_pos)) typename propagator_t::state (in_prop_states.at(globalIndex), out_nav_candidates.at(l_pos));
    }
}

}  // namespace traccc::device
