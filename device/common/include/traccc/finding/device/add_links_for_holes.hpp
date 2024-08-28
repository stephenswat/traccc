/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {

/// Function to add a dummy link in case of a hole

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
    unsigned int& n_total_candidates);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/add_links_for_holes.ipp"
