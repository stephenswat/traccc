/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

namespace traccc::device {

template <typename propagator_t, typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, const finding_config& cfg,
    typename detector_t::view_type det_data, const int n_params,
    vecmem::data::vector_view<typename propagator_t::state> prop_state_view) {

    // Type definitions
    using algebra_type = typename detector_t::algebra_type;
    using interactor_type = detray::pointwise_material_interactor<algebra_type>;

    // Detector
    detector_t det(det_data);

    // in param
    vecmem::device_vector<typename propagator_t::state> prop_states(prop_state_view);

    if (globalIndex >= n_params) {
        return;
    }

    auto& bound_param = prop_states.at(globalIndex)._stepping._bound_params;

    // Get intersection at surface
    const detray::tracking_surface sf{det, bound_param.surface_link()};
    const typename detector_t::geometry_context ctx{};

    // Apply interactor
    typename interactor_type::state interactor_state;
    interactor_type{}.update(
        ctx,
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, bound_param),
        bound_param, interactor_state,
        static_cast<int>(detray::navigation::direction::e_forward), sf);
}

}  // namespace traccc::device
