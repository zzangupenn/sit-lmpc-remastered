"""On-demand, counterfactual hull scoring without changing control or RNGs."""

from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from sit_lmpc.planners.common import unit_range
from sit_lmpc.planners.safe_set import build_hull, hull_distance


@jax.jit
def facet_cost(states, hull, relaxation, reward_limit, valid=None):
    if valid is None:
        valid = jnp.all(jnp.isfinite(states), axis=1)
    clean = jnp.where(valid[:, None], states, 0.)
    distances = hull_distance(clean, hull)
    excess = jnp.maximum(distances - relaxation, 0.0)
    return jnp.where(valid, distances, jnp.inf), unit_range(jnp.minimum(excess, reward_limit), valid)


def facet_selection(result, hull, distances, config, controller):
    """Reweight the frozen samples, reroll all proposals and apply SIT selection."""
    from sit_lmpc.planners.lmppi import candidate_predictions, propose, select_candidate
    from sit_lmpc.planners.common import to_frenet

    data = result.debug
    required = ("proposal_previous", "proposal_noise", "proposal_values", "proposal_lambdas",
                "planning_state", "planning_s", "planning_search_indices")
    if controller is None or any(name not in data for name in required):
        return dict(hull_selection_status="Hull selection unavailable: proposal context missing")
    if config.probabilistic_safety.enabled or config.risk_constraint.enabled:
        return dict(hull_selection_status="Hull selection unavailable with risk/chance gates enabled")
    facet_config = replace(config, safe_set_query="facets")
    dtype = controller.model.params.dtype
    previous, noise, values, lambdas, state, progress, indices = (
        jnp.asarray(data[name], dtype=dtype if name != "planning_search_indices" else None)
        for name in required
    )
    candidates = propose(previous, noise, jnp.asarray(data["states"], dtype), values,
                         jnp.asarray(distances, dtype), lambdas, facet_config)
    cartesian, _ = candidate_predictions(controller.model, state, candidates, facet_config)
    frenet = to_frenet(controller.track, cartesian, indices, progress, config.projection_block_size)
    candidate_distances = hull_distance(frenet[:, -1, :5], hull)
    index, violation, boundary = select_candidate(frenet, candidate_distances,
                                                 config.boundary_limit, config.safe_set_relaxation)
    packet = jax.device_get(dict(hull_candidate_states=frenet, hull_chosen_states=frenet[index],
                                hull_lambdas=lambdas[index], hull_actions=candidates[index],
                                hull_candidate_index=index, hull_violation=violation,
                                hull_boundary_violation=boundary))
    if not np.isfinite(packet["hull_chosen_states"]).all() or not np.isfinite(packet["hull_actions"]).all():
        return dict(hull_selection_status="No finite convex-hull candidate")
    return dict(hull_selection_status="ready", **packet)


def compare_safe_set_costs(result, safe_set, config, *, controller=None):
    """Score exactly this update's sampled endpoints, never new trajectories.

    The left cost is the actual pre-lambda reward term. The counterfactual uses
    the original local five-dimensional facet hull, tolerance and reward scaling.
    Cloning the host generator prevents debug selection from changing later
    policy samples/geometry. This function is called only for keyboard view 6.
    """
    if result.debug is None:
        raise ValueError("safe-set cost comparison requires learned-control samples")
    data = result.debug
    paths = np.asarray(data["states"])
    all_terminal = paths[:, -1, :5]
    physical = np.isfinite(paths).all(axis=(1, 2))
    current_all = np.asarray(data["hull"])[:, -1]
    shown = np.asarray(data.get("valid_samples", physical)) & physical & np.isfinite(current_all)
    if not shown.any():
        raise ValueError("no valid sampled terminal states or safe-set costs")
    terminal, current = all_terminal[shown], current_all[shown]
    method = result.metrics.get("safe_set_query_active", config.safe_set_query)
    if method == "nf_density":
        bounds = float(all_terminal[physical, 0].min()), float(all_terminal[physical, 0].max())
        selected, _ = safe_set.select_window(
            *safe_set.window(bounds, data["safe_set_selection_speed"]),
            rng=deepcopy(safe_set.rng),
        )
    else:
        # Facet/implicit/warm-up updates already chose their local points. Reuse
        # those exact points, including an accepted speculative window.
        selected = np.asarray(data["safe_set"])
    dtype = jnp.asarray(terminal).dtype
    hull = build_hull(selected[:, :5], config.hull_bucket, dtype=dtype)
    facet_valid = physical.copy()
    if "proposal_values" in data:
        facet_valid &= np.isfinite(np.asarray(data["proposal_values"]).reshape(len(paths), -1)).all(axis=1)
    if "proposal_noise" in data:
        facet_valid &= np.isfinite(data["proposal_noise"]).all(axis=(1, 2))
    distance, cost = jax.device_get(facet_cost(
        jnp.asarray(all_terminal, dtype), hull, config.safe_set_relaxation,
        config.safe_set_reward_limit, jnp.asarray(facet_valid),
    ))
    raw_current = np.asarray(data["sample_hull_distances"])
    current_inside = raw_current[shown] <= config.safe_set_threshold
    if method == "nf_density" and "sample_nll" in data:
        bound = data.get("density_relaxation_cutoff", data["density_cutoff"])
        current_inside = np.asarray(data["sample_nll"])[shown] <= bound
        if "density_window" in data:
            lo, hi = data["density_window"]
            current_inside &= (terminal[:, 0] > lo) & (terminal[:, 0] < hi)
    return dict(
        terminal_states=terminal,
        current_cost=current,
        hull_cost=np.asarray(cost)[shown],
        current_method=method,
        current_transform=result.metrics.get("safe_set_density_penalty_transform", config.safe_set_density.penalty_transform),
        current_calibration=result.metrics.get("safe_set_density_calibration"),
        current_cutoff=result.metrics.get("safe_set_density_cutoff"),
        current_inside=current_inside,
        hull_inside=np.asarray(distance)[shown] <= config.safe_set_relaxation,
        excluded_samples=int(np.sum(~shown)),
        safe_set=np.asarray(selected),
        candidate_states=np.asarray(data["candidate_states"]),
        chosen_states=np.asarray(data["chosen_states"]),
        current_lambdas=result.lambdas,
        lambda_terms=result.metrics.get("lambda_terms", ("hull", "boundary")),
        **facet_selection(result, hull, distance, config, controller),
    )
