"""Geometric chance events and bounded, active-only constraint proposals."""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


class SafetyStop(RuntimeError):
    """A deliberate stop before applying an unverified control, not a failed lap."""

    def __init__(self, reason, **metrics):
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics


def candidate_count(config, chance_active, risk_active):
    limits = [config.lambda_count**2]
    if chance_active:
        limits.append(config.probabilistic_safety.max_candidates)
    if risk_active:
        limits.append(config.risk_constraint.max_candidates)
    return min(limits)


def configured_candidate_counts(config):
    chance, risk = config.probabilistic_safety.enabled, config.risk_constraint.enabled
    return tuple(
        dict.fromkeys(
            candidate_count(config, c, r)
            for c, r in ((False, False), (chance, False), (False, risk), (chance, risk))
        )
    )


def lambda_table(base, config, chance_active, risk_active):
    """A deterministic bounded design, not an exponential Cartesian grid.

    Risk-only retains Stage 4's sampled combinations. Joint chance axes use deterministic
    radical-inverse coordinates, including an all-zero proposal; no external
    random source, repeated lockstep cycles, or inactive singleton axes.
    Sort the COMPLETE active vector so first-minimum selection retains SIT's
    smaller-lambda preference, including chance and risk penalties.
    """
    if not chance_active and not risk_active:
        return base
    count = candidate_count(config, chance_active, risk_active)
    selected = np.linspace(0, len(base) - 1, count, dtype=int)
    columns = [np.asarray(base)[selected]]
    ordinal = np.arange(count)
    settings = []
    if chance_active:
        settings += [(config.probabilistic_safety, 2), (config.probabilistic_safety, 3)]
    if risk_active:
        settings += [(config.risk_constraint, 5)]
    for setting, radix in settings:
        if not chance_active:
            indices = ordinal % setting.lambda_count
            if count < setting.lambda_count:
                indices = np.floor(ordinal * setting.lambda_count / count).astype(int)
        else:
            coordinates = []
            for number in ordinal:
                value, denominator = 0.0, 1.0
                while number:
                    number, remainder = divmod(number, radix)
                    denominator *= radix
                    value += remainder / denominator
                coordinates.append(value)
            indices = np.floor(np.asarray(coordinates) * setting.lambda_count).astype(int)
        levels = indices * setting.lambda_max / setting.lambda_count
        levels[0] = 0
        columns.append(levels[:, None])
    table = np.column_stack(columns)
    order = np.argsort(np.linalg.norm(table, axis=1), kind="stable")
    return jnp.asarray(table[order], dtype=base.dtype)


@jax.jit
def distribution_availability(frenet, valid, probabilities):
    positive = probabilities > 0
    finite = jnp.all(jnp.isfinite(frenet), axis=-1)
    support_valid = jnp.all(~positive[None, :, None] | ~valid | finite, axis=(1, 2))
    # A stopped scenario cannot resume or borrow a different trajectory's state.
    support_valid &= ~jnp.any(
        positive[None, :, None] & ~valid[:, :, :-1] & valid[:, :, 1:], axis=(1, 2)
    )
    weights_valid = (
        jnp.all(jnp.isfinite(probabilities))
        & jnp.all(probabilities >= 0)
        & jnp.isclose(jnp.sum(probabilities), 1, atol=1e-6)
    )
    return support_valid & weights_valid


@partial(jax.jit, static_argnames=("boundary_limit", "relaxation"))
def chance_probabilities(
    frenet, valid, probabilities, hull_distances, error_bound, boundary_limit, relaxation
):
    """Count each event once per complete scenario; never renormalize support.

    Missing steps/endpoints retain their mass conservatively as violations.
    Nonfinite declared-valid support makes verification unavailable, rather than
    being ignored by comparisons against NaN. Zero-mass branches cannot poison
    a finite belief. Upper probabilities include a provider-declared envelope.
    An explicit empirical gate uses zero added margin as a computational alias
    for the raw fraction, not a confidence bound; host diagnostics must say so.
    """
    positive = probabilities > 0
    support_valid = distribution_availability(frenet, valid, probabilities)
    support_valid &= jnp.all(
        ~positive[None, :] | ~valid[:, :, -1] | jnp.isfinite(hull_distances), axis=1
    )
    error_valid = jnp.isfinite(error_bound) & (error_bound >= 0) & (error_bound <= 1)
    boundary_events = jnp.any(~valid | (jnp.abs(frenet[..., 1]) > boundary_limit), axis=-1)
    hull_events = ~valid[:, :, -1] | (hull_distances > relaxation)
    boundary = jnp.sum(jnp.where(positive[None] & boundary_events, probabilities[None], 0), axis=1)
    hull = jnp.sum(jnp.where(positive[None] & hull_events, probabilities[None], 0), axis=1)
    return {
        "boundary": boundary,
        "terminal_hull": hull,
        "boundary_upper": jnp.minimum(boundary + error_bound, 1),
        "terminal_hull_upper": jnp.minimum(hull + error_bound, 1),
        "available": support_valid & error_valid,
    }


@partial(jax.jit, static_argnames=("mode", "boundary_limit", "relaxation"))
def select_constrained_candidate(
    frenet,
    hull_distances,
    chance,
    risk,
    cutoff,
    boundary_threshold,
    hull_threshold,
    mode,
    boundary_limit,
    relaxation,
    terminal_values=None,
):
    """Gate candidates, falling back to SIT-LMPC's least geometric violation.

    Fallback minimizes 1000 * boundary violation + hull violation on the belief
    prediction. It does not certify chance/risk feasibility or simply choose the
    largest lambdas. Candidates arrive in full-lambda norm order; equal geometric
    scores keep the first candidate in BOTH feasible selection and fallback.
    Terminal values can invalidate a nonfinite prediction, but cannot override
    this smaller-lambda preference.
    """
    boundary = jnp.abs(frenet[:, :, 1])
    boundary_sum = jnp.where(boundary > boundary_limit, boundary, 0).sum(axis=1)
    hull_violation = jnp.where(hull_distances > relaxation, hull_distances, 0)
    score = boundary_sum * 1000 + hull_violation
    valid_candidate = jnp.all(jnp.isfinite(frenet), axis=(1, 2)) & jnp.isfinite(hull_distances)
    if terminal_values is not None:
        valid_candidate &= jnp.isfinite(terminal_values)
    score = jnp.where(valid_candidate, score, jnp.inf)
    feasible = valid_candidate
    if chance is not None:
        feasible &= (
            chance["available"]
            & (chance["boundary_upper"] <= boundary_threshold)
            & (chance["terminal_hull_upper"] <= hull_threshold)
        )
    risk_score = None
    if risk is not None:
        risk_score = jnp.max(risk, axis=1) if mode == "all" else risk[:, -1]
        feasible &= jnp.isfinite(risk_score) & (risk_score <= cutoff)
    gated = jnp.where(feasible, score, jnp.inf)
    index = jnp.argmin(gated)
    fallback = ~jnp.any(feasible)
    index = jnp.where(fallback, jnp.argmin(score), index)
    return index, hull_violation[index], boundary_sum[index], feasible.sum(), fallback, risk_score
