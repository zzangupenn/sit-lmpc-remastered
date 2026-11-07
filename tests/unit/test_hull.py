import numpy as np
import pytest
from scipy.spatial import ConvexHull
from sit_lmpc.planners.safe_set import build_hull, hull_distance


def test_padding_preserves_all_hull_inequalities():
    rng = np.random.default_rng(12)
    points = rng.normal(size=(35, 5))
    original = ConvexHull(points).equations
    hull = build_hull(points, bucket=64)
    assert np.asarray(hull.mask).sum() == len(original)
    np.testing.assert_allclose(np.asarray(hull.equations)[: len(original)], original)
    queries = rng.normal(size=(100, 5)) * 3
    expected = np.maximum((queries @ original[:, :-1].T + original[:, -1]).max(axis=1), 0)
    np.testing.assert_allclose(hull_distance(queries, hull), expected, atol=1e-12)
    np.testing.assert_allclose(hull_distance(points, hull), 0, atol=1e-12)


def test_bucket_sizes_preserve_distances():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(20, 5))
    query = rng.normal(size=(7, 5))
    np.testing.assert_allclose(
        hull_distance(query, build_hull(points, 16)),
        hull_distance(query, build_hull(points, 128)),
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype,atol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_bounded_precompiled_evaluators_keep_every_facet(dtype, atol):
    from sit_lmpc.planners.safe_set import PrecompiledHullEvaluator

    rng = np.random.default_rng(11)
    points = rng.normal(size=(35, 5))
    hull = build_hull(points, bucket=10, dtype=dtype)
    scorer = PrecompiledHullEvaluator(7, 3, bucket=10, limit=40, dtype=dtype)
    queries = rng.normal(size=(7, 5)).astype(dtype)
    with pytest.raises(RuntimeError, match="Precompile"):
        scorer.evaluate(queries, hull)
    events = []
    scorer.precompile(lambda kind, **fields: events.append((kind, fields)))
    compiled = dict(scorer.compiled)
    assert len(compiled) == 6  # Three sizes, two query-batch sizes.
    assert scorer.sizes == (10, 20, 30)
    assert len(hull.mask) > 30  # Exercise overflow using existing bounded scorers.
    original = ConvexHull(points).equations.astype(dtype)
    for count in (7, 3):
        expected = np.maximum(
            (queries[:count] @ original[:, :-1].T + original[:, -1]).max(axis=1), 0
        )
        actual = scorer.evaluate(queries[:count], hull)
        np.testing.assert_allclose(actual, expected, atol=atol)
        assert actual.dtype == np.dtype(dtype)
    other = build_hull(rng.normal(size=(45, 5)), bucket=10, dtype=dtype)
    np.testing.assert_allclose(
        scorer.evaluate(queries, other), hull_distance(queries, other), atol=atol
    )
    scorer.precompile()
    assert scorer.compiled == compiled  # No new hull-size executables during updates.
    assert [event[0] for event in events] == ["hull_precompile_started", "hull_precompile_finished"]
    with pytest.raises(ValueError, match="query shape"):
        scorer.evaluate(queries[:2], hull)


def test_proposal_and_selection_do_not_recompile_for_new_hull_sizes():
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from sit_lmpc.config.schema import PlannerConfig
    from sit_lmpc.planners.lmppi import propose, select_candidate
    from sit_lmpc.planners.safe_set import PrecompiledHullEvaluator

    rng = np.random.default_rng(19)
    config = replace(PlannerConfig(), samples=7, horizon=2, lambda_count=2)
    scorer = PrecompiledHullEvaluator(7, 4, bucket=10, limit=40)
    scorer.precompile()
    previous = jnp.zeros((2, 2))
    da = jnp.asarray(rng.uniform(-0.2, 0.2, (7, 2, 2)))
    states = jnp.asarray(rng.normal(size=(7, 2, 7)))
    values = jnp.arange(14.0)
    lambdas = jnp.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.float64)
    candidate_states = jnp.asarray(rng.normal(size=(4, 2, 7)))
    counts = []
    padded_sizes = []
    for length in (12, 35):
        hull = build_hull(rng.normal(size=(length, 5)), bucket=10)
        padded_sizes.append(len(hull.mask))
        distances = scorer.evaluate(states[:, -1, :5], hull)
        jax.block_until_ready(propose(previous, da, states, values, distances, lambdas, config))
        candidate_distances = scorer.evaluate(candidate_states[:, -1, :5], hull)
        result = select_candidate(candidate_states, candidate_distances, 3.9, 0.03)
        jax.block_until_ready(result)
        counts.append((propose._cache_size(), select_candidate._cache_size()))
    assert padded_sizes[0] != padded_sizes[1]
    assert counts[0] == counts[1]
