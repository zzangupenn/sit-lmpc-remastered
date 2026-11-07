import json
import os
import subprocess
import sys
import textwrap
import pytest


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_runtime_precision_reaches_dynamics_hulls_and_values(tmp_path, dtype):
    # Isolate global JAX settings from the float64 numerical-regression fixtures.
    code = textwrap.dedent("""
        import json
        import sys
        import numpy as np
        from sit_lmpc.config import load_config
        from sit_lmpc.experiments.runtime import configure

        config = load_config(overrides={"simulation": {"dtype": sys.argv[1]}})
        configure(config, cpu=True)
        import jax
        import jax.numpy as jnp
        from sit_lmpc.adapters.f1tenth import vehicle_parameters
        from sit_lmpc.dynamics import STModel, step
        from sit_lmpc.planners.safe_set import build_hull, PrecompiledHullEvaluator
        from sit_lmpc.config.schema import ValueConfig
        from sit_lmpc.training.value import initial_state, fit_value, predict_value

        model = STModel.from_vehicle(vehicle_parameters(0.6), config.simulation)
        next_state = step(model, jnp.asarray([0, 0, 0, 7, 0, 0, 0]), jnp.zeros(2))
        rng = np.random.default_rng(3)
        hull = build_hull(rng.normal(size=(12, 5)), dtype=model.params.dtype)
        scorer = PrecompiledHullEvaluator(7, 3, limit=40, dtype=model.params.dtype)
        scorer.precompile()
        distances = scorer.evaluate(jnp.zeros((7, 5)), hull)
        value_config = ValueConfig(hidden_dims=8, layers=1, bins=4, batch_size=8,
                                   epochs=1, boundary_samples=0, samples=2)
        state = initial_state(value_config, jax.random.key(0))
        snapshot, losses = fit_value(state, rng.normal(size=(16, 6)),
                                     value_config, 4, 0, 1)
        values = predict_value(snapshot, jnp.zeros((7, 7)), jax.random.key(1), 2)
        assert np.isfinite(next_state).all() and np.isfinite(distances).all()
        assert np.isfinite(values).all() and np.isfinite(losses).all()
        print(json.dumps({
            "x64": jax.config.jax_enable_x64,
            "dtypes": [str(array.dtype) for array in (
                jnp.ones(1), model.params, model.action_scale, next_state,
                hull.equations, distances, snapshot.normalization.low, values)]
        }))
    """)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    # Explicit configuration must win over a conflicting inherited environment.
    environment["JAX_ENABLE_X64"] = "true" if dtype == "float32" else "false"
    result = subprocess.run(
        [sys.executable, "-c", code, dtype],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    record = json.loads(result.stdout)
    assert record["x64"] is (dtype == "float64")
    assert record["dtypes"] == [dtype] * 8
