"""The migrated defaults, preset composition and dependency boundary are intentional."""
from pathlib import Path
import importlib.util
import pytest
from sit_lmpc.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_default_is_known_dynamics_without_auxiliary_learners():
    c = load_config()
    assert c.planner.sit_lmpc_mode and c.planner.dynamics.backend == 'st'
    assert c.planner.friction == c.simulation.friction
    assert c.planner.mass_ratio == c.simulation.mass_ratio
    assert c.planner.safe_set_query == 'facets'
    assert not c.planner.belief.enabled
    assert not c.planner.probabilistic_safety.enabled
    assert not c.planner.risk_constraint.enabled and not c.risk.enabled
    assert load_config(ROOT / 'configs/default.yaml') == c


@pytest.mark.parametrize('name,chance', [('bnn-posterior', False), ('bnn-chance', True)])
def test_bnn_presets_inherit_common_settings(name, chance):
    base = load_config()
    c = load_config(ROOT / 'configs' / f'{name}.yaml')
    assert not c.planner.sit_lmpc_mode and not c.planner.belief.enabled
    assert c.planner.dynamics.backend == 'gaussian_bnn'
    assert c.planner.dynamics.belief_update == 'posterior'
    assert c.planner.dynamics.posterior_method == 'tempered'
    assert c.planner.probabilistic_safety.enabled == chance
    assert c.planner.safe_set_query == 'facets'
    assert c.planner.horizon == base.planner.horizon
    assert c.value == base.value and c.simulation == base.simulation
    assert not c.risk.enabled and not c.planner.risk_constraint.enabled


def test_relative_preset_inheritance_and_cli_override(tmp_path):
    (tmp_path / 'base.yaml').write_text('planner:\n  horizon: 7\n  samples: 32\nsimulation:\n  initial_speed: 8.0\n')
    child = tmp_path / 'child.yaml'
    child.write_text('extends: base.yaml\nplanner:\n  samples: 64\n')
    c = load_config(child, {'planner': {'samples': 128}})
    assert c.planner.horizon == 7 and c.planner.samples == 128
    assert c.simulation.initial_speed == 8


def test_extends_cycles_and_unknown_keys_are_rejected(tmp_path):
    a, b = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    a.write_text('extends: b.yaml\n'); b.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError, match='extends cycle'):
        load_config(a)
    a.write_text('extends: b.yaml\nunknown: {}\n'); b.write_text('{}\n')
    with pytest.raises(ValueError, match='configuration sections'):
        load_config(a)


def test_runtime_packages_do_not_resolve_to_sibling_checkouts():
    for name in ('sit_lmpc', 'dyna_dist', 'f1tenth_gym', 'utilsuite'):
        path = Path(importlib.util.find_spec(name).origin).resolve()
        # Vendored dependencies are built into this environment; the main package is editable.
        assert path.is_relative_to(ROOT), (name, path)
    assert not (ROOT / 'src/sit_lmpc/walker').exists()


def test_bnn_import_validates_and_preserves_checkpoint(tmp_path):
    import hashlib
    import jax
    import jax.numpy as jnp
    from dyna_dist.config import ModelConfig
    from dyna_dist.data.transforms import Normalizer
    from dyna_dist.inference.posterior import PopulationModel
    from dyna_dist.io.checkpoint import save_model
    from dyna_dist.models.features import initialize
    from sit_lmpc.cli.import_bnn import import_model

    config = ModelConfig(mode='dynamics', latent_dimension=4, hidden_sizes=(8,), dtype='float32')
    model = PopulationModel(initialize(config, jax.random.key(1)),
        Normalizer(jnp.zeros(6), jnp.ones(6), jnp.zeros(7), jnp.ones(7)), jnp.zeros(19), config)
    source = save_model(model, tmp_path / 'source')
    target = tmp_path / 'imported'
    record = import_model(source, target)
    assert record['latent_dimension'] == 4
    for name in ('manifest.json', 'model.msgpack'):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == record['sha256'][name]
        assert (source / name).read_bytes() == (target / name).read_bytes()
    with pytest.raises(FileExistsError):
        import_model(source, target)
