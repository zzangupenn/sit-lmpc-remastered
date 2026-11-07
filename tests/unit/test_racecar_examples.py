"""The published example must run without local research paths or missing weights."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from sit_lmpc.config import load_config
from sit_lmpc.data.trajectories import load_trajectory

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/racecar'


@pytest.mark.parametrize('arm', ['known-dynamics', 'bnn-unknown'])
def test_configs_and_recorded_outcomes_are_portable_and_consistent(arm):
    path = EXAMPLE/'configs'/f'{arm}.yaml'
    assert '/home/' not in path.read_text()
    config = load_config(path)
    assert config.simulation.max_laps == 300 and config.output.resume is None
    assert config.output.directory is None and config.planner.samples == 1024
    assert config.value.epochs == 300
    if arm == 'known-dynamics':
        assert config.planner.sit_lmpc_mode
        assert config.planner.friction == config.simulation.friction
        assert config.planner.mass_ratio == config.simulation.mass_ratio
    else:
        assert config.planner.probabilistic_safety.enabled and config.planner.dynamics.particles == 64
        assert config.planner.dynamics.belief_update == 'posterior'
        assert (ROOT/config.planner.dynamics.checkpoint/'model.msgpack').exists()
        assert config.planner.friction != config.simulation.friction
    with (EXAMPLE/'data'/f'{arm}-laps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300 and all(v['completed'] == 'True' for v in rows)
    metrics = json.loads((EXAMPLE/'metrics.json').read_text())['arms'][arm]
    assert min(float(v['lap_time']) for v in rows) == pytest.approx(metrics['best_completed_lap_seconds'])
    record = load_trajectory(EXAMPLE/metrics['best_lap_file'])
    assert record.outcome == 'success' and record.attempt_id == metrics['best_trial']
    assert np.isfinite(record.cartesian).all()


def test_bundled_population_and_exported_data_match_hashes():
    from dyna_dist.io.checkpoint import load_model
    provenance = json.loads((EXAMPLE/'provenance.json').read_text())
    for relative, checksum in provenance['saved_files_sha256'].items():
        assert hashlib.sha256((EXAMPLE/relative).read_bytes()).hexdigest() == checksum
    model = load_model(EXAMPLE/'models/vehicle-bnn-v2-l96')
    assert model.fingerprint == provenance['model_fingerprint']
    assert model.config.latent_dimension == 96 and model.config.domain == 'st'


def test_example_resume_uses_bundled_population_and_explicit_settings(tmp_path):
    spec = importlib.util.spec_from_file_location('racecar_example', EXAMPLE/'run_example.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefix = module.phase_config('bnn-unknown', tmp_path/'prefix', 10)
    continued = module.phase_config('bnn-unknown', tmp_path/'continued', 300,
                                  resume=tmp_path/'prefix/checkpoints/trial-000010')
    assert prefix['simulation']['max_laps'] == 10 and prefix['output']['resume'] is None
    assert continued['simulation']['max_laps'] == 300
    assert continued['planner']['dynamics']['checkpoint'] is None
    assert continued['output']['resume'].endswith('trial-000010')
    assert prefix['value'] == continued['value']
