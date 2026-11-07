import numpy as np
import pytest
import yaml
from sit_lmpc.config import load_config
from sit_lmpc.dynamics import step
from sit_lmpc.dynamics.st import PARAMETER_FIELDS
from sit_lmpc.experiments.runner import build_vehicle_dynamics, run


@pytest.mark.parametrize("actual,assumed", [(0.6, 0.8), (0.8, 0.6)])
def test_vehicle_parameter_sets_are_independent(actual, assumed):
    config = load_config(
        overrides={
            "simulation": {"friction": actual},
            "planner": {"friction": assumed},
        }
    )
    simulator, model = build_vehicle_dynamics(config)
    assert simulator["mu"] == actual
    assert float(model.params[PARAMETER_FIELDS.index("mu")]) == pytest.approx(assumed)
    simulator["mu"] = 0.3
    assert float(model.params[PARAMETER_FIELDS.index("mu")]) == pytest.approx(assumed)


def test_only_prediction_changes_when_planner_friction_changes():
    original = load_config()
    changed = load_config(overrides={"planner": {"friction": 0.8}})
    first_simulator, first_model = build_vehicle_dynamics(original)
    second_simulator, second_model = build_vehicle_dynamics(changed)
    assert first_simulator == second_simulator
    state = np.asarray([0, 0, 0.1, 8, 0, 0.15, 0.02])
    first = np.asarray(step(first_model, state, np.zeros(2)))
    second = np.asarray(step(second_model, state, np.zeros(2)))
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert np.max(np.abs(first - second)) > 1e-5


@pytest.mark.parametrize("section,settings", [
    ("simulation", {"map": "Spielberg"}),
    ("planner", {"half_width": 6.0}),
])
def test_incompatible_resume_is_rejected_before_output_or_compilation(tmp_path, section, settings):
    source = tmp_path / "saved"
    source.mkdir()
    (source / "config.yaml").write_text(
        yaml.safe_dump(load_config(overrides={"output": {"resume": None}}).as_dict())
    )
    (source / "manifest.json").write_text('{"schema": 2}')
    destination = tmp_path / "new"
    config = load_config(
        overrides={
            section: settings,
            "output": {"resume": str(source), "directory": str(destination), "quick_load": False},
        }
    )
    with pytest.raises(ValueError, match="resume requires the saved experiment configuration"):
        run(config)
    assert not destination.exists()
