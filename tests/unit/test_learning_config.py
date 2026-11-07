import pytest
from sit_lmpc.config import load_config


def test_learning_input_defaults_and_independent_modes(tmp_path):
    # Test schema defaults independently of the user's editable experiment preset.
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    config = load_config(path)
    assert config.value.input_mode == config.risk.input_mode == "state"
    config = load_config(path, overrides={"risk": {"input_mode": "state_action"}})
    assert config.risk.input_mode == "state_action" and config.value.input_mode == "state"


@pytest.mark.parametrize(
    "section,changes",
    [
        ("risk", {"input_mode": "action"}),
        ("value", {"input_mode": "action"}),
        ("risk", {"enabled": "false"}),
        ("risk", {"buffer_size": 0}),
        ("risk", {"failure_fraction": 0}),
        ("risk", {"failure_fraction": 1}),
        ("risk", {"failure_fraction": float("nan")}),
        ("risk", {"validation_fraction": 0.5}),
        ("risk", {"target_safe_fraction": 0}),
        ("risk", {"target_safe_fraction": 1.1}),
        ("risk", {"cutoff_min_states": 0}),
        ("risk", {"cutoff_min_episodes": 0}),
        ("risk", {"batch_size": 1}),
        ("risk", {"epochs": 0}),
    ],
)
def test_invalid_learning_configuration(section, changes):
    with pytest.raises(ValueError):
        load_config(overrides={section: changes})


def test_retired_continuation_action_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("value:\n  continuation_action: [0, 0]\n")
    with pytest.raises(ValueError, match="continuation_action"):
        load_config(path)
