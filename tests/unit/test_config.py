from dataclasses import replace
import pytest
from sit_lmpc.config import load_config


def test_cli_overrides_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("simulation:\n  seed: 12\n  max_laps: 3\nplanner:\n  samples: 128\n")
    config = load_config(path, {"simulation": {"seed": 99}})
    assert config.simulation.seed == 99
    assert config.simulation.max_laps == 3
    assert config.planner.samples == 128


@pytest.mark.parametrize("settings,expected", [
    ({"insight_steps": 7}, 7),
    ({"insight_steps": 0}, 0),
    ({"trace_steps": 0, "insight_steps": 7}, 0),
    ({"trace_steps": 8, "insight_steps": 0}, 8),
])
def test_shared_history_limit_migrates_legacy_yaml_and_overrides(tmp_path, settings, expected):
    import yaml

    path = tmp_path / "old-config.yaml"
    path.write_text(yaml.safe_dump({"diagnostics": settings}))
    for config in (load_config(path), load_config(overrides={"diagnostics": settings})):
        assert config.diagnostics.trace_steps == expected
        assert "insight_steps" not in config.as_dict()["diagnostics"]
    # Each layer is normalized before merging, so an old-name override works.
    assert load_config(path, {"diagnostics": {"insight_steps": 3}}).diagnostics.trace_steps == 3
    assert load_config(path, {"diagnostics": {"trace_steps": 0}}).diagnostics.trace_steps == 0


def test_shared_history_default_and_nf_preset_use_one_setting():
    from pathlib import Path
    import yaml

    assert load_config().diagnostics.trace_steps == 50
    preset = Path(__file__).resolve().parents[2]/"tests/fixtures/presets/nf-safe-set.yaml"
    settings = yaml.safe_load(preset.read_text())["diagnostics"]
    assert load_config(preset).diagnostics.trace_steps == settings["trace_steps"]
    assert "insight_steps" not in settings


def test_nf_does_not_require_a_dynamics_calibration_holdout():
    from pathlib import Path

    config = load_config(Path(__file__).resolve().parents[2] / "tests/fixtures/presets/nf-safe-set.yaml")
    assert config.planner.safe_set_query == "nf_density"
    assert config.planner.dynamics.calibration_fraction == .1
    assert config.planner.dynamics.validation_fraction == 0.
    no_holdout = load_config(overrides={"planner": {"safe_set_query": "nf_density",
                "dynamics": {"calibration_fraction": 0., "validation_fraction": 0.}}})
    assert no_holdout.planner.safe_set_query == "nf_density"


def test_sequence_overrides_are_immutable_for_jax_static_arguments():
    config = load_config(overrides={
        "planner": {"sample_std": [0.3, 0.2], "model_bank": {"frictions": [0.4, 0.6]}},
        "simulation": {"action_scale": [0.8, 7.0], "control_noise": [0.01, 0.02]},
    })
    assert config.planner.sample_std == (0.3, 0.2)
    assert config.planner.model_bank.frictions == (0.4, 0.6)
    assert config.simulation.action_scale == (0.8, 7.0)
    assert config.simulation.control_noise == (0.01, 0.02)
    hash(config.planner)
    hash(config.simulation)


def test_no_path_loads_packaged_yaml_from_any_working_directory(tmp_path, monkeypatch):
    from importlib.resources import files

    expected = load_config(files("sit_lmpc") / "config" / "default.yaml")
    monkeypatch.chdir(tmp_path)
    assert load_config() == expected


def test_no_path_reads_current_default_yaml_and_applies_overrides(tmp_path, monkeypatch):
    package = tmp_path / "package"
    (package / "config").mkdir(parents=True)
    default = package / "config" / "default.yaml"
    default.write_text("simulation:\n  initial_speed: 11.0\n  max_laps: 9\n")
    monkeypatch.setattr("sit_lmpc.config.schema.files", lambda _: package)
    assert load_config().simulation.initial_speed == 11.0
    assert load_config(overrides={"simulation": {"max_laps": 3}}).simulation.max_laps == 3
    default.write_text("simulation:\n  initial_speed: 12.0\n")
    assert load_config().simulation.initial_speed == 12.0


def test_explicit_yaml_replaces_default_file_and_keeps_schema_fallbacks(tmp_path):
    from sit_lmpc.config.schema import SimulationConfig

    custom = tmp_path / "custom.yaml"
    custom.write_text("simulation:\n  max_laps: 2\n")
    config = load_config(custom)
    assert config.simulation.max_laps == 2
    assert config.simulation.initial_speed == SimulationConfig().initial_speed


@pytest.mark.parametrize(
    "section,settings",
    [
        ("simulation", {"model": "residual_kinematic_ST"}),
        ("simulation", {"timestep": 0.11}),
        ("simulation", {"max_laps": 0}),
        ("simulation", {"dtype": "int32"}),
        ("planner", {"sit_lmpc_mode": "true"}),
        ("planner", {"sample_std": (0, 1)}),
        ("value", {"model": "bnn"}),
        ("planner", {"hull_precompile_limit": 10}),
        ("planner", {"hull_precompile_limit": 0}),
        ("planner", {"hull_precompile": "yes"}),
        ("risk", {"cutoff_mode": "automatic"}),
        ("risk", {"manual_cutoff": -0.1}),
        ("risk", {"manual_cutoff": 1.1}),
        ("risk", {"manual_cutoff": float("nan")}),
        ("risk", {"manual_cutoff": True}),
    ],
)
def test_invalid_configuration_rejected(section, settings):
    with pytest.raises(ValueError):
        load_config(overrides={section: settings})


def test_configuration_is_frozen():
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        load_config().simulation.seed = 1


def test_manual_cutoff_preset_and_learned_mode_override():
    preset = load_config()
    assert preset.risk.cutoff_mode == "manual"
    assert preset.risk.manual_cutoff == 0.5
    assert load_config(overrides={"risk": {"cutoff_mode": "learned"}}).risk.cutoff_mode == "learned"
    assert load_config(overrides={"risk": {"manual_cutoff": 0.25}}).risk.manual_cutoff == 0.25


def test_sit_lmpc_schema_default_off_and_is_loaded_from_yaml(tmp_path):
    from sit_lmpc.config.schema import PlannerConfig

    assert PlannerConfig().sit_lmpc_mode is False
    path = tmp_path / "baseline.yaml"
    path.write_text("planner:\n  sit_lmpc_mode: true\n")
    assert load_config(path).planner.sit_lmpc_mode is True
    path.write_text("planner:\n  sit_lmpc_mode: false\n")
    assert load_config(path).planner.sit_lmpc_mode is False


def test_old_recorded_sit_mode_location_loads_into_planner(tmp_path):
    path = tmp_path / "old-config.yaml"
    path.write_text("simulation:\n  sit_lmpc_mode: true\n")
    config = load_config(path)
    assert config.planner.sit_lmpc_mode is True
    assert "sit_lmpc_mode" not in config.as_dict()["simulation"]
    path.write_text("simulation:\n  sit_lmpc_mode: true\nplanner:\n  sit_lmpc_mode: false\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_config(path)


def test_documented_yaml_includes_every_supported_setting():
    from dataclasses import fields
    from importlib.resources import files
    import yaml
    from sit_lmpc.config.schema import SECTIONS

    path = files("sit_lmpc") / "config" / "default.yaml"
    raw = yaml.safe_load(path.read_text())
    assert set(raw) == set(SECTIONS)
    for section, cls in SECTIONS.items():
        assert set(raw[section]) == {field.name for field in fields(cls)}
    from sit_lmpc.config.schema import ModelBankConfig, RiskConstraintConfig, DynamicsConfig

    assert set(raw["planner"]["model_bank"]) == {field.name for field in fields(ModelBankConfig)}
    assert set(raw["planner"]["dynamics"]) == {field.name for field in fields(DynamicsConfig)}
    assert set(raw["planner"]["risk_constraint"]) == {
        field.name for field in fields(RiskConstraintConfig)
    }
    load_config(path).validate()


def test_model_bank_moves_under_planner_and_old_checkpoint_config_loads(tmp_path):
    import yaml

    from sit_lmpc.config.schema import ModelBankConfig

    path = tmp_path / "checkpoint-config.yaml"
    path.write_text(yaml.safe_dump({"model_bank": {"frictions": [0.3, 0.7]}}))
    restored = load_config(path)
    assert restored.planner.model_bank == ModelBankConfig((0.3, 0.7))
    assert "model_bank" not in restored.as_dict()
    assert restored.as_dict()["planner"]["model_bank"]["frictions"] == (0.3, 0.7)
    path.write_text(
        yaml.safe_dump(
            {
                "model_bank": {"frictions": [0.3]},
                "planner": {"model_bank": {"frictions": [0.7]}},
            }
        )
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_config(path)


def test_packaged_and_schema_precision_defaults_are_float32():
    from importlib.resources import files

    for config in (load_config(), load_config(files("sit_lmpc") / "config" / "default.yaml")):
        assert config.simulation.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_yaml_selects_precision(tmp_path, monkeypatch, dtype):
    import sys
    from sit_lmpc.cli.racecar import arguments

    path = tmp_path / "precision.yaml"
    path.write_text(f"simulation:\n  dtype: {dtype}\n")
    monkeypatch.setattr(sys, "argv", ["sit-racecar", "--config", str(path)])
    _, config = arguments()
    assert config.simulation.dtype == dtype


def test_gym_and_planner_schema_friction_defaults_match():
    from sit_lmpc.config import ExperimentConfig

    config = ExperimentConfig()
    assert config.simulation.friction == config.planner.friction == 0.6


def test_gym_and_planner_friction_follow_editable_preset():
    from importlib.resources import files
    import yaml

    path = files("sit_lmpc") / "config" / "default.yaml"
    raw = yaml.safe_load(path.read_text())
    for config in (load_config(), load_config(path)):
        assert config.simulation.friction == raw["simulation"]["friction"]
        assert config.planner.friction == raw["planner"]["friction"]


@pytest.mark.parametrize("section", ["simulation", "planner"])
@pytest.mark.parametrize("friction", [0, -0.1, float("nan"), float("inf"), True, "0.6", None])
def test_invalid_friction_is_rejected(section, friction):
    with pytest.raises(ValueError, match=f"{section}.friction"):
        load_config(overrides={section: {"friction": friction}})
