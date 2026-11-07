"""Resume a new experiment with only structural/data compatibility settings fixed."""

from pathlib import Path

RESUME_FIXED_FIELDS = {
    "simulation": ("model", "map"),
    "planner": ("half_width",),  # Existing success/failure labels use this boundary.
    "value": ("model", "input_mode", "hidden_dims", "layers", "bins", "activation"),
    "risk": ("input_mode", "hidden_dims", "layers"),
}


def resume_fixed_fields(checkpoint=None):
    """Network formats are fixed only when their trained weights are present."""
    fixed = dict(RESUME_FIXED_FIELDS)
    if checkpoint is not None:
        for name in ("value", "risk"):
            if not (Path(checkpoint) / f"{name}_model").is_dir():
                fixed.pop(name)
    return fixed


def resume_overrides(settings, checkpoint=None):
    """Keep compatible YAML entries; omitted entries retain checkpoint values."""
    from sit_lmpc.config.schema import RETIRED_DIAGNOSTIC_FIELDS, SECTIONS

    if not isinstance(settings, dict) or set(settings) - (set(SECTIONS) | {"model_bank"}):
        raise ValueError(f"configuration sections must be {list(SECTIONS)}")
    settings = dict(settings)
    for section in SECTIONS:
        if section in settings:
            if not isinstance(settings[section], dict):
                raise ValueError(f"{section} must be a mapping")
            settings[section] = dict(settings[section])
    if "model_bank" in settings:
        bank = settings.pop("model_bank")
        planner = settings.setdefault("planner", {})
        if not isinstance(bank, dict):
            raise ValueError("model_bank must be a mapping")
        if "model_bank" in planner and planner["model_bank"] != bank:
            raise ValueError("conflicting model_bank and planner.model_bank values")
        planner["model_bank"] = bank
    simulation = settings.get("simulation", {})
    if "sit_lmpc_mode" in simulation:
        mode = simulation.pop("sit_lmpc_mode")
        planner = settings.setdefault("planner", {})
        if "sit_lmpc_mode" in planner and planner["sit_lmpc_mode"] != mode:
            raise ValueError("conflicting simulation/planner sit_lmpc_mode values")
        planner["sit_lmpc_mode"] = mode
    fixed = resume_fixed_fields(checkpoint)
    return {
        section: {
            key: value
            for key, value in settings.get(section, {}).items()
            if not (section == "diagnostics" and key in RETIRED_DIAGNOSTIC_FIELDS)
            and key not in fixed.get(section, ())
        }
        for section in ("simulation", "planner", "value", "risk", "diagnostics", "output")
        if section in settings
    }
