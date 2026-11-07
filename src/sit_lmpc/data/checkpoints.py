"""BRO value checkpoint schema; original residual/BNN artifacts are untouched."""

from dataclasses import asdict
from pathlib import Path
import json
import jax.numpy as jnp
import orbax.checkpoint as ocp


def cast_snapshot(snapshot, dtype):
    """Convert floating model/optimizer/normalization leaves for a new run precision."""
    import jax

    return jax.tree.map(
        lambda value: value.astype(dtype)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) else value,
        snapshot,
    )


def save_value(directory, snapshot, config, context=None):
    from sit_lmpc.training.features import input_metadata

    if snapshot.model_type != config.model:
        raise ValueError("Value snapshot/config model differs")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "value_model"
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        checkpointer.save(str(path), {"snapshot": snapshot}, force=True)
    finally:
        checkpointer.close()
    metadata = {
        "schema": 1,
        "model": config.model,
        "value_config": asdict(config),
        "generation": int(snapshot.generation),
        **input_metadata(config.input_mode),
        "target": "legacy extended successful-lap time value (not paper task cost)",
        "context": context,
    }
    (directory / "value_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_risk(directory, snapshot, config, context):
    from sit_lmpc.training.features import input_metadata

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        checkpointer.save(str(directory / "risk_model"), {"snapshot": snapshot}, force=True)
    finally:
        checkpointer.close()
    metadata = {
        "schema": 1,
        "model": "sigmoid_mlp",
        "risk_config": asdict(config),
        "generation": int(snapshot.generation),
        **input_metadata(config.input_mode),
        "target": "failure before success, excluding censored outcomes",
        **context,
        "safe_cutoff": context["evaluation"]["safe_cutoff"],
    }
    (directory / "risk_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_risk(directory, config, seed=0):
    from sit_lmpc.training.risk import RiskSnapshot, initial_state
    from sit_lmpc.training.features import feature_names
    from sit_lmpc.data.normalization import Normalization
    from flax.training import orbax_utils
    import jax

    directory = Path(directory).resolve()
    metadata = json.loads((directory / "risk_metadata.json").read_text())
    if metadata.get("schema") != 1 or metadata.get("model") != "sigmoid_mlp":
        raise ValueError("unsupported risk checkpoint schema/model")
    for field in ("input_mode", "hidden_dims", "layers"):
        if metadata["risk_config"][field] != asdict(config)[field]:
            raise ValueError(f"Risk checkpoint architecture differs: {field}")
    count = len(feature_names(config.input_mode)) + 1
    template = RiskSnapshot(
        initial_state(config, jax.random.key(seed)),
        Normalization(jnp.zeros(count), jnp.ones(count)),
        0,
        config.input_mode,
    )
    target = {"snapshot": template}
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        args = ocp.args.PyTreeRestore(
            item=target, restore_args=orbax_utils.restore_args_from_target(target)
        )
        return checkpointer.restore(str(directory / "risk_model"), args=args)["snapshot"]
    finally:
        checkpointer.close()


def load_value(directory, config, seed=0, template_state=None):
    from sit_lmpc.training.value import ValueSnapshot, initial_state
    from sit_lmpc.data.normalization import Normalization
    import jax

    directory = Path(directory).resolve()
    metadata_path = directory / "value_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            "Expected a SIT-LMPC Remastered value checkpoint; residual/BNN artifacts are not supported"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != 1 or metadata.get("model") not in ("normalizing_flow", "mlp"):
        raise ValueError("Unsupported value checkpoint schema/model")
    if metadata["model"] != config.model:
        raise ValueError("Value checkpoint architecture differs: model")
    architecture = ("model", "hidden_dims", "layers", "activation", "input_mode")
    if config.model == "normalizing_flow":
        architecture += ("bins",)
    for field in architecture:
        if field == "input_mode" and field not in metadata["value_config"]:
            old_value = "state"
        else:
            old_value = metadata["value_config"][field]
        if old_value != asdict(config)[field]:
            raise ValueError(f"Value checkpoint architecture differs: {field}")
    from sit_lmpc.training.features import feature_names

    count = len(feature_names(config.input_mode)) + 1
    template = ValueSnapshot(
        initial_state(config, jax.random.key(seed)) if template_state is None else template_state,
        Normalization(jnp.zeros(count), jnp.ones(count)),
        0,
        config.input_mode,
        config.model,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        from flax.training import orbax_utils

        target = {"snapshot": template}
        args = ocp.args.PyTreeRestore(
            item=target, restore_args=orbax_utils.restore_args_from_target(target)
        )
        return checkpointer.restore(str(directory / "value_model"), args=args)["snapshot"]
    finally:
        checkpointer.close()
