"""Self-contained, immutable trial-boundary checkpoints (numerical data only)."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import yaml

from sit_lmpc.data.trajectories import OUTCOMES, load_trajectory

SCHEMA = 2


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _records(directory, state):
    """Read exactly the committed attempts, ignoring any later experiment tail."""
    records = []
    counts = dict.fromkeys(OUTCOMES, 0)
    for outcome in OUTCOMES:
        for path in (directory / "trajectories" / outcome).glob("attempt-*.npz"):
            if int(path.stem.removeprefix("attempt-")) > state["attempts"]:
                continue
            trajectory = load_trajectory(path)
            if trajectory.outcome != outcome or path.stem != f"attempt-{trajectory.attempt_id:06d}":
                raise ValueError(f"Trajectory identity differs from file location: {path}")
            counts[outcome] += 1
            records.append(
                {
                    "attempt_id": trajectory.attempt_id,
                    "lap": trajectory.lap,
                    "outcome": trajectory.outcome,
                    "termination_reason": trajectory.termination_reason,
                    "steps": len(trajectory.controls),
                    "duration": float(trajectory.times[-1]),
                    "file": str(path.relative_to(directory / "trajectories")),
                }
            )
    records.sort(key=lambda item: item["attempt_id"])
    if [item["attempt_id"] for item in records] != list(range(1, state["attempts"] + 1)):
        raise ValueError("committed trajectory attempt IDs are incomplete")
    if (
        counts != state["counts"]
        or sum(item["steps"] for item in records) != state["applied_steps"]
    ):
        raise ValueError("committed trajectory counts/steps differ from checkpoint")
    return records


def _copy_map(prefix, destination):
    """Bundle geometry and the occupancy image, including external custom maps."""
    prefix = Path(prefix).resolve()
    destination.mkdir()
    source_yaml = prefix.with_name(prefix.name + "_map.yaml")
    spec = yaml.safe_load(source_yaml.read_text())
    image = (prefix.parent / spec["image"]).resolve()
    image_name = prefix.name + "_map" + image.suffix
    shutil.copy2(image, destination / image_name)
    spec["image"] = image_name
    (destination / source_yaml.name).write_text(yaml.safe_dump(spec))
    geometry = []
    for suffix in ("_centerline.csv", "_raceline.csv"):
        source = prefix.with_name(prefix.name + suffix)
        if source.is_file():
            shutil.copy2(source, destination / source.name)
            geometry.append(source)
    if not geometry:
        raise FileNotFoundError(f"No centerline/raceline for checkpoint map: {prefix}")
    return str(Path("map") / prefix.name)


def _populate(staged, directory, state, arrays, value_ready, risk_ready, map_source, model_source):
    records = _records(directory, state)
    config_file = directory / "config.yaml"
    if map_source is None:
        from sit_lmpc.track.resources import map_path

        map_source = map_path(yaml.safe_load(config_file.read_text())["simulation"]["map"])
    state = {**state, "schema": SCHEMA, "map_prefix": _copy_map(map_source, staged / "map")}
    shutil.copy2(config_file, staged / "config.yaml")
    shutil.copy2(directory / "metadata.json", staged / "metadata.json")
    trajectories = staged / "trajectories"
    trajectories.mkdir()
    for outcome in OUTCOMES:
        (trajectories / outcome).mkdir()
    shutil.copy2(directory / "trajectories" / "context.json", trajectories / "context.json")
    for record in records:
        shutil.copy2(directory / "trajectories" / record["file"], trajectories / record["file"])
    (trajectories / "index.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records)
    )
    events = (directory / "events.jsonl").read_bytes()
    if state["events_bytes"] > len(events):
        raise ValueError("checkpoint event offset exceeds event log length")
    (staged / "events.jsonl").write_bytes(events[: state["events_bytes"]])
    if state["init_saved"] and (directory / "init.npz").is_file():
        shutil.copy2(directory / "init.npz", staged / "init.npz")
    np.savez_compressed(staged / "state.npz", **arrays)
    (staged / "state.json").write_text(json.dumps(state, allow_nan=False, indent=2) + "\n")
    for name, ready in (("value", value_ready), ("risk", risk_ready)):
        if ready:
            shutil.copytree(model_source / f"{name}_model", staged / f"{name}_model")
            shutil.copy2(model_source / f"{name}_metadata.json", staged / f"{name}_metadata.json")
    if (model_source / "risk_constraint").is_dir():
        shutil.copytree(model_source / "risk_constraint", staged / "risk_constraint")
    for name in ("dynamics_belief.json", "dynamics_belief.npz",
                 "dynamics_posterior.json", "dynamics_posterior.npz"):
        source = model_source / name
        if source.is_file():
            shutil.copy2(source, staged / name)
    if (model_source / "dynamics_model").is_dir():
        shutil.copytree(model_source / "dynamics_model", staged / "dynamics_model")
    if (model_source / "safe_set_density").is_dir():
        shutil.copytree(model_source / "safe_set_density", staged / "safe_set_density")
    hashes = {
        str(path.relative_to(staged)): _digest(path) for path in staged.rglob("*") if path.is_file()
    }
    (staged / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "trials": state["trials"],
                "experiment_directory": str(directory),
                "files": hashes,
            },
            indent=2,
        )
        + "\n"
    )


def save_checkpoint(directory, trials, state, arrays, value_ready, risk_ready, *, map_source=None):
    directory = Path(directory).resolve()
    if state.get("trials") != trials:
        raise ValueError("checkpoint state/trial identity mismatch")
    root = directory / "checkpoints"
    root.mkdir(exist_ok=True)
    target = root / f"trial-{trials:06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    staged = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        _populate(staged, directory, state, arrays, value_ready, risk_ready, map_source, directory)
        os.replace(staged, target)
        marker = root / ".latest-pending"
        marker.write_text(target.name + "\n")
        os.replace(marker, root / "latest")
        return target
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _read_checkpoint(path, *, allow_legacy=False):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA and not (allow_legacy and manifest.get("schema") == 1):
        raise ValueError("Legacy checkpoint needs conversion to a self-contained checkpoint")
    if manifest["schema"] == SCHEMA:
        required = {
            "config.yaml",
            "metadata.json",
            "state.json",
            "state.npz",
            "events.jsonl",
            "trajectories/context.json",
            "trajectories/index.jsonl",
        }
        if not required.issubset(manifest["files"]):
            raise ValueError("checkpoint manifest is missing required continuation files")
    for relative, expected in manifest["files"].items():
        file = path / relative
        if not file.resolve().is_relative_to(path.resolve()) or not file.is_file():
            raise ValueError(f"checkpoint file missing or outside checkpoint: {relative}")
        if _digest(file) != expected:
            raise ValueError(f"checkpoint checksum differs: {relative}")
    state = json.loads((path / "state.json").read_text())
    if state.get("schema") != manifest["schema"] or state.get("trials") != manifest["trials"]:
        raise ValueError("checkpoint state/manifest mismatch")
    if manifest["schema"] == SCHEMA:
        prefix = path / state["map_prefix"]
        if not prefix.resolve().is_relative_to(path / "map"):
            raise ValueError("checkpoint map prefix is external")
        if (
            str(prefix.with_name(prefix.name + "_map.yaml").relative_to(path))
            not in manifest["files"]
        ):
            raise ValueError("checkpoint map is missing from manifest")
        spec = yaml.safe_load(prefix.with_name(prefix.name + "_map.yaml").read_text())
        image = (prefix.parent / spec["image"]).resolve()
        if (
            not image.is_relative_to(path / "map")
            or str(image.relative_to(path)) not in manifest["files"]
        ):
            raise ValueError("checkpoint map image is missing or external")
        _records(path, state)
    with np.load(path / "state.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return state, arrays


def load_checkpoint(path):
    """Restore the exact selected checkpoint, wherever its folder was copied."""
    path = Path(path).expanduser().resolve()
    if not (path / "manifest.json").is_file():
        raise ValueError(f"resume must point directly to a checkpoint folder: {path}")
    state, arrays = _read_checkpoint(path)
    return path, state, arrays, []


def load_checkpoint_config(path, overrides=None):
    from sit_lmpc.config import load_config

    path = Path(path).expanduser().resolve()
    if not (path / "manifest.json").is_file():
        raise ValueError(f"resume must point directly to a checkpoint folder: {path}")
    if not (path / "config.yaml").is_file():
        raise ValueError(
            f"Legacy checkpoint needs conversion to a self-contained checkpoint: {path}"
        )
    return load_config(path / "config.yaml", overrides)


def resume_directory(checkpoint, config, requested=None):
    """Place every resumed run in its own experiment directory."""
    if requested is not None:
        destination = Path(requested).expanduser().resolve()
    else:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
        destination = (
            Path(config.output.root).expanduser().resolve()
            / Path(config.simulation.map).name
            / f"{config.output.exp_name}-{stamp}"
        )
    if destination == checkpoint or destination.is_relative_to(checkpoint):
        raise ValueError("resume output directory must be outside the immutable checkpoint")
    if destination.exists() and (requested is None or any(destination.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite nonempty continuation directory: {destination}"
        )
    return destination


def restore_checkpoint(directory, checkpoint, state, *, restore_constraint=True):
    """Copy an immutable checkpoint into a new, otherwise empty experiment."""
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty continuation directory: {directory}")
    for name in (
        "config.yaml",
        "trajectories",
        "events.jsonl",
        "map",
        "init.npz",
        "value_model",
        "value_metadata.json",
        "risk_model",
        "risk_metadata.json",
        "risk_constraint",
        "dynamics_belief.json",
        "dynamics_belief.npz",
        "dynamics_posterior.json",
        "dynamics_posterior.npz",
        "dynamics_model",
        "safe_set_density",
    ):
        if name == "risk_constraint" and not restore_constraint:
            continue
        source = checkpoint / name
        if source.is_dir():
            shutil.copytree(source, directory / name)
        elif source.is_file():
            shutil.copy2(source, directory / name)
    root = directory / "checkpoints"
    root.mkdir(exist_ok=True)
    local = root / f"trial-{state['trials']:06d}"
    shutil.copytree(checkpoint, local)
    manifest = json.loads((local / "manifest.json").read_text())
    manifest["experiment_directory"] = str(directory)
    manifest["copied_from"] = str(checkpoint)
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "latest").write_text(local.name + "\n")
    return local


def upgrade_checkpoint(path):
    """Convert an existing schema-1 checkpoint while retaining its original bundle."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") == SCHEMA:
        load_checkpoint(path)
        return None
    state, arrays = _read_checkpoint(path, allow_legacy=True)
    if path.parent.name != "checkpoints":
        raise ValueError("Legacy checkpoint conversion needs its original experiment directory")
    directory = path.parent.parent
    staged = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    try:
        _populate(
            staged,
            directory,
            state,
            arrays,
            (path / "value_model").is_dir(),
            (path / "risk_model").is_dir(),
            None,
            path,
        )
        _read_checkpoint(staged)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        backup = path.parent / "legacy" / f"{path.name}-{stamp}"
        backup.parent.mkdir(exist_ok=True)
        os.replace(path, backup)
        try:
            os.replace(staged, path)
        except Exception:
            os.replace(backup, path)
            raise
        return backup
    finally:
        if staged.exists():
            shutil.rmtree(staged)
