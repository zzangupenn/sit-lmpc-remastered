import hashlib
import json
import shutil

import numpy as np
import pytest
import yaml

from sit_lmpc.config import load_config
from sit_lmpc.data.trajectories import TrajectoryDataset
from sit_lmpc.data.experiment_checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    upgrade_checkpoint,
)


def empty_experiment(directory):
    directory.mkdir()
    config = load_config(
        overrides={
            "simulation": {
                "seed_mode": "fixed",
                "seed": 7,
                "map": "custom1",
                "friction": 0.5,
                "max_laps": 2,
                "max_steps": 20,
            },
            "planner": {
                "friction": 0.6,
                "sit_lmpc_mode": False,
                "risk_constraint": {"enabled": False},
            },
            "value": {"input_mode": "state"},
            "risk": {"enabled": False, "input_mode": "state"},
            "output": {"directory": str(directory), "resume": None, "quick_load": False},
            "diagnostics": {
                "render": False,
                "plots": False,
                "keyboard_controls": False,
            },
        }
    )
    (directory / "config.yaml").write_text(yaml.safe_dump(config.as_dict()))
    (directory / "metadata.json").write_text('{"schema": 1, "package": "sit-lmpc-remastered"}')
    (directory / "events.jsonl").write_text("")
    dataset = TrajectoryDataset(directory, config, 12)
    return {
        "schema": 2,
        "trials": 0,
        "attempts": 0,
        "applied_steps": 0,
        "counts": dataset.counts,
        "events_bytes": 0,
        "init_saved": False,
    }


def test_selected_checkpoint_is_rejected_when_damaged(tmp_path):
    directory = tmp_path / "run"
    state = empty_experiment(directory)
    checkpoint = save_checkpoint(directory, 0, state, {"state": np.arange(7)}, False, False)
    (checkpoint / "state.json").write_text("damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(checkpoint)


def test_checkpoint_never_overwrites_a_published_trial(tmp_path):
    directory = tmp_path / "run"
    state = empty_experiment(directory)
    save_checkpoint(directory, 0, state, {}, False, False)
    with pytest.raises(FileExistsError):
        save_checkpoint(directory, 0, state, {}, False, False)


def test_active_risk_pair_is_bundled_and_restored_independently(tmp_path):
    directory = tmp_path / "run"
    state = empty_experiment(directory)
    state["risk_constraint_generation"] = 2
    active = directory / "risk_constraint"
    (active / "risk_model").mkdir(parents=True)
    (active / "risk_model" / "weights").write_text("paired model")
    (active / "risk_metadata.json").write_text(
        json.dumps({"safe_cutoff": {"status": "ready", "generation": 2, "threshold": 0.3}})
    )
    checkpoint = save_checkpoint(directory, 0, state, {}, False, False)
    portable = tmp_path / "portable"
    shutil.copytree(checkpoint, portable)
    directory.rename(tmp_path / "source-unavailable")
    _, restored, _, _ = load_checkpoint(portable)
    assert restored["risk_constraint_generation"] == 2
    continued = tmp_path / "continued"
    restore_checkpoint(continued, portable, restored)
    assert (continued / "risk_constraint" / "risk_model" / "weights").read_text() == (
        "paired model"
    )


def test_checkpoint_is_readable_after_copy_and_rename_without_original_experiment(tmp_path):
    directory = tmp_path / "run"
    state = empty_experiment(directory)
    checkpoint = save_checkpoint(directory, 0, state, {"state": np.arange(7)}, False, False)
    portable = tmp_path / "arbitrary-folder-name"
    shutil.copytree(checkpoint, portable)
    directory.rename(tmp_path / "unavailable")
    path, restored, arrays, problems = load_checkpoint(portable)
    assert path == portable and restored["trials"] == 0 and not problems
    np.testing.assert_array_equal(arrays["state"], np.arange(7))
    assert (portable / "config.yaml").is_file()
    assert (portable / "map" / "custom1_map.png").is_file()
    with pytest.raises(ValueError, match="directly"):
        load_checkpoint(tmp_path)


def test_legacy_conversion_preserves_original_checkpoint(tmp_path):
    directory = tmp_path / "run"
    state = empty_experiment(directory)
    legacy = directory / "checkpoints" / "trial-000000"
    legacy.mkdir(parents=True)
    state["schema"] = 1
    (legacy / "state.json").write_text(json.dumps(state))
    np.savez_compressed(legacy / "state.npz", state=np.arange(7))
    hashes = {
        name: hashlib.sha256((legacy / name).read_bytes()).hexdigest()
        for name in ("state.json", "state.npz")
    }
    (legacy / "manifest.json").write_text(json.dumps({"schema": 1, "trials": 0, "files": hashes}))
    backup = upgrade_checkpoint(legacy)
    assert json.loads((backup / "manifest.json").read_text())["schema"] == 1
    _, restored, arrays, _ = load_checkpoint(legacy)
    assert restored["schema"] == 2
    np.testing.assert_array_equal(arrays["state"], np.arange(7))
    assert (legacy / "config.yaml").is_file() and (
        legacy / "trajectories" / "context.json"
    ).is_file()


def test_checkpoint_bundles_custom_map_with_external_image(tmp_path):
    from sit_lmpc.track.resources import map_path

    directory = tmp_path / "run"
    state = empty_experiment(directory)
    source = tmp_path / "external-map"
    source.mkdir()
    prefix = source / "testmap"
    reference = map_path("custom1")
    image = source / "image.png"
    shutil.copy2(reference.with_name("custom1_map.png"), image)
    shutil.copy2(
        reference.with_name("custom1_raceline.csv"), prefix.with_name("testmap_raceline.csv")
    )
    spec = yaml.safe_load(reference.with_name("custom1_map.yaml").read_text())
    spec["image"] = str(image)
    prefix.with_name("testmap_map.yaml").write_text(yaml.safe_dump(spec))
    checkpoint = save_checkpoint(directory, 0, state, {}, False, False, map_source=prefix)
    source.rename(tmp_path / "unavailable-map")
    _, restored, _, _ = load_checkpoint(checkpoint)
    bundled = checkpoint / restored["map_prefix"]
    assert bundled.name == "testmap"
    bundled_spec = yaml.safe_load(bundled.with_name("testmap_map.yaml").read_text())
    assert bundled_spec["image"] == "testmap_map.png"
    assert (bundled.parent / bundled_spec["image"]).is_file()
