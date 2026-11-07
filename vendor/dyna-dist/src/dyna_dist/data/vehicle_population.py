"""New vehicle populations with byte-identical, frozen test trajectories."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path

import numpy as np

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.config import VehicleCollectionConfig, resolve_seed, write_config
from dyna_dist.data.vehicle_collection import _rollout
from dyna_dist.data.vehicle_study import unseen_parameters
from dyna_dist.io.common import fingerprint, new_directory, versions, write_json
from dyna_dist.io.dataset import Dataset
from dyna_dist.models.profiles import vehicle_parameters


def build_vehicle_population(config, emit=print):
    """Replay the original action tapes for new train/validation combinations only."""
    config = replace(config.validate(), seed=resolve_seed(config.seed))
    source = Dataset(config.source_dataset)
    if source.domain.name != "st" or source.manifest["config"]["profile"] != "fullscale":
        raise ValueError("source must be a fullscale ST vehicle dataset")
    tests = source.members("test")
    if not tests:
        raise ValueError("source must contain the test physics to preserve")
    collection = VehicleCollectionConfig(**source.manifest["config"]).validate()
    rng = np.random.default_rng(config.seed)
    draws = {
        "train": unseen_parameters(source, config.training_instances, rng),
        "validation": unseen_parameters(source, config.validation_instances, rng),
    }
    combined = np.concatenate(list(draws.values()))
    if len(np.unique(combined, axis=0)) != len(combined):
        raise ValueError("duplicate train/validation physics combinations")
    first = source.members("train")[0]
    probes = {}
    for filename in first["files"]:
        shard = source.trajectory(filename)
        probes[Path(filename).stem] = {
            "initial_state": shard["states"][0], "controls": shard["controls"],
            "target_speed": shard["target_speed"], "target_steering": shard["target_steering"],
        }
    expected = sum(len(probe["controls"]) for probe in probes.values())
    root = new_directory(config.output)
    write_config(config, root / "study_config.yaml")
    manifest = {
        key: source.manifest[key] for key in (
            "schema_version", "state_fields", "control_fields", "state_units", "control_units",
            "source_fingerprint",
        )
    }
    if "domain" in source.manifest:
        manifest["domain"] = source.manifest["domain"]
    manifest.update(
        config={**source.manifest["config"], "output": str(root)}, versions=versions(),
        study={
            "seed": config.seed, "source_dataset": str(source.path.resolve()),
            "source_dataset_fingerprint": source.fingerprint,
            "training_instances": config.training_instances,
            "validation_instances": config.validation_instances,
            "preserved_test_instance_ids": [member["instance_id"] for member in tests],
            "shared_action_source_instance": first["instance_id"],
            "selection": "Independent Latin hypercubes for train and validation; inside original "
                         "bounds, normalized distance > 0.08 from all original physics",
        }, instances=[], shard_checksums={}, status="collecting",
    )
    write_json(root / "manifest.json", manifest)
    (root / "PLAN.md").write_text(
        "# Expanded vehicle population study\n\n"
        f"Generate {config.training_instances} new training and {config.validation_instances} new "
        f"validation physics combinations; retain all {len(tests)} original test combinations and "
        f"trajectory bytes. Collect {expected:,} transitions per combination with the same "
        f"{len(probes)} action tapes, timestep {collection.timestep:g} s, speeds up to 20 m/s, "
        "and unchanged four-parameter ranges. Keep all previous datasets/results.\n\n"
        "Repeat both training options at latent 4/8/16/32 with 20,000 neural updates, identical "
        "training/evaluation seeds and MPPI diagnostics. Report errors, coverage, energy and "
        "Wasserstein at 1/5/10/15 steps, loss curves and multiple GT/prediction sections.\n"
    )
    transitions = 0
    for split, values in draws.items():
        for index, (mu, front, rear, mass) in enumerate(values):
            mu, front, rear, mass = map(float, (mu, front, rear, mass))
            parameters = vehicle_parameters("fullscale", mu, mass, "fixed")
            parameters["C_Sf"] *= front
            parameters["C_Sr"] *= rear
            simulator = GymST(parameters, collection.timestep, collection.integrator_timestep)
            if simulator.source_fingerprint != manifest["source_fingerprint"]:
                raise ValueError("Gym ST implementation differs from the source dataset")
            member = {
                "instance_id": f"{split}-{index:05d}", "split": split, "anchor": None,
                "friction": mu, "front_cornering_scale": front, "rear_cornering_scale": rear,
                "mass_ratio": mass, "parameters": parameters, "files": [],
            }
            count = 0
            for probe_id, probe in probes.items():
                states, valid, reason = _rollout(simulator, probe, collection)
                if reason != "step_limit" or not valid.all():
                    raise ValueError(f"{member['instance_id']} {probe_id}: {reason} at {len(valid)}")
                length = len(valid)
                filename = f"trajectories/{member['instance_id']}/{probe_id}.npz"
                destination = root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination, states=states, controls=probe["controls"],
                    commanded_controls=probe["controls"], actuator_noise=np.zeros((length, 2)),
                    measured_states=states, valid=valid,
                    times=np.arange(length + 1) * collection.timestep,
                    termination=np.asarray(reason), target_speed=probe["target_speed"],
                    target_steering=probe["target_steering"],
                )
                manifest["shard_checksums"][filename] = hashlib.sha256(destination.read_bytes()).hexdigest()
                member["files"].append(filename)
                count += length
            if count != expected:
                raise ValueError(f"{member['instance_id']} has {count} transitions, expected {expected}")
            transitions += count
            manifest["instances"].append(member)
            write_json(root / "manifest.json", manifest)
            emit(f"[population] {member['instance_id']} | {index + 1}/{len(values)} {split} | "
                 f"{count:,} transitions | friction={mu:.3f}")
    for member in tests:
        copied = dict(member)
        copied["provenance"] = {
            filename: member.get("provenance", {}).get(filename, {
                "source_dataset_fingerprint": source.fingerprint, "source_shard": filename,
            }) for filename in member["files"]
        }
        count = 0
        for filename in member["files"]:
            shard = source.trajectory(filename)  # Verify the original checksum before linking.
            count += int(shard["valid"].sum())
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source.path / filename, destination)
            manifest["shard_checksums"][filename] = source.manifest["shard_checksums"][filename]
        if count != expected:
            raise ValueError("preserved test instance has a different transition budget")
        transitions += count
        manifest["instances"].append(copied)
    manifest.update(status="complete", transitions=transitions)
    manifest["fingerprint"] = fingerprint({
        "study": manifest["study"],
        "splits": [(member["instance_id"], member["split"]) for member in manifest["instances"]],
        "shards": manifest["shard_checksums"],
    })
    write_json(root / "manifest.json", manifest)
    counts = {"train": config.training_instances, "validation": config.validation_instances,
              "test": len(tests)}
    summary = {"counts": counts, "transitions_per_instance": expected,
               "total_transitions": transitions, "shared_action_probes": len(probes),
               "test_shards_identical_to_source": True, "fingerprint": manifest["fingerprint"]}
    write_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text(
        "# Vehicle population dataset\n\n"
        f"Split: **{counts['train']} train / {counts['validation']} validation / {counts['test']} test**. "
        f"{expected:,} transitions each; {transitions:,} total.\n\n"
        "Training/validation combinations are new, separate Latin-hypercube draws. Ranges: friction "
        "0.3–1.0, front/rear stiffness scales 0.9–1.1, mass ratio 0.9–1.1. The original test "
        "parameters and numeric trajectory shards are unchanged (verified SHA-256, hard links). "
        "Shared controls and initial states are identical across all physics; validation/test physics "
        "are never used for neural gradients, normalization, or population fitting.\n\n"
        f"Source: [{source.path.name}]({source.path.resolve()}); fingerprint `{source.fingerprint}`. "
        f"Dataset seed: {config.seed}. [Generation plan](PLAN.md) · [Summary](summary.json).\n"
    )
    emit(f"[population] complete | {counts} | {transitions:,} transitions | tests preserved")
    return root
