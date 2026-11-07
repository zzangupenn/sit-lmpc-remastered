"""Build a train/validation plus unseen-physics study from paired ST probes."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path

import numpy as np

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.config import VehicleCollectionConfig, resolve_seed, write_config
from dyna_dist.data.vehicle_collection import KNOBS, _rollout
from dyna_dist.io.common import fingerprint, new_directory, versions, write_json
from dyna_dist.io.dataset import Dataset
from dyna_dist.models.profiles import vehicle_parameters


def unseen_parameters(source, count, rng):
    """Latin-hypercube draws inside the source ranges, separated from existing draws."""
    config = source.manifest["config"]
    bounds = np.asarray([
        config["friction"], config["front_cornering_scale"],
        config["rear_cornering_scale"], config["mass_ratio"],
    ], dtype=float)
    existing = np.asarray([
        [member[key] for key in (
            "friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio"
        )]
        for member in source.instances
    ])
    existing_unit = (existing - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    for _ in range(100):
        unit = np.column_stack([
            (rng.permutation(count) + rng.random(count)) / count for _ in KNOBS
        ])
        distances = np.linalg.norm(unit[:, None, :] - existing_unit[None, :, :], axis=-1)
        if np.min(distances) > 0.08:
            return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    raise ValueError("could not draw sufficiently distinct unseen parameter combinations")


def build_vehicle_study(config, emit=print):
    """Reuse all original shards and replay their controls under new physics."""
    config.validate()
    config = replace(config, seed=resolve_seed(config.seed))
    source = Dataset(config.source_dataset)
    if source.domain.name != "st" or source.manifest["config"]["profile"] != "fullscale":
        raise ValueError("source must be a fullscale ST vehicle dataset")
    if config.validation_instances >= len(source.instances):
        raise ValueError("validation_instances must leave at least one training instance")
    collection_config = VehicleCollectionConfig(**source.manifest["config"]).validate()
    rng = np.random.default_rng(config.seed)
    random_members = [m for m in source.instances if m.get("anchor") is None]
    if config.validation_instances > len(random_members):
        raise ValueError("not enough non-anchor instances for physics validation")
    validation_ids = {
        random_members[i]["instance_id"]
        for i in rng.choice(len(random_members), config.validation_instances, replace=False)
    }
    new_values = unseen_parameters(source, config.unseen_instances, rng)

    first = source.instances[0]
    probes = {}
    for filename in first["files"]:
        shard = source.trajectory(filename)
        probes[Path(filename).stem] = {
            "initial_state": shard["states"][0],
            "controls": shard["controls"],
            "target_speed": shard["target_speed"],
            "target_steering": shard["target_steering"],
        }

    root = new_directory(config.output)
    write_config(config, root / "study_config.yaml")
    manifest = {
        "schema_version": source.manifest["schema_version"],
        "config": {**source.manifest["config"], "output": str(root)},
        "state_fields": source.manifest["state_fields"],
        "control_fields": source.manifest["control_fields"],
        "state_units": source.manifest["state_units"],
        "control_units": source.manifest["control_units"],
        "versions": versions(),
        "source_fingerprint": source.manifest["source_fingerprint"],
        "study": {
            "seed": config.seed,
            "source_dataset": str(source.path.resolve()),
            "source_dataset_fingerprint": source.fingerprint,
            "validation_instance_ids": sorted(validation_ids),
            "shared_action_source_instance": first["instance_id"],
            "unseen_selection": "Latin hypercube inside source bounds; normalized distance > 0.08",
        },
        "instances": [],
        "shard_checksums": {},
        "status": "collecting",
    }
    for member in source.instances:
        member = dict(member)
        member["split"] = (
            "validation" if member["instance_id"] in validation_ids else "train"
        )
        member["source_split"] = next(
            item["split"] for item in source.instances
            if item["instance_id"] == member["instance_id"]
        )
        member["provenance"] = {
            filename: {
                "source_dataset_fingerprint": source.fingerprint,
                "source_shard": filename,
            }
            for filename in member["files"]
        }
        for filename in member["files"]:
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source.path / filename, destination)
            manifest["shard_checksums"][filename] = source.manifest["shard_checksums"][filename]
        manifest["instances"].append(member)
    write_json(root / "manifest.json", manifest)
    emit(f"[vehicle-study] reused {len(source.instances)} physics instances "
         f"({len(source.instances) - len(validation_ids)} train, "
         f"{len(validation_ids)} validation)")

    expected = sum(len(probe["controls"]) for probe in probes.values())
    new_transitions = 0
    for offset, values in enumerate(new_values, start=len(source.instances)):
        mu, front, rear, mass = map(float, values)
        parameters = vehicle_parameters("fullscale", mu, mass, "fixed")
        parameters["C_Sf"] *= front
        parameters["C_Sr"] *= rear
        simulator = GymST(
            parameters, collection_config.timestep, collection_config.integrator_timestep
        )
        if simulator.source_fingerprint != manifest["source_fingerprint"]:
            raise ValueError("Gym ST implementation differs from the source dataset")
        member = {
            "instance_id": f"instance-{offset:05d}",
            "split": "test",
            "anchor": None,
            "friction": mu,
            "front_cornering_scale": front,
            "rear_cornering_scale": rear,
            "mass_ratio": mass,
            "parameters": parameters,
            "files": [],
        }
        count = 0
        for probe_id, probe in probes.items():
            states, valid, reason = _rollout(simulator, probe, collection_config)
            if reason != "step_limit" or not valid.all():
                raise ValueError(
                    f"unseen {member['instance_id']} {probe_id} stopped at "
                    f"{len(valid)} steps: {reason}"
                )
            length = len(valid)
            filename = f"trajectories/{member['instance_id']}/{probe_id}.npz"
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                states=states,
                controls=probe["controls"],
                commanded_controls=probe["controls"],
                actuator_noise=np.zeros((length, 2)),
                measured_states=states,
                valid=valid,
                times=np.arange(length + 1) * collection_config.timestep,
                termination=np.asarray(reason),
                target_speed=probe["target_speed"],
                target_steering=probe["target_steering"],
            )
            manifest["shard_checksums"][filename] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            member["files"].append(filename)
            count += length
        if count != expected:
            raise ValueError(f"unseen instance has {count} transitions, expected {expected}")
        new_transitions += count
        manifest["instances"].append(member)
        write_json(root / "manifest.json", manifest)
        emit(f"[vehicle-study] {member['instance_id']} | unseen test | "
             f"μ={mu:.3f} | Csf×{front:.3f} | Csr×{rear:.3f} | m×{mass:.3f}")

    source_transitions = source.manifest["transitions"]
    manifest["transitions"] = source_transitions + new_transitions
    manifest["status"] = "complete"
    manifest["fingerprint"] = fingerprint({
        "source_dataset_fingerprint": source.fingerprint,
        "study": manifest["study"],
        "splits": [(m["instance_id"], m["split"]) for m in manifest["instances"]],
        "shards": manifest["shard_checksums"],
    })
    write_json(root / "manifest.json", manifest)
    summary = {
        "train_instances": len(source.instances) - len(validation_ids),
        "validation_instances": len(validation_ids),
        "unseen_test_instances": config.unseen_instances,
        "transitions_per_instance": expected,
        "total_transitions": manifest["transitions"],
        "same_action_probes_per_instance": len(probes),
    }
    write_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text(
        "# Vehicle ST BNN study dataset\n\n"
        f"- Source: `{source.path.resolve()}` (fingerprint `{source.fingerprint}`)\n"
        f"- Existing physics: {summary['train_instances']} train, "
        f"{summary['validation_instances']} validation; none discarded.\n"
        f"- Newly generated unseen physics: {config.unseen_instances} test instances.\n"
        f"- Transitions: {manifest['transitions']:,} total, {expected:,} per instance.\n"
        f"- Every instance replays the same {len(probes)} recorded action tapes.\n"
        "- Test parameters lie inside the original four-knob ranges, but no test "
        "physics combination occurs in train/validation.\n"
        "- Validation and test measure generalization across physics, not across "
        "new action sequences.\n"
        "- Original trajectory shards are hard-linked into this standalone dataset; "
        "the source dataset is unchanged.\n"
    )
    return root
