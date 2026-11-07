"""Numeric trajectory shards; splits/grouping never depend on shuffled minibatches."""

from pathlib import Path
import hashlib
import json
import numpy as np
from dyna_dist.data.schema import Transitions, concatenate
from dyna_dist.domains import get_domain

DATASET_VERSION = 1


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["schema_version"] not in {DATASET_VERSION, 2}:
            raise ValueError("unsupported dataset schema")
        if self.manifest.get("status") != "complete":
            raise ValueError("dataset collection is incomplete")
        self.instances = self.manifest["instances"]
        self.timestep = self.manifest["config"]["timestep"]
        self.fingerprint = self.manifest["fingerprint"]
        self.domain = get_domain(self.manifest.get("domain", {}).get("name", "st"))
        if self.manifest["schema_version"] == 2 and (
            self.manifest["domain"]["transform"] != self.domain.transform
            or tuple(self.manifest["domain"]["state_fields"]) != self.domain.state_fields
            or tuple(self.manifest["domain"]["control_fields"]) != self.domain.control_fields
        ):
            raise ValueError("dataset domain layout/transform mismatch")

    def members(self, split):
        return [i for i in self.instances if i["split"] == split]

    def trajectory(self, filename):
        path = (self.path / filename).resolve()
        if not path.is_relative_to(self.path.resolve()):
            raise ValueError("trajectory path escapes dataset")
        checksum = self.manifest.get("shard_checksums", {}).get(filename)
        if checksum is None or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("trajectory checksum mismatch")
        with np.load(path, allow_pickle=False) as shard:
            result = {name: shard[name].copy() for name in shard.files}
        n = len(result["controls"])
        if (
            result["states"].shape != (n + 1, self.domain.state_dimension)
            or result["valid"].shape != (n,)
            or result["times"].shape != (n + 1,)
        ):
            raise ValueError("malformed trajectory shard")
        if not np.allclose(np.diff(result["times"]), self.timestep):
            raise ValueError("trajectory timestamps disagree with manifest interval")
        return result

    def transitions(self, instance, view="true", *, max_count=0, balance_stages=False, seed=0):
        if view not in {"true", "measured"}:
            raise ValueError("view must be true or measured")
        batches = []
        for filename in instance["files"]:
            trajectory = self.trajectory(filename)
            states = trajectory["states" if view == "true" else "measured_states"]
            indices = np.flatnonzero(trajectory["valid"])
            ids = tuple(
                str(value) for value in self.transition_ids(instance, filename, trajectory)[indices]
            )
            batches.append(
                Transitions(
                    states[indices],
                    trajectory["controls"][indices],
                    states[indices + 1],
                    ids,
                    self.timestep,
                    self.domain.name,
                    trajectory.get(
                        "learning_stages", np.zeros(len(trajectory["controls"]), dtype=int)
                    )[indices],
                )
            )
        result = concatenate(batches)
        if max_count and len(result) > max_count:
            rng = np.random.default_rng(seed)
            order = (
                balanced_indices(result.stages, max_count, rng)
                if balance_stages
                else rng.choice(len(result), max_count, replace=False)
            )
            result = result.take(np.sort(order))
        return result

    def transition_ids(self, instance, filename, trajectory):
        if "transition_ids" in trajectory:
            return trajectory["transition_ids"]
        origin = instance.get("provenance", {}).get(filename, {})
        fingerprint = origin.get("source_dataset_fingerprint", self.fingerprint)
        shard = origin.get("source_shard", filename)
        return np.asarray(
            [f"{fingerprint}:{shard}:{i}" for i in range(len(trajectory["controls"]))]
        )


def balanced_indices(stages, count, rng):
    """Without-replacement, near-equal stage sampling; redistribute exhausted groups."""
    groups = [list(rng.permutation(np.flatnonzero(stages == value))) for value in np.unique(stages)]
    selected = []
    while len(selected) < count:
        active = [group for group in groups if group]
        if not active:
            raise ValueError("not enough transitions for requested stage-balanced selection")
        for index in rng.permutation(len(active)):
            selected.append(active[index].pop())
            if len(selected) == count:
                break
    return np.asarray(selected)
