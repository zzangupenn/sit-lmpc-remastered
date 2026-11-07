"""Numeric domain-aware shards; training observations remain simulator-independent."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from dyna_dist.domains import get_domain
from dyna_dist.io.common import fingerprint, new_directory, write_json, versions


class TrajectoryDatasetWriter:
    def __init__(self, output, config, domain="halfcheetah"):
        self.root = new_directory(output)
        self.spec = get_domain(domain)
        config = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
        self.manifest = {
            "schema_version": 2,
            "status": "collecting",
            "domain": self.spec.metadata(),
            "config": config,
            "versions": versions(),
            "instances": [],
            "shard_checksums": {},
            "fingerprint": "incomplete",
            "transitions": 0,
        }
        write_json(self.root / "manifest.json", self.manifest)

    def instance(self, friction, split, metadata):
        metadata = {key: value for key, value in metadata.items() if key != "max_episode_steps"}
        for member in self.manifest["instances"]:
            if member["split"] == split and np.isclose(
                member["friction"], friction, rtol=0, atol=1e-12
            ):
                if json.dumps(
                    member["parameters"], sort_keys=True
                ) != json.dumps(metadata, sort_keys=True):
                    raise ValueError("same split/friction has conflicting physics metadata")
                return member
        member = {
            "instance_id": f"instance-{len(self.manifest['instances']):05d}",
            "friction": float(friction),
            "split": split,
            "parameters": metadata,
            "files": [],
        }
        self.manifest["instances"].append(member)
        return member

    def write(self, member, arrays, provenance):
        n = len(arrays["controls"])
        if not n:
            return
        if arrays["states"].shape != (n + 1, self.spec.state_dimension):
            raise ValueError("trajectory endpoint alignment mismatch")
        index = len(member["files"])
        filename = f"trajectories/{member['instance_id']}/segment-{index:06d}.npz"
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        self.manifest["shard_checksums"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        member["files"].append(filename)
        member.setdefault("provenance", {})[filename] = provenance
        self.manifest["transitions"] += int(np.asarray(arrays["valid"]).sum())
        write_json(self.root / "manifest.json", self.manifest)

    def import_member(self, source, member):
        """Keep an immutable source collection without re-encoding its numeric shards."""
        target = self.instance(member["friction"], member["split"], member["parameters"])
        source_root = Path(source.path).resolve()
        for filename in member["files"]:
            original = (source.path / filename).resolve()
            if not original.is_relative_to(source_root):
                raise ValueError("source trajectory escapes its dataset")
            expected = source.manifest["shard_checksums"][filename]
            if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
                raise ValueError(f"source trajectory checksum mismatch: {filename}")
            name = f"trajectories/{target['instance_id']}/segment-{len(target['files']):06d}.npz"
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(original, destination)
            except OSError:
                shutil.copy2(original, destination)
            with np.load(destination, allow_pickle=False) as shard:
                self.manifest["transitions"] += int(np.count_nonzero(shard["valid"]))
            self.manifest["shard_checksums"][name] = expected
            target["files"].append(name)
            target.setdefault("provenance", {})[name] = {
                **member.get("provenance", {}).get(filename, {}),
                "source_dataset_fingerprint": source.fingerprint,
                "source_shard": filename,
            }
        write_json(self.root / "manifest.json", self.manifest)

    def finish(self):
        self.manifest["status"] = "complete"
        self.manifest["fingerprint"] = fingerprint(
            {
                "domain": self.manifest["domain"],
                "config": self.manifest["config"],
                "instances": self.manifest["instances"],
                "shards": self.manifest["shard_checksums"],
            }
        )
        write_json(self.root / "manifest.json", self.manifest)
        return self.root
