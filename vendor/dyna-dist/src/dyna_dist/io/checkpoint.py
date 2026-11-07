"""Self-contained numeric model/posterior and complete optimizer/RNG snapshots."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
from flax import serialization
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
from dyna_dist.config import ModelConfig, decode
from dyna_dist.data.transforms import Normalizer
from dyna_dist.inference.posterior import PopulationModel, Posterior
from dyna_dist.models.gaussian import GaussianState
from .common import atomic_bytes, new_directory, versions, write_json

CHECKPOINT_VERSION = 1


def _payload(model):
    return serialization.msgpack_serialize(
        {
            "parameters": serialization.to_state_dict(jax.device_get(model.parameters)),
            "normalizer": serialization.to_state_dict(jax.device_get(model.normalizer)),
            "baseline_parameters": np.asarray(model.baseline_parameters),
        }
    )


def save_model(model, path, *, overwrite=False, extra=None):
    path = Path(path)
    if overwrite and path.exists() and any(path.iterdir()):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema_version") != CHECKPOINT_VERSION or manifest.get("kind") != "model":
            raise ValueError("Refusing to replace a directory not owned by dyna-dist model saving")
    else:
        new_directory(path)
    payload = _payload(model)
    checksum = hashlib.sha256(payload).hexdigest()
    atomic_bytes(path / "model.msgpack", payload)
    manifest = {
        "schema_version": CHECKPOINT_VERSION,
        "kind": "model",
        "model_config": asdict(model.config),
        "domain": model.domain.metadata(),
        "generation": model.generation,
        "fingerprint": checksum,
        "versions": versions(),
        **(extra or {}),
    }
    write_json(path / "manifest.json", manifest)
    return path


def load_model(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != CHECKPOINT_VERSION or manifest.get("kind") != "model":
        raise ValueError("unsupported checkpoint schema/kind")
    payload = (path / "model.msgpack").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["fingerprint"]:
        raise ValueError("model checkpoint checksum mismatch")
    config = decode(ModelConfig, manifest["model_config"])
    from dyna_dist.domains import get_domain

    spec = get_domain(config.domain)
    if "domain" in manifest and manifest["domain"]["transform"] != spec.transform:
        raise ValueError("model domain transform mismatch")
    if config.dtype == "float64" and not jax.config.jax_enable_x64:
        raise ValueError("Enable jax_enable_x64 before loading a float64 checkpoint")
    data = serialization.msgpack_restore(payload)
    data = jax.tree.map(jnp.asarray, data)
    if data["normalizer"]["input_mean"].shape != (spec.input_dimension,) or data["normalizer"][
        "target_mean"
    ].shape != (spec.state_dimension,):
        raise ValueError("model dimensions differ from domain")
    return PopulationModel(
        freeze(data["parameters"]),
        Normalizer(**data["normalizer"]),
        data["baseline_parameters"],
        config,
        manifest["generation"],
        manifest["fingerprint"],
    )


def save_posterior(posterior, path):
    save_model(posterior.model, path)
    payload = {
        "precision": np.asarray(posterior.state.precision),
        "information": np.asarray(posterior.state.information),
        "seen_ids": sorted(posterior.seen_ids),
        "generation": posterior.generation,
    }
    payload = serialization.msgpack_serialize(payload)
    atomic_bytes(Path(path) / "posterior.msgpack", payload)
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["posterior_checksum"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, manifest)


def load_posterior(path):
    model = load_model(path)
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    payload = (Path(path) / "posterior.msgpack").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("posterior_checksum"):
        raise ValueError("posterior checkpoint checksum mismatch")
    data = serialization.msgpack_restore(payload)
    state = GaussianState(jnp.asarray(data["precision"]), jnp.asarray(data["information"]))
    if (
        state.precision.shape != (model.config.latent_dimension,) * 2
        or state.information.shape != (model.config.latent_dimension,)
        or not np.isfinite(np.asarray(state.precision)).all()
        or not np.isfinite(np.asarray(state.information)).all()
    ):
        raise ValueError("invalid Gaussian posterior checkpoint")
    np.linalg.cholesky(np.asarray(state.precision))
    return Posterior(model, state, frozenset(data["seen_ids"]), int(data["generation"]))


def save_training(
    model, train_state, rng, path, config, dataset_fingerprint, best_loss, best_model,
    extra_state=None,
):
    save_model(
        model,
        path,
        extra={"training_config": asdict(config), "dataset_fingerprint": dataset_fingerprint},
    )
    data = {
        "optimizer": serialization.to_state_dict(jax.device_get(train_state.opt_state)),
        "step": int(train_state.step),
        "rng_state": json.dumps(rng.bit_generator.state),
        "best_loss": float(best_loss),
        "best_generation": best_model.generation,
        "best_parameters": serialization.to_state_dict(jax.device_get(best_model.parameters)),
    }
    if extra_state is not None:
        data["extra_state"] = jax.device_get(extra_state)
    payload = serialization.msgpack_serialize(data)
    atomic_bytes(Path(path) / "training.msgpack", payload)
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_checksum"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, manifest)


def restore_training(path, train_state, rng, dataset_fingerprint, *, return_extra=False):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("resume dataset differs from the saved training dataset")
    payload = (Path(path) / "training.msgpack").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("training_checksum"):
        raise ValueError("training checkpoint checksum mismatch")
    data = serialization.msgpack_restore(payload)
    opt_state = serialization.from_state_dict(train_state.opt_state, data["optimizer"])
    train_state = train_state.replace(
        opt_state=jax.tree.map(jnp.asarray, opt_state), step=data["step"]
    )
    rng.bit_generator.state = json.loads(data["rng_state"])
    result = (
        train_state,
        float(data["best_loss"]),
        freeze(jax.tree.map(jnp.asarray, data["best_parameters"])),
        int(data["best_generation"]),
    )
    return (*result, data.get("extra_state")) if return_extra else result
