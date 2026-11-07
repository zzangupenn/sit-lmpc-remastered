"""Recoverable outputs; refuse accidental overwrites of existing runs."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import tempfile


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def versions():
    result = {}
    for name in (
        "dyna-dist",
        "jax",
        "flax",
        "optax",
        "numpy",
        "f1tenth-gym",
        "gymnasium",
        "mujoco",
        "stable-baselines3",
        "torch",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
