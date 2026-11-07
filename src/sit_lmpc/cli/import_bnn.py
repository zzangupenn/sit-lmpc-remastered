"""Import a validated numeric ST population checkpoint without experiment data."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import shutil


def import_model(source, destination):
    from dyna_dist.io.checkpoint import load_model

    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    # Validate checksum, model format and domain before creating the destination.
    model = load_model(source)
    if model.config.domain != "st" or not model.config.bayesian:
        raise ValueError("Expected a Bayesian ST-domain dynamics population")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty model directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("manifest.json", "model.msgpack"):
        shutil.copy2(source / name, destination / name)
        hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    record = dict(source=str(source), fingerprint=model.fingerprint, latent_dimension=model.config.latent_dimension,
                  timestep=model.config.timestep, integrator_timestep=model.config.integrator_timestep,
                  copied_at=datetime.now(ZoneInfo("America/New_York")).isoformat(), sha256=hashes)
    (destination / "import-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", default="artifacts/vehicle-bnn-v2-l96")
    args = parser.parse_args()
    try:
        record = import_model(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"[bnn_imported] latent={record['latent_dimension']} | fingerprint={record['fingerprint']} | destination={Path(args.destination).resolve()}")
