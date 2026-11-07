"""Collection/training/evaluation use separate explicit configurations."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
from .config import (
    CollectionConfig,
    VehicleCollectionConfig,
    VehicleStudyConfig,
    TrainingConfig,
    EvaluationConfig,
    VehicleBNNEvaluationConfig,
    load_config,
    resolve_seed,
)


def collect_main(argv=None):
    parser = argparse.ArgumentParser(description="Collect paired Gym ST dynamics instances")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    from .data.collector import collect

    config = load_config(CollectionConfig, args.config, {"output": args.output, "seed": args.seed})
    print(f"[saved] dataset: {collect(config)}")
    return 0


def collect_vehicle_main(argv=None):
    parser = argparse.ArgumentParser(description="Collect paired full-scale vehicle ST trajectories")
    parser.add_argument("--config", default="configs/collect_vehicle.yaml")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    from .data.vehicle_collection import collect_vehicle

    config = load_config(
        VehicleCollectionConfig, args.config, {"output": args.output, "seed": args.seed}
    )
    print(f"[saved] vehicle dataset and plots: {collect_vehicle(config)}")
    return 0


def collect_vehicle_study_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reuse paired vehicle controls and add unseen physics for BNN testing"
    )
    parser.add_argument("--config", default="configs/vehicle_bnn_study.yaml")
    parser.add_argument("--source-dataset")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    from .data.vehicle_study import build_vehicle_study

    config = load_config(
        VehicleStudyConfig, args.config,
        {name: getattr(args, name) for name in ("source_dataset", "output", "seed")},
    )
    print(f"[saved] vehicle BNN study dataset: {build_vehicle_study(config)}")
    return 0


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train a hierarchical Gaussian dynamics model")
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--latent-dimension", type=int)
    parser.add_argument("--training-option", type=int, choices=(1, 2))
    parser.add_argument("--resume-path")
    parser.add_argument("--init-model-path")
    args = parser.parse_args(argv)
    if args.resume_path and args.init_model_path:
        parser.error("choose --resume-path or --init-model-path")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from .training.trainer import train

    config = load_config(
        TrainingConfig,
        args.config,
        {
            name: getattr(args, name)
            for name in ("dataset", "output", "seed", "init_model_path", "training_option")
        },
    )
    if args.latent_dimension is not None:
        config = replace(
            config, model=replace(config.model, latent_dimension=args.latent_dimension)
        ).validate()
    if args.resume_path:
        config = replace(config, resume_path=args.resume_path, init_model_path=None).validate()
    print(f"[saved] training run: {train(config)}")
    return 0


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate held-out dynamics functions and rollouts")
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--model", dest="model_path")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split")
    parser.add_argument("--plots", action="store_true", default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from .evaluation.evaluate import evaluate

    config = load_config(
        EvaluationConfig,
        args.config,
        {
            name: getattr(args, name)
            for name in ("dataset", "model_path", "output", "seed", "split", "plots")
        },
    )
    print(f"[saved] evaluation: {evaluate(config)}")
    return 0


def evaluate_vehicle_bnn_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare vehicle BNN function samples with sampled physical ST models"
    )
    parser.add_argument("--config", default="configs/vehicle_bnn/evaluate.yaml")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--refresh-path", help="Refresh saved evaluation metrics and compact report")
    args = parser.parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from .evaluation.vehicle_bnn import evaluate_vehicle_bnn, refresh_vehicle_bnn_evaluation

    if args.refresh_path:
        if args.output is not None or args.seed is not None:
            parser.error("--refresh-path uses the saved configuration; omit --output and --seed")
        print(f"[saved] refreshed vehicle BNN report: {refresh_vehicle_bnn_evaluation(args.refresh_path)}")
        return 0

    config = load_config(
        VehicleBNNEvaluationConfig, args.config,
        {"output": args.output, "seed": args.seed},
    )
    print(f"[saved] vehicle BNN evaluation: {evaluate_vehicle_bnn(config)}")
    return 0


def sample_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample fixed dynamics functions and replay controls"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--instance")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="runs/samples")
    args = parser.parse_args(argv)
    if args.context < 0 or min(args.particles, args.horizon) < 1:
        parser.error("context must be nonnegative; particles and horizon must be positive")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import json
    import jax
    import numpy as np
    from .evaluation.evaluate import rollout_window
    from .io.checkpoint import load_model
    from .io.common import new_directory, write_json
    from .io.dataset import Dataset
    from .training.trainer import precision

    manifest = json.loads((Path(args.model) / "manifest.json").read_text())
    precision(manifest["model_config"]["dtype"])
    posterior = load_model(args.model).prior()
    seed = resolve_seed(args.seed)
    truth = None
    if posterior.model.config.domain == "st":
        initial = np.array([0, 0, 0.02, 8, 0, 0, 0])
        controls = np.tile([0.02, 0.0], (args.horizon, 1))
    else:
        initial = np.zeros(18)
        controls = np.zeros((args.horizon, 6))
    if args.dataset:
        dataset = Dataset(args.dataset)
        if dataset.domain.name != posterior.model.config.domain:
            parser.error("dataset domain differs from trained model")
        if (
            dataset.domain.name == "st"
            and dataset.manifest["config"]["profile"] != posterior.model.config.baseline_profile
        ):
            parser.error("dataset vehicle profile differs from trained profile")
        members = [m for m in dataset.instances if m["instance_id"] == args.instance]
        if args.instance is None:
            members = dataset.members("test")
        if not members:
            parser.error("requested instance/test split does not exist")
        data = dataset.transitions(members[0])
        if args.context > len(data):
            parser.error("context exceeds available transitions")
        posterior = posterior.condition(data.take(np.arange(args.context)))
        window = rollout_window(dataset, members[0], posterior.seen_ids, args.horizon)
        if window is None:
            parser.error("no complete rollout disjoint from context")
        initial, controls, truth = window
    samples = posterior.sample_functions(jax.random.key(seed), args.particles)
    paths = np.asarray(samples.rollout(initial, controls))
    root = new_directory(args.output)
    payload = {
        "states": paths,
        "coefficients": np.asarray(samples.coefficients),
        "weights": np.asarray(samples.weights),
        "initial_state": initial,
        "controls": controls,
    }
    if truth is not None:
        payload["gym_truth"] = truth
    np.savez_compressed(root / "samples.npz", **payload)
    posterior.save(root / "posterior")
    write_json(
        root / "config.json",
        {**vars(args), "seed": seed, "model_fingerprint": posterior.model.fingerprint},
    )
    print(f"[saved] function samples: {root}")
    return 0
