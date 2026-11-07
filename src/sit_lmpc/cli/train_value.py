"""Offline value training from BRO lap records; no vehicle-dynamics learning."""

import argparse
from pathlib import Path
from sit_lmpc.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, help="SIT-LMPC Remastered experiment directory or laps.npz (state-only)"
    )
    parser.add_argument(
        "--config", help="YAML settings file (default: packaged config/default.yaml)"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--resume", help="Continue from a BRO value checkpoint")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    changes = {
        "diagnostics": {"render": False, "plots": False},
        "output": {"directory": args.save_dir},
    }
    if args.epochs is not None:
        changes["value"] = {"epochs": args.epochs}
    config = load_config(args.config, changes)
    from sit_lmpc.experiments.runtime import configure

    configure(config, args.cpu)
    from sit_lmpc.experiments.runner import output_directory
    from sit_lmpc.experiments.randomness import resolve_seed, ExperimentRandomness
    from sit_lmpc.diagnostics.events import Events
    from sit_lmpc.training.value import ValueTrainer
    from sit_lmpc.planners.safe_set import SafeSet
    from sit_lmpc.data.laps import load_laps
    from sit_lmpc.track import Track, map_path

    config = resolve_seed(config)
    directory = output_directory(config)
    randomness = ExperimentRandomness(config.simulation.seed)
    track = Track.from_track_path(map_path(config.simulation.map))
    safe = SafeSet(
        config.planner,
        config.simulation.timestep,
        float(track.s_frame_max),
        config.simulation.seed,
        randomness=randomness,
    )
    if Path(args.data).is_dir():
        from sit_lmpc.data.trajectories import load_trajectories

        for trajectory in load_trajectories(args.data, "success"):
            safe.add_lap(trajectory.as_lap(), trajectory)
    else:
        if config.value.input_mode == "state_action":
            raise ValueError(
                "state-action offline training requires an experiment directory with full trajectories"
            )
        for lap in load_laps(args.data):
            safe.add_lap(lap)
    data = safe.training_data(config.value.input_mode)
    events = Events(directory)
    trainer = ValueTrainer(
        config.value,
        config.planner.half_width,
        config.simulation.seed,
        directory,
        events.emit,
        randomness=randomness,
    )
    if args.resume:
        trainer.resume(args.resume)
    try:
        trainer.start(data)
    finally:
        trainer.wait()
    events.emit(
        "offline_training_finished", source=str(Path(args.data).resolve()), directory=str(directory)
    )


if __name__ == "__main__":
    main()
