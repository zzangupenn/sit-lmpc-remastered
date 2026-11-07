"""Inspect saved MPPI decisions, including the steps leading to a failure."""

import argparse
from pathlib import Path

from sit_lmpc.diagnostics.insight import load_insight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path, help="Experiment directory; opens its failure recording when available")
    source.add_argument("--trace", type=Path, help="Explicit BRO insight.npz or insight_before_crash.npz")
    parser.add_argument("--index", type=int, default=-1, help="Initial recorded frame, starting at 0; -1 selects the newest")
    parser.add_argument("--headless", action="store_true", help="Validate and describe the archive without opening Gym")
    args = parser.parse_args()
    if args.run is None and args.trace is None:
        paths = [path for name in ("insight.npz", "insight_before_crash.npz")
                 for path in Path("results").glob(f"*/*/{name}")]
        if not paths:
            parser.error("No insight recordings under results/. Start a new experiment with --trace-steps 50, "
                         "or pass --run /path/to/run.")
        args.run = max(paths, key=lambda path: path.stat().st_mtime).parent
    if args.run is not None:
        directory = args.run.expanduser().resolve()
        path = directory / "insight_before_crash.npz"
        if not path.is_file():
            path = directory / "insight.npz"
    else:
        path = args.trace.expanduser().resolve()
    if not path.is_file():
        parser.error(f"No insight recording found at {path}. Start a new run with --trace-steps 50 "
                     "(already enabled in configs/nf-safe-set.yaml). Older trace.json files do not contain "
                     "the sampled rollouts needed for insight.")
    try:
        recording = load_insight(path)
        frames = [frame for frame in recording["steps"] if frame.get("applied")]
        if not frames:
            raise ValueError("No completed physical steps in this recording")
        index = args.index if args.index >= 0 else len(frames) + args.index
        if not 0 <= index < len(frames):
            raise ValueError(f"index must select one of {len(frames)} recorded frames")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Loaded {len(frames)} frames from {path}; stop reason: {recording['reason']}")
    if args.headless:
        return
    from sit_lmpc.diagnostics.native_insight import replay_gym

    replay_gym(recording, frames, index, path.parent)


if __name__ == "__main__":
    main()
