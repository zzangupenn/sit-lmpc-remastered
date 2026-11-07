"""Replay numerical lap records once, or produce a headless summary."""

import argparse
from dataclasses import replace
from pathlib import Path
import time
import json
from sit_lmpc.diagnostics.lap_metrics import measured_boundary_metrics
from sit_lmpc.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=100000)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()
    if args.max_frames < 1 or args.interval < 0:
        parser.error("max-frames must be positive and interval nonnegative")
    directory = Path(args.run).expanduser().resolve()
    config = load_config(
        directory / "config.yaml", {"diagnostics": {"render": not args.headless, "plots": False}}
    )
    from sit_lmpc.experiments.runtime import configure

    configure(config)
    from sit_lmpc.data.laps import load_laps

    laps = load_laps(directory / "laps.npz")
    from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
    from sit_lmpc.track import Track, map_path

    metrics_path = directory / "lap_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    # Older runs saved the failed/incomplete lap separately without metrics.
    # Include its measured boundary excess, but do not invent lost predictions.
    partial_path = directory / "partial_lap.npz"
    if partial_path.exists() and not any(not item.get("completed", True) for item in metrics):
        partial_laps = load_laps(partial_path)
        if partial_laps:
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            while len(metrics) < len(laps):
                metrics.append({"lap": len(metrics) + 1, "completed": True})
            metrics.append(
                {
                    "lap": len(laps) + 1,
                    "completed": False,
                    "termination_reason": summary.get("status", "unknown"),
                    **measured_boundary_metrics(partial_laps[0], config.planner.half_width),
                }
            )
    plots = MatplotlibPlots(directory, interactive=False)
    try:
        plots.summaries(
            laps,
            metrics,
            Track.from_track_path(map_path(config.simulation.map)),
            config.planner.half_width,
        )
    finally:
        plots.close()
    if args.headless:
        print(f"Read {len(laps)} laps; wrote {directory / 'summary.png'}")
        return
    from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
    from sit_lmpc.track import map_path

    simulator = F1TenthSimulator(
        config.simulation,
        map_path(config.simulation.map),
        True,
        vehicle_parameters(config.simulation.friction),
    )
    try:
        count = 0
        for lap in laps:
            for state in lap.cartesian:
                simulator.reset(state)
                simulator.render()
                count += 1
                if count >= args.max_frames:
                    return
                time.sleep(args.interval)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
