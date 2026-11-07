import argparse
from pathlib import Path
import yaml
from sit_lmpc.config import load_config, schema
from sit_lmpc.config.resume import resume_overrides


def arguments(description="Dynamic-ST simulation with online value learning"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", help="YAML settings file (default: packaged config/default.yaml)"
    )
    parser.add_argument("--map")
    parser.add_argument("--max-laps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--safe-set-size", type=int)
    parser.add_argument("--trace-steps", "--insight-steps", dest="trace_steps", type=int,
                        help="Retain this many trace and insight steps; 0 disables both")
    parser.add_argument("--save-dir")
    parser.add_argument(
        "--resume",
        help="Resume directly from a self-contained checkpoint folder",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Disable only the simulator window"
    )
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument(
        "--plots", action="store_true", help="Enable live diagnostic plots (disabled by default)"
    )
    plotting.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable live diagnostic plots without disabling the simulator window",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-scan", action="store_true", help="Use the unrolled benchmark baseline"
    )
    args = parser.parse_args()
    changes = {"simulation": {}, "planner": {}, "value": {}, "diagnostics": {}, "output": {}}
    if args.trace_steps is not None:
        changes["diagnostics"]["trace_steps"] = args.trace_steps
    for field in ("map", "max_laps", "max_steps", "seed"):
        value = getattr(args, field)
        if value is not None:
            changes["simulation"][field] = value
    if args.seed is not None:
        changes["simulation"]["seed_mode"] = "fixed"
    for field in ("samples", "horizon", "safe_set_size"):
        value = getattr(args, field)
        if value is not None:
            changes["planner"][field] = value
    if args.epochs is not None:
        changes["value"]["epochs"] = args.epochs
    if args.save_dir is not None:
        changes["output"]["directory"] = args.save_dir
    if args.resume is not None:
        changes["output"]["resume"] = args.resume
    if args.plots:
        changes["diagnostics"]["plots"] = True
    if args.no_plots:
        changes["diagnostics"]["plots"] = False
    if args.headless:
        changes["diagnostics"]["render"] = False
    if args.no_scan:
        changes["planner"]["scan_rollout"] = False
    try:
        source = (
            Path(args.config)
            if args.config is not None
            else schema.files("sit_lmpc") / "config" / "default.yaml"
        )
        raw = schema.read_config_mapping(source)
        if not isinstance(raw, dict) or not isinstance(raw.get("output", {}), dict):
            raise ValueError("configuration and output must be mappings")
        resume = args.resume if args.resume is not None else raw.get("output", {}).get("resume")
        if resume:
            resume_path = Path(resume).expanduser().resolve()
            from sit_lmpc.data.experiment_checkpoint import load_checkpoint_config

            resume_changes = resume_overrides(raw, resume_path)
            for section, fields in changes.items():
                resume_changes.setdefault(section, {}).update(fields)
            resume_changes["output"]["resume"] = str(resume_path)
            saved = load_checkpoint_config(resume_path)
            if args.map is not None and args.map != saved.simulation.map:
                parser.error("--map cannot change on resume: saved trajectories use the checkpoint map")
            for field in ("model_bank", "risk_constraint"):
                planner_changes = resume_changes.get("planner", {})
                if field in planner_changes:
                    planner_changes[field] = {
                        **saved.as_dict()["planner"][field], **planner_changes[field],
                    }
            config = load_checkpoint_config(resume_path, resume_changes)
        else:
            config = load_config(args.config, changes)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return args, config


def execute(tracking_only=False):
    args, config = arguments(
        "Dynamic-ST MPPI tracking" if tracking_only else "Dynamic-ST online LMPPI"
    )
    from sit_lmpc.experiments.runtime import configure

    configure(config, args.cpu)
    from sit_lmpc.experiments.runner import run

    result = run(config, tracking_only=tracking_only)
    if result.status == "interrupted":
        raise SystemExit(130)
    if result.status not in ("lap_limit", "step_limit"):
        raise SystemExit(2)


def main():
    execute()


if __name__ == "__main__":
    main()
