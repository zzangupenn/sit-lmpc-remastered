"""Reevaluate frozen baseline models after an inference-only numerical correction."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dyna_dist.config import VehicleBNNEvaluationConfig, load_config, write_config
from dyna_dist.io.common import new_directory, write_json
from .training_options import average


def recheck_baseline(baseline, output):
    source = Path(baseline).resolve()
    root = new_directory(Path(output).resolve())
    plan = json.loads((source / "plan.json").read_text())
    if plan["status"] != "complete":
        raise ValueError("baseline must be complete")
    plan.update(status="running", model_source_root=str(source), inference="stored Gaussian mean/factor")
    write_json(root / "plan.json", plan)
    (root / "logs").mkdir()
    evaluator = Path(sys.executable).parent / "dyna-evaluate-vehicle-bnn"
    environment = dict(os.environ, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    try:
        for option in (1, 2):
            config = replace(load_config(VehicleBNNEvaluationConfig,
                                          source / f"evaluation-option-{option}/config.yaml"),
                             training_run=str(source / f"option-{option}"),
                             output=str(root / f"evaluation-option-{option}"))
            path = root / f"evaluate-option-{option}.yaml"
            write_config(config, path)
            print(f"[baseline-recheck] option {option}: frozen models, same evaluation inputs", flush=True)
            started = time.monotonic()
            with (root / f"logs/evaluation-option-{option}.log").open("w") as stream:
                subprocess.run([str(evaluator), "--config", str(path)], check=True,
                               env=environment, stdout=stream, stderr=subprocess.STDOUT)
            plan["timings"][f"evaluation-option-{option}"] = time.monotonic() - started
        fields = ("terminal_position_rmse_m", "simultaneous_tube_95_coverage", "path_energy_distance_m",
                  "wasserstein1_path_m", "terminal_radius_95_m", "simultaneous_tube_radius_m",
                  "physical_violation_fraction")
        rows = []
        for option in (1, 2):
            metrics = json.loads((root / f"evaluation-option-{option}/metrics.json").read_text())
            for latent in plan["latents"]:
                for horizon in (1, 5, 10, 15):
                    rows.append({"option": option, "latent": latent, "horizon": horizon,
                                 **{field: average(metrics, latent, horizon, field) for field in fields},
                                 "endpoint_rmse_by_split": {
                                     split: average(metrics, latent, horizon, "terminal_position_rmse_m", split)
                                     for split in ("train", "validation", "test")}})
        write_json(root / "comparison.json", {"rows": rows, "model_source_root": str(source)})
        lines = ["# Baseline re-evaluation with stable prior inference", "",
                 "No model was retrained or modified. This repeats the original evaluation inputs and "
                 "seeds, using the encoded Gaussian mean/factor directly rather than converting through "
                 "an ill-conditioned precision matrix. Observation noise remains excluded from function "
                 "rollouts. Original evaluation artifacts are preserved at the source study.", "",
                 "| Option | Latent | 15-step RMSE [m] | W1 [m] | Path coverage |",
                 "| ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            if row["horizon"] == 15:
                lines.append(f"| {row['option']} | {row['latent']} | {row['terminal_position_rmse_m']:.4f} | "
                             f"{row['wasserstein1_path_m']:.4f} | {row['simultaneous_tube_95_coverage']:.1%} |")
        lines += ["", f"[Original frozen models]({source}) · "
                  "[Option 1 evaluation](evaluation-option-1/REPORT.md) · "
                  "[Option 2 evaluation](evaluation-option-2/REPORT.md).", ""]
        (root / "REPORT.md").write_text("\n".join(lines))
        plan["status"] = "complete"
        write_json(root / "plan.json", plan)
        return root
    except Exception as exc:
        plan.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(root / "plan.json", plan)
        raise
