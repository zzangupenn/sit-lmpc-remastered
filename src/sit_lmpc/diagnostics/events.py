from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import threading
import numpy as np


_CONSOLE_FIELDS = {
    "insight_saved": ("file", "steps", "reason"),
    "safe_set_density_training": ("method", "generation", "epochs", "gradient_steps", "batch_size", "learning_rate", "cumulative_learning_rate", "training_laps", "training_states", "calibration_states", "cutoff", "relaxation_cutoff", "upper_cutoff", "calibration_coverage", "test_coverage"),
    "safe_set_density_status": ("status", "fallback", "reason", "generation"),
    "safe_set_density_warning": ("reason", "training_states", "min_training_states", "calibration_states", "min_calibration_states", "policy"),
    "dynamics_model_loaded": ("kind", "latent_dimension", "particles", "adaptation", "numerical_substep_mismatch"),
    "checkpoint_saved": ("file", "trials"),
    "hull_precompile_started": ("variants", "max_facets", "dtype"),
    "hull_precompile_finished": ("variants",),
    "initial_snapshot_saved": ("file",),
    "keyboard_paused": ("step", "message"),
    "keyboard_received": ("key", "message"),
    "lap_reset": ("next_attempt", "failures"),
    "learned_controller_started": ("step", "value_generation", "hull_facets"),
    "model_bank_warmup": ("frictions",),
    "offline_training_finished": ("source", "directory"),
    "partial_lap": ("attempt_id", "steps", "termination_reason"),
    "quick_load": ("safe_set_laps", "epochs"),
    "resume": ("checkpoint", "trials", "next_step"),
    "training": ("generation", "epochs", "loss", "gradient_steps"),
    "dynamics_training": ("generation", "status", "backend", "weights", "fit_rmse", "energy_score"),
    "dynamics_posterior_training": ("generation", "status", "posterior_method", "new_training_transitions", "training_transitions", "likelihood_power", "heldout_xy10_rmse", "heldout_coverage10", "heldout_xy15_rmse", "heldout_coverage15"),
    "dynamics_posterior_restored": ("generation", "training_transitions", "rebuilt", "likelihood_power"),
    "dynamics_observed": ("attempt_id", "valid_transitions", "rejected_transitions"),
    "dynamics_belief_restored": ("generation", "weights"),
    "probabilistic_safety_stop": ("reason", "detail", "feasible_candidates"),
    "probabilistic_safety_status": ("status", "generation", "step", "estimator", "samples", "reason"),
    "probabilistic_safety_fallback": ("step", "status", "policy"),
}
_QUIET_EVENTS = {"value_action_queries_warmed", "lambda_summary"}
_HIDDEN_FIELDS = {"time", "seconds", "elapsed", "directory", "evaluation", "sampling", "trace"}
_EVENT_COLORS = {
    "safe_set_density_training": "1;36",
    "safe_set_density_status": "1;33",
    "safe_set_density_warning": "1;33",
    "keyboard_started": "33",  # Plain yellow for all keyboard messages.
    "keyboard_paused": "33",
    "risk_training": "1;35",  # Bright magenta.
    "risk_cutoff_updated": "1;35",
    "risk_constraint_status": "1;33",  # Warm-up and activation state.
    "risk_no_feasible_candidate": "1;31",  # Explicitly unconstrained fallback.
    "probabilistic_safety_stop": "1;31",
    "probabilistic_safety_fallback": "1;31",
    "probabilistic_safety_status": "1;33",
    "training": "1;36",  # Bright cyan.
    "dynamics_training": "1;36",
    "dynamics_posterior_training": "1;36",
    "dynamics_posterior_warning": "1;33",
    "dynamics_belief_refit": "1;33",
    "checkpoint_saved": "1;32",  # Bright green.
    "lap": "1;32",
    "resume": "1;34",  # Bright blue.
}


def _display_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


def _display_fields(event, fields):
    keys = _CONSOLE_FIELDS.get(event)
    if keys is None:
        keys = [
            key
            for key, value in fields.items()
            if key not in _HIDDEN_FIELDS and not isinstance(value, dict)
        ][:4]
    return [f"{key}={_display_value(fields[key])}" for key in keys if fields.get(key) is not None]


def _lambda_summary_fields(summary):
    parts = []
    for name, values in (summary or {}).get("terms", {}).items():
        label = "ss" if name == "hull" else name
        parts.append(f"lambda_{label} mean={_display_value(values['mean'])} "
                     f"max={_display_value(values['max'])}")
    return parts


def console_event(event, fields):
    """A short terminal view; the JSONL event keeps the complete record."""
    if event == "insight_step":
        return f"[insight step {fields['position']}/{fields['total']}]"
    if event in _QUIET_EVENTS:
        return None
    if event == "start":
        mode = "tracking" if fields.get("tracking_only") else "learning"
        parts = [
            mode,
            f"seed={fields['seed']}",
            f"friction actual/model={_display_value(fields['simulator_friction'])}/"
            + ("ensemble" if fields.get("predictor_kind") == "ensemble" else _display_value(fields['predictor_friction'])),
            f"value={fields['value_input_mode']}",
        ]
        if fields.get("risk_input_mode") is not None:
            parts.append(f"risk={fields['risk_input_mode']}")
    elif event == "attempt":
        parts = [f"#{fields['attempt_id']} {fields['outcome']}"]
        if fields.get("lap") != fields["attempt_id"]:
            parts.append(f"lap={fields['lap']}")
        parts.append(f"{fields['steps']} steps")
        if fields["outcome"] != "success" and fields.get("termination_reason"):
            parts.append(fields["termination_reason"])
        if fields.get("out_of_bounds_steps"):
            parts.append(f"{fields['out_of_bounds_steps']} out-of-bounds steps")
        if fields["outcome"] != "success":
            parts.extend(_lambda_summary_fields(fields.get("lambda_summary")))
    elif event == "lap":
        parts = [
            f"trial {fields['lap']}",
            f"{_display_value(fields['laptime'])} s",
            f"avg speed {_display_value(fields['mean_speed'])} m/s",
            fields["mode"],
        ]
        for key, label in (("safe_set_violation", "safe-set"), ("boundary_violation", "boundary")):
            if fields.get(key):
                parts.append(f"{label} violation={_display_value(fields[key])}")
        parts.extend(_lambda_summary_fields(fields.get("lambda_summary")))
    elif event == "safe_set_density_warning":
        training_minimum = max(fields["min_training_states"], fields.get("target_training_states", 0))
        parts = [fields["reason"],
                 f"training={fields['training_states']}/{training_minimum}",
                 f"calibration={fields['calibration_states']}/{fields['min_calibration_states']}",
                 "continuing with nf_density" if fields["policy"] == "continue_nf_density"
                 else "cannot fit an empty NF partition"]
        if fields.get("safe_set_min_states"):
            parts.insert(1, f"NF history={fields['retained_states']}/{fields['safe_set_min_states']}")
    elif event == "selected_lambdas":
        parts = [f"step={fields['step']}"]
        for name, value in fields["lambdas"].items():
            label = "ss" if name == "hull" else name
            parts.append(f"lambda_{label}={_display_value(value)}")
    elif event == "risk_training":
        parts = _display_fields("training", fields)
        if fields.get("input_mode") is not None:
            parts.insert(1, f"mode={fields['input_mode']}")
        counts = (
            fields.get("sampling", {}).get("state_counts_before_balancing", {}).get("training", {})
        )
        if counts:
            parts.append(f"train states S/F={counts.get('success', 0)}/{counts.get('failure', 0)}")
        evaluation = fields.get("evaluation") or {}
        cutoff = evaluation.get("safe_cutoff") or {}
        if evaluation.get("status") == "held_out_both_outcomes":
            states = evaluation.get("states") or {}
            if states.get("brier") is not None:
                parts.append(f"held-out Brier={_display_value(states['brier'])}")
            if cutoff.get("status") == "ready" and cutoff.get("source") != "manual":
                parts.append(
                    f"cutoff<={_display_value(cutoff['threshold'])} "
                    f"(success={cutoff['observed_safe_fraction']:.1%}, n={cutoff['selected_states']})"
                )
            elif cutoff and cutoff.get("status") != "ready":
                parts.append(f"cutoff unavailable: {cutoff['status']}")
        elif evaluation.get("status") == "insufficient_held_out_outcomes":
            parts.append("held-out: needs both outcomes")
        if cutoff.get("status") == "ready" and cutoff.get("source") == "manual":
            parts.append(f"cutoff<={_display_value(cutoff['threshold'])} (manual)")
    elif event == "risk_not_ready":
        counts = fields.get("state_counts", {}).get("training", {})
        parts = ["waiting for success and failure data"]
        if counts:
            parts.append(f"train states S/F={counts.get('success', 0)}/{counts.get('failure', 0)}")
    elif event == "probabilistic_safety_fallback":
        parts = _display_fields(event, fields)
        metrics = fields.get("metrics") or {}
        for name, label in (("boundary", "boundary"), ("terminal_hull", "hull")):
            probability = metrics.get(f"{name}_probability")
            if probability is None:
                parts.append(f"P_violate({label})=unavailable")
                continue
            detail = [
                f"{key}={_display_value(metrics[f'{name}_probability_{suffix}'])}"
                for key, suffix in (("upper", "upper"), ("limit", "threshold"))
                if metrics.get(f"{name}_probability_{suffix}") is not None
            ]
            if metrics.get("probability_estimator") == "empirical_planning_samples":
                detail.insert(0, "empirical")
            parts.append(f"P_violate({label})={_display_value(probability)}"
                         + (f" ({', '.join(detail)})" if detail else ""))
        if metrics.get("chance_unavailable_reason"):
            parts.append(f"reason={metrics['chance_unavailable_reason']}")
    elif event == "finish":
        keys = (
            "status",
            "trials",
            "successful_laps",
            "failed_attempts",
            "applied_steps",
            "training_generations",
            "risk_training_generations",
        )
        parts = [
            f"{key}={_display_value(fields[key])}" for key in keys if fields.get(key) is not None
        ]
    elif event == "keyboard_started":
        parts = [fields["shortcuts"]] if fields.get("shortcuts") else []
    elif event == "error" or event.endswith("_error") or event == "keyboard_unavailable":
        parts = _display_fields(event, fields)
    else:
        parts = _display_fields(event, fields)
    line = f"[{event}]" + (" " + " | ".join(parts) if parts else "")
    if event == "finish" and fields.get("directory"):
        line += f"\n  results: {fields['directory']}"
    return line


def color_console_event(event, fields, line, *, enabled):
    """Color entire selected event messages; leave ordinary events unchanged."""
    if not enabled:
        return line
    color = _EVENT_COLORS.get(event)
    if event.startswith("keyboard_"):
        color = "33"
    elif event == "attempt" and fields.get("outcome") == "failure":
        color = "1;31"  # Bright red.
    elif event == "error" or event.endswith("_error") or event == "keyboard_unavailable":
        color = "1;31"
    elif event == "finish":
        color = "1;32" if fields.get("status") in ("lap_limit", "step_limit") else "1;33"
    if color is None:
        return line
    return f"\x1b[{color}m{line}\x1b[0m"


def _color_enabled(stream, environment):
    return stream.isatty() and "NO_COLOR" not in environment and environment.get("TERM") != "dumb"


class Events:
    def __init__(self, directory, trace_steps=50):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.trace = deque(maxlen=trace_steps)

    def emit(self, event, **fields):
        record = {"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields}
        line = json.dumps(record, allow_nan=False)
        with self.lock:
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(line + "\n")
            display = console_event(event, fields)
            if display is not None:
                print(
                    color_console_event(
                        event, fields, display, enabled=_color_enabled(sys.stdout, os.environ)
                    ),
                    flush=True,
                )

    def record(self, state, predicted, action, lap, mode, metrics):
        if not self.trace.maxlen:
            return
        self.trace.append(
            {
                "state": np.asarray(state).tolist(),
                "predicted": np.asarray(predicted).tolist(),
                "action": np.asarray(action).tolist(),
                "lap": lap,
                "mode": mode,
                "metrics": metrics,
            }
        )

    @property
    def tracing(self):
        return bool(self.trace.maxlen)

    def save_trace(self):
        (self.directory / "trace.json").write_text(
            json.dumps(list(self.trace), indent=2, allow_nan=False) + "\n"
        )
