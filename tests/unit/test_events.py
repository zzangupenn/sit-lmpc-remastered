import json
from types import SimpleNamespace

from sit_lmpc.diagnostics.events import (
    Events,
    _color_enabled,
    color_console_event,
    console_event,
)


def test_risk_training_console_is_compact_while_event_file_keeps_details(tmp_path, capsys):
    events = Events(tmp_path)
    evaluation = {
        "status": "insufficient_held_out_outcomes",
        "states": {"brier": 0.01, "reliability": [{"predicted": 0.2, "observed": 0.1}]},
    }
    sampling = {
        "state_counts_before_balancing": {"training": {"success": 6036, "failure": 369}},
        "active_attempts": {"success": list(range(20)), "failure": [21, 22]},
    }
    events.emit(
        "risk_training",
        generation=3,
        loss=0.104583919,
        gradient_steps=1200,
        input_mode="state",
        seconds=0.22,
        evaluation=evaluation,
        sampling=sampling,
    )

    output = capsys.readouterr().out.strip()
    assert output == (
        "[risk_training] generation=3 | mode=state | loss=0.1046 | "
        "gradient_steps=1200 | train states S/F=6036/369 | held-out: needs both outcomes"
    )
    saved = json.loads((tmp_path / "events.jsonl").read_text())
    assert saved["time"] and saved["evaluation"] == evaluation
    assert saved["sampling"] == sampling and saved["seconds"] == 0.22


def test_common_console_events_omit_timestamps_and_runtime(capsys, tmp_path):
    events = Events(tmp_path)
    events.emit("checkpoint_saved", file="checkpoints/trial-000010", trials=10, seconds=2.5)
    events.emit("value_action_queries_warmed", seconds=0.1)
    events.emit(
        "finish",
        status="step_limit",
        trials=10,
        successful_laps=8,
        failed_attempts=2,
        applied_steps=3000,
        training_generations=7,
        risk_training_generations=3,
        elapsed=123.45,
        directory=str(tmp_path),
    )
    output = capsys.readouterr().out
    assert "[checkpoint_saved] file=checkpoints/trial-000010 | trials=10" in output
    assert "[finish] status=step_limit | trials=10 | successful_laps=8" in output
    assert f"results: {tmp_path}" in output
    assert "value_action_queries_warmed" not in output
    assert "seconds" not in output and "elapsed" not in output and "time" not in output
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 3


def test_lap_prints_mean_and_max_while_summary_event_only_saves_counts(tmp_path, capsys):
    terms = {"hull": dict(mean=2.5, max=7.5, steps=12),
             "boundary": dict(mean=0., max=0., steps=12),
             "risk": dict(mean=1.25, max=3., steps=4)}
    Events(tmp_path).emit("lambda_summary", attempt_id=5, lap=5, outcome="success", steps=12, terms=terms)
    assert capsys.readouterr().out == ""
    saved = json.loads((tmp_path / "events.jsonl").read_text())
    assert saved["terms"] == terms and saved["steps"] == 12
    Events(tmp_path).emit("lap", lap=5, laptime=32.8, mean_speed=11.68, mode="lmppi",
                          lambda_summary=dict(steps=12, terms=terms))
    assert capsys.readouterr().out.strip() == (
        "[lap] trial 5 | 32.8 s | avg speed 11.68 m/s | lmppi | lambda_ss mean=2.5 max=7.5 | "
        "lambda_boundary mean=0 max=0 | lambda_risk mean=1.25 max=3"
    )


def test_chance_fallback_prints_selected_violation_probabilities_and_bounds(tmp_path, capsys):
    metrics = {
        "boundary_probability": 0.2,
        "boundary_probability_upper": 0.24748,
        "boundary_probability_threshold": 0.05,
        "terminal_hull_probability": 1.0,
        "terminal_hull_probability_upper": 1.0,
        "terminal_hull_probability_threshold": 0.05,
        "minimum_boundary_probability": 0.0,
        "chance_feasible_candidates": 0,
    }
    Events(tmp_path).emit("probabilistic_safety_fallback", step=1908,
                         status="no_feasible_candidate", policy="sit_conservative", metrics=metrics)
    line = capsys.readouterr().out.strip()
    assert line == (
        "[probabilistic_safety_fallback] step=1908 | status=no_feasible_candidate | "
        "policy=sit_conservative | P_violate(boundary)=0.2 (upper=0.2475, limit=0.05) | "
        "P_violate(hull)=1 (upper=1, limit=0.05)"
    )
    assert color_console_event("probabilistic_safety_fallback", {}, line, enabled=True) == (
        f"\x1b[1;31m{line}\x1b[0m"
    )
    assert json.loads((tmp_path / "events.jsonl").read_text())["metrics"] == metrics


def test_unavailable_chance_fallback_does_not_print_fake_zero_probabilities():
    line = console_event("probabilistic_safety_fallback", {
        "step": 1, "status": "unavailable", "policy": "sit_conservative",
        "metrics": {"chance_unavailable_reason": "dynamics_distribution_unavailable"},
    })
    assert "P_violate(boundary)=unavailable" in line
    assert "P_violate(hull)=unavailable" in line
    assert "reason=dynamics_distribution_unavailable" in line


def test_empirical_chance_console_does_not_claim_a_confidence_bound():
    line = console_event("probabilistic_safety_fallback", {"step": 7,
        "status": "no_feasible_candidate", "policy": "sit_conservative", "metrics": {
            "probability_estimator": "empirical_planning_samples",
            "boundary_probability": 4 / 64, "terminal_hull_probability": 0.0,
            "boundary_probability_upper": None, "terminal_hull_probability_upper": None,
            "boundary_probability_threshold": 0.05, "terminal_hull_probability_threshold": 0.05}})
    assert "P_violate(boundary)=0.0625 (empirical, limit=0.05)" in line
    assert "P_violate(hull)=0 (empirical, limit=0.05)" in line
    assert "upper=" not in line and "confidence=" not in line


def test_risk_training_displays_held_out_brier_only_with_both_outcomes():
    assert console_event(
        "risk_training",
        {
            "generation": 4,
            "loss": 0.2,
            "gradient_steps": 5,
            "evaluation": {"status": "held_out_both_outcomes", "states": {"brier": 0.025}},
        },
    ).endswith("held-out Brier=0.025")


def test_risk_training_displays_ready_cutoff_without_expanding_details():
    line = console_event(
        "risk_training",
        {
            "generation": 4,
            "loss": 0.2,
            "gradient_steps": 5,
            "evaluation": {
                "status": "held_out_both_outcomes",
                "states": {"brier": 0.025},
                "safe_cutoff": {
                    "status": "ready",
                    "threshold": 0.2,
                    "observed_safe_fraction": 0.95,
                    "selected_states": 40,
                },
            },
        },
    )
    assert line.endswith("cutoff<=0.2 (success=95.0%, n=40)")


def test_manual_cutoff_prints_when_validation_is_unavailable():
    line = console_event(
        "risk_training",
        {
            "generation": 1,
            "evaluation": {
                "status": "insufficient_held_out_outcomes",
                "safe_cutoff": {
                    "status": "ready",
                    "source": "manual",
                    "threshold": 0.5,
                    "observed_safe_fraction": None,
                },
            },
        },
    )
    assert line.endswith("cutoff<=0.5 (manual)")


def test_attempt_and_lap_console_show_essential_race_metrics():
    assert (
        console_event(
            "attempt",
            {
                "attempt_id": 20,
                "outcome": "failure",
                "lap": 20,
                "steps": 124,
                "termination_reason": "out_of_bounds",
                "out_of_bounds_steps": 1,
            },
        )
        == "[attempt] #20 failure | 124 steps | out_of_bounds | 1 out-of-bounds steps"
    )
    assert (
        console_event(
            "lap",
            {
                "lap": 19,
                "laptime": 31.7,
                "mean_speed": 13.0724,
                "mode": "lmppi",
                "safe_set_violation": 0.0,
                "boundary_violation": 0.0,
                "elapsed": 71.67,
            },
        )
        == "[lap] trial 19 | 31.7 s | avg speed 13.07 m/s | lmppi"
    )


def test_nf_minimum_warning_prints_counts_and_continuing_method():
    fields = dict(reason="below_configured_minimum", training_states=531, min_training_states=128,
                  calibration_states=62, min_calibration_states=64, policy="continue_nf_density")
    line = console_event("safe_set_density_warning", fields)
    assert line == ("[safe_set_density_warning] below_configured_minimum | training=531/128 | "
                    "calibration=62/64 | continuing with nf_density")
    assert color_console_event("safe_set_density_warning", fields, line, enabled=True) == (
        f"\x1b[1;33m{line}\x1b[0m"
    )
    fields.update(calibration_states=0, policy="stop_empty_partition")
    assert "cannot fit an empty NF partition" in console_event("safe_set_density_warning", fields)


def test_density_training_console_reports_actual_fit_effort():
    line = console_event("safe_set_density_training", dict(generation=3, epochs=2, gradient_steps=6,
        batch_size=256, training_states=541, calibration_states=61, cutoff_rule="quantile"))
    assert "epochs=2" in line and "gradient_steps=6" in line and "batch_size=256" in line


def test_density_history_warning_reports_training_sample_target():
    line = console_event("safe_set_density_warning", dict(reason="below_configured_minimum",
        training_states=900, min_training_states=128, target_training_states=1280,
        calibration_states=100, min_calibration_states=64, policy="continue_nf_density"))
    assert "training=900/1280" in line
    assert "continuing with nf_density" in line


def test_only_important_event_messages_are_colored_on_a_terminal():
    keyboard = "[keyboard_started] right Alt / 1: step debug; 4: cutoff samples"
    assert color_console_event("keyboard_started", {}, keyboard, enabled=True) == (
        f"\x1b[33m{keyboard}\x1b[0m"
    )
    assert color_console_event("keyboard_paused", {}, "[keyboard_paused]", enabled=True) == (
        "\x1b[33m[keyboard_paused]\x1b[0m"
    )
    assert color_console_event("keyboard_unavailable", {}, "[keyboard_unavailable]", enabled=True) == (
        "\x1b[33m[keyboard_unavailable]\x1b[0m"
    )
    risk = "[risk_training] generation=3 | loss=0.1"
    assert color_console_event("risk_training", {}, risk, enabled=True) == (
        "\x1b[1;35m[risk_training] generation=3 | loss=0.1\x1b[0m"
    )
    failure = "[attempt] #20 failure"
    assert color_console_event("attempt", {"outcome": "failure"}, failure, enabled=True) == (
        "\x1b[1;31m[attempt] #20 failure\x1b[0m"
    )
    assert color_console_event("attempt", {"outcome": "success"}, failure, enabled=True) == failure
    assert color_console_event("risk_training", {}, risk, enabled=False) == risk
    assert color_console_event("keyboard_started", {}, keyboard, enabled=False) == keyboard


def test_color_respects_redirected_output_and_no_color():
    terminal = SimpleNamespace(isatty=lambda: True)
    redirected = SimpleNamespace(isatty=lambda: False)
    assert _color_enabled(terminal, {"TERM": "xterm-256color"})
    assert not _color_enabled(redirected, {})
    assert not _color_enabled(terminal, {"NO_COLOR": "1"})
    assert not _color_enabled(terminal, {"TERM": "dumb"})
