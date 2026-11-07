from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip("matplotlib")
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
from sit_lmpc.planners.common import PlannerResult
from sit_lmpc.data.laps import Lap
from sit_lmpc.diagnostics.lap_metrics import summarize_lap


class Track:
    s_frame_max = 100

    def frenet_to_cartesian(self, s, ey, yaw):
        return (s, ey, yaw)


@pytest.mark.parametrize("spread", [0.0, 0.001, 1.0])
@pytest.mark.parametrize("with_safe_set", [False, True])
def test_sampled_dynamics_view_plots_all_functions_mean_and_truth_without_export(
    tmp_path, spread, with_safe_set
):
    from sit_lmpc.diagnostics.model_comparison import Comparison

    candidates = np.zeros((8, 5, 7))
    candidates[:, :, 0] = np.arange(5)
    candidates[:, :, 1] = spread * np.linspace(-1, 1, 8)[:, None] * np.linspace(0, 1, 5)
    mean = candidates.mean(axis=0)
    truth = mean.copy()
    truth[:, 1] += np.linspace(0, 0.1, 5)
    comparison = Comparison(
        kind="planned", step=12, actions=np.zeros((4, 2)), candidates=candidates,
        belief=mean, true_reference=truth, actual=mean[:1], frictions=(None,) * 8,
        belief_friction=None, true_friction=0.5, timestep=0.1, evaluation_seconds=0,
        model_weights=np.full(8, 1 / 8), belief_generation=7, belief_mode="gaussian_bnn",
        weight_kind="gaussian_function_samples",
    )
    safe_states = np.zeros((6, 8)) if with_safe_set else None
    if with_safe_set:
        safe_states[:, 0] = np.arange(6)
        safe_states[:, 1] = 0.2
        safe_states[-1, :2] = 1000  # Full safe set must not zoom out the local trajectory view.
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        figure = plots.model_comparison(comparison, Track(), 4, safe_set_states=safe_states)
        figure.canvas.draw()
        ax = figure.axes[0]
        labels = ax.get_legend_handles_labels()[1]
        assert labels == (["Safe-set states"] if with_safe_set else []) + [
            "Sampled dynamics functions (n=8)", "BNN sampled-function mean",
            "Ground-truth ST trajectory mu=0.5",
            "Measured initial state (future unobserved)",
        ]
        if with_safe_set:
            np.testing.assert_array_equal(ax.collections[0].get_offsets(), safe_states[:, :2])
            assert ax.get_xlim()[1] < 1000 and ax.get_ylim()[1] < 1000
        for line, path in zip(ax.lines[2:10], candidates, strict=True):
            np.testing.assert_array_equal(line.get_xdata(), path[:, 0])
            np.testing.assert_array_equal(line.get_ydata(), path[:, 1])
            assert line.get_color() == "lightsteelblue" and line.get_alpha() == 0.4
        assert ax.lines[10].get_color() == "royalblue"  # Mean remains distinct.
        truth_line = ax.lines[11]
        assert truth_line.get_color() == "tab:orange"
        assert truth_line.get_linewidth() == 0.9
        np.testing.assert_array_equal(truth_line.get_xydata(), truth[:, :2])
        title = figure._suptitle.get_text()
        assert "Sampled dynamics trajectories" in title
        assert "Belief generation 7" in title
        assert "one fixed function per path" in title
        assert len(figure.axes) == 2 and not ax.child_axes
        zoom = figure.axes[1]
        assert zoom.get_position().x0 > ax.get_position().x1
        assert np.ptp(zoom.get_xlim()) == pytest.approx(np.ptp(zoom.get_ylim()))
        assert zoom.get_xlabel() == "Δx (mm)" and zoom.get_ylabel() == "Δy (mm)"
        expected = (candidates[:, -1, :2] - mean[-1, :2]) * 1000
        np.testing.assert_allclose(zoom.collections[0].get_offsets(), expected)
        zoom_truth = next(
            line for line in zoom.lines if line.get_label() == "Ground-truth ST trajectory"
        )
        expected_truth = (truth[-2:, :2] - mean[-1, :2]) * 1000
        np.testing.assert_allclose(zoom_truth.get_xydata(), expected_truth)
        assert zoom_truth.get_color() == truth_line.get_color()
        assert zoom_truth.get_linewidth() == truth_line.get_linewidth()
        for coordinate, limits in zip(expected_truth[-1], (zoom.get_xlim(), zoom.get_ylim())):
            assert limits[0] < coordinate < limits[1]
        assert "8 sampled paths | terminal range:" in zoom.texts[0].get_text()
        assert plots.model_comparison(
            comparison, Track(), 4, safe_set_states=safe_states
        ) is figure
        assert len(figure.axes) == 2 and not figure.axes[0].child_axes
        assert len(plots.figures) == 1 and not list(tmp_path.iterdir())
    finally:
        plots.close()


def test_all_matplotlib_summary_exports(tmp_path):
    states = np.zeros((10, 7))
    states[:, 0] = np.arange(10)
    states[:, 3] = 7
    lap = Lap(states, states.copy(), np.zeros((10, 2)), np.arange(1, 11) / 10)
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries(
            [lap],
            [
                {
                    "safe_set_max": 0.1,
                    "safe_set_mean": 0.02,
                    "boundary_max": 0.3,
                    "boundary_mean": 0.04,
                }
            ],
            Track(),
            4,
        )
    finally:
        plots.close()
    for name in (
        "summary",
        "lap_time_speed",
        "best_lap_speed",
        "safe_set_violations",
        "boundary_violations",
        "predicted_boundary_violations",
    ):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_failed_attempt_is_in_iteration_plots_but_not_best_lap(tmp_path):
    states = np.zeros((3, 7))
    states[:, 3] = 7
    completed = Lap(states.copy(), states.copy(), np.zeros((3, 2)), np.arange(1, 4) / 10)
    failed_states = states.copy()
    failed_states[-1, 1] = -4.2
    partial = replace(completed, frenet=failed_states)
    metrics = [
        summarize_lap(
            completed, 1, 4, [(0, 0)] * 3, completed=True, termination_reason="lap_completed"
        ),
        summarize_lap(
            partial, 2, 4, [(0, 0)] * 3, completed=False, termination_reason="out_of_bounds"
        ),
    ]
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries([completed], metrics, Track(), 4)
        boundary = plots.summary_figures["boundary_violations"].axes[0]
        np.testing.assert_allclose(boundary.lines[0].get_ydata(), [0, 0.2])
        assert boundary.lines[1].get_label() == "unfinished: out_of_bounds"
        assert boundary.lines[1].get_marker() == "x"
        lap_times = plots.summary_figures["lap_time_speed"].axes[0]
        assert len(lap_times.lines[0].get_ydata()) == 2
        assert np.isnan(lap_times.lines[0].get_ydata()[1])
        assert lap_times.lines[1].get_marker() == "x"
        assert lap_times.lines[1].get_transform() is lap_times.get_xaxis_transform()
        np.testing.assert_array_equal(lap_times.lines[1].get_xdata(), [2])
        predicted = plots.summary_figures["predicted_boundary_violations"].axes[0]
        np.testing.assert_allclose(predicted.lines[0].get_ydata(), [0, 0])
        from matplotlib.layout_engine import TightLayoutEngine

        assert all(
            isinstance(f.get_layout_engine(), TightLayoutEngine)
            for f in plots.summary_figures.values()
        )
    finally:
        plots.close()


def test_failure_before_first_completed_lap_still_produces_boundary_summary(tmp_path):
    states = np.zeros((1, 7))
    states[0, 1] = 4.2
    lap = Lap(states, states.copy(), np.zeros((1, 2)), np.asarray([0.1]))
    metrics = summarize_lap(
        lap, 1, 4, [(0, 0)], completed=False, termination_reason="out_of_bounds"
    )
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries([], [metrics], Track(), 4)
        assert (tmp_path / "boundary_violations.png").exists()
        assert (tmp_path / "lap_time_speed.png").exists()
        assert not len(plots.summary_figures["lap_time_speed"].axes[0].get_yticks())
        for ax in plots.summary_figures["lap_time_speed"].axes:
            np.testing.assert_array_equal(ax.get_xticks(), np.rint(ax.get_xticks()))
        assert not (tmp_path / "best_lap_speed.png").exists()
        assert plots.summary_figures["boundary_violations"].axes[0].lines[0].get_ydata()[
            0
        ] == pytest.approx(0.2)
    finally:
        plots.close()


def test_legacy_completed_lap_measured_boundary_comes_from_states(tmp_path):
    states = np.zeros((1, 7))
    states[0, 1] = 4.3
    lap = Lap(states, states.copy(), np.zeros((1, 2)), np.asarray([0.1]))
    old_metrics = [{"boundary_max": 0, "boundary_mean": 0}]
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries([lap], old_metrics, Track(), 4)
        assert plots.summary_figures["boundary_violations"].axes[0].lines[0].get_ydata()[
            0
        ] == pytest.approx(0.3)
        assert "actual_boundary_max" not in old_metrics[0]
    finally:
        plots.close()


def test_multiple_failure_summaries_keep_metric_titles_and_do_not_repeat_legend(tmp_path):
    records = [
        {
            "attempt_id": i,
            "lap": 1,
            "completed": False,
            "outcome": "failure",
            "termination_reason": "out_of_bounds",
            "actual_boundary_max": 0.2,
            "actual_boundary_mean": 0.05,
            "boundary_max": 0,
            "boundary_mean": 0,
            "safe_set_max": 0,
            "safe_set_mean": 0,
        }
        for i in (1, 2)
    ]
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries([], records, Track(), 4)
        ax = plots.summary_figures["boundary_violations"].axes[0]
        assert ax.get_title() == "Measured boundary excess: max" and ax.get_xlabel() == "iteration"
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
        assert [text.get_text() for text in ax.legend_.get_texts()] == ["failed: out_of_bounds"]
    finally:
        plots.close()


def test_iteration_plots_include_failed_trials_and_missing_values_are_axis_markers(tmp_path):
    good_states = np.zeros((3, 7))
    good_states[:, 3] = 7
    good = Lap(good_states, good_states.copy(), np.zeros((3, 2)), np.asarray([1, 2, 3]))
    failed_states = good_states.copy()
    failed_states[:, 3] = 99
    failed = Lap(failed_states, failed_states.copy(), np.zeros((3, 2)), np.asarray([0.1, 0.2, 0.3]))
    records = [
        {
            **summarize_lap(
                failed, 1, 4, [(0, 0)] * 3, completed=False, termination_reason="out_of_bounds"
            ),
            "attempt_id": 1,
            "outcome": "failure",
        },
        {
            **summarize_lap(
                good, 2, 4, [(0, 0)] * 3, completed=True, termination_reason="lap_completed"
            ),
            "attempt_id": 2,
            "outcome": "success",
        },
        {
            "attempt_id": 3,
            "outcome": "failure",
            "completed": False,
            "termination_reason": "out_of_bounds",
            "safe_set_max": None,
            "safe_set_mean": np.inf,
            "boundary_max": np.nan,
        },
    ]
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        plots.summaries([good], records, Track(), 4)
        lap_time, mean_speed = plots.summary_figures["lap_time_speed"].axes
        np.testing.assert_array_equal(lap_time.lines[0].get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(
            lap_time.lines[0].get_ydata(), [np.nan, 3, np.nan], equal_nan=True
        )
        np.testing.assert_allclose(mean_speed.lines[0].get_ydata(), [99, 7, np.nan], equal_nan=True)
        for name in (
            "lap_time_speed",
            "safe_set_violations",
            "boundary_violations",
            "predicted_boundary_violations",
        ):
            for ax in plots.summary_figures[name].axes:
                assert ax.get_xlabel() == "iteration" and len(ax.lines[0].get_xdata()) == 3
                missing = [line for line in ax.lines if line.get_label() == "no value"]
                assert missing and all(
                    line.get_transform() is ax.get_xaxis_transform() for line in missing
                )
                assert all(line.get_marker() == "x" and line.get_color() == "r" for line in missing)
        # The short, much faster failed trial must never win best-lap selection.
        best = plots.summary_figures["best_lap_speed"].axes[0]
        np.testing.assert_array_equal(best.collections[0].get_array(), [7, 7, 7])
        assert best.get_title() == "Best lap: 3.00 s"
    finally:
        plots.close()


def test_risk_distribution_shows_every_raw_probability_and_reuses_window(tmp_path):
    plots = MatplotlibPlots(tmp_path, interactive=False)
    probabilities = np.asarray([[0.01, 0.15, 0.8], [0.2, 0.5, 0.99]])
    frenet = np.zeros((2, 3, 2))
    frenet[:, :, 0] = [[10, 11, 12], [20, 21, 22]]
    frenet[:, :, 1] = [[0, 1, 2], [-1, -2, -3]]
    diagnostic = {
        "probabilities": probabilities,
        "frenet": frenet,
        "generation": 4,
        "input_mode": "state_action",
    }
    try:
        figure = plots.risk_distribution(
            None, 7, status="Needs success and failure", half_width=2.5
        )
        assert "Needs success and failure" in figure.texts[0].get_text()
        assert not list(tmp_path.glob("risk_distribution*"))
        assert plots.risk_distribution(diagnostic, 8, half_width=2.5) is figure
        assert len(figure.axes) == 2  # One spatial plot and its probability color bar.
        spatial = figure.axes[0]
        np.testing.assert_array_equal(spatial.collections[0].get_array(), probabilities.ravel())
        np.testing.assert_array_equal(
            spatial.collections[0].get_offsets(),
            np.column_stack((-frenet[:, :, 1].ravel(), frenet[:, :, 0].ravel())),
        )
        assert spatial.get_xlim() == (-2.5, 2.5)
        assert spatial.collections[0].get_clim() == (0.0, 1.0)
        assert spatial.get_xlabel() == "-ey (m)" and spatial.get_ylabel() == "s (m)"
        assert len(plots.figures) == 1
        assert not list(tmp_path.glob("risk_distribution*"))
        clustered = {
            **diagnostic,
            "probabilities": np.asarray([[0.51, 0.52, 0.53], [0.52, 0.54, 0.55]]),
        }
        plots.risk_distribution(clustered, 9, half_width=2.5)
        assert len(figure.axes) == 2 and figure.axes[0].get_xlim() == (-2.5, 2.5)
        assert figure.axes[0].collections[0].get_clim() == (0.0, 1.0)
    finally:
        plots.close()
    assert plots.risk_figure is None


def test_risk_validation_shows_balanced_cutoff_samples_and_line_without_export(tmp_path):
    plots = MatplotlibPlots(tmp_path, interactive=False)
    data = {
        "generation": 5,
        "predictions": np.asarray([0.8, 0.1, 0.6, 0.2]),
        "labels": np.asarray([1, 0, 0, 1]),
        "cutoff": 0.6,
        "cutoff_status": "ready",
    }
    try:
        figure = plots.risk_validation(data)
        assert "generation 5" in figure._suptitle.get_text()
        assert len(figure.axes) == 2
        from matplotlib.colors import to_rgba

        assert len(figure.axes[0].collections) == 2
        safe, fail = figure.axes[0].collections
        np.testing.assert_allclose(safe.get_facecolors()[0], to_rgba("tab:blue", alpha=0.65))
        np.testing.assert_allclose(fail.get_facecolors()[0], to_rgba("tab:red", alpha=0.65))
        np.testing.assert_allclose(np.sort(safe.get_offsets()[:, 1]), [0.1, 0.6])
        np.testing.assert_allclose(np.sort(fail.get_offsets()[:, 1]), [0.2, 0.8])
        predicted = np.concatenate([safe.get_offsets()[:, 1], fail.get_offsets()[:, 1]])
        np.testing.assert_allclose(np.sort(predicted), np.sort(data["predictions"]))
        assert "label" not in figure.axes[0].get_ylabel()
        assert figure.axes[0].lines[0].get_ydata()[0] == 0.6
        assert figure.axes[1].lines[0].get_xdata()[0] == 0.6
        assert "Cutoff" in figure.axes[0].lines[0].get_label()
        assert (
            plots.risk_validation(
                {**data, "generation": 6, "cutoff": None, "cutoff_status": "insufficient_support"}
            )
            is figure
        )
        assert "insufficient_support" in figure._suptitle.get_text()
        assert not figure.axes[0].lines
        assert not list(tmp_path.glob("risk_validation*"))
    finally:
        plots.close()
    assert plots.validation_figure is None


@pytest.mark.parametrize("predictions,labels", [([], []), ([0.1, 0.8], [0, 1]), ([0.1, 0.2], [0, 0])])
def test_manual_cutoff_plot_keeps_line_with_or_without_validation_data(tmp_path, predictions, labels):
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        figure = plots.risk_validation({
            "generation": 1, "predictions": np.asarray(predictions), "labels": np.asarray(labels),
            "cutoff": 0.5, "cutoff_source": "manual", "cutoff_status": "ready",
        })
        assert len(figure.axes) == 2
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0.5, 0.5])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_xdata(), [0.5, 0.5])
        assert "cutoff=0.5 (manual)" in figure._suptitle.get_text()
        assert sum(len(collection.get_offsets()) for collection in figure.axes[0].collections) == len(predictions)
        if not predictions:
            assert figure.axes[0].texts
        assert not list(tmp_path.glob("risk_validation*"))
    finally:
        plots.close()


def test_nonblocking_debug_does_not_wait(tmp_path, monkeypatch):
    states = np.zeros((4, 3, 7))
    states[:, :, 0] = np.arange(12).reshape(4, 3)
    states[:, :, 1] = np.linspace(-1, 1, 12).reshape(4, 3)
    candidates = states[:2].copy()
    candidates[:, :, 1] += 3
    debug = {
        "states": states,
        "safe_set": np.zeros((8, 8)),
        "candidate_states": candidates,
        "chosen_states": states[0],
        "reward": np.zeros((4, 3)),
        "value": np.zeros((4, 3)),
        "hull": np.zeros((4, 3)),
        "boundary": np.zeros((4, 3)),
        "sample_actions": np.zeros((4, 3, 2)),
        "previous_actions": np.zeros((3, 2)),
        "chosen_actions": np.zeros((3, 2)),
    }
    debug["candidate_boundary"] = np.zeros((2, 3))
    result = PlannerResult(np.zeros(2), states[0], debug=debug)
    plots = MatplotlibPlots(tmp_path, interactive=False)
    plots.interactive = True
    monkeypatch.setattr(plots, "pump", lambda: None)
    # Render the six panels through Agg while testing the interactive pause gate.
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "show", lambda self: None)
    monkeypatch.setattr(
        plots, "wait_for_interaction", lambda *args: pytest.fail("nonblocking plotting waited")
    )
    try:
        plots.debug(result, 1, pause=False)
        assert len(plots.debug_figure.axes) == 11  # Six panels plus five color bars.
        assert plots.debug_figure.axes[5].get_title() == "Sampled risk (terminal)"
        assert "Risk model not ready" in plots.debug_figure.axes[5].texts[0].get_text()
        probabilities = np.linspace(0.01, 0.99, 12).reshape(4, 3)
        for mode in ("all", "terminal"):
            scored = replace(
                result,
                debug={**debug, "risk_mode": mode},
                risk={"probabilities": probabilities},
            )
            plots.debug(scored, 2, pause=False)
            risk_axis = plots.debug_figure.axes[5]
            assert risk_axis.get_title() == f"Sampled risk ({mode})"
            assert risk_axis.get_xlabel() == "-ey (m)"
            assert risk_axis.get_ylabel() == "s (m)"
            assert risk_axis.get_xlim() == plots.debug_figure.axes[0].get_xlim()
            assert risk_axis.get_ylim() == plots.debug_figure.axes[0].get_ylim()
            from matplotlib.collections import LineCollection

            paths = [item for item in risk_axis.collections if isinstance(item, LineCollection)]
            assert len(paths) == 1 and len(paths[0].get_segments()) == 2
            np.testing.assert_allclose(
                paths[0].get_segments()[0],
                np.column_stack((-candidates[0, :, 1], candidates[0, :, 0])),
            )
            shown_states = states.reshape(-1, 7) if mode == "all" else states[:, -1]
            shown_risk = probabilities.ravel() if mode == "all" else probabilities[:, -1]
            risk_points = next(
                item for item in risk_axis.collections if item.get_array() is not None
            )
            np.testing.assert_allclose(
                risk_points.get_offsets(),
                np.column_stack((-shown_states[:, 1], shown_states[:, 0])),
            )
            np.testing.assert_allclose(risk_points.get_array(), shown_risk)
            assert risk_points.get_clim() == (0, 1)
            assert len(plots.debug_figure.axes) == 12  # Six panels and six color bars.
    finally:
        plots.close()


@pytest.mark.parametrize("transform", ["linear", "log1p"])
def test_safe_set_cost_comparison_has_identical_queries_axes_and_color_scale(tmp_path, transform):
    states = np.zeros((8, 3, 7))
    states[:, :, 0] = np.arange(3) + np.linspace(1, 2, 8)[:, None]
    states[:, :, 1] = np.linspace(-1, 1, 8)[:, None]
    selected = states.reshape(-1, 7).copy()
    selected[-1, 0] = 10000  # Background states must not destroy the local zoom.
    diagnostic = dict(
        terminal_states=states[:, -1, :5], safe_set=selected,
        current_cost=np.linspace(0, 1, 8), hull_cost=np.linspace(1, 0, 8),
        current_inside=np.arange(8) < 6, hull_inside=np.arange(8) < 4,
        candidate_states=states[:2], chosen_states=states[0], current_method="nf_density",
        current_calibration="local",
        current_transform=transform,
        current_lambdas=(12., 0., .5, 2., 7.),
        lambda_terms=["hull", "boundary", "chance_boundary", "chance_hull", "risk"],
        hull_candidate_states=states[2:4], hull_chosen_states=states[2], hull_lambdas=(4., 6.),
    )
    plots = MatplotlibPlots(tmp_path, interactive=False)
    try:
        figure = plots.safe_set_costs(diagnostic, 20)
        figure.canvas.draw()
        left, right, colorbar = figure.axes
        assert "NF density" in left.get_title() and "75.0%" in left.get_title()
        assert "local calibration" in left.get_title()
        assert f"[{transform}]" in left.get_title()
        assert "Selected λ: ss=12, boundary=0\nchance_boundary=0.5, chance_hull=2\nrisk=7" in left.get_title()
        assert "Original facet hull" in right.get_title() and "50.0%" in right.get_title()
        assert "Selected λ: ss=4, boundary=6" in right.get_title()
        assert left.get_xlim() == right.get_xlim() and left.get_ylim() == right.get_ylim()
        assert left.get_ylim()[1] < 10000
        for axis, cost in zip((left, right), (diagnostic["current_cost"], diagnostic["hull_cost"]), strict=True):
            dots = next(item for item in axis.collections if item.get_array() is not None)
            np.testing.assert_array_equal(dots.get_array(), cost)
            np.testing.assert_array_equal(dots.get_offsets(), np.column_stack((-states[:, -1, 1], states[:, -1, 0])))
            assert dots.get_clim() == (0, 1)
            assert axis.get_xlabel() == "-ey (m)" and axis.get_ylabel() == "s (m)"
            assert any("Penalty 0: 1/8 (12.5%)\nPenalty 1: 1/8 (12.5%)" == text.get_text()
                       for text in axis.texts)
        assert "before λ" in colorbar.get_ylabel()
        diagnostic["current_cost"] = np.zeros(8)
        diagnostic["current_lambdas"] = (3., 1.)
        diagnostic["lambda_terms"] = ["hull", "boundary"]
        diagnostic["hull_lambdas"] = (8., 0.)
        assert plots.safe_set_costs(diagnostic, 21) is figure
        assert "Selected λ: ss=3, boundary=1" in figure.axes[0].get_title()
        assert "risk=" not in figure.axes[0].get_title()
        assert "Selected λ: ss=8, boundary=0" in figure.axes[1].get_title()
        assert any("Penalty 0: 8/8 (100.0%)\nPenalty 1: 0/8 (0.0%)" == text.get_text()
                   for text in figure.axes[0].texts)
        assert not list(tmp_path.iterdir())  # Debug display is not an automatic export.
    finally:
        plots.close()
    assert plots.safe_set_cost_figure is None


@pytest.mark.parametrize("panel", [0, 2, 3])
def test_real_qtagg_combined_safe_set_plot_advances_refreshes_and_closes(tmp_path, panel):
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6")
    script = """
import sys
from dataclasses import replace
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from matplotlib.backend_bases import MouseEvent, KeyEvent
from sit_lmpc.config import load_config
from sit_lmpc.diagnostics.keyboard import KeyboardControls
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
app = QApplication([])
p = MatplotlibPlots(sys.argv[1], interactive=True)
keyboard = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
p.keyboard_command = keyboard.command
keyboard.command('6')
states = np.zeros((8, 3, 7))
states[:, :, 0] = np.arange(3)
states[:, :, 1] = np.linspace(-1, 1, 8)[:, None]
data = dict(terminal_states=states[:, -1, :5], safe_set=states.reshape(-1, 7),
            current_cost=np.linspace(0, 1, 8), hull_cost=np.linspace(1, 0, 8),
            current_inside=np.ones(8, bool), hull_inside=np.zeros(8, bool),
            candidate_states=states[:2], chosen_states=states[0], current_method='nf_density',
            hull_candidate_states=states[2:4], hull_chosen_states=states[2],
            current_lambdas=(12., 0.), hull_lambdas=(4., 6.), lambda_terms=['hull', 'boundary'])
density = dict(status='unavailable', message='NF calibration is not ready yet.')
figure = p.safe_set_costs(data, 10, density=density)
def keep_paused():
    if keyboard.consume_close_request():
        p.close()
    return keyboard.safe_set_cost_mode()
def click():
    x, y = figure.axes[int(sys.argv[2])].transAxes.transform((.5, .5))
    figure.canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', figure.canvas, x, y, button=1))
QTimer.singleShot(75, click)
p.wait_for_interaction(figure, keep_paused)
assert keyboard.safe_set_cost_mode() and p.plt.fignum_exists(figure.number)
assert p.figures == [figure] and p.density_calibration_figure is None
assert len(figure.axes) == 5
assert 'Selected λ: ss=12, boundary=0' in figure.axes[0].get_title()
assert 'Selected λ: ss=4, boundary=6' in figure.axes[1].get_title()
sizes = [(ax.get_position().width, ax.get_position().height) for ax in figure.axes[:4]]
np.testing.assert_allclose(sizes, np.tile(sizes[0], (4, 1)), atol=.01)
before = np.asarray(figure.canvas.buffer_rgba()).copy()
draws = []
figure.canvas.mpl_connect('draw_event', lambda event: draws.append(event))
states[:, :, 0] += 1
states[:, :, 1] += np.linspace(-.2, .3, 8)[:, None]
data['current_cost'] = np.zeros(8)
data['current_lambdas'] = (3., 1.)
data['hull_lambdas'] = (8., 0.)
assert p.safe_set_costs(data, 11, density=density) is figure
QTimer.singleShot(75, click)
p.wait_for_interaction(figure, keep_paused)
assert draws and not np.array_equal(before, np.asarray(figure.canvas.buffer_rgba()))
assert 'Selected λ: ss=3, boundary=1' in figure.axes[0].get_title()
assert 'Selected λ: ss=8, boundary=0' in figure.axes[1].get_title()
points = next(item for item in figure.axes[0].collections if item.get_array() is not None)
np.testing.assert_array_equal(points.get_offsets(), np.column_stack((-states[:, -1, 1], states[:, -1, 0])))
assert any('Penalty 0: 8/8' in item.get_text() for item in figure.axes[0].texts)
QTimer.singleShot(75, lambda: figure.canvas.callbacks.process(
    'key_press_event', KeyEvent('key_press_event', figure.canvas, key='0')))
p.wait_for_interaction(figure, keep_paused)
assert not keyboard.should_pause() and not p.plt.fignum_exists(figure.number)
assert not p.figures
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(panel)],
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_qtagg_risk_plot_advances_and_zero_closes_it(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6")
    script = """
import sys
from dataclasses import replace
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from matplotlib.backend_bases import MouseEvent, KeyEvent
from sit_lmpc.config import load_config
from sit_lmpc.diagnostics.keyboard import KeyboardControls
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
app = QApplication([])
p = MatplotlibPlots(sys.argv[1], interactive=True)
keyboard = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
p.keyboard_command = keyboard.command
keyboard.command('1')
debug = p.plt.figure()
p.figures.append(debug)
QTimer.singleShot(75, lambda: debug.canvas.callbacks.process(
    'key_press_event', KeyEvent('key_press_event', debug.canvas, key='3')))
p.wait_for_interaction(debug, lambda: keyboard.poll()[0])
assert keyboard.risk_mode() and keyboard.should_pause()
p.plt.close(debug)
diagnostic = {
    'probabilities': np.array([[.1, .4, .9], [.2, .5, .8]]),
    'frenet': np.zeros((2, 3, 2)), 'generation': 2, 'input_mode': 'state'
}
figure = p.risk_distribution(diagnostic, 10, half_width=4.0)
QTimer.singleShot(75, lambda: figure.canvas.callbacks.process(
    'button_press_event', MouseEvent('button_press_event', figure.canvas, 10, 10, button=1)))
p.wait_for_interaction(figure, keyboard.should_pause)
assert keyboard.risk_mode() and p.plt.fignum_exists(figure.number)
QTimer.singleShot(75, lambda: figure.canvas.callbacks.process(
    'key_press_event', KeyEvent('key_press_event', figure.canvas, key='0')))
p.wait_for_interaction(figure, keyboard.should_pause)
assert not keyboard.risk_mode() and not p.plt.fignum_exists(figure.number)
p.close()
print('QtAgg risk plot advances and 0 closes it')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_qtagg_trial_summaries_pause_until_click_or_zero(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6")
    script = """
import sys
from dataclasses import replace
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from matplotlib.backend_bases import MouseEvent, KeyEvent
from sit_lmpc.config import load_config
from sit_lmpc.diagnostics.keyboard import KeyboardControls
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
app = QApplication([])
p = MatplotlibPlots(sys.argv[1], interactive=True)
keyboard = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
p.keyboard_command = keyboard.command
keyboard.command('2')
record = {'attempt_id': 1, 'completed': False, 'outcome': 'failure'}
def event(kind):
    figure = p.summary_figures['lap_time_speed']
    if kind == 'click':
        figure.canvas.callbacks.process('button_press_event', MouseEvent(
            'button_press_event', figure.canvas, 10, 10, button=1))
    else:
        figure.canvas.callbacks.process('key_press_event', KeyEvent(
            'key_press_event', figure.canvas, key='0'))
QTimer.singleShot(75, lambda: event('click'))
p.summaries([], [record], object(), 4, show=True, pause=True,
            keep_paused=lambda: keyboard.poll()[1])
assert keyboard.poll()[1] and p.summary_figures
QTimer.singleShot(75, lambda: event('0'))
p.summaries([], [record], object(), 4, show=True, pause=True,
            keep_paused=lambda: keyboard.poll()[1])
assert not keyboard.poll()[1] and not p.summary_figures
p.close()
print('QtAgg trial summaries paused and 0 closed them')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_qtagg_pause_advances_on_plot_click_in_isolated_offscreen_app(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6")
    script = """
import sys, time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
app = QApplication([])
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
from matplotlib.backend_bases import MouseEvent
p = MatplotlibPlots(sys.argv[1], interactive=True)
figure = p.plt.figure()
p.figures.append(figure)
other = p.plt.figure()
QTimer.singleShot(75, lambda: figure.canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', figure.canvas, 10, 10, button=1)))
started = time.perf_counter()
p.wait_for_interaction(figure)
elapsed = time.perf_counter() - started
assert 0.03 <= elapsed < 3, elapsed
p.close()
assert p.plt.fignum_exists(other.number)
p.plt.close(other)
print('QtAgg pause advanced on plot click and cleaned up only owned figures')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", ["plot", "global", "suspend"])
def test_zero_closes_owned_windows_during_real_qt_pause_and_allows_reopening(tmp_path, source):
    import os
    import subprocess
    import sys

    pytest.importorskip("PyQt6")
    script = """
import sys, threading, time
from dataclasses import replace
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
app = QApplication([])
from sit_lmpc.config import load_config
from sit_lmpc.diagnostics.keyboard import KeyboardControls
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
from matplotlib.backend_bases import KeyEvent
p = MatplotlibPlots(sys.argv[1], interactive=True)
keyboard = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
p.keyboard_command = keyboard.command
controller = p.plt.figure()
comparison = p.plt.figure()
p.debug_figure, p.comparison_figure = controller, comparison
p.figures.extend([controller, comparison])
summary, _ = p.figure('lap_summary', shape=(1, 1))
p.save(summary, 'retained.png')
other = p.plt.figure()
closed_on = []
close = p.close
def checked_close():
    closed_on.append(threading.current_thread())
    close()
p.close = checked_close
def keep_paused():
    if keyboard.consume_close_request():
        p.close()
    return keyboard.should_pause()
def stop():
    if sys.argv[2] == 'plot':
        comparison.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', comparison.canvas, key='0'))
    else:
        listener = threading.Thread(target=keyboard.command, args=(
            'toggle_listening' if sys.argv[2] == 'suspend' else '0',))
        listener.start()
        listener.join(timeout=1)
        assert not listener.is_alive()
keyboard.command('1')
QTimer.singleShot(75, stop)
started = time.perf_counter()
p.wait_for_interaction([controller, comparison], keep_paused)
keep_paused()
assert 0.03 <= time.perf_counter() - started < 3
assert not keyboard.should_pause() and keyboard.poll() == (False, False)
assert not p.figures and not p.summary_figures
assert p.debug_figure is None and p.comparison_figure is None
assert closed_on and all(t is threading.main_thread() for t in closed_on)
assert p.plt.fignum_exists(other.number)
assert (p.directory / 'retained.png').is_file()
if sys.argv[2] == 'suspend':
    assert keyboard.suspended
    keyboard.command('1')
    assert not keyboard.should_pause()
    keyboard.command('toggle_listening')
    assert not keyboard.suspended and not keyboard.should_pause()
keyboard.command('1')
assert keyboard.should_pause()
reopened, _ = p.figure('lap_summary', shape=(1, 1))
assert reopened is not summary and p.plt.fignum_exists(reopened.number)
p.close()
assert p.plt.fignum_exists(other.number)
p.plt.close(other)
print('0 released pause, closed only owned windows on the main thread, retained exports, and allowed reopening')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), source],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
