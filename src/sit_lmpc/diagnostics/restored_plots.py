"""Matplotlib versions of the SIT summaries and six-panel controller inspection."""

from pathlib import Path
import numpy as np
from sit_lmpc.diagnostics.lap_metrics import measured_boundary_metrics


def _penalty_counts_text(penalties):
    penalties = np.asarray(penalties)
    total = len(penalties)
    lines = []
    for value in (0, 1):
        count = int(np.count_nonzero(penalties == value))
        lines.append(f"Penalty {value}: {count}/{total} ({count / total if total else 0:.1%})")
    return "\n".join(lines)


def _selected_lambdas_text(lambdas, terms):
    weights = [f"{'ss' if name == 'hull' else name}={float(value):.4g}"
               for name, value in zip(terms, lambdas, strict=True)]
    return "Selected λ: " + "\n".join(
        ", ".join(weights[start:start + 2]) for start in range(0, len(weights), 2)
    )


def _log1p_nll_excess(scores, cutoff, scale):
    """Display the controller's compressed score, without changing raw data."""
    scores = np.asarray(scores, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        excess = np.maximum(scores - cutoff, 0.)
        log_excess = np.where(np.isfinite(excess), np.log(excess),
                              np.logaddexp(np.log(np.abs(scores)), np.log(np.abs(cutoff))))
        return np.logaddexp(0., log_excess - np.log(scale))


class MatplotlibPlots:
    def __init__(self, directory, interactive=False):
        import matplotlib

        # Headless exports must not depend on a display or Qt event loop.
        if not interactive:
            matplotlib.use("Agg")
        else:
            matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt

        self.plt, self.directory, self.interactive = plt, Path(directory), interactive
        self.figures = []
        self.debug_figure = None
        self.comparison_figure = None
        self.risk_figure = None
        self.validation_figure = None
        self.safe_set_cost_figure = None
        self.density_calibration_figure = None
        self.summary_figures = {}
        self.track_boundaries = None
        self.keyboard_command = None
        self.history_active = False
        self.waiting = False

    def boundaries(self, track, half_width):
        if self.track_boundaries is None:
            positions = np.linspace(0, float(track.s_frame_max), 400, endpoint=False)
            self.track_boundaries = [
                np.asarray([track.frenet_to_cartesian(s, side * half_width, 0) for s in positions])
                for side in (-1, 1)
            ]
        return self.track_boundaries

    def figure(self, name, shape=(1, 2), figsize=(10, 4)):
        figure = self.summary_figures.get(name)
        if figure is None or not self.plt.fignum_exists(figure.number):
            figure = self.plt.figure(figsize=figsize)
            self.summary_figures[name] = figure
            self.figures.append(figure)
        figure.clear()
        figure.set_layout_engine("tight")
        return figure, np.asarray(figure.subplots(*shape)).reshape(-1)

    def iteration_plot(self, ax, records, field, title, units):
        """Keep every trial on the x-axis without inventing missing measurements."""
        numbers = [item.get("attempt_id", index + 1) for index, item in enumerate(records)]
        values = np.asarray(
            [np.nan if item.get(field) is None else float(item[field]) for item in records]
        )
        values[~np.isfinite(values)] = np.nan
        ax.plot(numbers, values, ".-")
        if not np.isfinite(values).any():
            # No measurements: avoid displaying an invented numeric y-scale.
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        for number, value, item in zip(numbers, values, records, strict=True):
            if not np.isfinite(value):
                # Axes-relative y=0 is the x-axis, not a fabricated data value.
                ax.plot(
                    number,
                    0,
                    "rx",
                    markersize=9,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    label="no value",
                )
            elif not item.get("completed", True):
                outcome = "failed" if item.get("outcome") == "failure" else "unfinished"
                reason = item.get("termination_reason", "unknown")
                ax.plot(number, value, "rx", markersize=9, label=f"{outcome}: {reason}")
        ax.set(title=title, xlabel="iteration", ylabel=units)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=True))
            ax.legend(unique.values(), unique.keys(), fontsize=8)
        ax.grid(True)

    def summaries(
        self,
        laps,
        metrics,
        track,
        half_width,
        show=False,
        pause=False,
        keep_paused=lambda: True,
        export=True,
    ):
        if not laps and not metrics:
            return
        # Old runs contain only predicted penalties. Recover measured boundary
        # excess from their saved states without changing the original records.
        records = [dict(item) for item in metrics]
        completed_records = [item for item in records if item.get("completed", True)]
        for index, lap in enumerate(laps):
            if index >= len(completed_records):
                item = {"lap": index + 1, "completed": True}
                records.append(item)
                completed_records.append(item)
            for name, value in measured_boundary_metrics(lap, half_width).items():
                completed_records[index].setdefault(name, value)
            completed_records[index].setdefault("lap_time", lap.laptime)
            completed_records[index].setdefault("mean_speed", float(lap.cartesian[:, 3].mean()))
        if records:
            figure, axes = self.figure("lap_time_speed")
            for ax, field, title, units in zip(
                axes,
                ("lap_time", "mean_speed"),
                ("Lap time", "Mean speed"),
                ("seconds", "m/s"),
                strict=True,
            ):
                self.iteration_plot(ax, records, field, title, units)
            if export:
                self.save(figure, "lap_time_speed.png")
            # Keep the familiar summary.png entry point for existing workflows.
            if export:
                figure.savefig(self.directory / "summary.png", dpi=120)
        if laps:
            best = min(laps, key=lambda lap: lap.laptime)
            figure, axes = self.figure("best_lap_speed", shape=(1, 1), figsize=(7, 5))
            ax = axes[0]
            for boundary in self.boundaries(track, half_width):
                ax.plot(boundary[:, 0], boundary[:, 1], color="gray", linewidth=0.8)
            points = ax.scatter(
                best.cartesian[:, 0],
                best.cartesian[:, 1],
                c=best.cartesian[:, 3],
                s=5,
                cmap="viridis",
            )
            figure.colorbar(points, ax=ax, label="speed (m/s)")
            ax.set(
                title=f"Best lap: {best.laptime:.2f} s",
                xlabel="x (m)",
                ylabel="y (m)",
                aspect="equal",
            )
            if export:
                self.save(figure, "best_lap_speed.png")
        for name, field, label, units in (
            ("safe_set", "safe_set", "Predicted safe-set violation", "controller safe-set score (backend-specific)"),
            (
                "boundary",
                "actual_boundary",
                "Measured boundary excess",
                "excess beyond half-width (m)",
            ),
            (
                "predicted_boundary",
                "boundary",
                "Predicted boundary penalty",
                "controller horizon penalty",
            ),
        ):
            figure, axes = self.figure(f"{name}_violations")
            for ax, aggregation in zip(axes, ("max", "mean"), strict=True):
                self.iteration_plot(
                    ax, records, f"{field}_{aggregation}", f"{label}: {aggregation}", units
                )
            if export:
                self.save(figure, f"{name}_violations.png")
        if show and self.interactive:
            for figure in self.summary_figures.values():
                figure.show()
            if pause:
                self.wait_for_interaction(self.summary_figures.values(), keep_paused)
            else:
                self.pump()

    def save(self, figure, filename):
        figure.tight_layout()
        # tight_layout() is a one-off adjustment; retain it for later window
        # redraws/resizes as well as the saved PNG.
        figure.set_layout_engine("tight")
        figure.savefig(self.directory / filename, dpi=120)

    def unavailable(self, message):
        figure, axes = self.figure("historical_view_status", shape=(1, 1), figsize=(7, 3))
        axes[0].set_axis_off()
        axes[0].text(.5, .5, message, ha="center", va="center", wrap=True, transform=axes[0].transAxes)
        if self.interactive:
            figure.show()
        return figure

    def debug(self, result, step, pause=False, keep_paused=lambda: True):
        if result.debug is None:
            return
        data = result.debug
        if self.debug_figure is None or not self.plt.fignum_exists(self.debug_figure.number):
            self.debug_figure = self.plt.figure(figsize=(12, 5))
            self.figures.append(self.debug_figure)
        figure = self.debug_figure
        figure.clear()
        axes = np.asarray(figure.subplots(2, 3)).reshape(-1)
        states, selected = data["states"], data["safe_set"]
        terminal = states[:, -1]
        specs = (
            (data["reward"].sum(axis=1), "Sample reward"),
            (data["value"][:, -1], "Terminal value"),
            (data["hull"][:, -1], "Safe-set penalty"),
        )
        for ax, (colors, title) in zip(axes[[0, 1, 3]], specs, strict=True):
            ax.scatter(-selected[:, 1], selected[:, 0], color="blue", s=2, alpha=0.3)
            points = ax.scatter(-terminal[:, 1], terminal[:, 0], c=colors, s=4, cmap="viridis")
            chosen = data["chosen_states"][-1]
            ax.scatter(-chosen[1], chosen[0], color="red", marker="x")
            ax.set(title=title, xlabel="-ey (m)", ylabel="s (m)")
            figure.colorbar(points, ax=ax)
        for ax, points, colors, title in (
            (
                axes[2],
                data["candidate_states"],
                data["candidate_boundary"],
                "Candidate boundary violations",
            ),
            (axes[4], states, -data["boundary"], "Sample boundary rewards"),
        ):
            flat = points.reshape(-1, 7)
            dots = ax.scatter(-flat[:, 1], flat[:, 0], c=colors.ravel(), s=2, cmap="viridis")
            chosen = data["chosen_states"]
            ax.plot(-chosen[:, 1], chosen[:, 0], color="red", linewidth=1)
            ax.set(title=title, xlabel="-ey (m)", ylabel="s (m)")
            figure.colorbar(dots, ax=ax)
        risk_mode = data.get("risk_mode", result.metrics.get("risk_mode", "terminal"))
        if risk_mode not in ("all", "terminal"):
            raise ValueError("debug risk mode must be all or terminal")
        from matplotlib.collections import LineCollection

        candidates = np.asarray(data["candidate_states"])
        chosen = np.asarray(data["chosen_states"])
        risk_axis = axes[5]
        risk_axis.scatter(
            -selected[:, 1], selected[:, 0], color="blue", s=2, alpha=0.3, zorder=0
        )
        candidate_paths = np.stack((-candidates[:, :, 1], candidates[:, :, 0]), axis=-1)
        risk_axis.add_collection(
            LineCollection(candidate_paths, colors="0.4", linewidths=0.5, alpha=0.25, zorder=1)
        )
        risk = None if result.risk is None else np.asarray(result.risk["probabilities"])
        if risk is None:
            risk_axis.text(
                0.5, 0.5, "Risk model not ready", ha="center", va="center",
                transform=risk_axis.transAxes,
            )
        else:
            if (
                risk.shape != states.shape[:2]
                or not np.isfinite(risk).all()
                or np.any((risk < 0) | (risk > 1))
            ):
                raise ValueError("sampled risk must match debug states with values in [0, 1]")
            shown_states = states.reshape(-1, 7) if risk_mode == "all" else terminal
            shown_risk = risk.ravel() if risk_mode == "all" else risk[:, -1]
            points = risk_axis.scatter(
                -shown_states[:, 1], shown_states[:, 0], c=shown_risk,
                s=2 if risk_mode == "all" else 4, alpha=0.7,
                cmap="viridis", vmin=0, vmax=1, zorder=2,
            )
            figure.colorbar(points, ax=risk_axis, label="Predicted failure probability")
        risk_axis.plot(-chosen[:, 1], chosen[:, 0], color="red", linewidth=1, zorder=3)
        risk_axis.plot(-chosen[-1, 1], chosen[-1, 0], "rx", zorder=4)
        # The first and risk panels use identical Frenet bounds for comparison.
        positions = np.concatenate(
            (selected[:, :2], states.reshape(-1, 7)[:, :2], candidates.reshape(-1, 7)[:, :2]),
            axis=0,
        )
        lateral = -positions[:, 1]
        progress = positions[:, 0]
        xpad = max(0.05, float(np.ptp(lateral)) * 0.05)
        ypad = max(0.05, float(np.ptp(progress)) * 0.05)
        limits = (
            (float(lateral.min() - xpad), float(lateral.max() + xpad)),
            (float(progress.min() - ypad), float(progress.max() + ypad)),
        )
        for ax in (axes[0], risk_axis):
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
        risk_axis.set(
            title=f"Sampled risk ({risk_mode})",
            xlabel="-ey (m)",
            ylabel="s (m)",
        )
        figure.suptitle(
            f"Step {step} | λ={result.lambdas} | safe-set={result.safe_set_violation:.3g} | boundary={result.boundary_violation:.3g}"
        )
        figure.tight_layout()
        if self.interactive:
            figure.show()
            if pause:
                self.wait_for_interaction(figure, keep_paused)
            else:
                self.pump()

    def safe_set_costs(self, diagnostic, step, pause=False, keep_paused=lambda: True, *, density=None):
        """Same sampled terminal states, actual and counterfactual reward costs."""
        from matplotlib.collections import LineCollection

        if self.safe_set_cost_figure is None or not self.plt.fignum_exists(
            self.safe_set_cost_figure.number
        ):
            self.safe_set_cost_figure = self.plt.figure(figsize=(24, 5.5) if density is not None else (10, 5),
                                                       layout="constrained")
            self.figures.append(self.safe_set_cost_figure)
        figure = self.safe_set_cost_figure
        figure.clear()
        if density is not None:
            # A dedicated colorbar column leaves all four data panels equal.
            grid = figure.add_gridspec(1, 5, width_ratios=(1, 1, .04, 1, 1))
            left = figure.add_subplot(grid[0, 0])
            axes = [left, figure.add_subplot(grid[0, 1], sharex=left, sharey=left)]
            cutoff_axis = figure.add_subplot(grid[0, 3])
            mapping_axis = figure.add_subplot(grid[0, 4])
            colorbar_axis = figure.add_subplot(grid[0, 2])
            self.density_calibration(density, step, axis=cutoff_axis)
            self.density_mapping(density, mapping_axis)
        else:
            axes = figure.subplots(1, 2, sharex=True, sharey=True)
            colorbar_axis = None
        if diagnostic is None:
            for ax in axes:
                ax.text(.5, .5, "Sampled costs appear when learned control starts.",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
            if colorbar_axis is not None:
                colorbar_axis.set_axis_off()
            figure.suptitle(f"Safe-set inspection | Step {step}")
            if self.interactive:
                figure.show()
                self.pump()
            return figure
        terminal = diagnostic["terminal_states"]
        selected = diagnostic["safe_set"]
        candidates = diagnostic["candidate_states"]
        candidates = candidates[np.isfinite(candidates).all(axis=(1, 2))]
        chosen = diagnostic["chosen_states"]
        hull_candidates = diagnostic.get("hull_candidate_states", np.empty((0, *chosen.shape)))
        hull_candidates = hull_candidates[np.isfinite(hull_candidates).all(axis=(1, 2))]
        hull_chosen = diagnostic.get("hull_chosen_states")
        names = {
            "nf_density": "NF density", "facets": "Facet hull",
            "facets_warmup": "Facet hull (NF warm-up)", "implicit": "Implicit hull",
            "facets_local_calibration": "Facet hull (local NF calibration unavailable)",
        }
        method = names.get(diagnostic["current_method"], diagnostic["current_method"])
        transform = diagnostic.get("current_transform", (density or {}).get("penalty_transform"))
        if diagnostic["current_method"] == "nf_density" and transform is not None:
            method += f" [{transform}]"
        if diagnostic["current_method"] == "nf_density" and diagnostic.get("current_calibration"):
            method += f" ({diagnostic['current_calibration']} calibration)"
        for panel, (ax, title, cost, inside) in enumerate(zip(
            axes,
            (f"Current: {method}", "Original facet hull"),
            (diagnostic["current_cost"], diagnostic["hull_cost"]),
            (diagnostic["current_inside"], diagnostic["hull_inside"]),
            strict=True,
        )):
            panel_candidates = candidates if panel == 0 else hull_candidates
            panel_chosen = chosen if panel == 0 else hull_chosen
            ax.scatter(-selected[:, 1], selected[:, 0], color="0.7", s=3, alpha=0.4)
            ax.add_collection(LineCollection(
                np.stack((-panel_candidates[:, :, 1], panel_candidates[:, :, 0]), axis=-1),
                colors="0.6", linewidths=0.4, alpha=0.15,
            ))
            points = ax.scatter(
                -terminal[:, 1], terminal[:, 0], c=cost, cmap="viridis",
                vmin=0, vmax=1, s=10, zorder=3,
            )
            if panel_chosen is not None:
                selected_label = "Current selection" if panel == 0 else "Convex-hull selection"
                lambdas = diagnostic.get("current_lambdas" if panel == 0 else "hull_lambdas")
                if lambdas is not None:
                    terms = diagnostic.get("lambda_terms", ("hull", "boundary")) if panel == 0 else ("hull", "boundary")
                    title += "\n" + _selected_lambdas_text(lambdas, terms)
                ax.plot(-panel_chosen[:, 1], panel_chosen[:, 0], color="tab:red", linewidth=1.5,
                        zorder=4, label=selected_label)
                ax.plot(-panel_chosen[-1, 1], panel_chosen[-1, 0], "rx", zorder=5)
            if panel == 1:
                ax.plot(-chosen[:, 1], chosen[:, 0], color="tab:blue", linestyle="--",
                        linewidth=1, alpha=.7, label="Current selection (reference)")
                if hull_chosen is None:
                    ax.text(.02, .82, diagnostic.get("hull_selection_status", "Hull selection unavailable"),
                            transform=ax.transAxes, fontsize=7, va="top", wrap=True)
            ax.legend(fontsize=7, loc="lower left")
            ax.set(
                title=f"{title}\nMembership: {np.mean(inside):.1%}",
                xlabel="-ey (m)", ylabel="s (m)",
            )
            ax.text(.02, .98, _penalty_counts_text(cost), transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
            ax.grid(alpha=0.2)
        # Focus on the identical query/candidate cloud, not every retained lap.
        positions = np.concatenate((terminal[:, :2], candidates.reshape(-1, 7)[:, :2],
                                    hull_candidates.reshape(-1, 7)[:, :2]))
        lateral, progress = -positions[:, 1], positions[:, 0]
        xpad = max(0.05, float(np.ptp(lateral)) * 0.05)
        ypad = max(0.05, float(np.ptp(progress)) * 0.05)
        axes[0].set_xlim(lateral.min() - xpad, lateral.max() + xpad)
        axes[0].set_ylim(progress.min() - ypad, progress.max() + ypad)
        figure.colorbar(points, ax=list(axes) if colorbar_axis is None else None,
                        cax=colorbar_axis, label="Terminal safe-set cost (before λ)")
        figure.suptitle(
            f"Step {step} | Same {len(terminal)} sampled endpoints\n"
            "Costs use each method's reward normalization; membership is reported separately"
            + (f" | {diagnostic['excluded_samples']} invalid samples excluded"
               if diagnostic.get("excluded_samples") else "")
        )
        if self.interactive:
            figure.show()
            if pause:
                self.wait_for_interaction(figure, keep_paused)
            else:
                self.pump()
        return figure

    def risk_distribution(
        self, diagnostic, step, status=None, pause=False, keep_paused=lambda: True, *, half_width
    ):
        """Show every sampled risk output at its predicted state, without rescaling."""
        if self.risk_figure is None or not self.plt.fignum_exists(self.risk_figure.number):
            self.risk_figure = self.plt.figure(figsize=(6, 7))
            self.figures.append(self.risk_figure)
        figure = self.risk_figure
        figure.clear()
        figure.set_layout_engine("tight")
        if diagnostic is None:
            figure.text(
                0.5, 0.5, status or "Risk model not ready", ha="center", va="center", fontsize=13
            )
            figure.suptitle(f"Sampled risk-to-go | step {step}")
        else:
            risk = np.asarray(diagnostic["probabilities"])
            frenet = np.asarray(diagnostic["frenet"])
            if (
                risk.ndim != 2
                or frenet.shape != (*risk.shape, 2)
                or not np.isfinite(risk).all()
                or np.any((risk < 0) | (risk > 1))
            ):
                raise ValueError(
                    "sampled risk needs finite (samples, horizon) probabilities in [0, 1]"
                )
            spatial = figure.subplots()
            points = spatial.scatter(
                -frenet[:, :, 1].ravel(),
                frenet[:, :, 0].ravel(),
                c=risk.ravel(),
                s=3,
                alpha=0.6,
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            spatial.set(
                title="Where the sampled states lie",
                xlabel="-ey (m)",
                ylabel="s (m)",
                xlim=(-half_width, half_width),
            )
            figure.colorbar(points, ax=spatial, label="predicted failure probability")
            figure.suptitle(
                f"Sampled risk-to-go | step {step} | model generation "
                f"{diagnostic['generation']} | {diagnostic['input_mode']}"
            )
        if self.interactive:
            figure.show()
            if pause:
                self.wait_for_interaction(figure, keep_paused)
            else:
                self.pump()
        return figure

    def risk_validation(self, data, pause=False, keep_paused=lambda: True):
        """Show held-out predictions and the manual or learned cutoff."""
        if self.validation_figure is None or not self.plt.fignum_exists(
            self.validation_figure.number
        ):
            self.validation_figure = self.plt.figure(figsize=(9, 6))
            self.figures.append(self.validation_figure)
        figure = self.validation_figure
        figure.clear()
        figure.set_layout_engine("tight")
        probabilities = np.asarray(data["predictions"], dtype=float)
        labels = np.asarray(data["labels"], dtype=int)
        cutoff = data.get("cutoff")
        if (
            probabilities.ndim != 1
            or probabilities.shape != labels.shape
            or np.any(~np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
            or np.any((labels != 0) & (labels != 1))
            or (cutoff is not None and (not np.isfinite(cutoff) or not 0 <= cutoff <= 1))
        ):
            raise ValueError(
                "cutoff plot needs aligned probabilities, binary labels, and valid cutoff"
            )
        axes = figure.subplots(2, 1, height_ratios=(2, 1))
        if not len(probabilities):
            axes[0].text(
                0.5, 0.8, data.get("message", "No validation samples available yet"),
                ha="center", va="center", transform=axes[0].transAxes,
            )
        else:
            ordered = np.argsort(probabilities, kind="stable")
            predicted, actual = probabilities[ordered], labels[ordered]
            x = np.arange(len(predicted))
            for label, color, name in ((0, "tab:blue", "safe"), (1, "tab:red", "fail")):
                selected = actual == label
                axes[0].scatter(
                    x[selected], predicted[selected], s=8, alpha=0.65,
                    color=color, label=f"{name} predictions ({selected.sum()})",
                )
                axes[1].hist(
                    predicted[selected],
                    bins=np.linspace(0, 1, 31),
                    alpha=0.55,
                    color=color,
                    label=f"{name} samples",
                )
        source = data.get("cutoff_source")
        if cutoff is not None:
            label = f"Cutoff = {cutoff:.6g}" + (f" ({source})" if source else "")
            axes[0].axhline(cutoff, color="tab:purple", linestyle="--", label=label)
            axes[1].axvline(cutoff, color="tab:purple", linestyle="--", label=label)
        axes[0].set(
            xlabel="Validation state (sorted by prediction)",
            ylabel="Predicted failure probability",
            ylim=(-0.05, 1.05),
        )
        axes[1].set(xlabel="Predicted failure probability", ylabel="State count", xlim=(0, 1))
        for axis in axes:
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        figure.suptitle(
            f"Risk cutoff samples | model generation {data['generation']} | "
            f"{len(probabilities)} states"
            + (
                f" | cutoff={cutoff:.6g}" + (f" ({source})" if source else "")
                if cutoff is not None
                else f" | cutoff: {data.get('cutoff_status', 'unavailable')}"
            )
        )
        if self.interactive:
            figure.show()
            if pause:
                self.wait_for_interaction(figure, keep_paused)
            else:
                self.pump()
        return figure

    def density_mapping(self, data, ax):
        """Actual sampled penalties against raw NLL, with the frozen mapping."""
        transform = data.get("penalty_transform", "linear")
        ax.set_title(f"Raw NLL → {transform} penalty")
        cutoff, scale = data.get("active_cutoff"), data.get("compression_scale")
        normalizer = data.get("penalty_normalizer")
        message = data.get("message", "Active NF mapping is unavailable.")
        ready = (data.get("status") == "ready" and data.get("active_method") == "nf_density"
                 and cutoff is not None and np.isfinite(cutoff)
                 and scale is not None and np.isfinite(scale) and scale > 0)
        if ready and transform == "log1p" and (normalizer is None
                                                or not np.isfinite(normalizer) or normalizer <= 0):
            ready, message = False, "The active log1p normalizer was not recorded."
        if not ready:
            ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            return

        def mapping(scores):
            if transform == "log1p":
                return _log1p_nll_excess(scores, cutoff, scale) / normalizer
            return np.clip((np.asarray(scores, dtype=float) - cutoff) / scale, 0., 1.)

        safe = np.asarray(data["safe_set_scores"])
        calibration = np.asarray(data["calibration_scores"])
        terminal = np.asarray(data["terminal_scores"])
        window = data.get("active_window") if data.get("calibration_mode") == "local" else None
        if window is not None:
            progress = np.asarray(data["safe_set_states"])[:, 0]
            safe = safe[(progress > window[0]) & (progress < window[1])]
            calibration = np.asarray(data["selected_calibration_scores"])
        reference = np.concatenate((safe, calibration, [cutoff]))
        raw = np.concatenate((reference, terminal))
        raw = raw[np.isfinite(raw)]
        lo, hi = float(raw.min()), float(raw.max())
        curve = np.concatenate((np.linspace(lo, hi, 256), raw, [cutoff]))
        excess_max = hi - cutoff
        if excess_max > 0:
            # Resolve the bend near the cutoff even with very large outliers.
            curve = np.concatenate((curve, cutoff + np.geomspace(
                min(scale * 1e-6, excess_max), excess_max, 256)))
        curve = np.unique(curve)
        ax.plot(curve, mapping(curve), color="tab:purple", linewidth=1.5,
                label=f"{transform} mapping", zorder=2)
        for scores, color, label in ((safe, "0.6", "Safe set"),
                                     (calibration, "tab:blue", "Calibration")):
            finite = np.isfinite(scores)
            ax.scatter(scores[finite], mapping(scores[finite]), color=color, s=14,
                       alpha=.5, label=f"{label} (n={np.sum(finite)})", zorder=3)
        # Preserve exact inference costs (including support overrides) instead
        # of replacing the terminal observations with theoretical curve values.
        penalties = data.get("terminal_penalties")
        if penalties is not None:
            penalties = np.asarray(penalties)
            finite = np.isfinite(terminal) & np.isfinite(penalties)
            ax.scatter(terminal[finite], penalties[finite], color="tab:orange", marker="x",
                       s=28, alpha=.75, label=f"Terminal samples (n={np.sum(finite)})", zorder=4)
            if window is not None:
                progress = np.asarray(data["terminal_states"])[:, 0]
                outside = finite & ~((progress > window[0]) & (progress < window[1]))
                if outside.any():
                    ax.scatter(terminal[outside], penalties[outside], facecolors="none",
                               edgecolors="tab:red", s=60, linewidths=1,
                               label="Outside local window (penalty ≥ 1)", zorder=5)
        full_range = data.get("penalty_origin") == "safe_set_min"
        ax.axvline(cutoff, color="0.5" if full_range else "tab:red", linewidth=1, linestyle=":",
                   label="Minimum safe-set NLL" if full_range else "Zero-penalty cutoff")
        ax.axhline(1., color="0.4", linewidth=1, linestyle="--", label="Penalty 1")
        reference = reference[np.isfinite(reference)]
        symlog = np.ptp(raw) > 6 * max(float(np.ptp(reference)), 1.)
        if symlog:
            ax.set_xscale("symlog", linthresh=max(float(np.max(np.abs(reference))), 1.))
            # Pad in display coordinates: a linear margin at a huge positive
            # maximum would invent an equally huge negative NLL region.
            axis_transform = ax.xaxis.get_transform()
            bounds = axis_transform.transform(np.asarray([lo, hi]))
            padding = max(float(np.ptp(bounds)) * .04, 1e-3)
            ax.set_xlim(axis_transform.inverted().transform(bounds + [-padding, padding]))
        ax.set(xlabel="Raw NLL" + (" (symlog scale)" if symlog else ""),
               ylabel="Terminal safe-set penalty (before λ)")
        if transform == "log1p":
            compressed_axis = ax.secondary_yaxis("right", functions=(
                lambda penalty: penalty * normalizer, lambda compressed: compressed / normalizer))
            origin = "minimum" if full_range else "cutoff"
            compressed_axis.set_ylabel(f"log1p(max(NLL − {origin}, 0) / scale)", fontsize=8)
            compressed_axis.tick_params(labelsize=8)
            ax.set_title(f"Raw NLL → log1p penalty\nscale={scale:.3g} | normalizer={normalizer:.3g}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="upper left")

    def density_calibration(self, data, step, show=False, *, axis=None):
        """Raw NLL against progress, with the active NLL cutoff and upper bound."""
        if axis is None:
            if self.density_calibration_figure is None or not self.plt.fignum_exists(
                self.density_calibration_figure.number
            ):
                self.density_calibration_figure = self.plt.figure(figsize=(11, 5.5), layout="constrained")
                self.figures.append(self.density_calibration_figure)
            figure = self.density_calibration_figure
            figure.clear()
            ax = figure.subplots()
        else:
            figure, ax = axis.figure, axis
        logarithmic = data.get("penalty_transform") == "log1p"
        full_range = data.get("penalty_origin") == "safe_set_min"
        mode_label = " | log1p" if logarithmic else ""
        loss = data.get("training_loss")
        loss_label = f"NF training loss: {loss:.4g}" if loss is not None else "NF training loss: N/A"
        upper_quantile = data.get("penalty_upperbound_quantile")
        sigma = data.get("method") == "sigma"
        # Older recordings predate quantile-only calibration. Preserve their
        # numeric bounds without describing them as quantile-derived.
        relaxation_quantile = data.get("relaxation_quantile") if sigma or upper_quantile is not None else None
        relaxation_label = (f"method=sigma | penalty_sigma={data['penalty_sigma']:g} | cutoff quantile={relaxation_quantile:g}"
                            if sigma else f"relaxation_quantile={relaxation_quantile:g}"
                            if relaxation_quantile is not None else "")
        model_label = "\n".join(label for label in (loss_label, relaxation_label) if label)
        if data["status"] != "ready":
            ax.text(.5, .5, data["message"], ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if axis is None:
                figure.suptitle(f"NF threshold inspection | Step {step}{mode_label}\n{model_label}")
            else:
                ax.set_title(f"NF threshold inspection{mode_label}\n{model_label}")
        else:
            safe, safe_scores = data["safe_set_states"], data["safe_set_scores"]
            cal, cal_scores = data["calibration_states"], data["calibration_scores"]
            terminal, terminal_scores = data["terminal_states"], data["terminal_scores"]
            mode, window = data["calibration_mode"], data["active_window"]
            safe_s, cal_s, terminal_s = safe[:, 0], cal[:, 0], terminal[:, 0]
            calibration_count = len(cal_s)
            if mode == "global":
                # A measured state can appear at s and s+L in controller
                # geometry. Mark both as calibrated without counting it twice.
                cal_s = np.asarray(data.get("calibration_display_s", cal_s))
                cal_scores = np.asarray(data.get("calibration_display_scores", cal_scores))
            if mode == "local":
                # Highlight the selected local calibration states while keeping
                # the full safe set and every sampled endpoint visible.
                cal_s = np.asarray(data.get("selected_calibration_s", []))
                cal_scores = np.asarray(data.get("selected_calibration_scores", []))
                if window is not None:
                    lo, hi = window
                    ax.axvspan(lo, hi, color="tab:cyan", alpha=.1,
                               label=f"Selected local range: {lo:.3g}–{hi:.3g} m")
                    ax.axvspan(lo, hi, ymin=0, ymax=.015, color="tab:cyan", alpha=.9, zorder=5)
                else:
                    ax.text(.5, .5, "The local calibration window was not recorded.",
                            ha="center", va="center", transform=ax.transAxes)
            active = data["active_cutoff"]
            relaxation = data.get("active_relaxation_cutoff")
            upper = data.get("active_upper_cutoff")
            calibrated_upper = data.get("calibrated_upper_cutoff")
            sample_upper = (data.get("penalty_normalization") == "sample_max" and logarithmic
                            or upper is not None and calibrated_upper is not None and upper < calibrated_upper)
            if logarithmic and active is None:
                mode_label += " (inactive)"
            ax.scatter(safe_s, safe_scores, color="0.7", s=9, alpha=.35,
                       label=f"Safe set (n={len(safe_s)} rows)", zorder=1)
            calibration_label = (f"Calibration (n={calibration_count} unique; {len(cal_s)} displayed)"
                                 if mode == "global" and len(cal_s) != calibration_count else
                                 f"{'Local ' if mode == 'local' else ''}Calibration (n={len(cal_s)})")
            if sigma:
                calibration_label = calibration_label.replace("Calibration", "Held-out calibration")
            ax.scatter(cal_s, cal_scores, color="tab:blue", s=17, alpha=.65,
                       label=calibration_label, zorder=2)
            ax.scatter(terminal_s, terminal_scores, color="tab:orange", marker="x", s=20,
                       alpha=.65, label=f"Terminal samples (n={len(terminal_s)})", zorder=3)
            coverage = f"{100 * data['coverage']:g}%"
            penalty_bound = (f"μ + {data['penalty_sigma']:g}σ" if sigma else f"{100 * upper_quantile:g}% quantile"
                             if upper_quantile is not None else "recorded upper reference")
            # Preserve every raw score while keeping the active thresholds
            # readable when a terminal outlier is far outside the safe states.
            reference = np.concatenate((safe_scores, cal_scores,
                                        [] if active is None else [active],
                                        [] if relaxation is None else [relaxation],
                                        [] if upper is None or logarithmic else [upper]))
            reference = reference[np.isfinite(reference)]
            all_scores = np.concatenate((reference, terminal_scores[np.isfinite(terminal_scores)]))
            symlog = len(reference) and np.ptp(all_scores) > 6 * max(float(np.ptp(reference)), 1.)
            if symlog:
                linear_limit = max(float(np.max(np.abs(reference))), 1.)
                ax.set_yscale("symlog", linthresh=linear_limit)
            if active is not None:
                if window is not None:
                    lo, hi = window
                if not full_range:
                    cutoff_label = (f"Recorded penalty cutoff = {active:.3g}"
                                    if data.get("penalty_origin") == "legacy_cutoff"
                                    else f"Zero penalty / trajectory threshold ({coverage} quantile) = {active:.3g}" if sigma
                                    else f"Active cutoff ({coverage}) = {active:.3g}")
                    if mode == "global":
                        ax.axhline(active, color="tab:red", linewidth=2, label=cutoff_label)
                    elif window is not None:
                        ax.plot([lo, hi], [active, active], color="tab:red", linewidth=2.5,
                                label=cutoff_label, zorder=5)
                if relaxation is not None and not sigma:
                    level = (f" ({100 * relaxation_quantile:g}% quantile)"
                             if relaxation_quantile is not None else " (recorded)")
                    label = f"Optimal-trajectory threshold{level} = {relaxation:.3g}"
                    if mode == "global":
                        ax.axhline(relaxation, color="tab:purple", linestyle="-.", label=label)
                    elif window is not None:
                        ax.plot([lo, hi], [relaxation, relaxation], color="tab:purple",
                                linestyle="-.", label=label, zorder=5)
                if upper is not None:
                    limit_label = (f"Penalty 1 at sampled maximum = {upper:.3g}" if sample_upper
                                   else f"Penalty 1 at {penalty_bound} = {upper:.3g}")
                    if mode == "global":
                        ax.axhline(upper, color="firebrick", linestyle="--", linewidth=1.5,
                                   label=limit_label)
                        ax.axhspan(active, upper, color="tab:red", alpha=.04)
                    elif window is not None:
                        ax.plot([lo, hi], [upper, upper], color="firebrick", linestyle="--",
                                linewidth=1.5, label=limit_label)
                        ax.fill_between([lo, hi], active, upper, color="tab:red", alpha=.04)
                if sigma and sample_upper and calibrated_upper is not None:
                    label = f"Compression upper reference ({penalty_bound}) = {calibrated_upper:.3g}"
                    if mode == "global":
                        ax.axhline(calibrated_upper, color="0.35", linestyle=":", label=label)
                    elif window is not None:
                        ax.plot([lo, hi], [calibrated_upper, calibrated_upper], color="0.35",
                                linestyle=":", label=label)
            statistics = []
            if data.get("terminal_penalties") is not None:
                statistics.append(_penalty_counts_text(data["terminal_penalties"]))
            if mode == "local" and window is not None:
                outside = (terminal_s <= window[0]) | (terminal_s >= window[1])
                if outside.any():
                    statistics.append(f"Terminal samples outside window: {np.sum(outside)}")
            if data.get("excluded_samples"):
                statistics.append(f"Invalid samples excluded: {data['excluded_samples']}")
            if statistics:
                ax.text(.99, .02, "\n".join(statistics),
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
            ylabel = "Raw NLL" + ("; symlog" if symlog else "")
            ax.set(xlabel="Track progress s (m)", ylabel=ylabel,
                   title=((f"NF threshold inspection | Step {step} | {mode} calibration" if axis is None
                           else f"NF thresholds | {mode} calibration") + mode_label + f"\n{model_label}"))
            ax.grid(alpha=.25)
            ax.legend(fontsize=8, loc="upper left")
            if symlog:
                from matplotlib.ticker import FuncFormatter, MaxNLocator

                # Matplotlib's log locator otherwise leaves the negative NLL
                # calibration range unlabeled when it lies inside the linear band.
                lo, hi = ax.get_ylim()
                linear_ticks = MaxNLocator(nbins=6).tick_values(max(lo, -linear_limit), min(hi, linear_limit))
                log_ticks = ax.get_yticks()
                ticks = np.unique(np.concatenate((linear_ticks[np.abs(linear_ticks) < linear_limit],
                                                   log_ticks[np.abs(log_ticks) >= linear_limit])))
                ax.set_yticks(ticks[(ticks >= lo) & (ticks <= hi)])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            if data["active_method"] != "nf_density":
                ax.text(.01, .02, f"Controller uses {data['active_method']}: {data['calibration_status']}",
                        transform=ax.transAxes, fontsize=9, color="tab:red")
        if show and self.interactive:
            figure.show()
            self.pump()
        return figure

    def sampled_dynamics_zoom(self, zoom, comparison):
        """Zoom the real terminal spread; never jitter paths or inflate covariance."""
        paths = np.asarray(comparison.candidates, dtype=float)
        center = np.asarray(comparison.belief[-1, :2], dtype=float)
        offsets = (paths[..., :2] - center) * 1000  # Metres to millimetres only.
        endpoints = offsets[:, -1]
        span = np.ptp(endpoints, axis=0)
        for path in offsets:
            zoom.plot(path[-2:, 0], path[-2:, 1], color="lightsteelblue", linewidth=0.6, alpha=0.4)
        zoom.scatter(endpoints[:, 0], endpoints[:, 1], color="lightsteelblue", s=9)
        zoom.plot(0, 0, "x", color="royalblue", markersize=6, label="Belief endpoint")
        truth = (np.asarray(comparison.true_reference)[-2:, :2] - center) * 1000
        zoom.plot(
            truth[:, 0], truth[:, 1], color="tab:orange", linewidth=0.9,
            marker="o", markevery=[-1], markersize=3, zorder=4,
            label="Ground-truth ST trajectory",
        )
        # Padding only sets the view limits, including a zero-spread population.
        visible = np.vstack((endpoints, [0., 0.], truth[-1]))
        midpoint = (visible.min(axis=0) + visible.max(axis=0)) / 2
        # A square view stays readable even when spread is zero along one axis.
        radius = max(0.02, float(np.ptp(visible, axis=0).max()) * 0.65)
        lower, upper = midpoint - radius, midpoint + radius
        zoom.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                 title="Terminal samples (zoom)", xlabel="Δx (mm)", ylabel="Δy (mm)",
                 aspect="equal")
        zoom.legend(fontsize=8)
        zoom.grid(True, alpha=0.3)
        zoom.text(0.02, 0.02,
                  f"{len(paths)} sampled paths | terminal range:\n"
                  f"Δx={span[0]:.3g}, Δy={span[1]:.3g} mm",
                  transform=zoom.transAxes, fontsize=8,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})

    def model_comparison(self, comparison, track, half_width, show=False, *, safe_set_states=None):
        """Show Cartesian safe-set states with rollouts, plus a separate BNN zoom panel."""
        sampled = comparison.belief_mode == "gaussian_bnn"
        if self.comparison_figure is None or not self.plt.fignum_exists(
            self.comparison_figure.number
        ):
            self.comparison_figure = self.plt.figure(figsize=(11, 6) if sampled else (8, 6))
            self.figures.append(self.comparison_figure)
        figure = self.comparison_figure
        figure.clear()
        figure.set_layout_engine("tight")
        if sampled:
            ax, zoom = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.2, 1)})
            self.sampled_dynamics_zoom(zoom, comparison)
        else:
            ax = figure.subplots()
        for boundary in self.boundaries(track, half_width):
            ax.plot(boundary[:, 0], boundary[:, 1], color="lightgray", linewidth=0.8)
        if safe_set_states is not None and len(safe_set_states):
            safe_states = np.asarray(safe_set_states)
            ax.scatter(
                safe_states[:, 0], safe_states[:, 1], color="tab:green", s=6,
                alpha=0.35, zorder=1, label="Safe-set states",
            )
        for index, (mu, states) in enumerate(
            zip(comparison.frictions, comparison.candidates, strict=True)
        ):
            ax.plot(
                states[:, 0],
                states[:, 1],
                color="lightsteelblue",
                linewidth=0.8 if mu is None else 1.2,
                alpha=0.4 if mu is None else 1.0,
                label=(f"Sampled dynamics functions (n={len(comparison.candidates)})"
                       if index == 0 else "_nolegend_") if mu is None else f"Candidate mu={mu:g}" + (
                    f" | {'w' if comparison.weight_kind == 'ensemble_coefficients' else 'p'}={comparison.model_weights[index]:.3f}"
                    if comparison.model_weights is not None else ""
                ),
            )
        for states, color, style, linewidth, label in (
            (
                comparison.belief,
                "royalblue",
                "--",
                2.2,
                "BNN sampled-function mean" if comparison.belief_mode == "gaussian_bnn" else ("Belief fitted prediction ensemble"
                 if comparison.weight_kind == "ensemble_coefficients"
                 else "Belief probability-weighted ensemble") if comparison.belief_mode == "ensemble"
                else f"Belief ST mu={comparison.belief_friction:.3g}",
            ),
            (
                comparison.true_reference,
                "tab:orange",
                "-",
                0.9,
                f"Ground-truth ST trajectory mu={comparison.true_friction:.3g}",
            ),
        ):
            ax.plot(
                states[:, 0], states[:, 1], color=color, linestyle=style,
                linewidth=linewidth, label=label, zorder=4,
            )
        if comparison.kind == "recorded":
            ax.plot(
                comparison.actual[:, 0],
                comparison.actual[:, 1],
                color="black",
                linewidth=1.8,
                label="Measured Gym trajectory",
            )
        else:
            ax.plot(
                comparison.actual[:, 0],
                comparison.actual[:, 1],
                "ko",
                markersize=4,
                label="Measured initial state (future unobserved)",
            )
        xy = np.concatenate(
            (
                comparison.candidates.reshape(-1, 7),
                comparison.belief,
                comparison.true_reference,
                comparison.actual,
            )
        )[:, :2]
        padding = max(0.5, float(np.ptp(xy, axis=0).max()) * 0.1)
        lower, upper = xy.min(axis=0) - padding, xy.max(axis=0) + padding
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        label = (
            "Selected control horizon"
            if comparison.kind == "planned"
            else "Recorded applied-control replay"
        )
        title = "Model comparison"
        if comparison.belief_mode == "gaussian_bnn":
            title = "Sampled dynamics trajectories"
            label += " | one fixed function per path"
        figure.suptitle(
            f"{title} | Step {comparison.step} | Belief generation {comparison.belief_generation}\n{label}"
        )
        ax.set(
            xlabel="x (m)",
            ylabel="y (m)",
            aspect="equal",
        )
        ax.grid(True)
        if sampled:
            ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02))
        else:
            ax.legend(fontsize=8)
        if show and self.interactive:
            figure.show()
            self.pump()
        return figure

    def wait_for_interaction(self, figure, keep_paused=lambda: True):
        figures = [figure] if hasattr(figure, "canvas") else list(figure)
        done = [False]

        def finish(event):
            done[0] = True
            key = getattr(event, "key", None)
            if key == "0":
                # Handle a focused plot's keypress immediately, without waiting
                # for the global listener's later key-release notification.
                if self.keyboard_command is not None:
                    self.keyboard_command("0")
                self.close()
            elif key in ("1", "2", "3", "4", "5", "6") and self.keyboard_command is not None:
                self.keyboard_command(key)
            elif key in ("left", "right", "q") and self.keyboard_command is not None:
                self.keyboard_command(key)

        callbacks = [
            (item, item.canvas.mpl_connect(event, finish))
            for item in figures
            for event in ("key_press_event", "button_press_event", "close_event")
        ]
        self.waiting = True
        try:
            while (
                not done[0]
                and keep_paused()
                and any(self.plt.fignum_exists(f.number) for f in figures)
            ):
                self.plt.pause(0.05)
        finally:
            self.waiting = False
            for item, callback in callbacks:
                item.canvas.mpl_disconnect(callback)

    def pump(self):
        if not self.interactive:
            return
        figures = [figure for figure in self.figures if self.plt.fignum_exists(figure.number)]
        # pyplot.pause only redraws its active figure; queue every changed canvas.
        for figure in figures:
            if self.keyboard_command is not None and not getattr(figure, "_history_keys_bound", False):
                def key(event):
                    if not self.waiting and (event.key in ("left", "right", "q")
                            or self.history_active and event.key in ("0", "1", "2", "3", "4", "5", "6")):
                        self.keyboard_command(event.key)
                figure.canvas.mpl_connect("key_press_event", key)
                figure._history_keys_bound = True
            if figure.stale:
                figure.canvas.draw_idle()
        if figures:
            self.plt.pause(0.001)

    def close(self):
        """Close only this experiment's Matplotlib windows; allow later reuse."""
        for figure in self.figures:
            self.plt.close(figure)
        self.figures.clear()
        self.debug_figure = None
        self.comparison_figure = None
        self.risk_figure = None
        self.validation_figure = None
        self.safe_set_cost_figure = None
        self.density_calibration_figure = None
        self.summary_figures.clear()
