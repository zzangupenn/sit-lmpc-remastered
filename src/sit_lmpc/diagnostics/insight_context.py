"""Frozen inputs for the existing six diagnostic views, evaluated on demand."""

from copy import copy, deepcopy
from dataclasses import fields
from types import SimpleNamespace
import numpy as np

from sit_lmpc.training.features import inference_snapshot


class InsightContexts:
    """Share immutable trial/generation data across the bounded frame buffer."""

    def __init__(self, config, track):
        from sit_lmpc.diagnostics.density_calibration import DensityCalibrationDiagnostics

        self.config, self.track = config, track
        self.density_diagnostics = DensityCalibrationDiagnostics()
        self._summary_key = self._summary = self._validation_key = self._validation = None

    def capture(self, *, controller, safe_set, comparisons, density, risk, risk_snapshot,
                laps, metrics):
        summary = self.summary(laps, metrics)
        validation_key = id(getattr(risk, "snapshot", None))
        if validation_key != self._validation_key:
            self._validation_key = validation_key
            self._validation = None if risk is None else deepcopy(risk.validation_samples())
        safe = copy(safe_set)
        safe.frenet, safe.cartesian = list(safe_set.frenet), list(safe_set.cartesian)
        safe.state_ids = list(safe_set.state_ids)
        safe.controls, safe.control_masks = list(safe_set.controls), list(safe_set.control_masks)
        safe.rng = deepcopy(safe_set.rng)
        comparison = copy(comparisons) if comparisons is not None else None
        if comparison is not None:
            comparison.belief = controller.model
        training = getattr(density, "diagnostics", {})
        density_training = {key: training.get(key) for key in ("generation", "loss")}
        density_training.update({key: training[key] for key in (
            "status", "training_states", "calibration_states",
        ) if key in training})
        return FrozenContext(self.config, self.track, controller.model, safe, comparison,
            None if density is None else density.snapshot,
            None if density is None else density.validation,
            None if density is None else density.local_calibration,
            inference_snapshot(risk_snapshot), self._validation, summary, self.density_diagnostics,
            density_training)

    def summary(self, laps, metrics):
        key = len(laps), len(metrics)
        if key != self._summary_key:
            self._summary_key = key
            self._summary = dict(laps=[{f.name: getattr(lap, f.name) for f in fields(lap)} for lap in laps],
                                 metrics=deepcopy(metrics))
        return self._summary


class FrozenContext:
    def __init__(self, config, track, model, safe, comparison, density, validation, local,
                 risk, risk_validation, summary, density_diagnostics, density_training=None):
        self.config, self.track, self.model, self.safe = config, track, model, safe
        self.comparison, self.density, self.validation, self.local = comparison, density, validation, local
        self.risk, self.risk_validation, self.summary = risk, risk_validation, summary
        self.density_diagnostics = density_diagnostics
        self.density_training = density_training

    def get(self, frame, number):
        from sit_lmpc.diagnostics.native_insight import frame_result

        result = frame_result(frame)
        if number == 2:
            return self.summary
        if number == 3:
            diagnostic = result.risk
            if diagnostic is None and self.risk is not None and result.debug is not None:
                from sit_lmpc.planners.lmppi import sampled_risk_diagnostic

                actions = result.debug["sample_actions"]
                if self.risk.input_mode == "state_action":
                    actions = np.concatenate((actions, result.debug["sample_endpoint_actions"][:, None]), axis=1)
                diagnostic = sampled_risk_diagnostic(self.risk, result.debug["states"], actions, self.model.action_scale)
            return dict(diagnostic=diagnostic, status=None if diagnostic is not None else
                        "Risk model not ready" if self.config.risk.enabled else "Risk learning is disabled")
        if number == 4:
            return self.risk_validation if self.risk_validation is not None else dict(
                generation=0, predictions=np.empty(0), labels=np.empty(0, int),
                cutoff=self.config.risk.manual_cutoff if self.config.risk.cutoff_mode == "manual" else None,
                cutoff_source=self.config.risk.cutoff_mode, cutoff_status="model_not_trained",
                message="Risk model not ready" if self.config.risk.enabled else "Risk learning is disabled")
        if number == 5:
            if self.comparison is None:
                return dict(unavailable="Dynamics comparison was unavailable in this recording.")
            comparison = self.comparison.planned(frame["state"], frame["actions"][:self.config.planner.horizon], frame["step"])
            return dict(comparison={f.name: getattr(comparison, f.name) for f in fields(comparison)},
                        safe_set_states=self.safe.arrays()[1] if self.safe.cartesian else None)
        if number == 6:
            from sit_lmpc.diagnostics.safe_set_costs import compare_safe_set_costs

            density = self.density_diagnostics.evaluate(result, self.safe, self.density,
                self.validation, self.config.planner, self.local, training=self.density_training)
            if result.debug is None:
                return dict(diagnostic=None, density=density)
            diagnostic = compare_safe_set_costs(result, self.safe, self.config.planner,
                controller=SimpleNamespace(model=self.model, track=self.track))
            return dict(diagnostic=diagnostic, density=density)
        return dict(unavailable="Historical view unavailable.")
