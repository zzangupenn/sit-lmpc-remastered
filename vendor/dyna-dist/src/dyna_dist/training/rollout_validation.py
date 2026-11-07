"""Small fixed held-out-physics rollout check; never reads test members."""

from dataclasses import replace
import jax
import numpy as np

from dyna_dist.config import VehicleBNNEvaluationConfig


class RolloutValidation:
    def __init__(self, dataset, seed, particles):
        from dyna_dist.evaluation.vehicle_bnn import (
            PARAMETER_KEYS, fresh_probe, _physical_rollouts, _simulators,
        )

        self.seed, self.particles = seed, particles
        self.horizon = 15
        self.cases = []
        if dataset.domain.name != "st":
            raise ValueError("option 2 rollout checkpoint selection currently supports ST")
        members = dataset.members("validation")
        if not members:
            raise ValueError("rollout validation needs held-out physics")
        simulators = _simulators(dataset, np.asarray([
            [member.get(key, 1.0) for key in PARAMETER_KEYS] for member in members
        ]))
        probe_config = replace(
            VehicleBNNEvaluationConfig(), seed=seed, horizon=self.horizon,
            rollout_starts=(40,), rollout_signals=(7,),
        )
        for speed in (5.0, 10.0, 15.0, 20.0):
            tape, nominal = fresh_probe(dataset, probe_config, speed, 7)
            initial, controls = nominal[40], tape[40:40 + self.horizon]
            self.cases.append((initial, controls, _physical_rollouts(simulators, initial, controls)))

    def evaluate(self, model):
        from dyna_dist.evaluation.vehicle_bnn import trajectory_cover

        functions = model.sample_functions(jax.random.key(self.seed), self.particles)
        rows = []
        for initial, controls, truth in self.cases:
            paths = np.asarray(functions.rollout(initial, controls)[0])
            paths = np.concatenate((np.broadcast_to(initial, (len(paths), 1, len(initial))), paths), axis=1)
            if not np.isfinite(paths).all():
                return {"score": 1e20, "nonfinite": True}
            for horizon in (1, 5, 10, 15):
                # Energy supports unequal ensemble counts; W1 is unnecessary for
                # frequent checkpoint selection and is reserved for final analysis.
                a, b = paths[:, 1:horizon + 1, :2], truth[:, 1:horizon + 1, :2]
                from scipy.spatial.distance import cdist

                aa, bb = a.reshape(len(a), -1), b.reshape(len(b), -1)
                energy = (2 * cdist(aa, bb).mean() - cdist(aa, aa).mean()
                          - cdist(bb, bb).mean()) / np.sqrt(horizon)
                cover = trajectory_cover(paths, truth, horizon)
                rows.append((energy, cover["terminal_position_rmse_m"],
                             cover["simultaneous_tube_95_coverage"]))
        values = np.mean(rows, axis=0)
        return {"score": float(values[0]), "endpoint_rmse_m": float(values[1]),
                "path_coverage_95": float(values[2]), "nonfinite": False}
