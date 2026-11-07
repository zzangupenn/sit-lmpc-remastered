"""Explicit real-ST environment settings for chance-constraint integration tests."""
from sit_lmpc.config import load_config


def real_gym_config(tmp_path, backend_name, predictor_mode):
    config = load_config(
        overrides={
            "simulation": {
                "friction": 0.55,
                "seed_mode": "fixed",
                "seed": 31,
                "initial_speed": 6,
                "max_laps": 3,
                "max_steps": 4000,
                "dtype": "float32",
                "timestep": 0.1,
                "integrator_timestep": 0.02,
                "action_scale": (0.4, 3.5),
                "control_noise": (0.001, 0.001),
                "state_noise": 0.001,
            },
            "planner": {
                "friction": 0.7,
                "model_bank": {"frictions": (0.4, 0.5, 0.6, 0.7)},
                "belief": {
                    "enabled": True,
                    "backend": backend_name,
                    "predictor": predictor_mode,
                    "fit_horizon": None,
                    "fit_max_iterations": 30,
                    "fit_tolerance": 1e-6,
                    "batch_size": 128,
                    "history_samples": 256,
                    "residual_scales": (0.05, 0.05, 0.01, 0.1, 0.01, 0.1, 0.01),
                    "prior": None,
                    "evidence_strength": 32,
                    "recent_fraction": 0.5,
                    "max_standardized_rmse": 5,
                },
                "horizon": 10,
                "samples": 128,
                "iterations": 1,
                "lambda_count": 4,
                "safe_set_size": 1,
                "hull_selection_points": 100,
                "hull_precompile": True,
                "hull_precompile_limit": 80,
                "risk_constraint": {"enabled": False},
                "probabilistic_safety": {"enabled": False},
                "sit_lmpc_mode": False,
            },
            "value": {
                "epochs": 1,
                "batch_size": 32,
                "hidden_dims": 8,
                "layers": 1,
                "bins": 4,
                "samples": 2,
                "boundary_samples": 0,
                "input_mode": "state",
            },
            "risk": {"enabled": False, "input_mode": "state"},
            "diagnostics": {
                "render": False,
                "plots": False,
                "summary_plots": False,
                "keyboard_controls": False,
            },
            "output": {
                "directory": str(tmp_path / "real-run"),
                "quick_load": False,
                "resume": None,
                "checkpoint_every_trials": 1,
            },
        }
    )
    return config
