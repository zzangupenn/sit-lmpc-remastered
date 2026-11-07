# SIT-LMPC Remastered

Refactored racecar SIT-LMPC with online value learning, a Gaussian BNN dynamics distribution, calibrated Bayesian belief updates, and dynamics-based chance constraints.

## Main findings and capabilities

- The known-dynamics preset retains MPPI warm-up, convex-hull safe sets, NF/MLP value learning and the smallest-feasible-lambda selection behavior.
- The BNN preset uses the mean of coherent sampled-function predictions for nominal MPPI prediction, with one fixed function per scenario throughout each horizon.
- Attempt-boundary learning updates the latent Gaussian posterior using measured applied controls and successful/failed trajectories. Tempering uses separate calibration/test windows to avoid accepting an unjustifiably concentrated posterior.
- Chance constraints estimate boundary and terminal-hull violation probabilities under sampled dynamics. Thresholds and the probability estimator are configurable; empirical estimates are not physical safety guarantees.
- This repository has its own Python package and uv environment. Walker experiments, source-repository environments and historical results are excluded.

## Validated racecar examples

The [racecar examples](examples/racecar/README.md) contain portable configs, a small pretrained BNN, recorded data and plots from the completed 300-lap comparison. Both controllers completed all laps without a physical failure; best lap times were 13.7 s for known-dynamics SIT-LMPC and 14.0 s for BNN/chance control. The example report also documents chance fallbacks and imperfect held-out coverage.

```bash
uv sync --extra gpu --extra plots --extra dev
uv run python examples/racecar/run_example.py --controller known-dynamics
uv run python examples/racecar/run_example.py --controller bnn-unknown
```

## Run

```bash
uv sync --extra gpu --extra plots --extra dev
uv run sit-racecar
```

The default is the known-dynamics SIT-LMPC baseline. Edit `configs/default.yaml`; its target is the packaged, annotated default YAML. Simulator rendering and keyboard controls are independent. To run without the simulator window:

```bash
uv run sit-racecar --headless
uv run sit-racecar --config configs/bnn-chance.yaml --headless
```

The main BNN preset reads `artifacts/vehicle-bnn-v2-l96`. Import the included example model there, or provide your own compatible trained checkpoint:

```bash
uv run sit-import-bnn --source examples/racecar/models/vehicle-bnn-v2-l96
uv run sit-racecar --config configs/bnn-chance.yaml
```

Use `configs/bnn-posterior.yaml` for BNN learning without chance constraints. `--seed 42` selects a reproducible experiment; otherwise a fresh saved master seed is used. `--max-laps` includes physically failed attempts. `--resume` points directly to a self-contained experiment checkpoint and creates a new result folder.

**Table 1. Entry points.**

| Command | Purpose |
|---|---|
| `sit-racecar` | Warm-up, value learning, LMPC and optional BNN/chance constraints |
| `sit-track` | MPPI tracking without LMPC value learning |
| `sit-replay` | Saved experiment trajectories and plots |
| `sit-insight` | Recorded control-step diagnostics |
| `sit-train-value` | Offline NF or MLP value training |
| `sit-import-bnn` | Validate and copy a pretrained population checkpoint |
| `sit-train-dynamics` | Offline population-model training using dyna-dist |

## Code and experiment structure

**Table 2. Modules and saved data.**

| Location | Responsibility |
|---|---|
| `src/sit_lmpc/planners/` | MPPI/LMPPI kernels, safe sets and probabilistic constraints |
| `src/sit_lmpc/dynamics/` | Analytical ST and coherent Gaussian BNN predictors |
| `src/sit_lmpc/training/` | Value training, latent posterior updates and calibration |
| `src/sit_lmpc/data/` | Transitions, complete success/failure attempts and checkpoints |
| `src/sit_lmpc/experiments/` | Runtime and racecar experiment orchestration |
| `src/sit_lmpc/diagnostics/` | Console, keyboard, Qt and matplotlib diagnostics |
| `configs/` | Known-dynamics baseline and BNN presets |
| `examples/racecar/` | Reproducible known/unknown-dynamics examples, compact recorded outcomes and plots |
| `vendor/` | Pinned local source snapshots, with original license notices |
| `artifacts/` | Imported population checkpoints, excluded from Git |
| `results/<map>/<name>-<Eastern timestamp>/` | Resolved configuration, trajectories, models, metrics and checkpoints |

The BNN's pretrained neural features stay fixed during the online latent update. Retraining the population neural network is an explicit offline operation:

```bash
uv run sit-train-dynamics --config configs/train-dynamics.yaml --dataset /path/to/vehicle-study
```

Auxiliary racecar risk-to-go, finite-bank and density modules remain available internally for compatibility; the supplied presets disable their learning/constraints. They are separate from dynamics-based chance constraints.

See the [architecture and belief equations](docs/ARCHITECTURE_AND_DYNAMICS.md), [migration plan](docs/SIT_LMPC_REMASTERED_MIGRATION_PLAN.md), [source provenance](docs/MIGRATION_SOURCE_MANIFEST.json), and [validation results](docs/MIGRATION_VALIDATION.md).

## Development checks

```bash
uv run python scripts/test_project.py
uv run ruff check src tests scripts examples
uv build
```

The test runner uses separate Python processes for unit and integration suites. See [publication validation](docs/EXAMPLES_PUBLICATION_VALIDATION.md) for the tested setup and compiler-isolation note.
