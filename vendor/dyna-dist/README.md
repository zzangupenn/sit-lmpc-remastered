# dyna-dist

Dynamics-distribution learners and coherent dynamics-function samplers for BRO-LMPC.

The initial backend is implemented: a shared deterministic neural mean/basis plus a
low-dimensional Gaussian Bayesian last layer. It learns full dynamics or residuals
from the actual local F1TENTH Gym ST physics, with friction and mass varied between
dynamics instances. The HalfCheetah population objective learns from grouped
same-physics transitions and samples functions directly from the learned prior.
The older context-conditioned objective remains available for new studies.

Each sampled coefficient vector defines one dynamics function. It stays fixed at
every rollout step and across all candidate controls; there is no per-step model
resampling or injected process noise.

**Prototype limitation:** learned uncertainty has not been established as calibrated
for hard safety decisions. BRO wiring and the flow-prior backend are not implemented.
Previous trained dynamics checkpoints, evaluations, and result reports were cleared
for a fresh training study; data and reusable code remain.

See the [implementation plan](IMPLEMENTATION_PLAN.md) and
[API and artifact guide](docs/API.md).

The [HalfCheetah SAC example](examples/halfcheetah_rl/README.md) provides a separate
uv environment, friction-family training/data collection, retained intermediate
policies, and JAX actor exports. The shared learner now supports the HalfCheetah
domain as well as ST; see [its plan](docs/HALFCHEETAH_IMPLEMENTATION_PLAN.md).

Walker2d-v5 is also supported as `domain: walker2d`: 18 unclipped physical
qpos/qvel states, six native controls, translation-invariant 23D network inputs,
and a 0.008-second transition interval. Collection and BRO task configuration live
in the sibling `BRO-LMPC/examples/walker` isolated uv project. Its checksummed
fixed-physics trajectory datasets work directly with `dyna-train` and
`dyna-evaluate`; ST and HalfCheetah artifacts remain distinct and incompatible.

## Environment

Run from `/home/lucerna/Research/dyna-dist`. This is an independent Python 3.13 uv
project; its environment is `dyna-dist/.venv`, not BRO's environment.

```bash
uv sync --all-extras
```

This installs collection, plotting, tests, and CUDA 13 support. The collection extra
uses the explicitly configured sibling checkout `../f1tenth_gym`. For a CPU-only
development environment, use `uv sync --extra collection --extra plots --extra dev`.
For inference without Gym, use `uv sync` (CPU) or `uv sync --extra gpu` (CUDA).

After synchronizing, the commands below use `--no-sync` to preserve your selected
extras. Alternatively, use `uv run --all-extras ...` on this CUDA machine. `--locked`
is optional; the checked-in `uv.lock` records the dependency resolution.

## Collect, train, evaluate, sample

Edit the separate YAML presets before running:

- [collect_st.yaml](configs/collect_st.yaml): physical parameters, excitation, noise,
  instance splits, integration interval, seed, output.
- [collect_vehicle.yaml](configs/collect_vehicle.yaml): paired smooth-target/PID vehicle
  data with friction, front/rear cornering stiffness, and mass varied through 20 m/s.
- [vehicle_bnn_study.yaml](configs/vehicle_bnn_study.yaml): reuse the 33 original vehicle
  physics combinations for train/validation and generate 16 new unseen-physics tests.
- [vehicle_bnn/evaluate.yaml](configs/vehicle_bnn/evaluate.yaml): compare prior BNN
  trajectories with fresh sampled physical ST models and held-out ground truth.
- [vehicle_bnn/train_option1.yaml](configs/vehicle_bnn/train_option1.yaml): preserve
  joint-gradient training and explicitly refit its selected Gaussian population prior.
- [vehicle_bnn/train_option2.yaml](configs/vehicle_bnn/train_option2.yaml): hierarchical
  offline EM with full-instance posteriors, stable Gaussian calculations, canonical
  latents, and validation of sampled short-horizon trajectories.
- [train_gaussian.yaml](configs/train_gaussian.yaml): model mode, latent dimension,
  precision, Gaussian prior/likelihood, training budget, checkpoint/resume settings.
- [halfcheetah/train.yaml](configs/halfcheetah/train.yaml): population joint likelihood
  for coherent HalfCheetah functions without test-time context conditioning.
- [evaluate.yaml](configs/evaluate.yaml): held-out split, context sizes, particles,
  horizon, plots, output.

```bash
uv run --no-sync dyna-collect-st --config configs/collect_st.yaml
uv run --no-sync dyna-collect-vehicle --config configs/collect_vehicle.yaml \
  --output runs/vehicle_st/my-next-study
uv run dyna-collect-vehicle-study --config configs/vehicle_bnn_study.yaml \
  --output runs/vehicle_st/my-next-bnn-study
uv run --no-sync dyna-train --config configs/train_gaussian.yaml
uv run --no-sync dyna-evaluate --config configs/evaluate.yaml --plots
uv run --no-sync dyna-sample --model runs/train/best --dataset datasets/st \
  --context 32 --particles 64 --horizon 10 --seed 43 --output runs/samples
```

Set `model.mode: residual` to learn increments relative to the saved nominal ST
baseline; `dynamics` predicts the complete increment. Neither mode receives the
sample's true friction or mass. To force CPU execution, prefix a command with
`JAX_PLATFORMS=cpu`. CUDA is used automatically when available; the training,
evaluation, and sample commands disable JAX's bulk GPU preallocation by default.

The vehicle collector adapts the smooth random-sinusoid steering targets,
42-step speed-target changes, PID controller, and truncated-normal control jitter
from the local `vehicle_data_gen` code. It records each nominal action tape once
and replays it unchanged for every ST parameter combination, so its plots compare
the effect of dynamics rather than different feedback actions. Output includes
training/validation/test trajectory shards, a report, and paired-path,
parameter-effect, and shared-action plots. It uses the Gym ST physics without
map rendering or collision checks. Steering targets have a speed-dependent
lateral-acceleration cap for stable 20 m/s runs. The existing full-scale ST model applies
friction to lateral dynamics but not longitudinal acceleration.
The default vehicle preset allocates exactly 50,000 transitions per parameter
combination, evenly across 5–20 m/s starting speeds. Each speed has twelve
250-step segments and one 125-step segment.
The [vehicle BNN study plan](docs/VEHICLE_BNN_STUDY_PLAN.md) describes the
train/validation/test split, latent-size comparison, and evaluation criteria.
The matched latent-4/8/16/32 training preset is
[vehicle_bnn/train.yaml](configs/vehicle_bnn/train.yaml); launch the sequential
comparison with `uv run --no-sync python -u scripts/train_vehicle_bnn_study.py
--output runs/vehicle_st/NEW_TRAINING_RUN`. The launcher records a plan,
individual run histories, validation-best models, and periodic checkpoints.
Evaluate those checkpoints with `uv run dyna-evaluate-vehicle-bnn --config
configs/vehicle_bnn/evaluate.yaml --output runs/vehicle_st/NEW_EVALUATION`.
The evaluator measures 1–15-step prediction accuracy using fresh controls, seen
and unseen physics, and fixed sampled dynamics functions. It also compares
MPPI-style candidate decisions, local-corridor violation probabilities, first-action
replanning, and GPU throughput. Its compact Markdown report shows endpoint errors
at 1, 5, 10, and 15 steps, whole-path coverage, energy discrepancy, and exact
empirical Wasserstein-1 distance, with a physical-ST-versus-ST sampling baseline.
It embeds summary plots and links the detailed state, decision, and training plots.
Refresh a saved evaluation without rerunning BNN predictions using
`uv run --no-sync dyna-evaluate-vehicle-bnn --refresh-path runs/vehicle_st/EVALUATION`.
The standalone
tracking task does not include BRO's map, terminal value, or safe-set hull.

The [two-option training guide](docs/VEHICLE_BNN_TRAINING_OPTIONS.md) explains the
alternative fitting procedures, checkpoints, and matched comparison. Run all four
latent sizes for both options, followed by paired evaluation and a Markdown report:

```bash
uv run --no-sync python scripts/compare_vehicle_training_options.py \
  --output runs/vehicle_st/training-options-NEW
```

The launcher requires CUDA and records explicit resolved presets. Option 1 retains
the raw `best/` and adds `best_refit/`; the evaluator follows `summary.json`'s
`final_model`. Option 2's stored standard-normal latent is a canonical coordinate
choice: its learned population moments are absorbed into the neural output heads.
Its checkpoints also retain local vehicle posteriors for exact training resume.
The original 27/6/16 [matched option comparison](runs/vehicle_st/training-options-20261017-132753/REPORT.md)
favours option 1 + refit at the tested 20,000-update budget; option 2 remains an
experimental alternative. Neither currently establishes calibrated safety probabilities.

The [larger-population protocol](docs/VEHICLE_BNN_TRAINING_OPTIONS.md#larger-population-100--10--16)
generates 100 new training / 10 new validation combinations while preserving the
same 16 test vehicles. Pass `--dataset` to the study launcher to use that dataset;
the default remains the historical 27/6/16 study.
The completed [100/10/16 comparison](runs/vehicle_st/training-options-100-20261017-1408/DATASET_SIZE_COMPARISON.md)
includes all eight new models, unchanged-test verification, and the old/new plots.
Option 1 + refit improved W1 at all four latent sizes; latent 32 was strongest.
The latest [extended 4–128 latent comparison](runs/vehicle_st/training-options-high-latent-20261017-1447/latent-sweep/REPORT.md)
uses refreshed frozen baselines with stable prior-factor inference. Option 1,
latent 96 has the lowest observed 15-step W1 (0.00904 m; RMSE 0.0656 m; coverage
95.1%), at a higher prediction cost than latent 32. Original models/results are
preserved; numerical and performance audits are linked in that report.

Output directories must be new or empty. Re-running a preset requires changing its
`output` or using `--output`. A null seed creates a fresh master seed and saves it
in the resolved configuration; an integer is reproducible.

### Dyna dataset v2: original-generator racing excitation

[vehicle_dataset_v2.yaml](configs/vehicle_dataset_v2.yaml) calls the original
`vehicle_data_gen` main, steering generator and PID directly, without editing that
repository. It regenerates all 100/10/16 physics combinations, with 50,000 accepted
transitions each at 0.1 s, speeds up to 30 m/s, and speed-independent steering
amplitude/frequency variations. This replaces v1's restrictive steering excitation,
not the existing datasets or trained models. Each saved segment has its actual
terminal state and never crosses a reset or invalid-data gap.

```bash
uv run --no-sync python scripts/collect_vehicle_v2.py
uv run --no-sync python scripts/audit_vehicle_v2.py
```

Collection refuses a nonempty output directory; use `--resume` to continue the
same configuration after interruption (worker count may change), or `--output`
for a new collection. The audit compares joint state/action coverage with saved
SIT-LMPC runs and writes [a report with figures](runs/vehicle_st/dyna-dataset-v2/report/REPORT.md).
It does not train a BNN. Legacy fresh-probe BNN evaluation still needs its v1
action generator and 20 m/s bound updated before evaluating v2 high-speed models.

### Retained data for retraining

The prior trained models and result reports have been cleared. These **local,
git-ignored** data are currently available:

```text
data_collections/halfcheetah_frictions/rl/full_1m/  # five 1M-step RL runs
data_collections/halfcheetah_frictions/studies/bnn-retrain-20261015-223851-211901/data/  # current train/validation/test study
```

The older `datasets/` and `runs/` trees are now in Trash. The current study is
self-contained and does not require them. The local, Git-ignored study guide under
`data_collections/halfcheetah_frictions/studies/bnn-retrain-20261015-223851-211901/`
links the new ground-truth videos and four generated latent-size training configs.
The [pilot presets](configs/pilot_collect.yaml) remain as small-budget examples.

### Resume training

Set `resume_path` to a direct training checkpoint, choose a **new** output directory,
and set `steps` to the desired total step count. Keep the model configuration and
dataset identity unchanged. Set the seed to null or its saved value; optimizer and
RNG states are restored, including the previously best model.

```bash
uv run --no-sync dyna-train --config configs/pilot_train.yaml \
  --resume-path /absolute/path/to/new-run/checkpoints/step-000300 \
  --output runs/continued
```

For this example, first train a new run and increase `steps` above 300 in a copy of
the preset. A model or posterior snapshot is self-contained for inference;
training resume still requires the exact referenced dataset.

### Further train on new transitions

Build an independent dataset containing both historical and newly collected
same-physics groups. Set `init_model_path` to an existing `best/` model, point
`dataset` to the combined dataset, and choose a new `output` directory. This
starts a new optimizer run from the saved network/prior/likelihood and retains
its original normalization. It differs from exact checkpoint resume, which
requires the original dataset fingerprint.

```bash
uv run --no-sync dyna-train --config configs/halfcheetah/train.yaml \
  --dataset /absolute/path/to/combined/data \
  --init-model-path /absolute/path/to/previous/best \
  --output runs/halfcheetah/dynamics/updated
```

For direct population sampling, call `model.sample_functions(key, count)` on a
loaded model or use `dyna-sample` with its default `--context 0`.

## Physical and uncertainty conventions

State: `[x, y, steering_angle, speed, yaw, yaw_rate, slip_angle]`.
Native controls: `[steering_speed_rad_s, acceleration_m_s2]`.
Models predict seven body-frame increments at a fixed interval, initially 0.1 s.
Normalizers are fitted only on training instances.

The full-scale default sweeps friction 0.3–1.2 and mass ratios 0.8–1.2 around
1225.8878467253344 kg. Inertia is held fixed deliberately: scaling inertia with mass
would cancel the mass dependence in this ST model. These are synthetic-study ranges,
not a calibrated payload model. The separate `f1tenth` profile is never mixed with
full-scale training.

Low-speed probes stay below the 0.5 m/s ST switch with zero acceleration because
the local Gym RK4 can become unstable near the switch. Raw invalid terminal states
are preserved, but out-of-envelope/nonfinite/contact transitions are excluded from
learning. `gym_env` additionally requires an explicit existing local map prefix;
pure `gym_st` collection needs no map or renderer.

`predict()` separates epistemic and likelihood/discrepancy covariance. Function
rollouts sample only the Bayesian coefficients, not likelihood noise. The optional
BRO adapter emits equal-weight particles and keeps invalid particles' mass. Its
ordinary Monte Carlo probability error bound is unavailable, not zero. An independent
verification helper exists for fixed candidates, but it bounds sampling error under
the learned model, not model misspecification.

## Code structure and checks

```text
src/dyna_dist/
  config.py, cli.py   # validated presets and four commands
  adapters/          # actual Gym collection; standalone BRO scenario contract
  data/              # grouped collection, transition schema, body-frame transforms
  models/            # neural features, Gaussian algebra, prior interface, JAX ST
  training/          # grouped population or context predictive likelihood and trainer
  inference/         # immutable posteriors, function samples, hypothetical updates
  evaluation/        # held-out metrics and offline Matplotlib diagnostics
  io/                # numeric shards and self-contained checksummed artifacts
configs/, tests/, benchmarks/
```

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
JAX_PLATFORMS=cuda uv run --no-sync pytest -q
uv run --no-sync ruff check src tests benchmarks
uv build
```

Generated data, runs, environments, and caches are ignored. No BRO/Gym source files
are modified. Future flow priors require a new nonconjugate inference backend;
they are not a replacement prior file for the exact Gaussian update.

## MLP versus NF critic side experiment

The independent [PPO/SAC critic experiment](examples/value_critic_rl/README.md)
compares MLP and conditional spline-flow value/Q critics on Walker2d and
HalfCheetah, with a paired seed and 3M environment steps per run. It has its own
GPU uv environment, W&B progress/remaining-hours metrics, and policy checkpoints;
it does not change the dynamics learners or existing RL data collections.

The new [JAX PPO/SAC implementation](examples/value_critic_jax/README.md) runs a
separate matched comparison using 20 NF latent samples with Gaussian standard
deviation 0.1, following SIT-LMPC. Inference, GAE and optimizer epochs are
precompiled before collection; SAC also compiles twin-Q, actor, entropy and
target-network update scans. MuJoCo stays on CPU. Noisy studies train and
evaluate with configurable sensor/actuator noise and frozen calibration scales.
Older PyTorch jobs were
stopped on user request; saved results and checkpoints are retained. Only
actual training runs are logged to W&B; queue progress stays local.
