# Racecar examples: publication validation

## Main findings

- The published example contains two completed 300-lap histories, full portable configs, compact CSV metrics, two measured best-lap trajectories, regenerated figures and the small pretrained latent-96 model.
- A clean checkout of the staged Git files installed entirely offline. Its package imports resolved inside that checkout, and both CLI controllers applied physical controls using only the published files. The BNN required no local artifact directory or sibling repository.
- The two-phase example runner completed a one-lap prefix and resumed each controller to two total laps, preserving a self-contained checkpoint. The full recorded experiments separately exercised 300 attempts per controller, including learned LMPPI and chance constraints.
- Unit and integration suites run in separate Python processes through `scripts/test_project.py`. A combined single-process invocation encountered a native JAX compiler segmentation fault in this workspace; the root cause has not been established. This test-process issue did not occur in the completed GPU experiments or clean-checkout CLI checks.

**Table 1. Publication checks.**

| Check | Scope |
|---|---|
| Unit suite: **484 passed** | Controller kernels, config, posterior updates, probability selection, checkpoint logic, examples and hashes |
| Integration suite: **158 passed** | Simulator lifecycle, real-track planning, real Gym controls, posterior/resume, Qt offscreen diagnostics and value learning |
| Clean staged checkout | Offline install; local dependency isolation; known-ST and BNN CLI controls |
| Example workflow | Known/ST and BNN prefix creation followed by checkpoint resume |
| Recorded data | 600 completed trial rows, 300 belief-update rows, numeric best-lap archives and model checksum |
| Packaging | Wheel/source build; examples and model included in source archive; no local run directories |
| Git contents | Generated results, intermediate checkpoints, environments, caches and machine-specific monitoring files excluded |

## Reproduce the software checks

```bash
uv sync --extra gpu --extra plots --extra dev
uv run python scripts/test_project.py
uv run ruff check src tests scripts examples
uv pip check
uv build
```

The isolated test runner returns a nonzero exit code if either suite fails. It retains every selected test and does not suppress numerical or safety assertions. The separately installed dependencies and their versions are recorded in `uv.lock`.

See the [racecar examples](../examples/racecar/README.md) for run commands, measured outcomes and the observed chance-fallback and held-out-coverage limitations. This publication does not add an independent plant/seed replication to the recorded comparison.

Validated: 2026-11-07T15:12:23.247802-05:00 (US Eastern). **642 checks passed across the two isolated suites.**
