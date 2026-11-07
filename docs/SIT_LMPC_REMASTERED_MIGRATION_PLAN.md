# SIT-LMPC remastered: migration and validation

## Main findings and scope

- Build the refactored racecar controller in this repository, using the current BRO-LMPC racecar implementation as the source. Keep all source repositories unchanged.
- Retain MPPI warm-up, convex-hull safe sets, online NF/MLP value learning, finish-line extension, failed-attempt records, experiment checkpoints, keyboard diagnostics and plots.
- Add the Gaussian BNN function distribution, attempt-boundary posterior learning with calibrated tempering, expected-dynamics prediction and dynamics-based chance constraints. The analytical known-dynamics SIT-LMPC baseline remains a separate preset.
- Online learning updates the Gaussian latent belief with fixed pretrained neural features. Population neural-network training is a separate offline command; this migration does not silently add neural retraining during control.
- Leave Walker experiments and historical experiment orchestration outside this repository. Keep auxiliary racecar modules required by the migrated runner, with risk-to-go and density alternatives disabled in the supplied presets.

**Table 1. Implementation steps.**

| Step | Deliverable | Validation |
|---|---|---|
| 1 | Racecar package renamed to `sit_lmpc`, new CLI and project-local uv environment | Import, CLI help, package build |
| 2 | Dependency source snapshots, provenance and preserved license notices | No runtime imports from BRO-LMPC or sibling source trees |
| 3 | Annotated default known-dynamics preset and calibrated BNN/chance preset | Explicit config tests and checkpoint load |
| 4 | BNN artifact import and offline population-training entry points | Model fingerprint and one-step/coherent-rollout checks |
| 5 | Existing numerical and lifecycle tests plus bounded real simulator checks | ST baseline, BNN posterior publication, chance on/off, portable resume |
| 6 | User guide, architecture, equations and migration results | Reproducible run commands and verified artifacts |

## Runtime and output policy

Use Python 3.13 and the tested JAX/Flax/Optax versions from the source project. Create this repository's own `.venv`. Save results under `results/<map>/<experiment>-<Eastern timestamp>`. Ignore environments, caches, datasets, model artifacts and experiment outputs in Git. Do not commit or push automatically.

Copy the existing small v2 latent-96 checkpoint into an ignored local artifact directory so the BNN preset is usable here. Record its original path, file hashes and fingerprint; a fresh clone imports a model explicitly. Preserve the repository's existing MIT license and include the inherited Apache-2.0 license and notices for derived code.
