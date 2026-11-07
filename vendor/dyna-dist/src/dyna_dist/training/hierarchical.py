"""Option 2: alternating full-instance Gaussian E-steps and minibatch neural M-steps."""

from dataclasses import asdict, replace
from pathlib import Path
import json

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dyna_dist.config import resolve_seed, write_config
from dyna_dist.io.checkpoint import load_model, save_model, save_training, restore_training
from dyna_dist.io.common import new_directory, versions, write_json
from dyna_dist.io.dataset import Dataset
from .objectives import expected_complete_loss, predictive_loss
from .population import infer_and_canonicalize
from .trainer import build_model, prepare_instances, draw_batch, balanced_group_indices


def draw_em_batch(instances, means, covariances, rng, config):
    result = {key: [] for key in ("query_inputs", "query_targets", "latent_mean", "latent_covariance",
                                  "likelihood_weight")}
    average_count = np.mean([len(item[0]) for item in instances])
    for index in rng.integers(len(instances), size=config.batch_instances):
        x, y, _, groups = instances[index]
        order = (balanced_group_indices(groups, config.query_size, rng)
                 if config.balance_learning_stages
                 else rng.choice(len(x), config.query_size, replace=False))
        for key, value in zip(result, (x[order], y[order], means[index], covariances[index],
                                       len(x) / average_count), strict=True):
            result[key].append(value)
    dtype = instances[0][0].dtype
    return {key: jnp.asarray(np.stack(value), dtype=dtype) for key, value in result.items()}


def train_hierarchical(config, emit=print):
    if config.model.domain != "st":
        raise ValueError("option 2 currently supports the ST vehicle study")
    dataset = Dataset(config.dataset)
    if (dataset.domain.name != config.model.domain
            or not np.isclose(dataset.timestep, config.model.timestep)
            or dataset.manifest["config"]["profile"] != config.model.baseline_profile):
        raise ValueError("option 2 model domain, timestep, or profile differs from dataset")
    saved_config = None
    if config.resume_path:
        manifest = json.loads((Path(config.resume_path) / "manifest.json").read_text())
        saved_config = manifest["training_config"]
        if config.seed is None:
            config = replace(config, seed=saved_config["seed"])
        for key in ("training_option", "objective", "seed", "em_refresh_every", "em_transitions",
                    "em_chunk_size", "rollout_validation_every", "rollout_validation_particles",
                    "data_view", "max_transitions_per_instance", "balance_learning_stages", "model"):
            if json.dumps(asdict(config)[key], sort_keys=True) != json.dumps(saved_config[key], sort_keys=True):
                raise ValueError(f"option 2 resume requires the same {key}")
    else:
        config = replace(config, seed=resolve_seed(config.seed))
    jax.config.update("jax_enable_x64", True)
    if config.resume_path or config.init_model_path:
        model = load_model(config.resume_path or config.init_model_path)
        if model.config != config.model:
            raise ValueError("initial/resumed model configuration differs")
    else:
        model = build_model(config.model, dataset, config.seed, config.data_view,
                            config.max_transitions_per_instance, config.balance_learning_stages)
        jax.config.update("jax_enable_x64", True)
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    save_model(model, root / "initial")
    write_json(root / "data_split.json", {"fingerprint": dataset.fingerprint, "instances": [
        {"instance_id": item["instance_id"], "split": item["split"]} for item in dataset.instances
    ]})
    training = prepare_instances(model, dataset, "train", config.data_view,
                                 config.max_transitions_per_instance, config.balance_learning_stages,
                                 config.seed)
    validation = prepare_instances(model, dataset, "validation", config.data_view,
                                   config.max_transitions_per_instance, config.balance_learning_stages,
                                   config.seed)
    if not training or not validation or any(len(item[0]) < config.query_size for item in training):
        raise ValueError("option 2 requires nonempty train/validation groups with enough transitions")
    rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(1)[0])
    validation_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 714]))
    validation_batches = [draw_batch(validation, validation_rng, config)
                          for _ in range(config.validation_batches)]
    from .rollout_validation import RolloutValidation

    rollout_validator = RolloutValidation(dataset, config.seed ^ 81733, config.rollout_validation_particles)
    # The Gaussian prior is a closed-form M-step, not an Adam parameter. A
    # canonical N(0,I) gauge removes prior/head scale and translation redundancy.
    optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip),
                            optax.adamw(config.learning_rate, weight_decay=config.weight_decay))
    state = TrainState.create(apply_fn=None, params=model.parameters, tx=optimizer)
    history, best_score = [], float("inf")
    best_nll = last_nll = None
    if config.resume_path:
        state, best_score, best_parameters, best_generation, extra = restore_training(
            config.resume_path, state, rng, dataset.fingerprint, return_extra=True
        )
        if not extra or config.steps <= int(state.step):
            raise ValueError("option 2 resume needs latent state and a larger update budget")
        means, covariances, counts = (extra[key] for key in ("means", "covariances", "counts"))
        best_nll, last_nll = extra["best_nll"], extra["last_nll"]
        save_model(model.replace(parameters=best_parameters, generation=best_generation), root / "best")
    else:
        emit(f"[EM] initial E-step | instances={len(training)} | transitions={config.em_transitions or 'all'}")
        model, means, covariances, counts = infer_and_canonicalize(model, training, config)
        state = state.replace(params=model.parameters, opt_state=optimizer.init(model.parameters))

    @jax.jit
    def update(current, batch):
        # Keep prior parameters outside the differentiated parameter tree.
        from flax.core import freeze, unfreeze

        def objective(neural):
            parameters = unfreeze(current.params)
            parameters.update(neural)
            return expected_complete_loss(freeze(parameters), batch, config.model)

        neural = {key: current.params[key] for key in ("network", "noise_raw")}
        loss, gradient = jax.value_and_grad(objective)(neural)
        from flax.core import freeze as frozen

        gradients = frozen({key: gradient[key] if key in gradient else jnp.zeros_like(value)
                            for key, value in current.params.items()})
        next_state = current.apply_gradients(grads=gradients)
        # AdamW otherwise decays the fixed canonical Cholesky parameters too.
        parameters = unfreeze(next_state.params)
        parameters["prior_mean"] = current.params["prior_mean"]
        parameters["prior_raw"] = current.params["prior_raw"]
        return next_state.replace(params=frozen(parameters)), loss

    validate = jax.jit(lambda p, b: predictive_loss(p, b, config.model, "population_joint", stable=True))
    start_step = int(state.step)
    for step in range(start_step + 1, config.steps + 1):
        state, loss = update(state, draw_em_batch(training, means, covariances, rng, config))
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"nonfinite expected-complete likelihood at update {step}")
        refreshed = step % config.em_refresh_every == 0 or step == config.steps
        if refreshed:
            model = model.replace(parameters=state.params, generation=step)
            model, means, covariances, counts = infer_and_canonicalize(model, training, config)
            # A new coordinate system requires fresh Adam moments. This is an
            # explicit block-EM optimizer restart, not a hidden state mismatch.
            state = state.replace(params=model.parameters, opt_state=optimizer.init(model.parameters))
        if step == 1 or step % config.validation_every == 0 or step == config.steps:
            last_nll = float(np.mean([float(validate(state.params, batch)) for batch in validation_batches]))
            if not np.isfinite(last_nll):
                raise FloatingPointError("nonfinite validation joint likelihood")
            row = {"step": step, "train_expected_nll_per_component": loss,
                   "validation_nll_per_component": last_nll, "em_refreshed": refreshed}
            if step == 1 or step % config.rollout_validation_every == 0 or step == config.steps:
                current = model.replace(parameters=state.params, generation=step)
                score = rollout_validator.evaluate(current)
                row["rollout_validation"] = score
                if score["score"] < best_score:
                    best_score, best_nll = score["score"], last_nll
                    save_model(current, root / "best", overwrite=True)
            history.append(row)
            write_json(root / "history.json", history)
            emit(f"[option2] step={step} | expected NLL={loss:.5g} | validation joint NLL={last_nll:.5g}"
                 + (f" | rollout energy={row['rollout_validation']['score']:.5g}"
                    if "rollout_validation" in row else ""))
        if step % config.checkpoint_every == 0 or step == config.steps:
            current = model.replace(parameters=state.params, generation=step)
            save_training(current, state, rng, root / "checkpoints" / f"step-{step:06d}",
                          config, dataset.fingerprint, best_score, load_model(root / "best"),
                          extra_state={"means": means, "covariances": covariances,
                                       "counts": np.asarray(counts), "best_nll": best_nll,
                                       "last_nll": last_nll})
    write_json(root / "summary.json", {
        "status": "complete", "steps": config.steps, "training_option": 2,
        "best_validation_nll_per_component": best_nll,
        "last_validation_nll_per_component": last_nll, "best_rollout_energy_m": best_score,
        "dataset_fingerprint": dataset.fingerprint, "seed": config.seed,
        "final_model": "best", "checkpoint_selection": "validation rollout energy, 1/5/10/15 steps",
        "e_step_transition_counts": list(map(int, counts)), "versions": versions(),
        "canonical_prior": "N(0,I); population affine transform absorbed into network heads",
        "optimizer_restart": "after every E/prior/canonicalization block",
    })
    return root
