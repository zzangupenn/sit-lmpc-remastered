"""Small-dimension Bayesian meta-training with explicit data splits and resume."""

from dataclasses import asdict, replace
from pathlib import Path
import json
from flax.training.train_state import TrainState
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dyna_dist.config import ModelConfig, decode, resolve_seed, write_config
from dyna_dist.data.transforms import Normalizer, features, increments
from dyna_dist.inference.posterior import PopulationModel, baseline_increments
from dyna_dist.io.checkpoint import load_model, save_model, save_training, restore_training
from dyna_dist.io.common import new_directory, versions, write_json
from dyna_dist.io.dataset import Dataset, balanced_indices
from dyna_dist.domains import get_domain
from dyna_dist.models.features import initialize, prior_and_variance
from dyna_dist.models.profiles import PARAMETER_FIELDS, vehicle_parameters
from .objectives import predictive_loss


def precision(dtype):
    jax.config.update("jax_enable_x64", dtype == "float64")


def build_model(config, dataset, seed, view="true", max_count=0, balance_stages=False):
    config.validate()
    precision(config.dtype)
    if config.domain != dataset.domain.name:
        raise ValueError("dataset domain differs from model domain")
    if config.domain == "st" and config.baseline_profile != dataset.manifest["config"]["profile"]:
        raise ValueError("dataset vehicle profile differs from the model's declared profile")
    if not np.isclose(config.timestep, dataset.timestep):
        raise ValueError("dataset timestep differs from model timestep")
    dtype = getattr(jnp, config.dtype)
    spec = get_domain(config.domain)
    parameters = freeze(initialize(config, jax.random.key(seed)))
    baseline = vehicle_parameters(
        config.baseline_profile,
        config.baseline_friction,
        config.baseline_mass_ratio,
        config.baseline_inertia_policy,
    )
    baseline = jnp.asarray([baseline[k] for k in PARAMETER_FIELDS], dtype=dtype)
    if config.domain != "st":
        baseline = jnp.zeros(0, dtype=dtype)
    dummy = Normalizer(
        jnp.zeros(spec.input_dimension, dtype),
        jnp.ones(spec.input_dimension, dtype),
        jnp.zeros(spec.state_dimension, dtype),
        jnp.ones(spec.state_dimension, dtype),
    )
    model = PopulationModel(parameters, dummy, baseline, config)
    inputs, targets = [], []
    for member in dataset.members("train"):
        transitions = dataset.transitions(
            member, view, max_count=max_count, balance_stages=balance_stages, seed=seed
        )
        states, controls, next_states = (
            jnp.asarray(x, dtype=dtype)
            for x in (transitions.states, transitions.controls, transitions.next_states)
        )
        inputs.append(np.asarray(features(states, controls, config.domain)))
        baseline_delta = np.asarray(baseline_increments(model, states, controls))
        if config.mode == "residual" and (
            not np.isfinite(baseline_delta).all()
            or np.any(
                np.abs(baseline_delta[:, 5] + np.asarray(states)[:, 5])
                > dataset.manifest["config"]["max_abs_yaw_rate"]
            )
            or np.any(
                np.abs(baseline_delta[:, 6] + np.asarray(states)[:, 6])
                > dataset.manifest["config"]["max_abs_slip"]
            )
        ):
            raise ValueError(
                "nominal residual baseline leaves the collection envelope; "
                "check low-speed excitation or declare a finer baseline integration step"
            )
        targets.append(np.asarray(increments(states, next_states, config.domain)) - baseline_delta)
    if not inputs:
        raise ValueError("dataset has no training instances")
    normalizer = Normalizer.fit(
        np.concatenate(inputs), np.concatenate(targets), dtype, config.domain
    )
    return model.replace(normalizer=normalizer)


def prepare_instances(
    model, dataset, split, view="true", max_count=0, balance_stages=False, seed=0
):
    prepared = []
    dtype = model.normalizer.input_mean.dtype
    for member in dataset.members(split):
        transitions = dataset.transitions(
            member, view, max_count=max_count, balance_stages=balance_stages, seed=seed
        )
        states, controls, targets = (
            jnp.asarray(x, dtype=dtype)
            for x in (transitions.states, transitions.controls, transitions.next_states)
        )
        stages = transitions.stages
        groups = tuple(np.flatnonzero(stages == value) for value in np.unique(stages))
        prepared.append(
            (
                np.asarray(
                    model.normalizer.normalize_inputs(
                        features(states, controls, model.config.domain)
                    )
                ),
                np.asarray(model.targets(states, controls, targets)),
                stages,
                groups,
            )
        )
    return prepared


def balanced_group_indices(groups, count, rng):
    """Select a small, stage-balanced subset without permuting every stored transition."""
    limits = np.asarray([len(group) for group in groups])
    if count > int(limits.sum()):
        raise ValueError("not enough transitions for requested stage-balanced selection")
    quota = np.zeros(len(groups), dtype=int)
    assigned = 0
    while assigned < count:
        active = np.flatnonzero(quota < limits)
        for index in rng.permutation(active):
            quota[index] += 1
            assigned += 1
            if assigned == count:
                break
    selected = np.concatenate(
        [rng.choice(group, size=int(size), replace=False) for group, size in zip(groups, quota)]
    )
    return rng.permutation(selected)


def draw_batch(instances, rng, config):
    population = config.objective == "population_joint"
    suitable = [
        (x, y, stages, groups)
        for x, y, stages, groups in instances
        if len(x) >= config.query_size + (0 if population else config.context_min)
    ]
    if not suitable:
        raise ValueError("no instance has enough valid transitions for the batch settings")
    keys = (
        ("query_inputs", "query_targets")
        if population
        else ("context_inputs", "context_targets", "context_mask", "query_inputs", "query_targets")
    )
    result = {k: [] for k in keys}
    for member in rng.integers(len(suitable), size=config.batch_instances):
        inputs, targets, stages, groups = suitable[member]
        if population:
            order = (
                balanced_group_indices(groups, config.query_size, rng)
                if config.balance_learning_stages
                else rng.choice(len(inputs), config.query_size, replace=False)
            )
            result["query_inputs"].append(inputs[order])
            result["query_targets"].append(targets[order])
            continue
        upper = min(config.context_max, len(inputs) - config.query_size)
        count = int(rng.integers(config.context_min, upper + 1))
        order = (
            balanced_indices(stages, count + config.query_size, rng)
            if config.balance_learning_stages
            else rng.permutation(len(inputs))[: count + config.query_size]
        )
        context, query = order[:count], order[count:]
        cx, cy, mask = (
            np.zeros((config.context_max, inputs.shape[1]), inputs.dtype),
            np.zeros((config.context_max, targets.shape[1]), targets.dtype),
            np.zeros(config.context_max, inputs.dtype),
        )
        cx[:count], cy[:count], mask[:count] = inputs[context], targets[context], 1
        for name, value in zip(keys, (cx, cy, mask, inputs[query], targets[query]), strict=True):
            result[name].append(value)
    return {name: jnp.asarray(np.stack(value)) for name, value in result.items()}


def train(config, emit=print):
    config.validate()
    if config.training_option == 2:
        from .hierarchical import train_hierarchical

        return train_hierarchical(config, emit)
    dataset = Dataset(config.dataset)
    source_model = None
    if config.resume_path:
        saved = json.loads((Path(config.resume_path) / "manifest.json").read_text())
        saved_config = saved["training_config"]
        if config.objective != saved_config.get("objective", "context_predictive"):
            raise ValueError("training resume requires the same objective")
        if asdict(config.model) != asdict(decode(ModelConfig, saved_config["model"])):
            # YAML lists and dataclass tuples have equivalent serialized representations.
            if json.dumps(asdict(config.model), sort_keys=True) != json.dumps(
                asdict(decode(ModelConfig, saved_config["model"])), sort_keys=True
            ):
                raise ValueError("training resume requires the same model format/configuration")
        if config.seed is not None and config.seed != saved_config["seed"]:
            raise ValueError("training resume restores RNG state; seed must match or be null")
        config = replace(config, seed=saved_config["seed"])
        precision(config.model.dtype)
        model = load_model(config.resume_path)
    elif config.init_model_path:
        config = replace(config, seed=resolve_seed(config.seed))
        precision(config.model.dtype)
        model = load_model(config.init_model_path)
        if model.config != config.model:
            raise ValueError("initial model format/configuration differs from training config")
        if model.config.domain != dataset.domain.name or not np.isclose(
            model.config.timestep, dataset.timestep
        ):
            raise ValueError("initial model domain or timestep differs from the dataset")
        if model.config.domain == "st" and model.config.baseline_profile != dataset.manifest[
            "config"
        ]["profile"]:
            raise ValueError("initial model vehicle profile differs from the dataset")
        source_model = {"path": config.init_model_path, "fingerprint": model.fingerprint}
    else:
        config = replace(config, seed=resolve_seed(config.seed))
        model = build_model(
            config.model,
            dataset,
            config.seed,
            config.data_view,
            config.max_transitions_per_instance,
            config.balance_learning_stages,
        )
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    save_model(model, root / "initial")
    write_json(
        root / "data_split.json",
        {
            "fingerprint": dataset.fingerprint,
            "instances": [
                {"instance_id": x["instance_id"], "split": x["split"]} for x in dataset.instances
            ],
        },
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    state = TrainState.create(apply_fn=None, params=model.parameters, tx=tx)
    rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(1)[0])
    best_loss = float("inf")
    if config.resume_path:
        state, best_loss, best_parameters, best_generation = restore_training(
            config.resume_path, state, rng, dataset.fingerprint
        )
        if config.steps <= int(state.step):
            raise ValueError("steps must exceed the restored training step")
        save_model(
            model.replace(parameters=best_parameters, generation=best_generation), root / "best"
        )
    training = prepare_instances(
        model,
        dataset,
        "train",
        config.data_view,
        config.max_transitions_per_instance,
        config.balance_learning_stages,
        config.seed,
    )
    validation = prepare_instances(
        model,
        dataset,
        "validation",
        config.data_view,
        config.max_transitions_per_instance,
        config.balance_learning_stages,
        config.seed,
    )
    if not validation:
        raise ValueError("dataset has no held-out validation instances")
    validation_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 714]))
    validation_batches = [
        draw_batch(validation, validation_rng, config) for _ in range(config.validation_batches)
    ]

    @jax.jit
    def update(current, batch):
        loss, gradient = jax.value_and_grad(predictive_loss)(
            current.params, batch, config.model, config.objective
        )
        return current.apply_gradients(grads=gradient), loss

    validate = jax.jit(lambda p, b: predictive_loss(p, b, config.model, config.objective))
    history = []
    last_validation = None
    if source_model is not None:
        last_validation = float(
            np.mean([float(validate(state.params, b)) for b in validation_batches])
        )
        if not np.isfinite(last_validation):
            raise FloatingPointError("nonfinite initial held-out likelihood")
        best_loss = last_validation
        save_model(model, root / "best")
        history.append(
            {
                "step": 0,
                "train_nll_per_component": None,
                "validation_nll_per_component": last_validation,
                "mean_likelihood_std_normalized": None,
            }
        )
        write_json(root / "history.json", history)
        emit(f"[train] initialized from {config.init_model_path} | held-out NLL={last_validation:.5g}")
    for step in range(int(state.step) + 1, config.steps + 1):
        state, loss = update(state, draw_batch(training, rng, config))
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        if step == 1 or step % config.validation_every == 0 or step == config.steps:
            last_validation = float(
                np.mean([float(validate(state.params, b)) for b in validation_batches])
            )
            if not np.isfinite(last_validation):
                raise FloatingPointError("nonfinite held-out predictive likelihood")
            population = model.replace(parameters=state.params, generation=step)
            _, variance = prior_and_variance(state.params, config.model)
            metrics = {
                "step": step,
                "train_nll_per_component": loss,
                "validation_nll_per_component": last_validation,
                "mean_likelihood_std_normalized": float(jnp.sqrt(variance).mean()),
            }
            history.append(metrics)
            write_json(root / "history.json", history)
            if config.progress_plot:
                from dyna_dist.evaluation.plots import plot_training_history

                plot_training_history(history, root / "loss_curve.png")
            if last_validation < best_loss:
                best_loss = last_validation
                save_model(population, root / "best", overwrite=True)
            emit(f"[train] step={step} | train NLL={loss:.5g} | held-out NLL={last_validation:.5g}")
        if step % config.checkpoint_every == 0 or step == config.steps:
            population = model.replace(parameters=state.params, generation=step)
            save_training(
                population,
                state,
                rng,
                root / "checkpoints" / f"step-{step:06d}",
                config,
                dataset.fingerprint,
                best_loss,
                load_model(root / "best"),
            )
    refit_metadata = None
    if config.prior_refit:
        from .population import refit_population

        refitted, refit_metadata = refit_population(
            load_model(root / "best"), training, config.refit_transitions,
            config.seed, config.refit_iterations, config.em_chunk_size,
        )
        save_model(refitted, root / "best_refit", extra={"population_refit": refit_metadata})
        write_json(root / "prior_refit.json", refit_metadata)
        emit(f"[prior-refit] train instances={len(training)} | saved best_refit")
    write_json(
        root / "summary.json",
        {
            "status": "complete",
            "steps": config.steps,
            "best_validation_nll_per_component": best_loss,
            "last_validation_nll_per_component": last_validation,
            "dataset_fingerprint": dataset.fingerprint,
            "initial_model": source_model,
            "seed": config.seed,
            "training_option": 1,
            "final_model": "best_refit" if config.prior_refit else "best",
            "prior_refit": refit_metadata,
            "versions": versions(),
        },
    )
    return root
