"""Collect v2 by calling vehicle_data_gen's actual main, steering generator and PID.

Workers import the original repository's bundled Gym before any installed Gym fork.
Instrumentation observes real simulator endpoints; it never modifies the dynamics.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import yaml

from dyna_dist.io.common import fingerprint, new_directory, versions, write_json


SOURCE_FILES = ("data_gen_rand_uniform.py", "planner.py", "config/config.yaml")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quotas(total, groups):
    return np.full(groups, total // groups, dtype=int) + (np.arange(groups) < total % groups)


def valid_states(states, config):
    return (
        np.isfinite(states).all(axis=1)
        & (states[:, 3] >= config["min_speed"])
        & (states[:, 3] <= config["speed_limit"])
        & (np.abs(states[:, 2]) <= config["max_abs_steering"])
        & (np.abs(states[:, 5]) <= config["max_abs_yaw_rate"])
        & (np.abs(states[:, 6]) <= config["max_abs_slip"])
    )


def contiguous_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return (
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


class EndpointRecorder:
    """Observe only source main() steps, not warm-up or reset's hidden steps."""

    def __init__(self, source_file, profile):
        self.source_file, self.profile = str(source_file), profile
        self.segments, self.current = [], None
        self.parameters = []

    def flush(self):
        if self.current is not None and self.current["controls"]:
            self.segments.append(
                {
                    key: np.asarray(value) if isinstance(value, list) else value
                    for key, value in self.current.items()
                }
            )
        self.current = None

    def attach(self, env):
        car = env.unwrapped.sim.agents[0]
        self.parameters.append(
            dict(zip(car.sequential_params, car.params_arr.tolist(), strict=True))
        )
        step, reset = env.step, env.reset

        def observed_step(actions, *args, **kwargs):
            caller = sys._getframe(1)
            record = (
                caller.f_code.co_filename == self.source_file and caller.f_code.co_name == "main"
            )
            before = car.state.copy() if record else None
            result = step(actions, *args, **kwargs)
            if record:
                after = car.state.copy()
                if self.current is not None and not np.array_equal(
                    self.current["states"][-1], before
                ):
                    self.flush()
                if self.current is None:
                    self.current = dict(
                        states=[before],
                        controls=[],
                        target_speed=[],
                        target_steering=[],
                        profile=self.profile[0],
                        peaks=self.profile[1],
                    )
                self.current["states"].append(after)
                self.current["controls"].append(np.asarray(actions)[0].copy())
                self.current["target_speed"].append(float(caller.f_locals["vel"]))
                self.current["target_steering"].append(float(caller.f_locals["steer"]))
            return result

        def observed_reset(*args, **kwargs):
            self.flush()
            return reset(*args, **kwargs)

        env.step, env.reset = observed_step, observed_reset
        return env


def _collect_member(task):
    config, member, ordinal = task
    source, root = Path(config["source"]), Path(config["output"])
    member = deepcopy(member)
    member.pop("provenance", None)
    member["files"] = []
    member_root = root / "raw" / member["instance_id"]
    if (member_root / "member.json").is_file():
        cached = json.loads((member_root / "member.json").read_text())
        audit = json.loads((member_root / "audit.json").read_text())
        if any(
            sha256(root / filename) != checksum
            for filename, checksum in cached["checksums"].items()
        ):
            raise ValueError("completed member checksum mismatch on recovery")
        return cached["member"], cached["checksums"], audit
    raw = member_root
    retry = 0
    if raw.exists():
        retry = 1
        while (member_root / f"retry-{retry:03d}").exists():
            retry += 1
        raw = member_root / f"retry-{retry:03d}"
    raw.mkdir(parents=True)
    original_config = yaml.safe_load((source / "config/config.yaml").read_text())
    source_config = deepcopy(original_config)
    source_config["experiment"].update(
        name="dyna-dataset-v2",
        save_dir=str(raw) + "/",
        render=False,
        plot=False,
        acc_vs_control=True,
    )
    source_config["simulation"].update(
        steering_length=config["segments_per_round"] * config["segment_steps"],
        reset_step=config["segment_steps"],
        integration_dt=config["timestep"],
        vel_sample_up=config["speed_target_span"],
        density_curb=0,
        noise=[0, 0, 0],
    )
    source_config["environment"].update(
        timestep=config["timestep"],
        integrator_timestep=config["integrator_timestep"],
        model="st",
        integrator="rk4",
        enable_rendering=False,
        enable_scan=False,
        compute_frenet=False,
        steer_delay_buffer_size=0,
    )
    source_config["vehicle_params"] = {
        **original_config.get("vehicle_params", {}),
        **member["parameters"],
    }
    # The original loop sets MB tire coefficients, but ST actually reads mu.
    source_config["frictions"] = [member["parameters"]["mu"]]
    config_path = raw / "source_config.yaml"
    config_path.write_text(yaml.safe_dump(source_config))
    os.environ["VEHICLE_DATA_CONFIG"] = str(config_path)
    sys.path.insert(0, str(source))
    source_file = source / "data_gen_rand_uniform.py"
    old_argv = sys.argv
    sys.argv = [str(source_file), str(config["speeds"][0])]
    import gymnasium as gym

    with (raw / "source.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        loaded = runpy.run_path(str(source_file), run_name="original_vehicle_generator")
        namespace = loaded["main"].__globals__
        if not Path(loaded["f1tenth_gym"].__file__).resolve().is_relative_to(source):
            raise ValueError("original bundled Gym was not imported")
        original_steers, original_make = namespace["get_steers"], gym.make
        profile, counter = [0, 3], [0]
        profiles = [(i, peak) for i in range(len(config["amplitudes"])) for peak in config["peaks"]]

        def steering_targets(sample_length, parameters, segment_length=1, peak_num=10):
            amplitude_index, peaks = profiles[counter[0] % len(profiles)]
            counter[0] += 1
            profile[0] = amplitude_index
            profile[1] = peaks
            scaled = dict(parameters)
            scaled["s_max"] *= config["amplitudes"][amplitude_index]
            scaled["s_min"] *= config["amplitudes"][amplitude_index]
            return original_steers(sample_length, scaled, segment_length, peaks)

        namespace["get_steers"] = steering_targets
        checksums, audits, all_features = {}, [], []
        runtime_parameters = None
        for speed_index, (speed, count) in enumerate(
            zip(
                config["speeds"],
                quotas(config["transitions_per_instance"], len(config["speeds"])),
                strict=True,
            )
        ):
            remaining = quotas(int(count), len(config["amplitudes"]))
            audit = dict(
                speed=speed,
                accepted=0,
                candidate_transitions=0,
                rejected=0,
                unused_valid=0,
                rounds=0,
                seeds=[],
            )
            shard_index = 0
            for round_index in range(config["max_rounds"]):
                seed = int(
                    np.random.SeedSequence(
                        [config["seed"], ordinal, speed_index, round_index]
                    ).generate_state(1)[0]
                )
                audit["seeds"].append(seed)
                np.random.seed(seed)
                counter[0] = 0
                recorder = EndpointRecorder(source_file, profile)
                case = raw / f"speed-{speed:g}-round-{round_index:02d}"
                case.mkdir()
                namespace.update(start_vel=speed, SAVE_DIR=str(case) + "/")
                namespace["ENVIRONMENT_CONFIG"]["seed"] = seed
                gym.make = lambda *a, **kw: recorder.attach(original_make(*a, **kw))
                try:
                    namespace["main"]()
                    recorder.flush()
                finally:
                    gym.make = original_make
                runtime_parameters = recorder.parameters[0]
                for name in ("mu", "C_Sf", "C_Sr", "m", "I"):
                    if not np.isclose(
                        runtime_parameters[name], member["parameters"][name], rtol=1e-12
                    ):
                        raise ValueError(f"incorrect actual simulator parameter: {name}")
                for segment in recorder.segments:
                    states, controls = segment["states"], segment["controls"]
                    valid = valid_states(states, config)
                    mask = valid[:-1] & valid[1:] & np.isfinite(controls).all(axis=1)
                    audit["candidate_transitions"] += len(mask)
                    audit["rejected"] += int((~mask).sum())
                    group = segment["profile"]
                    for start, stop in contiguous_runs(mask):
                        length = min(stop - start, int(remaining[group]))
                        audit["unused_valid"] += stop - start - length
                        if not length:
                            continue
                        end = start + length
                        namespace_path = f"retry-{retry:03d}/" if retry else ""
                        filename = (
                            f"trajectories/{member['instance_id']}/{namespace_path}"
                            f"speed-{speed:g}-signal-{shard_index:04d}.npz"
                        )
                        destination = root / filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        selected = states[start : end + 1]
                        np.savez_compressed(
                            destination,
                            states=selected,
                            controls=controls[start:end],
                            commanded_controls=controls[start:end],
                            actuator_noise=np.zeros((length, 2)),
                            measured_states=selected,
                            valid=np.ones(length, dtype=bool),
                            times=np.arange(length + 1) * config["timestep"],
                            termination=np.asarray("step_limit"),
                            target_speed=segment["target_speed"][start:end],
                            target_steering=segment["target_steering"][start:end],
                            amplitude=np.asarray(config["amplitudes"][group]),
                            seed=np.asarray(seed),
                            peaks=np.asarray(segment["peaks"]),
                        )
                        checksums[filename] = sha256(destination)
                        member["files"].append(filename)
                        # Small deterministic plotting/audit sample, not the training data.
                        all_features.append(
                            np.column_stack((selected[:-1, [2, 3, 5, 6]], controls[start:end]))[::5]
                        )
                        remaining[group] -= length
                        audit["accepted"] += length
                        shard_index += 1
                audit["rounds"] += 1
                if not remaining.any():
                    break
            if remaining.any():
                raise RuntimeError(
                    f"{member['instance_id']} speed {speed}: quota shortfall {remaining}"
                )
            audits.append(audit)
        from dyna_dist.adapters.gym_st import GymST

        simulator = GymST(member["parameters"], config["timestep"], config["integrator_timestep"])
        replay_errors = []
        for filename in member["files"][:: max(1, len(member["files"]) // 64)]:
            with np.load(root / filename) as shard:
                predicted = simulator.step(shard["states"][0], shard["controls"][0])
                replay_errors.append(float(np.abs(predicted - shard["states"][1]).max()))
        if max(replay_errors) > 2e-5:
            raise ValueError(f"independent ST replay mismatch: {max(replay_errors)}")
        audit = dict(
            instance_id=member["instance_id"],
            per_speed=audits,
            actual_parameters=runtime_parameters,
            replay_count=len(replay_errors),
            replay_max_abs_error=max(replay_errors),
            source_fingerprint=simulator.source_fingerprint,
            transitions=sum(a["accepted"] for a in audits),
        )
        if audit["transitions"] != config["transitions_per_instance"]:
            raise ValueError("incorrect transition count")
        audit["raw_directory"] = str(raw.relative_to(root))
        np.savez_compressed(member_root / "audit_sample.npz", features=np.concatenate(all_features))
        write_json(member_root / "audit.json", audit)
        write_json(member_root / "member.json", dict(member=member, checksums=checksums))
    sys.argv = old_argv
    return member, checksums, audit


def collect(config, *, limit_instances=0, resume=False, emit=print):
    config = deepcopy(config)
    for key in ("source", "physics_dataset", "output"):
        config[key] = str(Path(config[key]).resolve())
    if (
        config["segment_steps"] < 16
        or not config["amplitudes"]
        or any(p < 2 for p in config["peaks"])
        or max(config["speeds"]) > config["speed_limit"]
        or config["transitions_per_instance"] < len(config["speeds"]) * len(config["amplitudes"])
    ):
        raise ValueError("invalid collection settings")
    source = Path(config["source"])
    hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    old = json.loads((Path(config["physics_dataset"]) / "manifest.json").read_text())
    if old.get("status") != "complete":
        raise ValueError("physics source dataset is incomplete")
    metadata_keys = (
        "profile",
        "friction",
        "front_cornering_scale",
        "rear_cornering_scale",
        "mass_ratio",
    )
    for key in metadata_keys:
        if key in config and config[key] != old["config"][key]:
            raise ValueError(f"{key} must match the reused physics population")
        config[key] = old["config"][key]
    members = old["instances"][: limit_instances or None]
    root = Path(config["output"])
    if resume:
        manifest = json.loads((root / "manifest.json").read_text())

        def without_workers(c):
            result = {k: v for k, v in c.items() if k != "workers"}
            for key in metadata_keys:
                result.setdefault(key, old["config"][key])
            return result

        if (
            without_workers(manifest["config"]) != without_workers(config)
            or manifest["generator_hashes"] != hashes
            or manifest["physics_source_fingerprint"] != old["fingerprint"]
        ):
            raise ValueError("resume configuration/source differs from collection")
        if not {m["instance_id"] for m in manifest["instances"]}.issubset(
            {m["instance_id"] for m in members}
        ):
            raise ValueError("resume instance selection excludes completed members")
        manifest["config"] = config
        manifest["status"] = "collecting"
        (root / "collection_config.yaml").write_text(yaml.safe_dump(config))
    else:
        new_directory(root)
        manifest = {
            key: old[key]
            for key in (
                "state_fields",
                "control_fields",
                "state_units",
                "control_units",
                "source_fingerprint",
            )
        }
        manifest.update(
            schema_version=1,
            collection_kind="original_vehicle_v2",
            name="dyna dataset v2",
            status="collecting",
            config=config,
            versions=versions(),
            generator_hashes=hashes,
            physics_source_fingerprint=old["fingerprint"],
            instances=[],
            shard_checksums={},
        )
        write_json(root / "manifest.json", manifest)
        (root / "collection_config.yaml").write_text(yaml.safe_dump(config))
    complete = {member["instance_id"] for member in manifest["instances"]}
    tasks = [
        (config, member, ordinal)
        for ordinal, member in enumerate(members)
        if member["instance_id"] not in complete
    ]
    try:
        with ProcessPoolExecutor(
            max_workers=config["workers"],
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as pool:
            futures = [pool.submit(_collect_member, task) for task in tasks]
            for future in as_completed(futures):
                member, checksums, audit = future.result()
                if audit["source_fingerprint"] != manifest["source_fingerprint"]:
                    raise ValueError("original and installed ST implementation differ")
                manifest["instances"].append(member)
                manifest["shard_checksums"].update(checksums)
                write_json(root / "manifest.json", manifest)
                emit(
                    f"[v2] {len(manifest['instances'])}/{len(members)} | {member['instance_id']} | "
                    f"{audit['transitions']:,} transitions | replay error={audit['replay_max_abs_error']:.2g}"
                )
                sys.stdout.flush()
        manifest["instances"].sort(
            key=lambda m: next(
                i for i, v in enumerate(members) if v["instance_id"] == m["instance_id"]
            )
        )
        if any(sha256(source / name) != value for name, value in hashes.items()):
            raise ValueError("source generator changed during collection")
        manifest["source_files_unchanged"] = True
        manifest["transitions"] = len(members) * config["transitions_per_instance"]
        manifest["status"] = "complete"
        manifest["fingerprint"] = fingerprint(
            {k: v for k, v in manifest.items() if k != "fingerprint"}
        )
        write_json(root / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        write_json(root / "manifest.json", manifest)
        raise
    return root
