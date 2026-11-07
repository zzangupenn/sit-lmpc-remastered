"""Reproducible state or transition partitions; no fitting or GPU work."""

import numpy as np


def state_partition_indices(ids, calibration_fraction, validation_fraction, seed=0):
    """Random individual states per attempt, independent of rollout windows.

    Attempt/step identities keep assignments stable on repeated fits and resume.
    All states have exactly one role; no gaps or minimum trajectory length apply.
    Temporal neighbours can have different roles, so these are not IID samples.
    """
    if not (0 <= calibration_fraction < 0.5 and 0 <= validation_fraction < 0.5):
        raise ValueError("state hold-out fractions must be in [0, 0.5)")
    groups = [[], [], []]
    for attempt in np.unique(ids[:, 0]):
        rows = np.flatnonzero(ids[:, 0] == attempt)
        rows = rows[np.argsort(ids[rows, 1], kind="stable")]
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(attempt) % 2**32, 0x4E465354])
        )
        rows = rng.permutation(rows)
        n = len(rows)
        nc = min(n - 1, int(np.ceil(n * calibration_fraction)))
        nt = min(n - nc - 1, int(np.ceil(n * validation_fraction)))
        selections = rows[nc + nt :], rows[:nc], rows[nc : nc + nt]
        for group, selected in zip(groups, selections, strict=True):
            group.extend(selected)
    return tuple(np.sort(np.asarray(group, dtype=int)) for group in groups)


def partition_indices(batch, config, seed=0):
    """Return fit/calibration/test/gap row indices without crossing attempts.

    Random-window mode draws start indices from the entire valid trajectory.
    Windows may overlap within one role, but calibration/test/fitting never
    share transition IDs. All remaining rows train; no guard sections are thrown
    away. Adjacent roles remain temporally dependent; this is not an independence
    or conformal guarantee. Legacy contiguous mode retains its original gaps.
    """
    mode = getattr(config, "calibration_split", "contiguous")
    if mode not in ("random_windows", "contiguous"):
        raise ValueError("calibration_split must be random_windows or contiguous")
    groups = [[], [], [], []]
    horizon = config.calibration_horizon
    for episode in np.unique(batch.ids[:, 0]):
        rows = np.flatnonzero(batch.ids[:, 0] == episode)
        rows = rows[np.argsort(batch.ids[rows, 1], kind="stable")]
        n = len(rows)
        if mode == "contiguous":
            nt = min(n - 1, int(np.ceil(n * config.validation_fraction)))
            nc = min(n - nt - 1, int(np.ceil(n * config.calibration_fraction)))
            gap = min(horizon, (n - nt - nc - 1) // 2)
            end = n - nt - nc - 2 * gap
            selections = (
                rows[:end],
                rows[end + gap : end + gap + nc],
                rows[n - nt :] if nt else [],
                np.concatenate((rows[end : end + gap], rows[end + gap + nc : n - nt])),
            )
            for group, selected in zip(groups, selections, strict=True):
                group.extend(selected)
            continue
        if n < 3 * horizon:
            groups[0].extend(rows)
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(episode) % 2**32, 0x53414645])
        )
        if horizon == 1:
            starts = np.arange(n)
        else:
            links = (np.diff(batch.ids[rows, 1]) == 1) & np.all(
                batch.states[rows[1:]] == batch.next_states[rows[:-1]], axis=1
            )
            starts = np.flatnonzero(
                np.convolve((~links).astype(int), np.ones(horizon - 1, dtype=int), mode="valid")
                == 0
            )
        owner = np.zeros(n, dtype=np.int8)
        remaining = n
        order = (1, 2) if rng.integers(2) else (2, 1)
        targets = {
            1: int(np.ceil(n * config.calibration_fraction)),
            2: int(np.ceil(n * config.validation_fraction)),
        }
        for role in order:
            count = 0
            for start in rng.permutation(starts):
                if count >= targets[role]:
                    break
                section = owner[start : start + horizon]
                added = int(np.sum(section == 0))
                if (
                    np.any((section != 0) & (section != role))
                    or not added
                    or remaining - added < 1
                ):
                    continue
                owner[start : start + horizon] = role
                remaining -= added
                count += added
        for role in range(3):
            groups[role].extend(rows[owner == role])
    # Preserve the original batch order so adjacent rows/windows stay aligned.
    return tuple(np.sort(np.asarray(group, dtype=int)) for group in groups)
