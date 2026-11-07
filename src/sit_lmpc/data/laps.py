"""Portable lap records: numerical arrays only, no pickle payloads."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Lap:
    cartesian: np.ndarray
    frenet: np.ndarray
    controls: np.ndarray
    times: np.ndarray

    def with_unwrapped_progress(self, track_length):
        """Learning view for numeric lap files without a full trajectory."""
        frenet = self.frenet.copy()
        if len(frenet):
            delta = (np.diff(frenet[:, 0]) + track_length / 2) % track_length - track_length / 2
            frenet[:, 0] = frenet[0, 0] % track_length + np.r_[0, np.cumsum(delta)]
        return Lap(self.cartesian, frenet, self.controls, self.times)

    @property
    def laptime(self):
        return float(self.times[-1])


class LapCollector:
    def __init__(self):
        self.cartesian, self.frenet, self.controls, self.times = [], [], [], []

    def append(self, cartesian, frenet, control, time):
        self.cartesian.append(np.asarray(cartesian).copy())
        self.frenet.append(np.asarray(frenet).copy())
        self.controls.append(np.asarray(control).copy())
        self.times.append(time)

    def finish(self):
        if not self.times:
            raise ValueError("cannot finish an empty lap")
        lap = Lap(
            np.asarray(self.cartesian),
            np.asarray(self.frenet),
            np.asarray(self.controls),
            np.asarray(self.times),
        )
        self.__init__()
        return lap


def save_laps(path, laps):
    payload = {"schema": np.asarray(1)}
    for i, lap in enumerate(laps):
        for name in ("cartesian", "frenet", "controls", "times"):
            payload[f"lap_{i:04d}/{name}"] = getattr(lap, name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_laps(path):
    with np.load(path, allow_pickle=False) as data:
        if "schema" not in data or int(data["schema"]) != 1:
            raise ValueError(
                "unsupported SIT-LMPC Remastered lap format; residual/BNN artifacts are not supported"
            )
        prefixes = sorted({key.split("/")[0] for key in data.files if key.startswith("lap_")})
        laps = [
            Lap(
                *(
                    data[f"{prefix}/{name}"].copy()
                    for name in ("cartesian", "frenet", "controls", "times")
                )
            )
            for prefix in prefixes
        ]
    for lap in laps:
        if (
            lap.cartesian.shape != lap.frenet.shape
            or lap.cartesian.ndim != 2
            or lap.cartesian.shape[1] != 7
        ):
            raise ValueError("invalid lap state shapes")
        if lap.controls.shape != (len(lap.times), 2) or len(lap.cartesian) != len(lap.times):
            raise ValueError("invalid lap action/time shapes")
        if not all(
            np.isfinite(a).all() for a in (lap.cartesian, lap.frenet, lap.controls, lap.times)
        ):
            raise ValueError("lap contains nonfinite data")
    return laps
