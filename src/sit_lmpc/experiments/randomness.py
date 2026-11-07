"""One experiment-owned host generator and JAX key stream."""

from dataclasses import replace
import secrets
import zlib
import numpy as np
from scipy.stats import truncnorm


def resolve_seed(config):
    """Resolve auto seeds before persisting config or initializing any model."""
    config.validate()
    seed = secrets.randbits(32) if config.simulation.seed_mode == "auto" else config.simulation.seed
    return replace(config, simulation=replace(config.simulation, seed=seed))


class ExperimentRandomness:
    def __init__(self, seed):
        import jax

        self.seed = seed
        self.host = np.random.default_rng(seed)
        self.key = jax.random.key(seed)

    def next_key(self):
        import jax

        self.key, key = jax.random.split(self.key)
        return key

    def fork(self, name):
        """Named streams share the master seed without perturbing baseline draws."""
        import jax

        tag = zlib.crc32(name.encode("utf-8"))
        child = object.__new__(ExperimentRandomness)
        child.seed = self.seed
        child.host = np.random.default_rng(np.random.SeedSequence([self.seed, tag]))
        child.key = jax.random.fold_in(jax.random.key(self.seed), tag)
        return child

    def state_noise(self, std, size=7):
        if std == 0:
            return np.zeros(size)
        return truncnorm.rvs(-3, 3, loc=0, scale=std, size=size, random_state=self.host)
