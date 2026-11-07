from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class Normalization:
    low: jax.Array
    high: jax.Array

    @classmethod
    def fit(cls, data):
        arr = np.asarray(data)
        if arr.ndim != 2 or len(arr) == 0 or not np.isfinite(arr).all():
            raise ValueError("normalization requires finite nonempty 2D data")
        return cls(jnp.asarray(arr.min(axis=0)), jnp.asarray(arr.max(axis=0)))

    def encode_inputs(self, inputs):
        span = self.high[:-1] - self.low[:-1]
        scale = jnp.where(span > 1e-8, span, 1.0)
        return (inputs - self.low[:-1]) / scale

    def encode_targets(self, targets):
        span = self.high[-1] - self.low[-1]
        return (targets - self.low[-1]) / jnp.where(span > 1e-8, span, 1.0)

    def decode_targets(self, targets):
        return targets * (self.high[-1] - self.low[-1]) + self.low[-1]
