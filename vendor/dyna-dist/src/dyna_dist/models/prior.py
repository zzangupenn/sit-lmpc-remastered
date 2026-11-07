"""Extension boundary: a flow prior will require nonconjugate posterior inference."""

from typing import Protocol
import jax
from flax import struct
import jax.numpy as jnp


class Prior(Protocol):
    family: str

    def sample(self, key, count): ...
    def log_prob(self, coefficients): ...


class InferenceBackend(Protocol):
    def condition(self, transitions): ...
    def sample_functions(self, key, count): ...
    def predict(self, state, physical_control): ...


@struct.dataclass
class GaussianPrior:
    mean: jax.Array
    covariance: jax.Array
    family: str = struct.field(pytree_node=False, default="gaussian")

    def sample(self, key, count):
        return self.mean + jnp.matmul(
            jax.random.normal(key, (count, len(self.mean)), dtype=self.mean.dtype),
            jnp.linalg.cholesky(self.covariance).T,
            precision=jax.lax.Precision.HIGHEST,
        )

    def log_prob(self, coefficients):
        factor = jnp.linalg.cholesky(self.covariance)
        whitened = jax.scipy.linalg.solve_triangular(
            factor, (coefficients - self.mean).T, lower=True
        ).T
        return -0.5 * (jnp.sum(whitened**2, axis=-1) + len(self.mean) * jnp.log(2 * jnp.pi)) - (
            jnp.log(jnp.diag(factor)).sum()
        )
