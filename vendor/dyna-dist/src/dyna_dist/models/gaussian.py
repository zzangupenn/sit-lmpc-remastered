"""Conjugate Gaussian updates and joint-query likelihood via low-rank identities."""

from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular


@struct.dataclass
class GaussianState:
    precision: jax.Array
    information: jax.Array

    def moments(self):
        factor = jnp.linalg.cholesky(self.precision)
        mean = cho_solve((factor, True), self.information)
        covariance = cho_solve((factor, True), jnp.eye(len(mean), dtype=mean.dtype))
        return mean, (covariance + covariance.T) / 2


def from_moments(mean, covariance):
    factor = jnp.linalg.cholesky(covariance)
    precision = cho_solve((factor, True), jnp.eye(len(mean), dtype=mean.dtype))
    return GaussianState(
        precision, jnp.matmul(precision, mean, precision=jax.lax.Precision.HIGHEST)
    )


@jax.jit
def condition(state, basis, residual, variance, mask):
    weights = mask[:, None] / variance[None]
    precision = state.precision + jnp.einsum(
        "nkd,nke,nk->de", basis, basis, weights, precision=jax.lax.Precision.HIGHEST
    )
    information = state.information + jnp.einsum(
        "nkd,nk,nk->d", basis, residual, weights, precision=jax.lax.Precision.HIGHEST
    )
    return GaussianState((precision + precision.T) / 2, information)


@jax.jit
def joint_nll(state, basis, residual, variance, mask):
    """Exact joint predictive likelihood for the declared linear-Gaussian head.

    Cost is O(N*d^2 + d^3), not a Cholesky on a (7*N)^2 covariance.
    Masked rows are integrated out, not treated as observed zeros.
    """
    mean, covariance = state.moments()
    return joint_nll_from_cholesky(
        mean, jnp.linalg.cholesky(covariance), basis, residual, variance, mask
    )


@jax.jit
def joint_nll_from_cholesky(mean, prior_factor, basis, residual, variance, mask):
    """Joint Gaussian NLL using the prior factor without precision inversion.

    The quadratic is evaluated as an augmented least-squares residual instead
    of subtracting two nearly equal large numbers at small likelihood noise.
    """
    scale = jnp.sqrt(mask[:, None] / variance[None])
    residual = residual - jnp.einsum("nkd,d->nk", basis, mean, precision=jax.lax.Precision.HIGHEST)
    whitened = (residual * scale).reshape(-1)
    matrix = (basis * scale[..., None]).reshape(-1, len(mean))
    matrix = jnp.matmul(
        matrix, prior_factor, precision=jax.lax.Precision.HIGHEST
    )
    small = jnp.eye(len(mean), dtype=matrix.dtype) + jnp.matmul(
        matrix.T, matrix, precision=jax.lax.Precision.HIGHEST
    )
    factor = jnp.linalg.cholesky(small)
    rhs = jnp.matmul(matrix.T, whitened, precision=jax.lax.Precision.HIGHEST)
    coefficients = cho_solve((factor, True), rhs)
    unexplained = whitened - jnp.matmul(matrix, coefficients, precision=jax.lax.Precision.HIGHEST)
    quadratic = jnp.vdot(unexplained, unexplained) + jnp.vdot(coefficients, coefficients)
    logdet = jnp.sum(mask) * jnp.sum(jnp.log(variance)) + 2 * jnp.log(jnp.diag(factor)).sum()
    observations = jnp.sum(mask) * basis.shape[1]
    return (quadratic + logdet + observations * jnp.log(2 * jnp.pi)) / 2


@jax.jit
def joint_nll_from_qr(mean, prior_factor, basis, residual, variance, mask):
    """Same exact likelihood, without forming normal equations (option 2)."""
    scale = jnp.sqrt(mask[:, None] / variance[None])
    centered = residual - jnp.einsum("nkd,d->nk", basis, mean)
    target = (centered * scale).reshape(-1)
    matrix = (basis * scale[..., None]).reshape(-1, len(mean)) @ prior_factor
    augmented = jnp.concatenate((matrix, jnp.eye(len(mean), dtype=matrix.dtype)))
    q, r = jnp.linalg.qr(augmented, mode="reduced")
    coefficients = solve_triangular(
        r, q.T @ jnp.concatenate((target, jnp.zeros_like(mean))), lower=False
    )
    unexplained = target - matrix @ coefficients
    quadratic = jnp.vdot(unexplained, unexplained) + jnp.vdot(coefficients, coefficients)
    logdet = jnp.sum(mask) * jnp.log(variance).sum() + 2 * jnp.log(jnp.abs(jnp.diag(r))).sum()
    observations = jnp.sum(mask) * basis.shape[1]
    return (quadratic + logdet + observations * jnp.log(2 * jnp.pi)) / 2
