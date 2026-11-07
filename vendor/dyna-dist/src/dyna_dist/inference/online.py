"""Stable streaming Gaussian inference with a fixed feature map and base prior.

The accumulator stores *untempered* likelihood statistics in prior-whitened
coordinates. A changed likelihood power can therefore be applied to all evidence
consistently, without replaying it or changing the learned population artifact.
Observation identity/deduplication belongs to the caller.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


@dataclass
class GaussianEvidence:
    mean: np.ndarray
    factor: np.ndarray
    variance: np.ndarray
    gram: np.ndarray
    information: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, mean, factor, variance):
        mean, factor, variance = [np.array(x, dtype=np.float64, copy=True)
                                  for x in (mean, factor, variance)]
        d = len(mean)
        if (mean.shape != (d,) or factor.shape != (d, d) or variance.ndim != 1
                or not all(np.isfinite(x).all() for x in (mean, factor, variance))
                or np.any(variance <= 0)):
            raise ValueError("invalid fixed Gaussian prior/likelihood")
        return cls(mean, factor, variance, np.zeros((d, d)), np.zeros(d))

    def copy(self):
        return GaussianEvidence(*(x.copy() for x in
            (self.mean, self.factor, self.variance, self.gram, self.information)), self.count)

    def add(self, basis, residual):
        """Add new y-b observations once; all arithmetic after features is float64."""
        basis, residual = np.asarray(basis, np.float64), np.asarray(residual, np.float64)
        n, k = residual.shape
        if basis.shape != (n, k, len(self.mean)) or k != len(self.variance):
            raise ValueError("Gaussian evidence dimensions differ from the fixed model")
        matrix = (basis @ self.factor / np.sqrt(self.variance)[None, :, None]).reshape(-1, len(self.mean))
        target = ((residual - basis @ self.mean) / np.sqrt(self.variance)).reshape(-1)
        gram = self.gram + matrix.T @ matrix
        information = self.information + matrix.T @ target
        if not np.isfinite(gram).all() or not np.isfinite(information).all():
            raise FloatingPointError("nonfinite Gaussian likelihood evidence")
        self.gram, self.information = (gram + gram.T) / 2, information
        self.count += n

    def moments(self, power=1.0):
        """Exact conditional Gaussian for power=1; generalized Bayes otherwise."""
        if not np.isfinite(power) or not 0 < power <= 1:
            raise ValueError("likelihood power must be in (0, 1]")
        chol = np.linalg.cholesky(np.eye(len(self.mean)) + power * self.gram)
        white_mean = cho_solve((chol, True), power * self.information)
        mean = self.mean + self.factor @ white_mean
        factor = solve_triangular(chol, self.factor.T, lower=True).T
        if not np.isfinite(mean).all() or not np.isfinite(factor).all():
            raise FloatingPointError("nonfinite Gaussian posterior")
        return mean, factor


@dataclass
class RecalibratableGaussianEvidence:
    """Per-output, unscaled statistics for a diagonal working likelihood.

    Changing a variance or likelihood power reweights the original unique
    observations, rather than conditioning an already conditioned posterior.
    The original GaussianEvidence path is intentionally unchanged.
    """

    mean: np.ndarray
    factor: np.ndarray
    variance: np.ndarray
    gram_components: np.ndarray
    information_components: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, mean, factor, variance):
        prior = GaussianEvidence.create(mean, factor, variance)
        k, d = len(prior.variance), len(prior.mean)
        return cls(prior.mean, prior.factor, prior.variance,
                   np.zeros((k, d, d)), np.zeros((k, d)))

    def copy(self):
        return type(self)(*(x.copy() for x in (
            self.mean, self.factor, self.variance,
            self.gram_components, self.information_components)), self.count)

    def add(self, basis, residual):
        basis, residual = np.asarray(basis, np.float64), np.asarray(residual, np.float64)
        if (residual.ndim != 2 or residual.shape[1] != len(self.variance)
                or basis.shape != (*residual.shape, len(self.mean))):
            raise ValueError("recalibratable Gaussian evidence dimensions differ")
        matrix = basis @ self.factor
        target = residual - basis @ self.mean
        gram = self.gram_components + np.einsum("nkd,nke->kde", matrix, matrix)
        information = self.information_components + np.einsum("nkd,nk->kd", matrix, target)
        if not np.isfinite(gram).all() or not np.isfinite(information).all():
            raise FloatingPointError("nonfinite recalibratable likelihood evidence")
        self.gram_components = (gram + gram.swapaxes(-1, -2)) / 2
        self.information_components = information
        self.count += len(residual)

    def weighted(self, variance=None):
        variance = self.variance if variance is None else np.asarray(variance, np.float64)
        if (variance.shape != self.variance.shape or not np.isfinite(variance).all()
                or np.any(variance <= 0)):
            raise ValueError("effective variances must be finite, positive and aligned")
        return GaussianEvidence(
            self.mean, self.factor, variance,
            np.sum(self.gram_components / variance[:, None, None], axis=0),
            np.sum(self.information_components / variance[:, None], axis=0), self.count,
        )

    def moments(self, power=1.0, variance=None):
        return self.weighted(variance).moments(power)
