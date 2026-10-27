"""Validated physical and whitened phase-graph models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _as_complex(value: object) -> ComplexArray:
    return np.asarray(value, dtype=np.complex128)


def _as_float(value: object) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class PilotGroupModel:
    """One AP and one reused-pilot group.

    ``los[:, i]`` already includes the tracked LoS phase.  ``covariances[i]``
    is the diffuse covariance, while ``beta[i]`` is the despread pilot
    coefficient.  Residual phases have support ``[-delta[i], delta[i]]``.
    """

    los: ComplexArray
    covariances: ComplexArray
    beta: ComplexArray
    noise_covariance: ComplexArray
    delta: FloatArray

    def __post_init__(self) -> None:
        los = _as_complex(self.los)
        cov = _as_complex(self.covariances)
        beta = _as_complex(self.beta)
        noise = _as_complex(self.noise_covariance)
        delta = _as_float(self.delta)
        object.__setattr__(self, "los", los)
        object.__setattr__(self, "covariances", cov)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "noise_covariance", noise)
        object.__setattr__(self, "delta", delta)
        self.validate()

    @property
    def antenna_count(self) -> int:
        return int(self.los.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.los.shape[1])

    def validate(self, tolerance: float = 1e-10) -> None:
        if self.los.ndim != 2:
            raise ValueError("los must have shape (N, P)")
        n, p = self.los.shape
        if self.covariances.shape != (p, n, n):
            raise ValueError("covariances must have shape (P, N, N)")
        if self.beta.shape != (p,) or self.delta.shape != (p,):
            raise ValueError("beta and delta must have shape (P,)")
        if self.noise_covariance.shape != (n, n):
            raise ValueError("noise_covariance must have shape (N, N)")
        if not np.all(np.isfinite(self.los)) or not np.all(np.isfinite(self.beta)):
            raise ValueError("los and beta must be finite")
        if not np.all(np.isfinite(self.covariances)) or not np.all(
            np.isfinite(self.noise_covariance)
        ):
            raise ValueError("covariances and noise_covariance must be finite")
        if not np.all(np.isfinite(self.delta)):
            raise ValueError("delta must be finite")
        if np.any(self.delta < 0.0) or np.any(self.delta > np.pi):
            raise ValueError("each residual-phase half-width must lie in [0, pi]")
        for index, matrix in enumerate(self.covariances):
            hermitian = 0.5 * (matrix + matrix.conj().T)
            if np.max(np.abs(matrix - matrix.conj().T)) > tolerance:
                raise ValueError(f"covariances[{index}] is not Hermitian")
            if np.min(np.linalg.eigvalsh(hermitian)) < -tolerance:
                raise ValueError(f"covariances[{index}] is not positive semidefinite")
        noise_h = 0.5 * (self.noise_covariance + self.noise_covariance.conj().T)
        if np.max(np.abs(self.noise_covariance - self.noise_covariance.conj().T)) > tolerance:
            raise ValueError("noise_covariance is not Hermitian")
        if np.min(np.linalg.eigvalsh(noise_h)) <= tolerance:
            raise ValueError("noise_covariance must be positive definite")

    def diffuse_pilot_covariance(self) -> ComplexArray:
        weighted = np.einsum(
            "p,pij->ij", np.abs(self.beta) ** 2, self.covariances, optimize=True
        )
        value = self.noise_covariance + weighted
        return 0.5 * (value + value.conj().T)

    def weighted_los(self) -> ComplexArray:
        return self.los * self.beta[np.newaxis, :]


@dataclass(frozen=True)
class PhaseGraph:
    """Whitened sufficient statistics for the phase posterior."""

    x: ComplexArray
    A: ComplexArray
    c: ComplexArray
    G: ComplexArray
    delta: FloatArray
    cholesky: ComplexArray

    @property
    def group_size(self) -> int:
        return int(self.A.shape[1])

    @property
    def observation_dimension(self) -> int:
        return int(self.A.shape[0])


def build_phase_graph(model: PilotGroupModel, observation: object) -> PhaseGraph:
    """Whiten the pilot observation and construct ``c=A^H x`` and ``G=A^H A``."""

    y = _as_complex(observation)
    if y.shape != (model.antenna_count,):
        raise ValueError("observation must have shape (N,)")
    if not np.all(np.isfinite(y)):
        raise ValueError("observation must be finite")
    covariance = model.diffuse_pilot_covariance()
    cholesky = np.linalg.cholesky(covariance)
    x = np.linalg.solve(cholesky, y)
    A = np.linalg.solve(cholesky, model.weighted_los())
    c = A.conj().T @ x
    G = A.conj().T @ A
    G = 0.5 * (G + G.conj().T)
    return PhaseGraph(x=x, A=A, c=c, G=G, delta=model.delta.copy(), cholesky=cholesky)
