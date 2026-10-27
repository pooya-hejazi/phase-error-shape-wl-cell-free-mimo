"""Exact transfer from phase moments to channel moments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import PilotGroupModel

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ChannelMoments:
    mean: ComplexArray
    covariance: ComplexArray | None


def channel_transfer(
    model: PilotGroupModel, observation: object, target: int
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    k = int(target)
    if not 0 <= k < model.group_size:
        raise IndexError("target index is outside the pilot group")
    y = np.asarray(observation, dtype=np.complex128)
    if y.shape != (model.antenna_count,):
        raise ValueError("observation must have shape (N,)")
    if not np.all(np.isfinite(y)):
        raise ValueError("observation must be finite")
    V = model.diffuse_pilot_covariance()
    solve_y = np.linalg.solve(V, y)
    solve_D = np.linalg.solve(V, model.weighted_los())
    Rk = model.covariances[k]
    linear = model.beta[k].conj() * Rk @ solve_y
    selector = np.zeros((1, model.group_size), dtype=np.complex128)
    selector[0, k] = 1.0
    B = model.los[:, k : k + 1] @ selector - model.beta[k].conj() * Rk @ solve_D
    S = Rk - abs(model.beta[k]) ** 2 * Rk @ np.linalg.solve(V, Rk)
    S = 0.5 * (S + S.conj().T)
    return linear, B, S


def channel_moments(
    model: PilotGroupModel,
    observation: object,
    target: int,
    z: object,
    Z: object | None = None,
) -> ChannelMoments:
    phase_mean = np.asarray(z, dtype=np.complex128)
    if phase_mean.shape != (model.group_size,):
        raise ValueError("z must have shape (P,)")
    if not np.all(np.isfinite(phase_mean)):
        raise ValueError("z must be finite")
    linear, B, S = channel_transfer(model, observation, target)
    mean = linear + B @ phase_mean
    if Z is None:
        return ChannelMoments(mean=mean, covariance=None)
    phase_second = np.asarray(Z, dtype=np.complex128)
    if phase_second.shape != (model.group_size, model.group_size):
        raise ValueError("Z must have shape (P, P)")
    if not np.all(np.isfinite(phase_second)):
        raise ValueError("Z must be finite")
    tolerance = 1e-8
    if np.max(np.abs(phase_second - phase_second.conj().T), initial=0.0) > tolerance:
        raise ValueError("Z must be Hermitian")
    if np.max(np.abs(np.diag(phase_second) - 1.0), initial=0.0) > tolerance:
        raise ValueError("Z must have unit diagonal for unit phasors")
    phase_covariance = phase_second - np.outer(phase_mean, phase_mean.conj())
    phase_covariance_h = 0.5 * (phase_covariance + phase_covariance.conj().T)
    if np.min(np.linalg.eigvalsh(phase_covariance_h), initial=0.0) < -tolerance:
        raise ValueError("Z - z z^H must be positive semidefinite")
    covariance = S + B @ phase_covariance_h @ B.conj().T
    covariance = 0.5 * (covariance + covariance.conj().T)
    return ChannelMoments(mean=mean, covariance=covariance)
