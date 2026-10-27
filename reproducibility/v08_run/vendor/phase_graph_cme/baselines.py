"""Protocol-compatible affine, phase-unaware, LS, and oracle estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .model import PilotGroupModel

ComplexArray = NDArray[np.complex128]


def bounded_uniform_phasor_mean(delta: object) -> NDArray[np.float64]:
    widths = np.asarray(delta, dtype=np.float64)
    return np.sinc(widths / np.pi)


def bounded_uniform_phasor_second_moment(delta: object) -> NDArray[np.float64]:
    """Return E[exp(2j epsilon)] for epsilon uniform on each bounded arc."""

    widths = np.asarray(delta, dtype=np.float64)
    return np.sinc(2.0 * widths / np.pi)


def _target_index(model: PilotGroupModel, target: int) -> int:
    index = int(target)
    if not 0 <= index < model.group_size:
        raise IndexError("target index is outside the pilot group")
    return index


def _observation(model: PilotGroupModel, observation: object) -> ComplexArray:
    value = np.asarray(observation, dtype=np.complex128)
    if value.shape != (model.antenna_count,):
        raise ValueError("observation must have shape (N,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("observation must be finite")
    return value


def clmmse(model: PilotGroupModel, observation: object, target: int) -> ComplexArray:
    """Best complex-affine estimate under the declared bounded-phase prior.

    The residual-phase channel is generally improper.  This covariance-only
    estimator is therefore not labeled as the best widely-linear estimator.
    """

    k = _target_index(model, target)
    y = _observation(model, observation)
    rho = bounded_uniform_phasor_mean(model.delta).astype(np.complex128)
    means = model.los * rho[np.newaxis, :]
    channel_covariances = model.covariances.copy()
    for i in range(model.group_size):
        channel_covariances[i] += (1.0 - abs(rho[i]) ** 2) * np.outer(
            model.los[:, i], model.los[:, i].conj()
        )
    mean_y = means @ model.beta
    psi = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, channel_covariances, optimize=True
    )
    innovation = np.linalg.solve(psi, y - mean_y)
    return means[:, k] + model.beta[k].conj() * channel_covariances[k] @ innovation


def widely_linear_mmse(
    model: PilotGroupModel, observation: object, target: int
) -> ComplexArray:
    """Best real/widely-linear affine estimate from y and conjugate(y).

    This is a stronger affine baseline whenever the bounded residual-phase
    channel has nonzero pseudo-covariance.
    """

    k = _target_index(model, target)
    y = _observation(model, observation)
    rho = bounded_uniform_phasor_mean(model.delta).astype(np.complex128)
    nu = bounded_uniform_phasor_second_moment(model.delta).astype(np.complex128)
    means = model.los * rho[np.newaxis, :]
    covariances = model.covariances.copy()
    pseudo_covariances = np.empty_like(covariances)
    for i in range(model.group_size):
        covariances[i] += (1.0 - abs(rho[i]) ** 2) * np.outer(
            model.los[:, i], model.los[:, i].conj()
        )
        pseudo_covariances[i] = (nu[i] - rho[i] ** 2) * np.outer(
            model.los[:, i], model.los[:, i]
        )
    mean_y = means @ model.beta
    covariance_y = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, covariances, optimize=True
    )
    pseudo_y = np.einsum(
        "p,pij->ij", model.beta**2, pseudo_covariances, optimize=True
    )
    augmented_covariance = np.block(
        [
            [covariance_y, pseudo_y],
            [pseudo_y.conj(), covariance_y.conj()],
        ]
    )
    cross_covariance = np.concatenate(
        (
            model.beta[k].conj() * covariances[k],
            model.beta[k] * pseudo_covariances[k],
        ),
        axis=1,
    )
    innovation = y - mean_y
    augmented_innovation = np.concatenate((innovation, innovation.conj()))
    correction = cross_covariance @ np.linalg.solve(
        augmented_covariance, augmented_innovation
    )
    if not np.all(np.isfinite(correction)):
        raise FloatingPointError("nonfinite widely-linear MMSE correction")
    return means[:, k] + correction


def phase_unaware_lmmse(
    model: PilotGroupModel, observation: object, target: int
) -> ComplexArray:
    """LMMSE baseline that ignores the tracked phase and uses zero channel mean."""

    k = _target_index(model, target)
    y = _observation(model, observation)
    channel_covariances = model.covariances.copy()
    for i in range(model.group_size):
        channel_covariances[i] += np.outer(model.los[:, i], model.los[:, i].conj())
    psi = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, channel_covariances, optimize=True
    )
    return model.beta[k].conj() * channel_covariances[k] @ np.linalg.solve(psi, y)


def collision_blind_ls(
    model: PilotGroupModel, observation: object, target: int
) -> ComplexArray:
    """Conventional LS estimate that attributes the whole pilot statistic to one UE."""

    k = _target_index(model, target)
    if abs(model.beta[k]) == 0.0:
        raise ZeroDivisionError("target pilot coefficient is zero")
    return _observation(model, observation) / model.beta[k]


def perfect_phase_oracle(
    model: PilotGroupModel, observation: object, target: int, u: object
) -> ComplexArray:
    """Conditional Gaussian MMSE estimate when residual phases are revealed."""

    k = _target_index(model, target)
    phases = np.asarray(u, dtype=np.complex128)
    if phases.shape != (model.group_size,):
        raise ValueError("u must have shape (P,)")
    y = _observation(model, observation)
    V = model.diffuse_pilot_covariance()
    residual = y - model.weighted_los() @ phases
    diffuse_mean = model.beta[k].conj() * model.covariances[k] @ np.linalg.solve(V, residual)
    return model.los[:, k] * phases[k] + diffuse_mean
