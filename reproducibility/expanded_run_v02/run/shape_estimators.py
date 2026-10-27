"""Packet-local shape-aware affine estimators (B1 and B2).

R2C01_DIRA_SHAPE_v08. The algebra is copied from
``src/phase_graph_cme/baselines.py`` with one change: the moments
``rho = m1`` and ``nu = m2`` are supplied by the shape model instead of being
derived from a uniform half-width. The batching and moment plumbing are
packet-local; no extra library module is imported. The library is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from shape_model import ShapePilotGroupModel

from phase_graph_cme.model import PilotGroupModel

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeModelBatch:
    los: ComplexArray
    covariances: ComplexArray
    beta: ComplexArray
    noise_covariance: ComplexArray
    m1: FloatArray
    m2: FloatArray
    observation: ComplexArray


def _target(model: ShapePilotGroupModel, target: int) -> int:
    index = int(target)
    if not 0 <= index < model.group_size:
        raise IndexError("target index is outside the pilot group")
    return index


def _observation(model: ShapePilotGroupModel, observation: object) -> ComplexArray:
    value = np.asarray(observation, dtype=np.complex128)
    if value.shape != (model.antenna_count,):
        raise ValueError("observation must have shape (N,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("observation must be finite")
    return value


def clmmse_shape(
    model: ShapePilotGroupModel, observation: object, target: int
) -> ComplexArray:
    """Best complex-affine estimate using m1 only."""

    k = _target(model, target)
    y = _observation(model, observation)
    rho = model.m1.astype(np.complex128)
    means = model.los * rho[np.newaxis, :]
    covariances = model.covariances.copy()
    for i in range(model.group_size):
        covariances[i] += (1.0 - abs(rho[i]) ** 2) * np.outer(
            model.los[:, i], model.los[:, i].conj()
        )
    mean_y = means @ model.beta
    psi = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, covariances, optimize=True
    )
    innovation = np.linalg.solve(psi, y - mean_y)
    return means[:, k] + model.beta[k].conj() * covariances[k] @ innovation


def widely_linear_mmse_shape(
    model: ShapePilotGroupModel, observation: object, target: int
) -> ComplexArray:
    """Best real-affine estimate using m1 and m2."""

    k = _target(model, target)
    y = _observation(model, observation)
    rho = model.m1.astype(np.complex128)
    nu = model.m2.astype(np.complex128)
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


def stack_shape_models(
    models: Sequence[ShapePilotGroupModel], observations: Sequence[object]
) -> ShapeModelBatch:
    if not models or len(models) != len(observations):
        raise ValueError("models and observations must have equal nonzero length")
    if any(model.group_size != 4 for model in models):
        raise ValueError("this packet requires P=4 local groups")
    n = models[0].antenna_count
    if any(model.antenna_count != n for model in models):
        raise ValueError("all models must have the same antenna count")
    observation = np.asarray(observations, dtype=np.complex128)
    if observation.shape != (len(models), n) or not np.all(np.isfinite(observation)):
        raise ValueError("observations must be finite with shape (B,N)")
    return ShapeModelBatch(
        los=np.stack([model.los for model in models]),
        covariances=np.stack([model.covariances for model in models]),
        beta=np.stack([model.beta for model in models]),
        noise_covariance=np.stack([model.noise_covariance for model in models]),
        m1=np.stack([model.m1 for model in models]),
        m2=np.stack([model.m2 for model in models]),
        observation=observation,
    )


def batched_b1_shape(batch: ShapeModelBatch) -> ComplexArray:
    """Vectorized P=4 widely-linear estimator using (m1, m2)."""

    if batch.los.ndim != 3 or batch.los.shape[2] != 4:
        raise ValueError("batched B1 requires shape (B,N,4)")
    rho = batch.m1.astype(np.complex128)
    nu = batch.m2.astype(np.complex128)
    means = batch.los * rho[:, None, :]
    los_covariance = np.einsum(
        "bnp,bmp->bpnm", batch.los, batch.los.conj(), optimize=True
    )
    los_pseudo = np.einsum("bnp,bmp->bpnm", batch.los, batch.los, optimize=True)
    covariance = batch.covariances + (1.0 - np.abs(rho) ** 2)[:, :, None, None] * los_covariance
    pseudo = (nu - rho**2)[:, :, None, None] * los_pseudo
    mean_y = np.einsum("bnp,bp->bn", means, batch.beta, optimize=True)
    covariance_y = batch.noise_covariance + np.einsum(
        "bp,bpnm->bnm", np.abs(batch.beta) ** 2, covariance, optimize=True
    )
    pseudo_y = np.einsum("bp,bpnm->bnm", batch.beta**2, pseudo, optimize=True)
    augmented = np.concatenate(
        (
            np.concatenate((covariance_y, pseudo_y), axis=2),
            np.concatenate((pseudo_y.conj(), covariance_y.conj()), axis=2),
        ),
        axis=1,
    )
    innovation = batch.observation - mean_y
    solved = np.linalg.solve(
        augmented,
        np.concatenate((innovation, innovation.conj()), axis=1)[..., None],
    )[..., 0]
    cross = np.concatenate(
        (
            batch.beta.conj()[:, :, None, None] * covariance,
            batch.beta[:, :, None, None] * pseudo,
        ),
        axis=3,
    )
    result = means + np.einsum("bpnm,bm->bpn", cross, solved, optimize=True).transpose(0, 2, 1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite batched B1 estimate")
    return result.astype(np.complex128)


def batched_b2_shape(batch: ShapeModelBatch) -> ComplexArray:
    """Vectorized P=4 complex-affine estimator using m1 only."""

    rho = batch.m1.astype(np.complex128)
    means = batch.los * rho[:, None, :]
    los_covariance = np.einsum(
        "bnp,bmp->bpnm", batch.los, batch.los.conj(), optimize=True
    )
    covariance = batch.covariances + (1.0 - np.abs(rho) ** 2)[:, :, None, None] * los_covariance
    mean_y = np.einsum("bnp,bp->bn", means, batch.beta, optimize=True)
    covariance_y = batch.noise_covariance + np.einsum(
        "bp,bpnm->bnm", np.abs(batch.beta) ** 2, covariance, optimize=True
    )
    solved = np.linalg.solve(covariance_y, (batch.observation - mean_y)[..., None])[..., 0]
    correction = np.einsum(
        "bp,bpnm,bm->bpn", batch.beta.conj(), covariance, solved, optimize=True
    )
    result = means + correction.transpose(0, 2, 1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite batched B2 estimate")
    return result.astype(np.complex128)


def to_library_model(model: ShapePilotGroupModel, delta: float) -> PilotGroupModel:
    """Build the library model for the uniform witness at the equivalent delta."""

    return PilotGroupModel(
        los=model.los,
        covariances=model.covariances,
        beta=model.beta,
        noise_covariance=model.noise_covariance,
        delta=np.full(model.group_size, float(delta), dtype=np.float64),
    )
