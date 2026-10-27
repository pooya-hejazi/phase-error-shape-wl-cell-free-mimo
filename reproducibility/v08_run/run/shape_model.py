"""Packet-local shape-aware channel frame, pilot model and frame assembly.

R2C01_DIRA_SHAPE_v08. The channel law is the contract's generator with one
change: the residual phase is drawn from a matched shape law instead of the
uniform law. The diffuse and pilot-noise draws are copied from
``src/phase_graph_cme/scenario.py`` so that a shared seed and a shared uniform
field give identical diffuse/noise draws across shapes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shape_core import ShapeSpec, draw_phases

from phase_graph_cme.scenario import ScenarioConfig, Topology

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeChannelFrame:
    uniform: FloatArray
    residual_phases: FloatArray
    channels: ComplexArray
    pilot_observations: ComplexArray


@dataclass(frozen=True)
class ShapePilotGroupModel:
    """One AP and one reused-pilot group with arbitrary matched moments."""

    los: ComplexArray
    covariances: ComplexArray
    beta: ComplexArray
    noise_covariance: ComplexArray
    m1: FloatArray
    m2: FloatArray

    def __post_init__(self) -> None:
        los = np.asarray(self.los, dtype=np.complex128)
        covariances = np.asarray(self.covariances, dtype=np.complex128)
        beta = np.asarray(self.beta, dtype=np.complex128)
        noise = np.asarray(self.noise_covariance, dtype=np.complex128)
        m1 = np.asarray(self.m1, dtype=np.float64)
        m2 = np.asarray(self.m2, dtype=np.float64)
        object.__setattr__(self, "los", los)
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "noise_covariance", noise)
        object.__setattr__(self, "m1", m1)
        object.__setattr__(self, "m2", m2)
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
        if self.beta.shape != (p,) or self.m1.shape != (p,) or self.m2.shape != (p,):
            raise ValueError("beta, m1 and m2 must have shape (P,)")
        if self.noise_covariance.shape != (n, n):
            raise ValueError("noise_covariance must have shape (N, N)")
        if not np.all(np.isfinite(self.los)) or not np.all(np.isfinite(self.beta)):
            raise ValueError("los and beta must be finite")
        if not np.all(np.isfinite(self.covariances)) or not np.all(
            np.isfinite(self.noise_covariance)
        ):
            raise ValueError("covariances and noise_covariance must be finite")
        if not np.all(np.isfinite(self.m1)) or not np.all(np.isfinite(self.m2)):
            raise ValueError("m1 and m2 must be finite")
        if np.any(self.m1 < 0.0) or np.any(self.m1 > 1.0):
            raise ValueError("each m1 must lie in [0, 1]")
        if np.any(self.m2 < -1.0) or np.any(self.m2 > 1.0):
            raise ValueError("each m2 must lie in [-1, 1]")
        lens = self.m2 - self.m1 * self.m1
        bound = 1.0 - self.m1 * self.m1
        if np.any(np.abs(lens) > bound + tolerance):
            raise ValueError("each (m1, m2) pair violates the lens")
        for index, matrix in enumerate(self.covariances):
            hermitian = 0.5 * (matrix + matrix.conj().T)
            if np.max(np.abs(matrix - matrix.conj().T)) > tolerance:
                raise ValueError(f"covariances[{index}] is not Hermitian")
            if np.min(np.linalg.eigvalsh(hermitian)) < -tolerance:
                raise ValueError(f"covariances[{index}] is not positive semidefinite")
        noise_h = 0.5 * (self.noise_covariance + self.noise_covariance.conj().T)
        if np.max(np.abs(self.noise_covariance - noise_h)) > tolerance:
            raise ValueError("noise_covariance is not Hermitian")
        if np.min(np.linalg.eigvalsh(noise_h)) <= tolerance:
            raise ValueError("noise_covariance must be positive definite")


def generate_shape_frame(
    topology: Topology,
    config: ScenarioConfig,
    spec: ShapeSpec,
    *,
    seed: int,
) -> ShapeChannelFrame:
    """Generate channels and reused-pilot statistics under a matched shape law."""

    rng = np.random.default_rng(int(seed))
    uniform = rng.uniform(
        0.0, 1.0, size=(config.access_points, config.users)
    ).astype(np.float64)
    phases = draw_phases(spec.shape, spec, uniform)
    channels = np.empty_like(topology.tracked_los)
    for access_point in range(config.access_points):
        for user in range(config.users):
            covariance = topology.diffuse_covariances[access_point, user]
            factor = np.linalg.cholesky(covariance)
            standard = (
                rng.standard_normal(config.antennas_per_ap)
                + 1j * rng.standard_normal(config.antennas_per_ap)
            ) / math.sqrt(2.0)
            diffuse = factor @ standard
            channels[access_point, :, user] = (
                topology.tracked_los[access_point, :, user]
                * np.exp(1j * phases[access_point, user])
                + diffuse
            )

    noise = math.sqrt(config.pilot_noise_variance / 2.0) * (
        rng.standard_normal(
            (config.access_points, config.antennas_per_ap, config.pilot_symbols)
        )
        + 1j
        * rng.standard_normal(
            (config.access_points, config.antennas_per_ap, config.pilot_symbols)
        )
    )
    observations = noise.astype(np.complex128)
    coefficient = config.pilot_coefficient
    for pilot in range(config.pilot_symbols):
        group = np.flatnonzero(topology.pilot_indices == pilot)
        observations[:, :, pilot] += coefficient * np.sum(
            channels[:, :, group], axis=2
        )
    return ShapeChannelFrame(
        uniform=uniform,
        residual_phases=phases,
        channels=channels,
        pilot_observations=observations,
    )


def shape_pilot_group(
    topology: Topology,
    frame: ShapeChannelFrame,
    config: ScenarioConfig,
    spec: ShapeSpec,
    *,
    access_point: int,
    pilot: int,
) -> tuple[ShapePilotGroupModel, ComplexArray, NDArray[np.int64], ComplexArray]:
    """Return one local shape model, statistic, user indices and oracle phasors."""

    l = int(access_point)
    t = int(pilot)
    if not 0 <= l < config.access_points or not 0 <= t < config.pilot_symbols:
        raise IndexError("access point or pilot index outside scenario")
    users = np.flatnonzero(topology.pilot_indices == t).astype(np.int64)
    beta = np.full(users.size, config.pilot_coefficient, dtype=np.complex128)
    model = ShapePilotGroupModel(
        los=topology.tracked_los[l][:, users],
        covariances=topology.diffuse_covariances[l, users],
        beta=beta,
        noise_covariance=(
            config.pilot_noise_variance
            * np.eye(config.antennas_per_ap, dtype=np.complex128)
        ),
        m1=np.full(users.size, spec.m1, dtype=np.float64),
        m2=np.full(users.size, spec.m2, dtype=np.float64),
    )
    observation = frame.pilot_observations[l, :, t]
    oracle_phasors = np.exp(1j * frame.residual_phases[l, users]).astype(
        np.complex128
    )
    return model, observation, users, oracle_phasors
