"""Deterministic synthetic cell-free Rician scenarios for paired experiments.

The generator intentionally exposes every normalization used by the research
experiments.  It is not presented as a calibrated deployment model; source-
specific parameter reproduction is implemented and reported separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .model import PilotGroupModel

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ScenarioConfig:
    access_points: int
    antennas_per_ap: int
    users: int
    pilot_symbols: int
    coherence_symbols: int
    area_side: float = 1.0
    minimum_distance: float = 0.03
    pathloss_exponent: float = 3.7
    rician_kappa: float = 5.0
    residual_phase_half_width_deg: float = 45.0
    pilot_power: float = 1.0
    payload_power: float = 1.0
    pilot_noise_variance: float = 1.0
    payload_noise_variance: float = 1.0
    spatial_correlation: float = 0.0
    pilot_assignment: str = "cyclic"
    phase_error_model: str = "matched_uniform"
    phase_mismatch_scale: float = 1.0
    correlated_phase_weight: float = 0.0

    def __post_init__(self) -> None:
        integer_values = (
            self.access_points,
            self.antennas_per_ap,
            self.users,
            self.pilot_symbols,
            self.coherence_symbols,
        )
        if any(
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) < 1
            for value in integer_values
        ):
            raise ValueError("system sizes must be positive integers")
        if self.pilot_symbols > self.users:
            raise ValueError("pilot_symbols cannot exceed users in a reuse experiment")
        if self.pilot_symbols >= self.coherence_symbols:
            raise ValueError("pilot_symbols must be smaller than coherence_symbols")
        positive = (
            self.area_side,
            self.minimum_distance,
            self.pathloss_exponent,
            self.pilot_power,
            self.payload_power,
            self.pilot_noise_variance,
            self.payload_noise_variance,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("geometry, powers, noises, and pathloss exponent must be finite and positive")
        if not math.isfinite(self.rician_kappa) or self.rician_kappa < 0.0:
            raise ValueError("rician_kappa must be finite and nonnegative")
        if (
            not math.isfinite(self.residual_phase_half_width_deg)
            or not 0.0 <= self.residual_phase_half_width_deg <= 180.0
        ):
            raise ValueError("residual phase half-width must lie in [0, 180] degrees")
        if not math.isfinite(self.spatial_correlation) or not 0.0 <= self.spatial_correlation < 1.0:
            raise ValueError("spatial_correlation must lie in [0,1)")
        if self.pilot_assignment not in {"cyclic", "balanced_random"}:
            raise ValueError("unsupported pilot_assignment")
        if self.phase_error_model not in {
            "matched_uniform",
            "wrapped_gaussian_mismatch",
            "correlated_uniform_mismatch",
        }:
            raise ValueError("unsupported phase_error_model")
        if not math.isfinite(self.phase_mismatch_scale) or self.phase_mismatch_scale <= 0.0:
            raise ValueError("phase_mismatch_scale must be finite and positive")
        if (
            not math.isfinite(self.correlated_phase_weight)
            or not 0.0 <= self.correlated_phase_weight <= 1.0
        ):
            raise ValueError("correlated_phase_weight must lie in [0,1]")

    @property
    def residual_phase_half_width(self) -> float:
        return math.radians(self.residual_phase_half_width_deg)

    @property
    def pilot_coefficient(self) -> float:
        return math.sqrt(self.pilot_symbols * self.pilot_power)


@dataclass(frozen=True)
class Topology:
    ap_positions: FloatArray
    user_positions: FloatArray
    distance: FloatArray
    large_scale_gain: FloatArray
    tracked_los: ComplexArray
    diffuse_covariances: ComplexArray
    pilot_indices: IntArray


@dataclass(frozen=True)
class ChannelFrame:
    residual_phases: FloatArray
    channels: ComplexArray
    pilot_observations: ComplexArray


def _wraparound_geometry(
    ap_positions: FloatArray, user_positions: FloatArray, area_side: float
) -> tuple[FloatArray, FloatArray]:
    displacement = user_positions[np.newaxis, :, :] - ap_positions[:, np.newaxis, :]
    side = float(area_side)
    displacement = (displacement + 0.5 * side) % side - 0.5 * side
    distance = np.sqrt(np.sum(displacement**2, axis=-1))
    bearing = np.arctan2(displacement[..., 1], displacement[..., 0])
    return distance.astype(np.float64), bearing.astype(np.float64)


def _pilot_indices(config: ScenarioConfig, rng: np.random.Generator) -> IntArray:
    base = np.arange(config.users, dtype=np.int64) % config.pilot_symbols
    if config.pilot_assignment == "balanced_random":
        base = base[rng.permutation(config.users)]
    return base


def generate_topology(config: ScenarioConfig, *, seed: int) -> Topology:
    """Generate one paired topology with median-normalized large-scale gains."""

    rng = np.random.default_rng(int(seed))
    ap = rng.uniform(0.0, config.area_side, size=(config.access_points, 2))
    users = rng.uniform(0.0, config.area_side, size=(config.users, 2))
    distance, bearing = _wraparound_geometry(ap, users, config.area_side)
    distance = np.maximum(distance, config.minimum_distance)
    gain = distance ** (-config.pathloss_exponent)
    gain /= float(np.median(gain))

    antenna = np.arange(config.antennas_per_ap, dtype=np.float64)
    steering_lkn = np.exp(
        1j
        * math.pi
        * np.sin(bearing)[..., np.newaxis]
        * antenna[np.newaxis, np.newaxis, :]
    )
    tracked_phase = rng.uniform(-math.pi, math.pi, size=bearing.shape)
    los_power = gain * config.rician_kappa / (config.rician_kappa + 1.0)
    tracked_los_lkn = (
        np.sqrt(los_power)[..., np.newaxis]
        * steering_lkn
        * np.exp(1j * tracked_phase)[..., np.newaxis]
    )

    index_difference = antenna[:, np.newaxis] - antenna[np.newaxis, :]
    diffuse_power = gain / (config.rician_kappa + 1.0)
    covariance = np.empty(
        (
            config.access_points,
            config.users,
            config.antennas_per_ap,
            config.antennas_per_ap,
        ),
        dtype=np.complex128,
    )
    for access_point in range(config.access_points):
        for user in range(config.users):
            correlation = config.spatial_correlation ** np.abs(index_difference)
            phase = np.exp(
                1j * math.pi * math.sin(float(bearing[access_point, user])) * index_difference
            )
            covariance[access_point, user] = (
                diffuse_power[access_point, user] * correlation * phase
            )

    return Topology(
        ap_positions=ap.astype(np.float64),
        user_positions=users.astype(np.float64),
        distance=distance,
        large_scale_gain=gain.astype(np.float64),
        tracked_los=np.transpose(tracked_los_lkn, (0, 2, 1)).astype(np.complex128),
        diffuse_covariances=covariance,
        pilot_indices=_pilot_indices(config, rng),
    )


def _draw_residual_phases(
    config: ScenarioConfig, rng: np.random.Generator
) -> FloatArray:
    shape = (config.access_points, config.users)
    delta = config.residual_phase_half_width
    if delta == 0.0:
        return np.zeros(shape, dtype=np.float64)
    if config.phase_error_model == "matched_uniform":
        return rng.uniform(-delta, delta, size=shape).astype(np.float64)
    if config.phase_error_model == "wrapped_gaussian_mismatch":
        standard_deviation = max(delta / math.sqrt(3.0), np.finfo(float).eps)
        raw = rng.normal(
            0.0, config.phase_mismatch_scale * standard_deviation, size=shape
        )
        return np.angle(np.exp(1j * raw)).astype(np.float64)

    weight = config.correlated_phase_weight
    common = rng.uniform(-delta, delta, size=(config.access_points, 1))
    independent = rng.uniform(-delta, delta, size=shape)
    mixed = weight * common + (1.0 - weight) * independent
    return mixed.astype(np.float64)


def generate_channel_frame(
    topology: Topology, config: ScenarioConfig, *, seed: int
) -> ChannelFrame:
    """Generate channels and the ordinary reused-pilot sufficient statistics."""

    rng = np.random.default_rng(int(seed))
    phases = _draw_residual_phases(config, rng)
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
    return ChannelFrame(
        residual_phases=phases,
        channels=channels,
        pilot_observations=observations,
    )


def pilot_group(
    topology: Topology,
    frame: ChannelFrame,
    config: ScenarioConfig,
    *,
    access_point: int,
    pilot: int,
) -> tuple[PilotGroupModel, ComplexArray, IntArray, ComplexArray]:
    """Return one local model, statistic, user indices, and oracle phasors."""

    l = int(access_point)
    t = int(pilot)
    if not 0 <= l < config.access_points or not 0 <= t < config.pilot_symbols:
        raise IndexError("access point or pilot index outside scenario")
    users = np.flatnonzero(topology.pilot_indices == t).astype(np.int64)
    beta = np.full(users.size, config.pilot_coefficient, dtype=np.complex128)
    model = PilotGroupModel(
        los=topology.tracked_los[l][:, users],
        covariances=topology.diffuse_covariances[l, users],
        beta=beta,
        noise_covariance=(
            config.pilot_noise_variance
            * np.eye(config.antennas_per_ap, dtype=np.complex128)
        ),
        delta=np.full(users.size, config.residual_phase_half_width),
    )
    observation = frame.pilot_observations[l, :, t]
    oracle_phasors = np.exp(1j * frame.residual_phases[l, users]).astype(
        np.complex128
    )
    return model, observation, users, oracle_phasors
