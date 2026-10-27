"""Packet-local scenario, receiver and assembly helpers for R2C01_DIRA_SHAPE_v08."""

from __future__ import annotations

import math

import numpy as np

from phase_graph_cme.receivers import mr_combiner, rzf_combiner
from phase_graph_cme.scenario import ScenarioConfig


def scenario_for(case: dict, cell: dict, system: dict, spec) -> ScenarioConfig:
    """Build the library scenario used for topology generation only.

    The packet's frame generator does not call the library's residual-phase
    draw; the ``residual_phase_half_width_deg`` field is therefore metadata for
    the topology generator and is never used as the shape law.
    """

    if cell["shape"] in {"uniform", "binary", "zero_c"}:
        delta_deg = float(math.degrees(spec.parameter["delta_rad"]))
    elif cell["shape"] == "wrapped_gaussian":
        delta_deg = 180.0
    else:
        delta_deg = 0.0
    return ScenarioConfig(
        access_points=system["access_points"],
        antennas_per_ap=case["antennas_per_ap"],
        users=system["users"],
        pilot_symbols=system["pilot_symbols"],
        coherence_symbols=system["coherence_symbols"],
        area_side=system["area_side"],
        minimum_distance=system["minimum_distance"],
        pathloss_exponent=system["pathloss_exponent"],
        rician_kappa=cell["kappa"],
        residual_phase_half_width_deg=delta_deg,
        pilot_power=system["pilot_power"],
        payload_power=system["payload_power"],
        pilot_noise_variance=system["pilot_noise_variance"],
        payload_noise_variance=system["payload_noise_variance"],
        spatial_correlation=system["spatial_correlation"],
        pilot_assignment=system["pilot_assignment"],
        # Accepted library value; the packet draws residual phases itself.
        phase_error_model="matched_uniform",
    )


def combiner(estimates: np.ndarray, receiver: dict) -> np.ndarray:
    if receiver["regularization"] is None:
        return mr_combiner(estimates)
    return rzf_combiner(estimates, receiver["regularization"])


def assemble(
    local: np.ndarray,
    groups: list[np.ndarray],
    coords: list[tuple[int, int]],
    n_ant: int,
    L: int,
    K: int,
) -> np.ndarray:
    """Place local group estimates into the (L*n_ant, K) receiver matrix."""

    out = np.zeros((L, n_ant, K), dtype=np.complex128)
    cover = np.zeros((L, K), dtype=np.int64)
    for index, ((ap, _pilot), users) in enumerate(zip(coords, groups)):
        out[ap][:, users] = local[index]
        cover[ap, users] += 1
    if not np.all(cover == 1):
        raise RuntimeError("every user must be covered by exactly one local group per AP")
    return out.reshape(L * n_ant, K)
