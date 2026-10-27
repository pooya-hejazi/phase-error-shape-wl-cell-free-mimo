"""Primary-cell experiment on a 3GPP TR 38.901 UMa channel.

The base experiment uses a power-law pathloss and an exponential spatial correlation. This script
replaces both with standard ingredients: the UMa pathloss and LoS probability of TR 38.901, log-normal
shadowing, and a Laplacian power angular spectrum on a half-wavelength ULA. The estimator, the pilot
model, the phase laws and the metric are unchanged.

Two variants are run:
  (a) all links are LoS with kappa = 20, so the result is directly comparable with the base design;
  (b) the standard LoS probability is used, with kappa = 20 on LoS links and kappa = 0 otherwise.

    .venv\\Scripts\\python.exe -B reproducibility\\standard_channel\\run_standard.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "reproducibility/expanded_run_v02/run"))

from phase_graph_cme.scenario import ScenarioConfig, Topology  # noqa: E402
from shape_core import shape_spec  # noqa: E402
from shape_estimators import (  # noqa: E402
    batched_b1_shape,
    batched_b2_shape,
    stack_shape_models,
)
from shape_model import generate_shape_frame, shape_pilot_group  # noqa: E402

FC_GHZ = 3.5
H_AP = 25.0
H_UT = 1.5
SHADOW_LOS_DB = 4.0
SHADOW_NLOS_DB = 6.0
AS_LOS_DEG = 5.0
AS_NLOS_DEG = 10.0
KAPPA = 20.0
TOPOLOGIES = 64
REALIZATIONS = 32
BASE_SEED = 202610290001


def uma_pathloss_db(distance_2d: np.ndarray, los: np.ndarray) -> np.ndarray:
    """TR 38.901 UMa pathloss in dB for fc = 3.5 GHz and h_AP = 25 m."""

    d3d = np.sqrt(distance_2d**2 + (H_AP - H_UT) ** 2)
    d3d = np.maximum(d3d, 10.0)
    pl_los = 28.0 + 22.0 * np.log10(d3d) + 20.0 * math.log10(FC_GHZ)
    pl_nlos = (
        13.54
        + 39.08 * np.log10(d3d)
        + 20.0 * math.log10(FC_GHZ)
        - 0.6 * (H_UT - 1.5)
    )
    return np.where(los, pl_los, pl_nlos)


def los_probability(distance_2d: np.ndarray) -> np.ndarray:
    d = np.maximum(distance_2d, 1e-3)
    raw = np.where(d <= 18.0, 1.0, 18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d))
    return np.clip(raw, 0.0, 1.0)


_TABLE_CACHE: dict[tuple[float, int], np.ndarray] = {}


def laplacian_covariance(angles: np.ndarray, half_width_rad: float,
                         antennas: int, spacing: float = 0.5) -> np.ndarray:
    """Toeplitz covariance of a half-wavelength ULA under a wrapped Laplacian PAS.

    ``half_width_rad`` is the Laplace scale, not the RMS angular spread; the RMS spread is
    ``sqrt(2) * half_width_rad`` for a Laplace distribution.
    """

    key = (round(half_width_rad, 9), int(antennas))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    grid = np.linspace(-math.pi, math.pi, 721, endpoint=False)
    table = np.empty((angles.size, antennas, antennas), dtype=np.complex128)
    for a, centre in enumerate(angles):
        delta = (grid - centre + math.pi) % (2.0 * math.pi) - math.pi
        weight = np.exp(-np.abs(delta) / half_width_rad)
        weight /= weight.sum()
        phase = np.exp(1j * 2.0 * math.pi * spacing * np.sin(grid))
        for i in range(antennas):
            for j in range(antennas):
                table[a, i, j] = np.sum(weight * phase ** (i - j))
    _TABLE_CACHE[key] = table
    return table


def generate_topology(
    config: ScenarioConfig, *, seed: int, standard_los: bool
) -> tuple[Topology, dict]:
    rng = np.random.default_rng(int(seed))
    ap = rng.uniform(0.0, config.area_side, size=(config.access_points, 2))
    users = rng.uniform(0.0, config.area_side, size=(config.users, 2))
    displacement = users[None, :, :] - ap[:, None, :]
    side = float(config.area_side)
    displacement = (displacement + 0.5 * side) % side - 0.5 * side
    distance = np.maximum(np.sqrt(np.sum(displacement**2, axis=-1)), config.minimum_distance)
    bearing = np.arctan2(displacement[..., 1], displacement[..., 0])

    distance_m = distance * 1000.0
    if standard_los:
        los = rng.uniform(size=distance.shape) < los_probability(distance_m)
    else:
        los = np.ones_like(distance, dtype=bool)

    shadow = np.where(
        los,
        rng.normal(0.0, SHADOW_LOS_DB, size=distance.shape),
        rng.normal(0.0, SHADOW_NLOS_DB, size=distance.shape),
    )
    pathloss = uma_pathloss_db(distance_m, los) + shadow
    gain = 10.0 ** (-pathloss / 10.0)
    gain /= float(np.median(gain))

    kappa = np.where(los, KAPPA, 0.0)
    los_power = gain * kappa / (kappa + 1.0)
    diffuse_power = gain / (kappa + 1.0)

    antenna = np.arange(config.antennas_per_ap, dtype=np.float64)
    steering = np.exp(1j * math.pi * np.sin(bearing)[..., None] * antenna[None, None, :])
    tracked_phase = rng.uniform(-math.pi, math.pi, size=bearing.shape)
    tracked_los = np.transpose(
        np.sqrt(los_power)[..., None]
        * steering
        * np.exp(1j * tracked_phase)[..., None],
        (0, 2, 1),
    )

    as_deg = np.where(los, AS_LOS_DEG, AS_NLOS_DEG)
    index = ((bearing + math.pi) / (2.0 * math.pi) * 180).astype(int) % 180
    covariances = np.empty(
        (config.access_points, config.users, config.antennas_per_ap,
         config.antennas_per_ap),
        dtype=np.complex128,
    )
    for as_value in (AS_LOS_DEG, AS_NLOS_DEG):
        table = laplacian_covariance(
            np.linspace(-math.pi, math.pi, 180, endpoint=False),
            math.radians(as_value),
            config.antennas_per_ap,
        )
        mask = np.isclose(as_deg, as_value)
        covariances[mask] = diffuse_power[mask][:, None, None] * table[index[mask]]

    pilot = np.arange(config.users, dtype=np.int64) % config.pilot_symbols
    pilot = pilot[rng.permutation(config.users)]
    topology = Topology(
        ap_positions=ap.astype(np.float64),
        user_positions=users.astype(np.float64),
        distance=distance.astype(np.float64),
        large_scale_gain=gain.astype(np.float64),
        tracked_los=tracked_los.astype(np.complex128),
        diffuse_covariances=covariances,
        pilot_indices=pilot.astype(np.int64),
    )
    meta = {
        "los_fraction": float(los.mean()),
        "median_gain": float(np.median(gain)),
        "kappa_mean": float(kappa.mean()),
    }
    return topology, meta


def cell_mse(config, topology, spec, seed: int) -> tuple[float, float]:
    frame = generate_shape_frame(topology, config, spec, seed=seed)
    models, observations = [], []
    for ap in range(config.access_points):
        for pilot in range(config.pilot_symbols):
            model, y, _, _ = shape_pilot_group(
                topology, frame, config, spec, access_point=ap, pilot=pilot
            )
            models.append(model)
            observations.append(y)
    batch = stack_shape_models(models, observations)
    b1 = batched_b1_shape(batch)
    b2 = batched_b2_shape(batch)
    truth = frame.channels.transpose(0, 2, 1)
    energy = 0.0
    error1 = 0.0
    error2 = 0.0
    for ap in range(config.access_points):
        for pilot in range(config.pilot_symbols):
            group = ap * config.pilot_symbols + pilot
            users = np.flatnonzero(topology.pilot_indices == pilot)
            target = truth[ap, users, :]
            energy += float(np.sum(np.abs(target) ** 2))
            error1 += float(np.sum(np.abs(b1[group].T - target) ** 2))
            error2 += float(np.sum(np.abs(b2[group].T - target) ** 2))
    return error1, error2, energy


def run_variant(standard_los: bool) -> dict:
    config = ScenarioConfig(
        access_points=32,
        antennas_per_ap=4,
        users=16,
        pilot_symbols=4,
        coherence_symbols=200,
        area_side=1.0,
        minimum_distance=0.03,
        pathloss_exponent=3.7,
        rician_kappa=KAPPA,
        pilot_assignment="balanced_random",
        pilot_noise_variance=1.0,
        payload_noise_variance=1.0,
    )
    ratios = {"U": [], "G": [], "B": []}
    meta_rows = []
    started = time.monotonic()
    for t in range(TOPOLOGIES):
        topology, meta = generate_topology(
            config, seed=BASE_SEED + 1000 * int(standard_los) + t,
            standard_los=standard_los,
        )
        meta_rows.append(meta)
        for name, shape in (("U", "uniform"), ("G", "wrapped_gaussian"), ("B", "binary")):
            spec = shape_spec(shape, 0.90)
            e1 = e2 = e = 0.0
            for r in range(REALIZATIONS):
                a, b, c = cell_mse(config, topology, spec, BASE_SEED + 7919 * r + t)
                e1 += a
                e2 += b
                e += c
            ratios[name].append(e2 / e1)
    elapsed = time.monotonic() - started
    u = np.array(ratios["U"])
    g = np.array(ratios["G"])
    b = np.array(ratios["B"])
    effect = float((b.mean() - 1.0) / (g.mean() - 1.0))
    order = int(np.sum((b > u) & (u > g)))
    return {
        "standard_los": bool(standard_los),
        "topologies": TOPOLOGIES,
        "realizations": REALIZATIONS,
        "mean_r_U": float(u.mean()),
        "mean_r_G": float(g.mean()),
        "mean_r_B": float(b.mean()),
        "effect": effect,
        "full_order": order,
        "B_gt_U": int(np.sum(b > u)),
        "U_gt_G": int(np.sum(u > g)),
        "los_fraction_mean": float(np.mean([m["los_fraction"] for m in meta_rows])),
        "seconds": elapsed,
    }


def validation() -> dict:
    config = ScenarioConfig(
        access_points=8, antennas_per_ap=4, users=8, pilot_symbols=4,
        coherence_symbols=200, area_side=1.0, minimum_distance=0.03,
        pathloss_exponent=3.7, rician_kappa=KAPPA,
        pilot_assignment="balanced_random",
    )
    topology, meta = generate_topology(config, seed=BASE_SEED, standard_los=False)
    cov = topology.diffuse_covariances[0, 0]
    hermitian = float(np.max(np.abs(cov - cov.conj().T)))
    eigen = float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.conj().T))))
    diag = np.real(np.diag(cov))
    corr = np.abs(cov[0, 1]) / math.sqrt(diag[0] * diag[1])
    return {
        "hermitian_error": hermitian,
        "min_eigenvalue": eigen,
        "unit_diagonal_relative_error": float(np.max(np.abs(diag - diag[0]) / diag[0])),
        "correlation_lag1": float(corr),
        "los_fraction_forced": meta["los_fraction"],
        "los_probability_at_100m": float(los_probability(np.array(100.0))),
        "los_probability_at_300m": float(los_probability(np.array(300.0))),
    }


def main() -> None:
    checks = validation()
    print("validation", json.dumps(checks, indent=1))
    results = {"validation": checks, "variants": []}
    for standard_los in (False, True):
        row = run_variant(standard_los)
        results["variants"].append(row)
        print(json.dumps(row, indent=1))
    out = HERE.parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
