"""Actual calibrated-network experiment (new, not frozen).

Simulates independent residual-phase links and computes calibrated channel MSE and
uplink UatF SE with shared reference information per RF chain. Results are written
to evidence/calibrated_network.json; nothing is overwritten in the frozen packets.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "expanded_run_v02/run"))
from phase_graph_cme.scenario import ScenarioConfig, generate_topology
from shape_core import shape_spec
from shape_model import ShapePilotGroupModel
from shape_estimators import clmmse_shape, widely_linear_mmse_shape
from phase_graph_cme.receivers import rzf_combiner

M1 = 0.90
TOPOLOGIES = int(os.environ.get("CAL_TOPOLOGIES", "32"))
REALIZATIONS = int(os.environ.get("CAL_REALIZATIONS", "8"))
CAL_M = 1024
LAWS = ("binary", "uniform", "wrapped_gaussian", "von_mises")
ESTIMATORS = ("oracle_B1", "cal_B1", "oracle_B2", "cal_B2")
SIGMA_W2 = 0.06


def vm_kappa(m1: float) -> float:
    return brentq(lambda x: ive(1, x) / ive(0, x) - m1, 1e-8, 1e5)


def phase_sample(law: str, size, rng):
    if law == "binary":
        d = math.acos(M1)
        return np.where(rng.random(size) < 0.5, -d, d)
    if law == "uniform":
        d = shape_spec("uniform", M1).parameter["delta_rad"]
        return rng.uniform(-d, d, size)
    if law == "wrapped_gaussian":
        return np.angle(np.exp(1j * math.sqrt(-2.0 * math.log(M1)) * rng.standard_normal(size)))
    if law == "von_mises":
        return rng.vonmises(0.0, vm_kappa(M1), size)
    raise ValueError(law)


def true_moments(law: str) -> tuple[float, float]:
    if law == "von_mises":
        k = vm_kappa(M1)
        return M1, float(ive(2, k) / ive(0, k))
    s = shape_spec(law, M1)
    return s.m1, s.m2


def clip_moments(m1: float, c: float) -> tuple[float, float]:
    return m1, min(0.0, max(-(1.0 - m1 * m1), c))


def calibrate(law: str, m: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    eps = phase_sample(law, m, rng)
    w = math.sqrt(SIGMA_W2 / 2.0) * (
        rng.standard_normal(m) + 1j * rng.standard_normal(m)
    )
    z = np.exp(1j * eps) + w
    mean = z.mean()
    m1h = float(np.clip(mean.real, 1e-6, 1.0 - 1e-9))
    c = float(np.real(np.sum((z - mean) ** 2)) / (m - 1))
    return clip_moments(m1h, c)


def make_frame(topology, config, law: str, seed: int, shared: bool = False):
    rng = np.random.default_rng(seed)
    l, k = config.access_points, config.users
    if shared:
        phases = np.broadcast_to(phase_sample(law, (l, 1), rng), (l, k)).copy()
    else:
        phases = phase_sample(law, (l, k), rng)
    channels = np.empty_like(topology.tracked_los)
    for ap in range(l):
        for user in range(k):
            cov = topology.diffuse_covariances[ap, user]
            z = (rng.standard_normal(config.antennas_per_ap)
                 + 1j * rng.standard_normal(config.antennas_per_ap)) / math.sqrt(2.0)
            channels[ap, :, user] = (
                topology.tracked_los[ap, :, user] * np.exp(1j * phases[ap, user])
                + np.linalg.cholesky(cov) @ z
            )
    noise = math.sqrt(config.pilot_noise_variance / 2.0) * (
        rng.standard_normal((l, config.antennas_per_ap, config.pilot_symbols))
        + 1j * rng.standard_normal((l, config.antennas_per_ap, config.pilot_symbols))
    )
    observations = noise.astype(np.complex128)
    beta = config.pilot_coefficient
    for pilot in range(config.pilot_symbols):
        group = np.flatnonzero(topology.pilot_indices == pilot)
        observations[:, :, pilot] += beta * np.sum(channels[:, :, group], axis=2)
    return channels, observations


def estimate(channels, observations, topology, config, m1a: float, ca: float, wl: bool):
    if abs(ca) > 1.0 - m1a * m1a + 1e-10:
        print("LENS DEBUG", m1a, ca)
        raise ValueError(f"lens violation before estimator: m1a={m1a}, ca={ca}")
    estimates = np.empty_like(channels)
    for ap in range(config.access_points):
        for pilot in range(config.pilot_symbols):
            users = np.flatnonzero(topology.pilot_indices == pilot).astype(np.int64)
            model = ShapePilotGroupModel(
                los=topology.tracked_los[ap][:, users],
                covariances=topology.diffuse_covariances[ap, users],
                beta=np.full(users.size, config.pilot_coefficient, dtype=np.complex128),
                noise_covariance=config.pilot_noise_variance * np.eye(
                    config.antennas_per_ap, dtype=np.complex128),
                m1=np.full(users.size, m1a),
                m2=np.full(users.size, m1a * m1a + ca),
            )
            y = observations[ap, :, pilot]
            for j, u in enumerate(users):
                estimates[ap, :, u] = (
                    widely_linear_mmse_shape(model, y, j) if wl else clmmse_shape(model, y, j)
                )
    return estimates


def uatf(channel: np.ndarray, combiner: np.ndarray, config) -> np.ndarray:
    h = channel[None, :, :]
    v = combiner[None, :, :]
    inner = np.einsum("smk,smi->ski", v.conj(), h, optimize=True)
    coh = np.mean(np.diagonal(inner, axis1=1, axis2=2), axis=0)
    second = np.mean(np.abs(inner) ** 2, axis=0)
    norm = np.mean(np.sum(np.abs(v) ** 2, axis=1), axis=0)
    powers = np.ones(config.users)
    num = powers * np.abs(coh) ** 2
    den = second @ powers - num + config.payload_noise_variance * norm
    sinr = num / den
    return (1.0 - config.pilot_symbols / config.coherence_symbols) * np.log2(1.0 + sinr)


def main() -> None:
    config = ScenarioConfig(
        access_points=32, antennas_per_ap=4, users=16, pilot_symbols=4,
        coherence_symbols=200, area_side=1.0, minimum_distance=0.03,
        pathloss_exponent=3.7, rician_kappa=20.0, pilot_assignment="balanced_random",
        spatial_correlation=0.4,
    )
    records = []
    for t in range(TOPOLOGIES):
        topology = generate_topology(config, seed=202611010000 + t)
        for law in LAWS:
            m1, m2 = true_moments(law)
            cal_m1, cal_c = calibrate(law, CAL_M, 202612020000 + 97 * t)
            per_law = {}
            for est in ESTIMATORS:
                is_b1 = est.endswith("B1")
                is_cal = est.startswith("cal_")
                if is_cal:
                    ma, ca = cal_m1, cal_c if is_b1 else 0.0
                else:
                    ma, ca = m1, (m2 - m1 * m1) if is_b1 else 0.0
                errs = []
                se_users = []
                per_realization = []
                for r in range(REALIZATIONS):
                    truth, obs = make_frame(topology, config, law, 202613030000 + 7919 * r + 17 * t)
                    estimate_array = estimate(truth, obs, topology, config, ma, ca, wl=is_b1)
                    err = float(np.sum(np.abs(truth - estimate_array) ** 2))
                    energy = float(np.sum(np.abs(truth) ** 2))
                    errs.append(err)
                    est_flat = estimate_array.reshape((config.access_points * config.antennas_per_ap,
                                            config.users))
                    truth_flat = truth.reshape((config.access_points * config.antennas_per_ap,
                                                config.users))
                    se_user = uatf(truth_flat, rzf_combiner(est_flat, 1.0), config)
                    se_users.append(se_user)
                    per_realization.append({"realization": r, "mse": err, "energy": energy,
                                            "se_per_user": se_user.tolist()})
                per_law[est] = {
                    "mse": float(np.mean(errs)),
                    "energy": float(np.mean([x["energy"] for x in per_realization])),
                    "se_per_user": np.mean(se_users, axis=0).tolist(),
                    "se_mean": float(np.mean(se_users)),
                    "per_realization": per_realization,
                }
            records.append({"topology": t, "law": law, "estimators": per_law})
        if (t + 1) % 8 == 0:
            print("topologies completed", t + 1)
    payload = {
        "experiment": "actual calibrated network experiment",
        "m1": M1,
        "calibration_frames": CAL_M,
        "topologies": TOPOLOGIES,
        "realizations": REALIZATIONS,
        "laws": list(LAWS),
        "estimators": list(ESTIMATORS),
        "model": "independent residuals, one calibration estimate per topology/RF chain",
        "se": "RZF lambda=1, UatF prelog .98, payload power=1 per user",
        "records": records,
    }
    out = Path(__file__).resolve().parent / "output" / os.environ.get(
        "CAL_OUTPUT", "calibrated_network.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
