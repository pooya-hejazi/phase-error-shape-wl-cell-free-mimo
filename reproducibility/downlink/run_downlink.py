"""Downlink MR precoding on the same estimator pair.

Both estimators are evaluated on the same frames under the same true phase law, so the comparison is
paired. The use-and-then-forget bound used here is

    SINR_k = |E[h_k^H v_k]|^2 / (sum_i E|h_k^H v_i|^2 - |E[h_k^H v_k]|^2 + sigma^2),

which includes the variance of the desired effective channel in the denominator.

    .venv\\Scripts\\python.exe -B reproducibility\\downlink\\run_downlink.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "reproducibility/expanded_run_v02/run"))

from phase_graph_cme.scenario import ScenarioConfig, generate_topology  # noqa: E402
from shape_core import shape_spec  # noqa: E402
from shape_estimators import (  # noqa: E402
    batched_b1_shape,
    batched_b2_shape,
    stack_shape_models,
)
from shape_model import generate_shape_frame, shape_pilot_group  # noqa: E402

TOPOLOGIES = 32
REALIZATIONS = 16
BASE_SEED = 202610310001
NOISE = 1.0
TAU_C = 200
TAU_P = 4


def estimates(config, topology, spec, seed: int, m2_offset=None):
    frame = generate_shape_frame(topology, config, spec, seed=seed)
    models, observations = [], []
    for ap in range(config.access_points):
        for pilot in range(config.pilot_symbols):
            offset = None if m2_offset is None else m2_offset[ap]
            model, y, _, _ = shape_pilot_group(
                topology, frame, config, spec, access_point=ap, pilot=pilot,
                m2_est=None if offset is None else float(spec.m2) + offset,
            )
            models.append(model)
            observations.append(y)
    batch = stack_shape_models(models, observations)
    b1 = batched_b1_shape(batch)
    b2 = batched_b2_shape(batch)
    n = config.antennas_per_ap
    out = {}
    for label, value in (("B1", b1), ("B2", b2)):
        matrix = np.zeros((config.access_points * n, config.users), dtype=np.complex128)
        for ap in range(config.access_points):
            for pilot in range(config.pilot_symbols):
                group = ap * config.pilot_symbols + pilot
                users = np.flatnonzero(topology.pilot_indices == pilot)
                matrix[ap * n:(ap + 1) * n, users] = value[group]
        out[label] = matrix
    truth = frame.channels.reshape(config.access_points * n, config.users)
    return truth, out


def accumulate(truth, est, sum_self, sum_self_sq, sum_cross):
    users = truth.shape[1]
    v = est / np.maximum(np.linalg.norm(est, axis=0), 1e-30)[None, :]
    for k in range(users):
        inner = truth[:, k].conj() @ v[:, k]
        sum_self[k] += inner
        sum_self_sq[k] += abs(inner) ** 2
        for i in range(users):
            if i == k:
                continue
            sum_cross[k, i] += abs(truth[:, k].conj() @ v[:, i]) ** 2


def finalise(sum_self, sum_self_sq, sum_cross, frames):
    num = np.abs(sum_self / frames) ** 2
    den = (
        sum_cross.sum(axis=1) / frames
        + sum_self_sq / frames
        - num
        + NOISE
    )
    sinr = num / np.maximum(den, 1e-30)
    return (TAU_C - TAU_P) / TAU_C * np.log2(1.0 + sinr)


def self_test() -> dict:
    """Deterministic and random single-user checks of the UatF denominator."""

    sigma = NOISE
    h = np.array([1 + 1j, 2 - 1j, 0.5 + 0j, -1j])
    e = h.reshape(-1, 1)
    self_sum = np.zeros(1, dtype=np.complex128)
    self_sq = np.zeros(1)
    cross = np.zeros((1, 1))
    accumulate(e, e, self_sum, self_sq, cross)
    deterministic = finalise(self_sum, self_sq, cross, 1)[0]
    expected = (TAU_C - TAU_P) / TAU_C * np.log2(
        1.0 + float(np.vdot(h, h).real) / sigma
    )

    rng = np.random.default_rng(7)
    self_sum = np.zeros(1, dtype=np.complex128)
    self_sq = np.zeros(1)
    cross = np.zeros((1, 1))
    for _ in range(400):
        draw = (rng.standard_normal(4) + 1j * rng.standard_normal(4))
        column = draw.reshape(-1, 1)
        accumulate(column, column, self_sum, self_sq, cross)
    random_case = finalise(self_sum, self_sq, cross, 400)[0]

    return {
        "deterministic_se": float(deterministic),
        "deterministic_expected": float(expected),
        "deterministic_match": bool(abs(deterministic - expected) < 1e-12),
        "random_se": float(random_case),
        "random_denominator_has_variance": bool(random_case < deterministic),
    }


def paired_summary(per_topology: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(20261031)

    def paired(a: str, b: str) -> dict:
        diff = per_topology[a] - per_topology[b]
        draws = [
            float(np.mean(rng.choice(diff, size=diff.size, replace=True)))
            for _ in range(5000)
        ]
        return {
            "mean": float(diff.mean()),
            "interval": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "positive_topologies": int(np.sum(diff > 0)),
            "n": int(diff.size),
        }

    return {
        "B1_minus_B2_uniform": paired("B1_U", "B2_U"),
        "B1_minus_B2_gaussian": paired("B1_G", "B2_G"),
        "B1_minus_B2_binary": paired("B1_B", "B2_B"),
        "B1_B_minus_B1_U": paired("B1_B", "B1_U"),
        "B1_U_minus_B1_G": paired("B1_U", "B1_G"),
        "CAL_minus_B1_B": paired("CAL_B", "B1_B"),
    }


def main() -> None:
    config = ScenarioConfig(
        access_points=32, antennas_per_ap=4, users=16, pilot_symbols=4,
        coherence_symbols=TAU_C, area_side=1.0, minimum_distance=0.03,
        pathloss_exponent=3.7, rician_kappa=20.0,
        pilot_assignment="balanced_random",
        pilot_noise_variance=NOISE, payload_noise_variance=NOISE,
    )
    laws = {"U": "uniform", "G": "wrapped_gaussian", "B": "binary"}
    rows: dict[str, list[np.ndarray]] = {
        "B1_U": [], "B2_U": [], "B1_G": [], "B2_G": [], "B1_B": [], "B2_B": [], "CAL_B": []
    }
    started = time.monotonic()
    for t in range(TOPOLOGIES):
        topology = generate_topology(config, seed=BASE_SEED + t)
        for code, shape in laws.items():
            spec = shape_spec(shape, 0.90)
            sums = {
                label: (
                    np.zeros(config.users, dtype=np.complex128),
                    np.zeros(config.users),
                    np.zeros((config.users, config.users)),
                )
                for label in ("B1", "B2")
            }
            cal_sums = (
                np.zeros(config.users, dtype=np.complex128),
                np.zeros(config.users),
                np.zeros((config.users, config.users)),
            )
            for r in range(REALIZATIONS):
                truth, est = estimates(config, topology, spec, BASE_SEED + 7919 * r + t)
                for label in ("B1", "B2"):
                    accumulate(truth, est[label], *sums[label])
                if code == "B":
                    rng = np.random.default_rng(BASE_SEED + 100000 + 97 * t)
                    offset = 0.0078 * rng.standard_normal(config.access_points)
                    truth_c, est_c = estimates(
                        config, topology, spec, BASE_SEED + 7919 * r + t, offset
                    )
                    accumulate(truth_c, est_c["B1"], *cal_sums)
            for label in ("B1", "B2"):
                rows[f"{label}_{code}"].append(finalise(*sums[label], REALIZATIONS))
            if code == "B":
                rows["CAL_B"].append(finalise(*cal_sums, REALIZATIONS))
    elapsed = time.monotonic() - started

    summary = {}
    per_topology = {}
    for name, values in rows.items():
        array = np.array(values)
        per_topology[name] = array.mean(axis=1)
        mean_user = array.mean(axis=0)
        summary[name] = {
            "mean_se": float(mean_user.mean()),
            "p10_se": float(np.quantile(mean_user, 0.10)),
            "p50_se": float(np.quantile(mean_user, 0.50)),
        }
    result = {
        "self_test": self_test(),
        "topologies": TOPOLOGIES,
        "realizations": REALIZATIONS,
        "estimand": "UatF downlink SE with the desired-channel variance in the denominator",
        "seconds": elapsed,
        "summary": summary,
        "paired": paired_summary(per_topology),
    }
    (HERE.parent / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["self_test"], indent=1))
    print(json.dumps(summary, indent=1))
    for key, value in result["paired"].items():
        print(key, value)
    print("seconds", elapsed)


if __name__ == "__main__":
    main()
