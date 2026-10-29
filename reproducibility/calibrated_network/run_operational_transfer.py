"""New operational-overhead, shared-LO and reference/data-mismatch experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_calibrated_network import (
    LAWS, ESTIMATORS, ScenarioConfig, generate_topology, calibrate, make_frame,
    true_moments, estimate, rzf_combiner, uatf,
)

M1 = 0.90
TOPOLOGIES = 8
REALIZATIONS = 4


def config_for_experiment():
    return ScenarioConfig(
        access_points=32, antennas_per_ap=4, users=16, pilot_symbols=4,
        coherence_symbols=200, area_side=1.0, minimum_distance=0.03,
        pathloss_exponent=3.7, rician_kappa=20.0, pilot_assignment="balanced_random",
        spatial_correlation=0.4,
    )


def evaluate(config, topology, law: str, ma: float, ca: float, wl: bool, seed_base: int,
             shared: bool):
    errs, se_users, energies = [], [], []
    for r in range(REALIZATIONS):
        truth, obs = make_frame(topology, config, law, seed_base + 7919 * r, shared=shared)
        est = estimate(truth, obs, topology, config, ma, ca, wl=wl)
        errs.append(float(np.sum(np.abs(truth - est) ** 2)))
        energies.append(float(np.sum(np.abs(truth) ** 2)))
        h = truth.reshape((config.access_points * config.antennas_per_ap, config.users))
        e = est.reshape((config.access_points * config.antennas_per_ap, config.users))
        se_users.append(uatf(h, rzf_combiner(e, 1.0), config))
    return {
        "mse": float(np.mean(errs)),
        "nmse": float(np.mean(errs) / np.mean(energies)),
        "se_per_user": np.mean(se_users, axis=0).tolist(),
        "se_mean": float(np.mean(se_users)),
    }


def paired_boot(x: np.ndarray, seed: int = 20261115) -> list[float]:
    if x.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(x, size=x.size, replace=True))) for _ in range(5000)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    config = config_for_experiment()
    transfer = []
    for shared in (False, True):
        for law in LAWS:
            m1, m2 = true_moments(law)
            gain, ratio = [], []
            for t in range(TOPOLOGIES):
                topology = generate_topology(config, seed=202611210000 + t)
                m1c, cc = calibrate(law, 1024, 202612220000 + 97 * t)
                b1 = evaluate(config, topology, law, m1c, cc, True, 202613230000 + 17 * t, shared)
                b2 = evaluate(config, topology, law, m1c, 0.0, False, 202613230000 + 17 * t, shared)
                gain.append(b1["se_mean"] - b2["se_mean"])
                ratio.append(b1["nmse"] / b2["nmse"])
            transfer.append({
                "law": law,
                "shared_lo": shared,
                "se_gain_B1_minus_B2_mean": float(np.mean(gain)),
                "se_gain_interval": paired_boot(np.asarray(gain)),
                "nmse_ratio_B1_over_B2_mean": float(np.mean(ratio)),
            })
        print("shared" if shared else "independent", "done")

    mismatch = []
    pairs = [("binary", "uniform"), ("uniform", "binary"),
             ("wrapped_gaussian", "von_mises"), ("von_mises", "wrapped_gaussian")]
    for ref, data_law in pairs:
        m1, m2 = true_moments(data_law)
        gains, ratios = [], []
        for t in range(TOPOLOGIES):
            topology = generate_topology(config, seed=202611240000 + t)
            m1c, cc = calibrate(ref, 1024, 202612250000 + 97 * t)
            b1 = evaluate(config, topology, data_law, m1c, cc, True, 202613260000 + 17 * t, False)
            oracle_b2 = evaluate(config, topology, data_law, m1, 0.0, False,
                                 202613260000 + 17 * t, False)
            gains.append(b1["se_mean"] - oracle_b2["se_mean"])
            ratios.append(b1["nmse"] / oracle_b2["nmse"])
        mismatch.append({
            "reference_law": ref,
            "data_law": data_law,
            "se_gain_B1_minus_oracle_B2_mean": float(np.mean(gains)),
            "se_gain_interval": paired_boot(np.asarray(gains)),
            "nmse_ratio_B1_over_oracle_B2_mean": float(np.mean(ratios)),
        })
        print("mismatch", ref, data_law, "done")

    operational = []
    for law in LAWS:
        for m in (64, 256, 1024, 4096):
            for horizon in (32, 128, 512):
                m1, m2 = true_moments(law)
                vals = {est: [] for est in ("cal_B1", "cal_B2")}
                for t in range(TOPOLOGIES):
                    topology = generate_topology(config, seed=202611270000 + t)
                    m1c, cc = calibrate(law, m, 202612280000 + 97 * t + m)
                    for est in vals:
                        out = evaluate(config, topology, law, m1c, cc if est == "cal_B1" else 0.0,
                                       est == "cal_B1", 202613290000 + 17 * t, False)
                        overhead = m * 4.0 / (m * 4.0 + horizon * (200.0 - 4.0))
                        vals[est].append((1.0 - overhead) * out["se_mean"])
                operational.append({
                    "law": law,
                    "calibration_frames": m,
                    "horizon_blocks": horizon,
                    "overhead_fraction": m * 4.0 / (m * 4.0 + horizon * 196.0),
                    **{f"{est}_net_se_mean": float(np.mean(vals[est])) for est in vals},
                })
            print("operational", law, m, "done")

    payload = {
        "experiment": "operational transferability and overhead",
        "m1": M1,
        "topologies": TOPOLOGIES,
        "realizations": REALIZATIONS,
        "transfer": transfer,
        "mismatch": mismatch,
        "operational": operational,
        "notes": "New experiment. Overhead uses 4 reference symbols per calibration frame and 196 data symbols per evaluation block.",
    }
    out = Path(__file__).resolve().parent / "output" / "operational_transfer.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
