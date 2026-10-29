"""Calibration experiment on the actual phase laws (round 3, replaces the Gaussian surrogate).

For each law the residual phase is drawn from the declared distribution, the reference-link
observation is formed as ``z = exp(j eps) + w`` with proper complex noise of variance
``sigma_w^2 = (sigma_g^2 + s/beta^2)/ell^2``, the moments are estimated from ``M`` frames, projected
onto the feasible set, and the risk is evaluated with the estimated pair. No Gaussian surrogate and no
zero-mean recentring are used.

Run from the repository root:

    .venv\\Scripts\\python.exe -B analysis\\calibration_law.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from calibration import LAWS, hessian, mse, oracle_gain

HERE = Path(__file__).resolve()
OUT = HERE.parent / "results" / "cp2_calibration_law.json"

SEED = 2026103101
REPLICATES = 2000
FRAMES = (256, 1024, 4096, 16384)
SIGMA_W2 = (1.0 / 20.0 + 0.01)
Z90 = float(norm.ppf(0.975))
BOUND = 1.6448536269514722  # sqrt(chi2_2 at 0.95) / sqrt(2); unused, kept explicit


def draw_phases(law: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if law == "binary":
        delta = math.acos(0.9)
        return np.where(rng.random(size) < 0.5, -delta, delta)
    if law == "uniform":
        from phase_laws import uniform_delta_for_m1

        delta = uniform_delta_for_m1(0.9)
        return (2.0 * rng.random(size) - 1.0) * delta
    if law == "wrapped_gaussian":
        from phase_laws import wg_sigma_for_m1

        sigma = wg_sigma_for_m1(0.9)
        return np.angle(np.exp(1j * sigma * rng.standard_normal(size)))
    if law == "von_mises":
        from phase_laws import vm_kappa_for_m1

        kappa = vm_kappa_for_m1(0.9)
        return rng.vonmises(0.0, kappa, size)
    raise ValueError(law)


def project(m1: float, c: float) -> tuple[float, float, bool]:
    lens = 1.0 - m1 * m1
    lower = -lens
    clipped = min(max(c, lower), 0.0)
    return m1, clipped, abs(clipped - c) > 1e-12


def law_moments(law: str, size: int = 400000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    eps = draw_phases(law, size, rng)
    z = np.exp(1j * eps) + math.sqrt(SIGMA_W2 / 2.0) * (
        rng.standard_normal(size) + 1j * rng.standard_normal(size)
    )
    centred = z - z.mean()
    return {
        "mean_real": float(z.mean().real),
        "e2": float(np.mean(centred**2).real),
        "e3": float(np.mean(centred**3).real),
        "e4": float(np.mean(np.abs(centred) ** 4)),
        "variance": float(np.mean(np.abs(centred) ** 2)),
        "kurtosis_excess": float(
            np.mean(np.abs(centred) ** 4) / max(np.mean(np.abs(centred) ** 2) ** 2, 1e-30) - 2.0
        ),
    }


def law_seed(law: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{law}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def run_law(law: str, m1_true: float, c_true: float) -> dict:
    seed = law_seed(law)
    rng = np.random.default_rng(seed)
    mse_b2, mse_b1, gain = oracle_gain(m1_true, c_true)
    rows = []
    moments = law_moments(law)
    for frames in FRAMES:
        m1_hat = np.empty(REPLICATES)
        c_hat = np.empty(REPLICATES)
        raw_c = np.empty(REPLICATES)
        active = np.zeros(REPLICATES, dtype=bool)
        risky = np.empty(REPLICATES)
        chunk = 200
        done = 0
        while done < REPLICATES:
            block = min(chunk, REPLICATES - done)
            eps = draw_phases(law, block * frames, rng).reshape(block, frames)
            w = math.sqrt(SIGMA_W2 / 2.0) * (
                rng.standard_normal((block, frames)) + 1j * rng.standard_normal((block, frames))
            )
            z = np.exp(1j * eps) + w
            zbar = z.mean(axis=1)
            m_hat = zbar.real
            centred = z - zbar[:, None]
            c_raw = np.real(np.sum(centred * centred, axis=1)) / max(frames - 1, 1)
            for i in range(block):
                mm, cc, act = project(float(m_hat[i]), float(c_raw[i]))
                m1_hat[done + i] = mm
                c_hat[done + i] = cc
                raw_c[done + i] = c_raw[i]
                active[done + i] = act
                risky[done + i] = mse(m1_true, c_true, mm, cc)
            done += block
        capture = (mse_b2 - risky) / gain
        cov = np.cov(np.vstack([m1_hat - m1_true, c_hat - c_true]), ddof=1)
        rows.append(
            {
                "frames": frames,
                "bias_m1": float(m1_hat.mean() - m1_true),
                "bias_c": float(c_hat.mean() - c_true),
                "raw_bias_c": float(raw_c.mean() - c_true),
                "se_m1": float(math.sqrt(cov[0, 0])),
                "se_c": float(math.sqrt(cov[1, 1])),
                "cov_m1_c": float(cov[0, 1]),
                "corr_m1_c": float(cov[0, 1] / math.sqrt(max(cov[0, 0] * cov[1, 1], 1e-30))),
                "projection_fraction": float(active.mean()),
                "capture_mean": float(capture.mean()),
                "capture_sd": float(capture.std(ddof=1)),
                "capture_p05": float(np.quantile(capture, 0.05)),
                "capture_p10": float(np.quantile(capture, 0.10)),
                "capture_p50": float(np.quantile(capture, 0.50)),
                "capture_p90": float(np.quantile(capture, 0.90)),
                "capture_p95": float(np.quantile(capture, 0.95)),
                "capture_ci95": [
                    float(capture.mean() - Z90 * capture.std(ddof=1) / math.sqrt(REPLICATES)),
                    float(capture.mean() + Z90 * capture.std(ddof=1) / math.sqrt(REPLICATES)),
                ],
                "fraction_worse_than_b2": float((capture < 0.0).mean()),
                "mcse_capture": float(capture.std(ddof=1) / math.sqrt(REPLICATES)),
                "diagnostic_draws": {
                    "m1_hat": [float(x) for x in m1_hat[:500]],
                    "c_hat": [float(x) for x in c_hat[:500]],
                    "capture": [float(x) for x in capture[:500]],
                },
            }
        )
    return {
        "law": law,
        "seed": seed,
        "m1": m1_true,
        "c": c_true,
        "gain": gain,
        "moments": moments,
        "curve": rows,
    }


def summarize_law(entry: dict) -> dict:
    m1, c = entry["m1"], entry["c"]
    _, _, gain = oracle_gain(m1, c)
    curve = sorted(entry["curve"], key=lambda row: row["frames"])
    measured = None
    for left, right in zip(curve, curve[1:]):
        if left["capture_mean"] < 0.99 <= right["capture_mean"]:
            fraction = ((0.99 - left["capture_mean"])
                        / (right["capture_mean"] - left["capture_mean"]))
            measured = math.exp(
                math.log(left["frames"])
                + fraction * (math.log(right["frames"]) - math.log(left["frames"]))
            )
            break

    hess = hessian(m1, c)
    hmat = np.array([[hess["h11"], hess["h12"]], [hess["h12"], hess["h22"]]])
    anchor = next(row for row in curve if row["frames"] == 1024)
    cov_anchor = np.array([
        [anchor["se_m1"] ** 2, anchor["cov_m1_c"]],
        [anchor["cov_m1_c"], anchor["se_c"] ** 2],
    ])
    model_excess = 0.5 * float(np.trace(hmat @ cov_anchor))
    model_m99 = 1024.0 * model_excess / (0.01 * gain)
    return {
        "gain": gain,
        "frames_for_99pct_measured": measured,
        "frames_for_99pct_model": model_m99,
        "excess_at_1024_model": model_excess,
        "capture_at_1024": anchor["capture_mean"],
        "capture_at_4096": next(row["capture_mean"] for row in curve if row["frames"] == 4096),
        "capture_at_16384": next(row["capture_mean"] for row in curve if row["frames"] == 16384),
    }


def main() -> None:
    payload = {
        "experiment": "calibration on the actual phase laws",
        "seed": SEED,
        "replicates": REPLICATES,
        "frames": list(FRAMES),
        "sigma_w2": SIGMA_W2,
        "estimators": {
            "m1": "real sample mean of z",
            "c": "real unbiased sample pseudo-variance, projected onto the lens and c<=0",
        },
        "note": "no Gaussian surrogate and no zero-mean recentring",
        "laws": {},
    }
    for law, (m1, c) in LAWS.items():
        payload["laws"][law] = run_law(law, m1, c)
        row = payload["laws"][law]["curve"][1]
        print(
            f"{law:17s} M={row['frames']} bias_c={row['bias_c']:+.4f} "
            f"se_c={row['se_c']:.4f} proj={row['projection_fraction']:.3f} "
            f"capture={row['capture_mean']:.4f} +/- {row['mcse_capture']:.4f}"
        )
    payload["summary"] = {
        law: summarize_law(entry) for law, entry in payload["laws"].items()
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
