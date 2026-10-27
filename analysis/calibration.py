"""Calibrated widely-linear estimation: envelope property, sample complexity, capture (CP2).

The estimator is B1 with the moments ``(m1, c)`` replaced by calibration estimates obtained from a
window of ``M`` frames.  The claim tested here is the envelope (second-order) property: because the
affine coefficients minimise the MSE at the true moments, the excess MSE is quadratic in the moment
error, so a modest calibration window recovers almost the whole oracle gain.

Run from the repository root:

    $env:PYTHONDONTWRITEBYTECODE = "1"
    .venv\\Scripts\\python.exe -B experiments\\upgrade\\calibration.py
    Remove-Item Env:\\PYTHONDONTWRITEBYTECODE
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results"
OUT.mkdir(parents=True, exist_ok=True)

# Operating point copied from the manuscript's scalar table.
L = 1.0
BETA = 1.0
S = 0.01
KAPPA = 20.0
SIGMA_G2 = 1.0 / KAPPA

LAWS = {
    # (m1, c) for the three laws at m1 = 0.90
    "binary": (0.90, -0.190000000000),
    "uniform": (0.90, -0.174422132908),
    "wrapped_gaussian": (0.90, -0.153900000000),
}


def link_terms(m1: float, c: float) -> tuple[float, float, float]:
    v = L * L * (1.0 - m1 * m1) + SIGMA_G2
    p = L * L * c
    cy = BETA * BETA * v + S
    return v, p, cy


def coefficients(m1a: float, ca: float) -> tuple[float, float]:
    """Matched affine coefficients for the assumed moments."""

    va, pa, cya = link_terms(m1a, ca)
    d = cya * cya - BETA**4 * pa * pa
    if d <= 0.0:
        raise ValueError("degenerate assumed moments")
    a = BETA * (va * cya - BETA**2 * pa * pa) / d
    b = BETA * pa * S / d
    return a, b


def mse(m1_true: float, c_true: float, m1a: float | None = None,
        ca: float | None = None) -> float:
    """Exact MSE of the affine estimator built for ``(m1a, ca)`` at the truth.

    With ``x = h - ell*m1a`` and ``e = beta*x + n``, the error is
    ``(a*beta - 1) x + b*beta x^* + a n + b n^*``, so the MSE is a quadratic form in
    ``(E|x|^2, E[x^2], s)``.  This form is valid for a mean mismatch as well.
    """

    m1a = m1_true if m1a is None else m1a
    ca = c_true if ca is None else ca
    v, p, _ = link_terms(m1_true, c_true)
    a, b = coefficients(m1a, ca)
    d = L * (m1_true - m1a)
    ex2 = v + d * d
    exx = p + d * d
    u = a * BETA - 1.0
    w = b * BETA
    return float((u * u + w * w) * ex2 + 2.0 * u * w * exx + (a * a + b * b) * S)


def oracle_gain(m1: float, c: float) -> tuple[float, float, float]:
    v, p, cy = link_terms(m1, c)
    mse_b2 = v * S / cy
    mse_b1 = mse(m1, c)
    return mse_b2, mse_b1, mse_b2 - mse_b1


def hessian(m1: float, c: float, h: float = 1e-4) -> dict:
    """Numerical Hessian of the true MSE in the assumed ``(m1a, ca)``."""

    f = lambda u, w: mse(m1, c, u, w)
    f00 = f(m1, c)
    f10 = f(m1 + h, c)
    f_10 = f(m1 - h, c)
    f01 = f(m1, c + h)
    f0_1 = f(m1, c - h)
    f11 = f(m1 + h, c + h)
    f1_1 = f(m1 + h, c - h)
    f_11 = f(m1 - h, c + h)
    f_1_1 = f(m1 - h, c - h)
    h11 = (f10 - 2.0 * f00 + f_10) / (h * h)
    h22 = (f01 - 2.0 * f00 + f0_1) / (h * h)
    h12 = (f11 - f1_1 - f_11 + f_1_1) / (4.0 * h * h)
    g1 = (f10 - f_10) / (2.0 * h)
    g2 = (f01 - f0_1) / (2.0 * h)
    return {
        "mse_at_truth": f00,
        "grad_m1": g1,
        "grad_c": g2,
        "h11": h11,
        "h22": h22,
        "h12": h12,
    }


def calibration_covariance(m1: float, c: float, frames: int,
                           diffuse_var: float = SIGMA_G2,
                           noise: float = S, replicates: int = 20000,
                           seed: int = 20261026, chunk: int = 2000) -> dict:
    """Monte Carlo covariance of the moment estimates ``(m1_hat, c_hat)``.

    The calibration window observes ``z_f = exp(j eps_f) + w_f`` with proper complex
    ``w_f ~ CN(0, sigma_w^2)``, which is the per-frame pilot observation of a reference link
    divided by ``beta * l``.
    """

    rng = np.random.default_rng(seed)
    sigma_w2 = (diffuse_var + noise / BETA**2) / (L * L)
    sigma_z2 = 1.0 - m1 * m1 + sigma_w2
    if abs(c) > sigma_z2:
        raise ValueError("pseudo-variance outside the circularity lens")
    # z = p*u + q*conj(u) with the SAME circular u gives E|z|^2 = p^2+q^2 and E[z^2] = 2pq.
    # Solving p^2+q^2 = sigma_z2 and 2pq = c yields
    #   p = (sqrt(sigma^2+c) + sqrt(sigma^2-c))/2,  q = (sqrt(sigma^2+c) - sqrt(sigma^2-c))/2.
    root_sum = math.sqrt(sigma_z2 + c)
    root_diff = math.sqrt(sigma_z2 - c)
    p = 0.5 * (root_sum + root_diff)
    q = 0.5 * (root_sum - root_diff)
    estimates = np.empty((replicates, 2))
    done = 0
    while done < replicates:
        block = min(chunk, replicates - done)
        u = (rng.standard_normal((block, frames))
             + 1j * rng.standard_normal((block, frames))) / math.sqrt(2.0)
        z = p * u + q * np.conj(u) + m1
        zbar = z.mean(axis=1)
        m1_hat = zbar.real
        centred = z - zbar[:, None]
        c_hat = np.real(np.sum(centred * centred, axis=1)) / max(frames - 1, 1)
        estimates[done:done + block, 0] = m1_hat
        estimates[done:done + block, 1] = c_hat
        done += block
        del z, u, centred
    cov_hat = np.cov(estimates, rowvar=False, ddof=1)
    return {
        "frames": frames,
        "replicates": replicates,
        "mean": estimates.mean(axis=0).tolist(),
        "bias": (estimates.mean(axis=0) - np.array([m1, c])).tolist(),
        "cov": cov_hat.tolist(),
        "se_m1": float(math.sqrt(cov_hat[0, 0])),
        "se_c": float(math.sqrt(cov_hat[1, 1])),
    }


def capture_curve(m1: float, c: float, cov: np.ndarray, draws: int = 200000,
                  seed: int = 7) -> dict:
    """Exact-risk capture fraction under Gaussian moment errors."""

    mse_b2, mse_b1, gain = oracle_gain(m1, c)
    rng = np.random.default_rng(seed)
    err = rng.multivariate_normal(np.zeros(2), cov, size=draws)
    m1a = m1 + err[:, 0]
    ca = c + err[:, 1]
    # Feasibility: clip to the lens and to c <= 0 (the physical sign for these laws).
    lens = 1.0 - np.clip(m1a, 0.0, 1.0) ** 2
    ca = np.clip(ca, -lens, 0.0)
    m1a = np.clip(m1a, 1e-6, 1.0 - 1e-9)
    risky = np.array([mse(m1, c, float(u), float(w)) for u, w in zip(m1a, ca)])
    captured = (mse_b2 - risky) / gain
    safe = np.maximum(captured, 0.0)
    return {
        "mean_capture": float(captured.mean()),
        "median_capture": float(np.median(captured)),
        "p05_capture": float(np.quantile(captured, 0.05)),
        "p95_capture": float(np.quantile(captured, 0.95)),
        "mean_capture_clipped": float(safe.mean()),
        "fraction_worse_than_b2": float(np.mean(captured < 0.0)),
        "mse_b2": mse_b2,
        "mse_b1": mse_b1,
        "gain": gain,
    }


def mse_ab(m1_true: float, c_true: float, m1a: float, a: float, b: float) -> float:
    """MSE of an arbitrary affine pair ``(a, b)`` built with assumed mean ``ell*m1a``."""

    v, p, _ = link_terms(m1_true, c_true)
    d = L * (m1_true - m1a)
    ex2 = v + d * d
    exx = p + d * d
    u = a * BETA - 1.0
    w = b * BETA
    return float((u * u + w * w) * ex2 + 2.0 * u * w * exx + (a * a + b * b) * S)


def shrinkage_capture(m1: float, c: float, cov: np.ndarray, draws: int = 50000,
                      seed: int = 11) -> dict:
    """Risk-estimated shrinkage between B2 and calibrated B1.

    The receiver forms ``h_lam = (1-lam) h_B2 + lam h_B1`` with the *same* calibrated mean, and
    chooses ``lam`` in ``[0, 1]`` to minimise the plug-in risk.  ``lam = 0`` is always available,
    so the plug-in risk never exceeds the plug-in B2 risk.
    """

    mse_b2, mse_b1, gain = oracle_gain(m1, c)
    rng = np.random.default_rng(seed)
    err = rng.multivariate_normal(np.zeros(2), cov, size=draws)
    m1a = np.clip(m1 + err[:, 0], 1e-6, 1.0 - 1e-9)
    lens = 1.0 - m1a * m1a
    ca = np.clip(c + err[:, 1], -lens, 0.0)

    lam = np.linspace(0.0, 1.0, 101)
    risky = np.empty(draws)
    oracle_risky = np.empty(draws)
    chosen = np.empty(draws)
    for i, (u, w) in enumerate(zip(m1a, ca)):
        a2, b2 = coefficients(float(u), 0.0)
        a1, b1 = coefficients(float(u), float(w))
        plug = np.array([
            mse_ab(float(u), float(w), float(u), (1 - t) * a2 + t * a1, t * b1)
            for t in lam
        ])
        t = float(lam[int(np.argmin(plug))])
        chosen[i] = t
        risky[i] = mse_ab(m1, c, float(u), (1 - t) * a2 + t * a1, t * b1)
        oracle_risky[i] = mse_ab(m1, c, float(u), a2, b2)
    captured = (mse_b2 - risky) / gain
    base = (mse_b2 - oracle_risky) / gain
    return {
        "mean_capture": float(captured.mean()),
        "p05_capture": float(np.quantile(captured, 0.05)),
        "fraction_worse_than_b2": float(np.mean(captured < 0.0)),
        "mean_b2_with_calibrated_mean": float(base.mean()),
        "mean_lambda": float(chosen.mean()),
    }


def minimax_shrinkage(m1: float, c: float, cov: np.ndarray, draws: int = 600,
                      seed: int = 13, alpha: float = 0.95) -> dict:
    """Shrinkage whose weight minimises the worst-case risk over a confidence ellipse.

    The receiver knows its own calibration covariance ``cov`` (from ``M`` and the sample
    moments), so it can form a level-``alpha`` ellipse around the estimate and choose ``lam`` to
    minimise the worst-case MSE over that ellipse.  ``lam = 0`` (B2) is always in the class.
    """

    from scipy.stats import chi2

    mse_b2, mse_b1, gain = oracle_gain(m1, c)
    rng = np.random.default_rng(seed)
    err = rng.multivariate_normal(np.zeros(2), cov, size=draws)
    lam = np.linspace(0.0, 1.0, 41)
    radius = math.sqrt(float(chi2.ppf(alpha, 2)))
    root = np.linalg.cholesky(cov + 1e-18 * np.eye(2))
    angles = np.linspace(0.0, 2.0 * math.pi, 25, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius

    captured = np.empty(draws)
    chosen = np.empty(draws)
    for i, (e1, e2) in enumerate(err):
        u = float(np.clip(m1 + e1, 1e-6, 1.0 - 1e-9))
        w = float(np.clip(c + e2, -(1.0 - u * u), 0.0))
        a2, b2 = coefficients(u, 0.0)
        a1, b1 = coefficients(u, w)
        pairs = [((1 - t) * a2 + t * a1, t * b1) for t in lam]
        worst = np.full(lam.size, -np.inf)
        for direction in unit:
            p1 = float(np.clip(u + (root[0, 0] * direction[0]), 1e-6, 1.0 - 1e-9))
            p2 = float(np.clip(w + (root[1, 0] * direction[0] + root[1, 1] * direction[1]),
                               -(1.0 - p1 * p1), 0.0))
            for j, (aa, bb) in enumerate(pairs):
                worst[j] = max(worst[j], mse_ab(p1, p2, u, aa, bb))
        j = int(np.argmin(worst))
        chosen[i] = lam[j]
        aa, bb = pairs[j]
        captured[i] = (mse_b2 - mse_ab(m1, c, u, aa, bb)) / gain
    return {
        "mean_capture": float(captured.mean()),
        "p05_capture": float(np.quantile(captured, 0.05)),
        "fraction_worse_than_b2": float(np.mean(captured < 0.0)),
        "mean_lambda": float(chosen.mean()),
        "alpha": alpha,
    }


def second_order_prediction(m1: float, c: float, cov: np.ndarray) -> dict:
    hess = hessian(m1, c)
    hmat = np.array([[hess["h11"], hess["h12"]], [hess["h12"], hess["h22"]]])
    excess = 0.5 * float(np.trace(hmat @ cov))
    _, _, gain = oracle_gain(m1, c)
    return {
        "hessian": hess,
        "excess_mse": excess,
        "predicted_capture": float(1.0 - excess / gain),
    }


def generator_check(m1: float, c: float, replicates: int = 200000,
                    seed: int = 4242) -> dict:
    """Verify that the observation generator carries the declared moments."""

    rng = np.random.default_rng(seed)
    sigma_w2 = (SIGMA_G2 + S / BETA**2) / (L * L)
    sigma_z2 = 1.0 - m1 * m1 + sigma_w2
    root_sum = math.sqrt(sigma_z2 + c)
    root_diff = math.sqrt(sigma_z2 - c)
    p = 0.5 * (root_sum + root_diff)
    q = 0.5 * (root_sum - root_diff)
    u = (rng.standard_normal(replicates) + 1j * rng.standard_normal(replicates)) / math.sqrt(2.0)
    z = p * u + q * np.conj(u) + m1
    return {
        "m1_empirical": float(z.mean().real),
        "m1_target": float(m1),
        "pseudo_empirical": float(np.mean((z - m1) ** 2).real),
        "pseudo_target": float(c),
        "variance_empirical": float(np.mean(np.abs(z - m1) ** 2)),
        "variance_target": float(sigma_z2),
    }


def main() -> None:
    payload = {"checkpoint": "CP2", "operating_point": {
        "ell": L, "beta": BETA, "noise": S, "kappa": KAPPA,
    }, "laws": {}}

    payload["generator_check"] = {}
    for name, (m1, c) in LAWS.items():
        check = generator_check(m1, c)
        payload["generator_check"][name] = check
        print(
            f"generator {name:17s} m1 {check['m1_empirical']:.6f}/{m1:.6f} "
            f"pseudo {check['pseudo_empirical']:+.6f}/{c:+.6f}"
        )

    frames_grid = (16, 64, 256, 1024, 4096, 16384)
    for name, (m1, c) in LAWS.items():
        entry = {"m1": m1, "c": c, "oracle": oracle_gain(m1, c), "curve": []}
        hess = hessian(m1, c)
        entry["hessian"] = hess
        entry["gradient_ratio"] = (
            abs(hess["grad_m1"]) / max(abs(hess["mse_at_truth"]), 1e-30),
            abs(hess["grad_c"]) / max(abs(hess["mse_at_truth"]), 1e-30),
        )
        for frames in frames_grid:
            cov_info = calibration_covariance(m1, c, frames, replicates=4000)
            cov = np.array(cov_info["cov"])
            curve = capture_curve(m1, c, cov)
            pred = second_order_prediction(m1, c, cov)
            safe = minimax_shrinkage(m1, c, cov, draws=400)
            entry["curve"].append({
                "frames": frames,
                "se_m1": cov_info["se_m1"],
                "se_c": cov_info["se_c"],
                "cov_m1_c": float(cov[0, 1]),
                "bias_m1": cov_info["bias"][0],
                "bias_c": cov_info["bias"][1],
                "mean_capture": curve["mean_capture"],
                "p05_capture": curve["p05_capture"],
                "fraction_worse_than_b2": curve["fraction_worse_than_b2"],
                "predicted_capture": pred["predicted_capture"],
                "excess_mse": pred["excess_mse"],
                "safe_capture": safe["mean_capture"],
                "safe_p05": safe["p05_capture"],
                "safe_fraction_worse": safe["fraction_worse_than_b2"],
                "safe_mean_lambda": safe["mean_lambda"],
            })
            print(
                f"{name:17s} M={frames:6d} se_m1={cov_info['se_m1']:.2e} "
                f"se_c={cov_info['se_c']:.2e} capture={curve['mean_capture']:.4f} "
                f"safe={safe['mean_capture']:.4f} lam={safe['mean_lambda']:.2f} "
                f"pred={pred['predicted_capture']:.4f} "
                f"worse={curve['fraction_worse_than_b2']:.4f}"
            )
        payload["laws"][name] = entry

    # Sample complexity: measured crossing of 99% capture (log-linear interpolation) and the
    # second-order prediction from the Hessian.
    summary = {}
    for name, entry in payload["laws"].items():
        m1, c = LAWS[name]
        _, _, gain = oracle_gain(m1, c)
        curve = sorted(entry["curve"], key=lambda r: r["frames"])
        measured = None
        for left, right in zip(curve, curve[1:]):
            if left["mean_capture"] < 0.99 <= right["mean_capture"]:
                t = (0.99 - left["mean_capture"]) / (right["mean_capture"] - left["mean_capture"])
                measured = math.exp(
                    math.log(left["frames"])
                    + t * (math.log(right["frames"]) - math.log(left["frames"]))
                )
                break
        hess = entry["hessian"]
        hmat = np.array([[hess["h11"], hess["h12"]], [hess["h12"], hess["h22"]]])
        anchor = [r for r in curve if r["frames"] == 1024][0]
        cov_anchor = np.array([
            [anchor["se_m1"] ** 2, anchor["cov_m1_c"]],
            [anchor["cov_m1_c"], anchor["se_c"] ** 2],
        ])
        model_excess = 0.5 * float(np.trace(hmat @ cov_anchor))
        model_m99 = 1024.0 * model_excess / (0.01 * gain)
        summary[name] = {
            "gain": gain,
            "frames_for_99pct_measured": measured,
            "frames_for_99pct_model": model_m99,
            "excess_at_1024_model": model_excess,
            "capture_at_1024": anchor["mean_capture"],
            "capture_at_4096": [r for r in curve if r["frames"] == 4096][0]["mean_capture"],
            "capture_at_16384": [r for r in curve if r["frames"] == 16384][0]["mean_capture"],
        }
    payload["summary"] = summary

    path = OUT / "cp2_calibration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
