"""Physical phase-error laws and the shape-ordering lens (CP1).

Every law is parameterized by the physical quantity that produces it, not by ``m1``:

* ``wrapped_gaussian`` - residual of a Wiener (Lorentzian) oscillator after tracking with
  delay ``tau``: ``eps ~ N(0, sigma^2)`` with ``sigma^2 = 2*pi*beta*tau``.
* ``von_mises`` - steady-state phase error of a first-order PLL with loop SNR ``rho``:
  ``eps ~ Tikhonov(kappa)`` with ``kappa = 2*rho``.
* ``uniform`` - maximum-entropy law on a bounded residual interval ``[-delta, delta]``.
* ``binary`` - two-state residual (cycle slip / BPSK phase ambiguity), the lens boundary.

Run from the repository root:

    $env:PYTHONDONTWRITEBYTECODE = "1"
    .venv\\Scripts\\python.exe -B experiments\\upgrade\\phase_laws.py
    Remove-Item Env:\\PYTHONDONTWRITEBYTECODE
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results"
OUT.mkdir(parents=True, exist_ok=True)

M1_GRID = (0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30)


def sinc(x: float) -> float:
    return 1.0 if x == 0.0 else math.sin(x) / x


def uniform_moments(delta: float) -> tuple[float, float]:
    return sinc(delta), sinc(2.0 * delta)


def uniform_delta_for_m1(m1: float) -> float:
    """Solve ``sinc(delta) = m1`` on ``[0, pi]`` (strictly decreasing)."""
    if m1 <= 0.0:
        return math.pi
    if m1 >= 1.0:
        return 0.0
    return brentq(lambda d: sinc(d) - m1, 0.0, math.pi, xtol=1e-14, rtol=1e-15)


def wrapped_gaussian_moments(sigma: float) -> tuple[float, float]:
    return math.exp(-0.5 * sigma**2), math.exp(-2.0 * sigma**2)


def wg_sigma_for_m1(m1: float) -> float:
    if m1 <= 0.0:
        raise ValueError("wrapped Gaussian cannot realize m1 <= 0")
    return math.sqrt(-2.0 * math.log(m1))


def von_mises_moments(kappa: float) -> tuple[float, float]:
    if kappa <= 0.0:
        return 0.0, 0.0
    # exponentially scaled Bessel functions avoid overflow for large kappa
    return float(ive(1, kappa) / ive(0, kappa)), float(ive(2, kappa) / ive(0, kappa))


def vm_kappa_for_m1(m1: float) -> float:
    if m1 <= 0.0:
        return 0.0
    if m1 >= 1.0:
        return math.inf
    low, high = 1e-9, 1.0
    while von_mises_moments(high)[0] < m1:
        high *= 2.0
    return brentq(lambda k: von_mises_moments(k)[0] - m1, low, high, xtol=1e-13, rtol=1e-15)


def binary_moments(delta: float) -> tuple[float, float]:
    return math.cos(delta), math.cos(2.0 * delta)


def binary_delta_for_m1(m1: float) -> float:
    return math.acos(min(max(m1, 0.0), 1.0))


def c_of(m1: float, m2: float) -> float:
    return m2 - m1 * m1


def lens_bound(m1: float) -> float:
    return 1.0 - m1 * m1


def law_rows(m1: float) -> dict:
    """All four laws matched to ``m1`` with their physical parameters."""

    sigma = wg_sigma_for_m1(m1)
    kappa = vm_kappa_for_m1(m1)
    delta_u = uniform_delta_for_m1(m1)
    delta_b = binary_delta_for_m1(m1)
    rows = {}
    for name, (m2, params) in {
        "binary": (binary_moments(delta_b)[1], {"delta_rad": delta_b}),
        "uniform": (uniform_moments(delta_u)[1], {"delta_rad": delta_u}),
        "wrapped_gaussian": (wrapped_gaussian_moments(sigma)[1], {"sigma_rad": sigma}),
        "von_mises": (von_mises_moments(kappa)[1], {"kappa": kappa}),
    }.items():
        rows[name] = {
            "m1": m1,
            "m2": m2,
            "c": c_of(m1, m2),
            "abs_c": abs(c_of(m1, m2)),
            "abs_c_over_lens": abs(c_of(m1, m2)) / lens_bound(m1),
            **params,
        }
    return rows


def ordering_grid(samples: int = 200001) -> dict:
    """Verify ``|c_B| >= |c_U| >= |c_G|`` on a dense grid."""

    grid = np.linspace(1e-6, 1.0 - 1e-6, samples)
    mins = {"B_minus_U": math.inf, "U_minus_G": math.inf}
    argmins = {"B_minus_U": None, "U_minus_G": None}
    worst = {"B_minus_U": 0.0, "U_minus_G": 0.0}
    for m1 in grid:
        sigma = math.sqrt(-2.0 * math.log(m1))
        delta_u = uniform_delta_for_m1(float(m1))
        cb = 1.0 - m1 * m1
        cu = m1 * m1 - sinc(2.0 * delta_u)
        cg = m1 * m1 * (1.0 - m1 * m1)
        d_bu = cb - cu
        d_ug = cu - cg
        if d_bu < mins["B_minus_U"]:
            mins["B_minus_U"] = d_bu
            argmins["B_minus_U"] = float(m1)
        if d_ug < mins["U_minus_G"]:
            mins["U_minus_G"] = d_ug
            argmins["U_minus_G"] = float(m1)
        worst["B_minus_U"] = max(worst["B_minus_U"], abs(d_bu))
        worst["U_minus_G"] = max(worst["U_minus_G"], abs(d_ug))
        del sigma
    return {
        "samples": samples,
        "min_B_minus_U": mins["B_minus_U"],
        "argmin_B_minus_U": argmins["B_minus_U"],
        "min_U_minus_G": mins["U_minus_G"],
        "argmin_U_minus_G": argmins["U_minus_G"],
        "pass": bool(mins["B_minus_U"] >= -1e-12 and mins["U_minus_G"] >= -1e-12),
    }


def sinc_inequality_grid(samples: int = 200001) -> dict:
    """Check ``sinc(delta)^3 >= cos(delta)``, the reduced form of ``|c_U| >= |c_G|``."""

    deltas = np.linspace(1e-9, math.pi - 1e-9, samples)
    values = np.array([sinc(float(d)) ** 3 - math.cos(float(d)) for d in deltas])
    return {
        "samples": samples,
        "min_value": float(values.min()),
        "argmin_delta": float(deltas[int(values.argmin())]),
        "pass": bool(values.min() >= -1e-12),
    }


def von_mises_order(samples: int = 20001) -> dict:
    """Check ``|c_vM| < |c_G|`` on a grid; this is numerical evidence, not a theorem."""

    grid = np.linspace(1e-4, 0.999, samples)
    worst = (math.inf, None)
    for m1 in grid:
        cg = m1 * m1 * (1.0 - m1 * m1)
        cv = abs(von_mises_moments(vm_kappa_for_m1(float(m1)))[1] - m1 * m1)
        if cg - cv < worst[0]:
            worst = (cg - cv, float(m1))
    return {
        "samples": samples,
        "minimum_gap_cG_minus_cvM": float(worst[0]),
        "argmin_m1": worst[1],
        "pass": bool(worst[0] > 0.0),
        "note": "numerical grid; no analytic proof is claimed",
    }


def physical_table() -> list[dict]:
    rows = []
    for m1 in M1_GRID:
        laws = law_rows(m1)
        sigma = laws["wrapped_gaussian"]["sigma_rad"]
        kappa = laws["von_mises"]["kappa"]
        rows.append(
            {
                "m1": m1,
                "lens": lens_bound(m1),
                "binary": {"delta_deg": math.degrees(laws["binary"]["delta_rad"]),
                           "c": laws["binary"]["c"]},
                "uniform": {"delta_deg": math.degrees(laws["uniform"]["delta_rad"]),
                            "c": laws["uniform"]["c"]},
                "wrapped_gaussian": {
                    "sigma_deg": math.degrees(sigma),
                    "beta_tau": sigma**2 / (2.0 * math.pi),
                    "c": laws["wrapped_gaussian"]["c"],
                },
                "von_mises": {
                    "kappa": kappa,
                    "loop_snr_db": 10.0 * math.log10(max(kappa, 1e-12) / 2.0)
                    if kappa > 0
                    else None,
                    "c": laws["von_mises"]["c"],
                    "abs_c_over_lens": laws["von_mises"]["abs_c_over_lens"],
                },
                "ordering": {
                    "abs_c_B": laws["binary"]["abs_c"],
                    "abs_c_U": laws["uniform"]["abs_c"],
                    "abs_c_G": laws["wrapped_gaussian"]["abs_c"],
                    "abs_c_vM": laws["von_mises"]["abs_c"],
                },
            }
        )
    return rows


def main() -> None:
    table = physical_table()
    ordering = ordering_grid()
    sinc_check = sinc_inequality_grid()
    payload = {
        "checkpoint": "CP1",
        "lens_bound": "|c| <= 1 - m1^2",
        "physical_table": table,
        "ordering_grid": ordering,
        "sinc_reduction_check": sinc_check,
        "von_mises_order": von_mises_order(),
        "notes": [
            "wrapped_gaussian: sigma^2 = 2*pi*beta*tau, m1 = exp(-sigma^2/2)",
            "von_mises: kappa = 2*rho_loop, m1 = I1(kappa)/I0(kappa)",
            "binary attains the lens boundary by Theorem 1",
            "uniform is the maximum-entropy law on [-delta, delta]",
        ],
    }
    path = OUT / "cp1_phase_laws.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "ordering_pass": ordering["pass"],
            "min_B_minus_U": ordering["min_B_minus_U"],
            "min_U_minus_G": ordering["min_U_minus_G"],
            "sinc_pass": sinc_check["pass"],
            "sinc_min": sinc_check["min_value"],
            "wrote": str(path.relative_to(ROOT)),
        },
        indent=2,
    ))
    for row in table:
        print(
            f"m1={row['m1']:.2f} |c|B={row['ordering']['abs_c_B']:.6f} "
            f"|c|U={row['ordering']['abs_c_U']:.6f} |c|G={row['ordering']['abs_c_G']:.6f} "
            f"|c|vM={row['ordering']['abs_c_vM']:.6f} "
            f"sigma={row['wrapped_gaussian']['sigma_deg']:.2f}deg "
            f"kappa={row['von_mises']['kappa']:.3f}"
        )


if __name__ == "__main__":
    main()
