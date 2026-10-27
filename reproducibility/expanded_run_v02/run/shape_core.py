"""Packet-local phase-error shapes and matched circular moments.

R2C01_DIRA_SHAPE_v08. This module is frozen with the packet and does not modify
``src/phase_graph_cme``. It implements the three natural shapes (uniform,
wrapped Gaussian, negative binary), the ``c = 0`` mixture of DERIVATION.md
section 4, and the fixed/proper structural controls.

Every shape is parameterized by a target ``m1 = E[cos(eps)]`` and returns the
matched ``m2 = E[cos(2 eps)]``. The phase draw uses one uniform variate per
link; the caller keeps that variate shared across shapes so the diffuse and
noise draws can be common random numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SHAPES = ("uniform", "wrapped_gaussian", "binary", "zero_c", "fixed", "proper")


@dataclass(frozen=True)
class ShapeSpec:
    """One matched phase-error law."""

    shape: str
    target_m1: float
    m1: float
    m2: float
    parameter: dict[str, Any]

    @property
    def c(self) -> float:
        return float(self.m2 - self.m1 * self.m1)


def _uniform_delta_rad(m1: float) -> float:
    """Solve ``sinc(delta/pi) = m1`` on ``[0, pi]``; sinc is strictly decreasing there."""

    target = float(m1)
    if not 0.0 <= target <= 1.0:
        raise ValueError("m1 must lie in [0, 1]")
    if target == 1.0:
        return 0.0
    if target == 0.0:
        return math.pi
    low, high = 0.0, math.pi
    for _ in range(200):
        middle = 0.5 * (low + high)
        value = math.sin(middle) / middle
        if value > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def shape_spec(shape: str, target_m1: float) -> ShapeSpec:
    """Return the matched moments and the draw parameters for one shape."""

    if shape not in SHAPES:
        raise ValueError(f"unsupported shape {shape!r}")
    target = float(target_m1)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target m1 must lie in [0, 1]")

    if shape == "uniform":
        delta = _uniform_delta_rad(target)
        m2 = 1.0 if delta == 0.0 else math.sin(2.0 * delta) / (2.0 * delta)
        return ShapeSpec(
            shape=shape,
            target_m1=target,
            m1=target,
            m2=float(m2),
            parameter={
                "delta_rad": float(delta),
                "delta_deg": float(math.degrees(delta)),
                "support": "[-delta, delta]",
            },
        )

    if shape == "wrapped_gaussian":
        if target == 0.0:
            raise ValueError("wrapped Gaussian cannot realize m1 = 0")
        sigma_squared = -2.0 * math.log(target)
        sigma = math.sqrt(sigma_squared)
        return ShapeSpec(
            shape=shape,
            target_m1=target,
            m1=target,
            m2=float(target**4),
            parameter={
                "sigma_squared": float(sigma_squared),
                "sigma_rad": float(sigma),
                "sigma_deg": float(math.degrees(sigma)),
                "support": "wrapped normal on [-pi, pi)",
            },
        )

    if shape == "binary":
        delta = math.acos(min(max(target, 0.0), 1.0))
        return ShapeSpec(
            shape=shape,
            target_m1=target,
            m1=target,
            m2=float(2.0 * target * target - 1.0),
            parameter={
                "delta_rad": float(delta),
                "delta_deg": float(math.degrees(delta)),
                "support": "{-delta, +delta}",
            },
        )

    if shape == "zero_c":
        delta = math.acos(min(max(target, 0.0), 1.0))
        return ShapeSpec(
            shape=shape,
            target_m1=target,
            m1=target,
            m2=float(target * target),
            parameter={
                "delta_rad": float(delta),
                "delta_deg": float(math.degrees(delta)),
                "support": "50% constant +/-delta, 50% {0, pi} mixture",
            },
        )

    if shape == "fixed":
        if target != 1.0:
            raise ValueError("fixed shape requires m1 = 1")
        return ShapeSpec(
            shape=shape, target_m1=1.0, m1=1.0, m2=1.0,
            parameter={"support": "{0}"},
        )

    # proper: the LoS term is absent (kappa = 0); the phase law is irrelevant.
    if target != 0.0:
        raise ValueError("proper shape requires m1 = 0")
    return ShapeSpec(
        shape=shape, target_m1=0.0, m1=0.0, m2=0.0,
        parameter={"support": "irrelevant (no LoS)"},
    )


def draw_phases(shape: str, spec: ShapeSpec, uniform: FloatArray) -> FloatArray:
    """Map one shared uniform field to a phase field.

    ``uniform`` has shape ``(access_points, users)`` and values in ``[0, 1)``.
    The mapping consumes no random numbers beyond the field, so callers that
    reuse the same field across shapes also reuse the diffuse and noise draws.
    """

    u = np.asarray(uniform, dtype=np.float64)
    if u.ndim != 2 or np.any(u < 0.0) or np.any(u >= 1.0):
        raise ValueError("uniform field must be 2-D with values in [0, 1)")
    if shape != spec.shape:
        raise ValueError("shape and spec disagree")

    if shape in {"fixed", "proper"}:
        return np.zeros_like(u)

    if shape == "uniform":
        delta = float(spec.parameter["delta_rad"])
        return (2.0 * u - 1.0) * delta

    if shape == "wrapped_gaussian":
        from scipy.special import ndtri  # imported after the driver's pinned-version checks

        sigma = float(spec.parameter["sigma_rad"])
        # ndtri(0) is -inf; clip to the open interval for a finite phase.
        safe = np.clip(u, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
        raw = sigma * ndtri(safe)
        return np.angle(np.exp(1j * raw)).astype(np.float64)

    if shape == "binary":
        delta = float(spec.parameter["delta_rad"])
        return np.where(u >= 0.5, delta, -delta).astype(np.float64)

    # zero_c: 50% constant +/-delta, 50% {0, pi} with P(0) = (1 + m1) / 2.
    delta = float(spec.parameter["delta_rad"])
    phases = np.empty_like(u)
    constant = u < 0.5
    phases[constant] = np.where(u[constant] < 0.25, -delta, delta)
    rest = ~constant
    threshold = 0.75 + 0.25 * float(spec.m1)
    phases[rest] = np.where(u[rest] < threshold, 0.0, math.pi)
    return phases.astype(np.float64)


def empirical_moments(phases: FloatArray) -> tuple[float, float]:
    """Return the realized first and second circular moments."""

    values = np.asarray(phases, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("phases must be finite and nonempty")
    m1 = float(np.mean(np.cos(values)))
    m2 = float(np.mean(np.cos(2.0 * values)))
    return m1, m2


def moment_standard_errors(phases: FloatArray) -> tuple[float, float]:
    """Return the Monte Carlo standard errors of the two moment estimates."""

    values = np.asarray(phases, dtype=np.float64)
    n = values.size
    if n < 2:
        raise ValueError("at least two phases are required")
    x1 = np.cos(values)
    x2 = np.cos(2.0 * values)
    se1 = float(np.std(x1, ddof=1) / math.sqrt(n))
    se2 = float(np.std(x2, ddof=1) / math.sqrt(n))
    return se1, se2


def moment_check(
    phases: FloatArray,
    spec: ShapeSpec,
    sigma: float,
    atol: float,
) -> dict:
    """Check the realized moments against the design with a numerical floor.

    The binary law is degenerate: ``cos(eps)`` and ``cos(2 eps)`` are constant
    across all draws.  A floating-point ``np.std`` of such a field is tiny but
    not exactly zero, so the raw z-score can be enormous even though the
    realized moment matches the design to machine precision.  The absolute
    floor ``atol`` removes that artifact without loosening the check for
    non-degenerate shapes: the effective standard error is
    ``max(se, atol / sigma)``, so ``sigma * se_eff >= atol``.  With
    ``atol = 1e-12`` the binary cells pass, while a real moment error of
    ``1e-3`` still fails by many orders of magnitude.
    """

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    m1_emp, m2_emp = empirical_moments(phases)
    m1_se, m2_se = moment_standard_errors(phases)
    floor = atol / sigma
    m1_se_eff = max(m1_se, floor)
    m2_se_eff = max(m2_se, floor)
    z1 = (m1_emp - float(spec.m1)) / m1_se_eff
    z2 = (m2_emp - float(spec.m2)) / m2_se_eff
    return {
        "m1_emp": float(m1_emp),
        "m2_emp": float(m2_emp),
        "m1_se": float(m1_se),
        "m2_se": float(m2_se),
        "m1_se_effective": float(m1_se_eff),
        "m2_se_effective": float(m2_se_eff),
        "moment_z1": float(z1),
        "moment_z2": float(z2),
        "moment_atol": float(atol),
        "moment_sigma": float(sigma),
        "moment_pass": bool(abs(z1) <= sigma and abs(z2) <= sigma),
    }


def moment_unit_checks() -> list[dict]:
    """Independently integrate the three natural laws and the c = 0 mixture.

    This is a unit check of ``shape_spec``, not a system result. It runs once
    per execution, after the pinned imports, and is logged in the output.
    """

    from scipy.integrate import quad
    from scipy.stats import norm

    checks: list[dict] = []
    for shape in ("uniform", "wrapped_gaussian", "binary", "zero_c"):
        for target in (0.99, 0.90, 0.70):
            spec = shape_spec(shape, target)
            if shape == "uniform":
                delta = float(spec.parameter["delta_rad"])

                def density(x: float, width: float = delta) -> float:
                    return 1.0 / (2.0 * width) if abs(x) <= width else 0.0

                m1_int = quad(
                    lambda x: math.cos(x) * density(x), -delta, delta, limit=200
                )[0]
                m2_int = quad(
                    lambda x: math.cos(2.0 * x) * density(x),
                    -delta,
                    delta,
                    limit=200,
                )[0]
            elif shape == "wrapped_gaussian":
                sigma = float(spec.parameter["sigma_rad"])

                def density(x: float, scale: float = sigma) -> float:
                    return float(
                        sum(
                            norm.pdf(x + 2.0 * math.pi * k, loc=0.0, scale=scale)
                            for k in range(-8, 9)
                        )
                    )

                m1_int = quad(
                    lambda x: math.cos(x) * density(x), -math.pi, math.pi, limit=400
                )[0]
                m2_int = quad(
                    lambda x: math.cos(2.0 * x) * density(x),
                    -math.pi,
                    math.pi,
                    limit=400,
                )[0]
            elif shape == "binary":
                delta = float(spec.parameter["delta_rad"])
                m1_int = math.cos(delta)
                m2_int = math.cos(2.0 * delta)
            else:
                # zero_c: 1/4 at -delta, 1/4 at +delta, (1+m1)/4 at 0,
                # (1-m1)/4 at pi. The atom weights are exact by construction.
                delta = float(spec.parameter["delta_rad"])
                m1_int = 0.5 * math.cos(delta) + 0.5 * float(spec.m1)
                m2_int = 0.5 * math.cos(2.0 * delta) + 0.5
            error1 = abs(float(m1_int) - float(spec.m1))
            error2 = abs(float(m2_int) - float(spec.m2))
            checks.append(
                {
                    "shape": shape,
                    "target_m1": target,
                    "m1_spec": float(spec.m1),
                    "m2_spec": float(spec.m2),
                    "m1_integrated": float(m1_int),
                    "m2_integrated": float(m2_int),
                    "abs_error_m1": error1,
                    "abs_error_m2": error2,
                    "tolerance": 1e-8,
                    "pass": bool(error1 <= 1e-8 and error2 <= 1e-8),
                }
            )
    return checks
