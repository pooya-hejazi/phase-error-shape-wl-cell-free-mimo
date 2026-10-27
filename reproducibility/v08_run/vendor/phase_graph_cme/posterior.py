"""Phase-posterior density and conditional-dependence diagnostics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .model import PhaseGraph

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def phasors(phases: object) -> ComplexArray:
    return np.exp(1j * np.asarray(phases, dtype=np.float64)).astype(np.complex128)


def log_likelihood_phasors(graph: PhaseGraph, u: object) -> float:
    """Return ``-||x-Au||^2``; the omitted Gaussian constant cancels."""

    values = np.asarray(u, dtype=np.complex128)
    if values.shape != (graph.group_size,):
        raise ValueError("phasor vector has the wrong shape")
    residual = graph.x - graph.A @ values
    return -float(np.vdot(residual, residual).real)


def expanded_log_kernel(graph: PhaseGraph, u: object) -> float:
    """Return the posterior log kernel excluding prior and ``-||x||^2``."""

    values = np.asarray(u, dtype=np.complex128)
    return float(2.0 * np.vdot(values, graph.c).real - np.vdot(values, graph.G @ values).real)


def log_posterior_kernel(graph: PhaseGraph, phases: object) -> float:
    angles = np.asarray(phases, dtype=np.float64)
    if angles.shape != (graph.group_size,):
        raise ValueError("phase vector has the wrong shape")
    if np.any(np.abs(angles) > graph.delta + 1e-13):
        return -math.inf
    return log_likelihood_phasors(graph, phasors(angles))


def arc_diameter_factors(delta: object) -> FloatArray:
    widths = np.asarray(delta, dtype=np.float64)
    return np.where(widths <= np.pi / 2.0, np.sin(widths), 1.0)


def dobrushin_envelope(graph: PhaseGraph) -> FloatArray:
    """Return the independently audited symmetric influence upper bound."""

    s = arc_diameter_factors(graph.delta)
    envelope = np.tanh(2.0 * np.outer(s, s) * np.abs(graph.G))
    np.fill_diagonal(envelope, 0.0)
    return envelope.astype(np.float64)


def dobrushin_eta(graph: PhaseGraph) -> float:
    return float(np.max(np.sum(dobrushin_envelope(graph), axis=1), initial=0.0))


def certified_random_scan_sweeps(
    graph: PhaseGraph, total_variation_tolerance: float
) -> int | None:
    """Sufficient full-sweep count, or ``None`` when the certificate is absent."""

    tolerance = float(total_variation_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("total_variation_tolerance must lie in (0,1)")
    eta = dobrushin_eta(graph)
    if eta >= 1.0:
        return None
    numerator = math.log(graph.group_size / tolerance)
    return max(0, math.ceil(numerator / (1.0 - eta)))
