"""Bounded circular mean-field COMPARATOR; not a new inference claim.

Coordinate ascent is on a finite quadrature grid. Convergence and agreement
between two grids are diagnostics, not continuous-posterior error bounds.
This module is additive: it does not alter the qualified V3 controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .baselines import widely_linear_mmse
from .model import PhaseGraph, PilotGroupModel, build_phase_graph
from .moments import channel_moments


STARTS = ("prior", "positive_sine", "negative_sine")


@dataclass(frozen=True)
class MeanFieldConfig:
    coarse_order: int = 64
    fine_order: int = 128
    max_sweeps: int = 200
    mean_tolerance: float = 1e-9
    grid_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        for name in ("coarse_order", "fine_order", "max_sweeps"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if not 2 <= self.coarse_order < self.fine_order:
            raise ValueError("require 2 <= coarse_order < fine_order")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be positive")
        for name in ("mean_tolerance", "grid_tolerance"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MeanFieldFit:
    z: np.ndarray
    marginals: tuple[np.ndarray, ...]
    objective_history: tuple[float, ...]
    converged: bool
    fixed_point_residual: float
    sweeps: int
    coordinate_updates: int
    order: int
    start: str


@dataclass(frozen=True)
class MeanFieldGroupResult:
    estimates: np.ndarray  # (N, P), shared inference for all group targets
    phase_mean: np.ndarray | None  # None on B1 fallback: never fake phase moments
    status: str
    fallback_reason: str | None
    grid_discrepancy: float
    coarse: MeanFieldFit
    fine: MeanFieldFit
    all_fits: tuple[MeanFieldFit, ...]
    fine_start_mean_spread: float
    fine_objective_gap: float
    total_sweeps: int
    total_coordinate_updates: int
    elapsed_seconds: float
    method: str = "BOUNDED_MF_VB_B1_FALLBACK_01"


def _validate_graph(graph: PhaseGraph) -> None:
    if graph.A.ndim != 2 or min(graph.A.shape) < 1:
        raise ValueError("A must have nonempty shape (N,P)")
    n, p = graph.A.shape
    if graph.x.shape != (n,) or graph.c.shape != (p,) or graph.G.shape != (p, p):
        raise ValueError("inconsistent sufficient-statistic shapes")
    if graph.delta.shape != (p,):
        raise ValueError("delta must have shape (P,)")
    if any(not np.isfinite(v).all() for v in (graph.A, graph.x, graph.c, graph.G, graph.delta)):
        raise ValueError("finite model statistics required")
    if np.any(graph.delta < 0) or np.any(graph.delta > np.pi):
        raise ValueError("delta outside [0,pi]")
    if not np.allclose(graph.G, graph.A.conj().T @ graph.A, rtol=1e-11, atol=1e-12):
        raise ValueError("G is inconsistent with A")
    if not np.allclose(graph.c, graph.A.conj().T @ graph.x, rtol=1e-11, atol=1e-12):
        raise ValueError("c is inconsistent with A and x")


def _rules(delta: np.ndarray, order: int):
    nodes, weights = leggauss(order)
    weights = weights / np.sum(weights)
    result = []
    for width in delta:
        theta = nodes * width if width else np.array([0.0])
        prior = weights.copy() if width else np.array([1.0])
        result.append((np.exp(1j * theta), np.log(prior)))
    return result


def _mass(log_values: np.ndarray) -> np.ndarray:
    if not np.isfinite(log_values).all():
        raise FloatingPointError("nonfinite variational logits")
    result = np.exp(log_values - logsumexp(log_values))
    # This normalizes roundoff only; it is not a probability/PSD projection.
    result /= np.sum(result)
    return result


def _objective(graph, means, marginals, rules) -> float:
    value = 2 * np.vdot(means, graph.c).real - np.vdot(means, graph.G @ means).real
    value -= np.dot(np.diag(graph.G).real, 1 - np.abs(means) ** 2)
    for q, (_, log_prior) in zip(marginals, rules):
        positive = q > 0
        value -= np.sum(q[positive] * (np.log(q[positive]) - log_prior[positive]))
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite finite-grid variational objective")
    return float(value)


def fit_mean_field(
    graph: PhaseGraph,
    *,
    order: int = 128,
    max_sweeps: int = 200,
    tolerance: float = 1e-9,
    start: str = "prior",
) -> MeanFieldFit:
    """One deterministic coordinate-ascent run; no global-optimum claim."""
    _validate_graph(graph)
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("order must be an integer >=2")
    if isinstance(max_sweeps, (bool, np.bool_)) or not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if start not in STARTS:
        raise ValueError("unknown deterministic initialization")
    rules = _rules(graph.delta, order)
    tilt = {"prior": 0.0, "positive_sine": 2.0, "negative_sine": -2.0}[start]
    q = [_mass(logw + tilt * u.imag) for u, logw in rules]
    means = np.array([mass @ rule[0] for mass, rule in zip(q, rules)], dtype=complex)
    history = [_objective(graph, means, q, rules)]
    updates = 0
    converged = False
    residual = float("inf")
    for sweep in range(1, max_sweeps + 1):
        old = means.copy()
        for i, (u, logw) in enumerate(rules):
            if graph.delta[i] == 0:
                continue
            eta = graph.c[i] - graph.G[i] @ means + graph.G[i, i] * means[i]
            q[i] = _mass(logw + 2 * np.real(u.conj() * eta))
            means[i] = q[i] @ u
            updates += 1
        value = _objective(graph, means, q, rules)
        if value < history[-1] - 1e-11 * (1 + abs(history[-1])):
            raise FloatingPointError("finite-grid coordinate objective decreased")
        history.append(value)
        residual = 0.0
        for i, (u, logw) in enumerate(rules):
            eta = graph.c[i] - graph.G[i] @ means + graph.G[i, i] * means[i]
            optimal_mean = _mass(logw + 2 * np.real(u.conj() * eta)) @ u
            residual = max(residual, float(abs(optimal_mean - means[i])))
        if np.max(np.abs(means - old)) <= tolerance and residual <= tolerance:
            converged = True
            break
    return MeanFieldFit(
        z=means.copy(), marginals=tuple(m.copy() for m in q),
        objective_history=tuple(history), converged=converged,
        fixed_point_residual=residual, sweeps=sweep, coordinate_updates=updates,
        order=int(order), start=start,
    )


def mean_field_group_estimate(
    model: PilotGroupModel, observation: object, config: MeanFieldConfig | None = None
) -> MeanFieldGroupResult:
    """Full-cost dual-grid, multistart comparator with explicit B1 fallback.

    Unexpected numerical/resource errors propagate. Only declared convergence
    or grid-agreement failures produce an ordinary affine fallback.
    """
    started = perf_counter()
    cfg = config if config is not None else MeanFieldConfig()
    if not isinstance(cfg, MeanFieldConfig):
        raise TypeError("config must be MeanFieldConfig")
    graph = build_phase_graph(model, observation)
    fits = tuple(
        fit_mean_field(graph, order=order, max_sweeps=cfg.max_sweeps,
                       tolerance=cfg.mean_tolerance, start=start)
        for order in (cfg.coarse_order, cfg.fine_order) for start in STARTS
    )
    coarse = max(fits[:len(STARTS)], key=lambda v: v.objective_history[-1])
    fine = max(fits[len(STARTS):], key=lambda v: v.objective_history[-1])
    discrepancy = float(np.max(np.abs(coarse.z - fine.z)))
    fine_fits = fits[len(STARTS):]
    start_spread = max(float(np.max(np.abs(a.z - b.z)))
                       for a in fine_fits for b in fine_fits)
    fine_objectives = sorted((v.objective_history[-1] for v in fine_fits), reverse=True)
    reason = None
    if not coarse.converged or not fine.converged:
        reason = "SELECTED_SOLUTION_NONCONVERGENCE"
    elif discrepancy > cfg.grid_tolerance:
        reason = "CROSS_ORDER_DISAGREEMENT"
    if reason is None:
        estimates = np.column_stack([
            channel_moments(model, observation, k, fine.z).mean
            for k in range(model.group_size)
        ])
        status = "MF_ACCEPTED_DIAGNOSTICALLY_NOT_CERTIFIED"
        phase_mean = fine.z.copy()
    else:
        estimates = np.column_stack([
            widely_linear_mmse(model, observation, k)
            for k in range(model.group_size)
        ])
        status = "B1_FALLBACK"
        phase_mean = None
    if not np.isfinite(estimates).all():
        raise FloatingPointError("nonfinite group channel estimate")
    return MeanFieldGroupResult(
        estimates=estimates, phase_mean=phase_mean, status=status,
        fallback_reason=reason, grid_discrepancy=discrepancy,
        coarse=coarse, fine=fine, all_fits=fits,
        fine_start_mean_spread=start_spread,
        fine_objective_gap=fine_objectives[0] - fine_objectives[1],
        total_sweeps=sum(v.sweeps for v in fits),
        total_coordinate_updates=sum(v.coordinate_updates for v in fits),
        elapsed_seconds=perf_counter() - started,
    )
