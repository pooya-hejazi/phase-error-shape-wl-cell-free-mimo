"""Reference quadrature and controlled phase-graph samplers."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import logsumexp

from .diagnostics import ChainDiagnostics, summarize_chains
from .model import PhaseGraph
from .posterior import certified_random_scan_sweeps, dobrushin_eta, log_likelihood_phasors

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class ConditionalSamplingError(RuntimeError):
    """Raised when an exact one-site rejection draw exhausts its declared cap."""

    def __init__(
        self,
        message: str,
        *,
        logical_proposals: int | None = None,
        generated_candidate_values: int | None = None,
    ) -> None:
        super().__init__(message)
        self.logical_proposals = logical_proposals
        self.generated_candidate_values = generated_candidate_values


class GibbsSamplingError(ConditionalSamplingError):
    """Structured Gibbs failure retaining deterministic work provenance."""

    def __init__(self, message: str, *, audit: dict[str, Any]) -> None:
        super().__init__(
            message,
            logical_proposals=int(audit["logical_proposal_count"]),
            generated_candidate_values=int(audit["generated_candidate_value_count"]),
        )
        self.audit = audit


@dataclass(frozen=True)
class MomentEstimate:
    z: ComplexArray
    Z: ComplexArray
    method: str
    status: str
    sample_count: int
    proposal_count: int = 0
    acceptance_rate: float | None = None
    diagnostics: ChainDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retained_samples: ComplexArray | None = None


@dataclass(frozen=True)
class ForestQuadratureResult:
    """Singleton moments and edge moments for a finite-grid forest model."""

    z: ComplexArray
    edge_second_moments: dict[tuple[int, int], complex]
    order: int
    edge_count: int
    pruned_edge_count: int
    status: str = "DISCRETE_FOREST_SUM_PRODUCT_EXACT"


def phase_quadrature(delta: float, order: int) -> tuple[FloatArray, FloatArray]:
    """Nodes and normalized prior weights for a uniform bounded phase."""

    width = float(delta)
    if not 0.0 <= width <= np.pi:
        raise ValueError("delta must lie in [0, pi]")
    if order < 1:
        raise ValueError("order must be positive")
    if width == 0.0:
        return np.array([0.0]), np.array([1.0])
    nodes, weights = leggauss(int(order))
    return (width * nodes).astype(np.float64), (0.5 * weights).astype(np.float64)


def tensor_quadrature(
    graph: PhaseGraph, order: int, maximum_states: int = 1_000_000
) -> MomentEstimate:
    """Evaluate a product Gauss-Legendre rule exactly by enumeration.

    This is exact for the finite quadrature rule, not for the continuous
    posterior.  The caller must calibrate quadrature error separately.
    """

    rules = [phase_quadrature(float(width), order) for width in graph.delta]
    state_count = math.prod(len(nodes) for nodes, _ in rules)
    if state_count > int(maximum_states):
        raise ValueError(f"tensor rule has {state_count} states, above the declared cap")
    phases = np.empty((state_count, graph.group_size), dtype=np.float64)
    log_weights = np.empty(state_count, dtype=np.float64)
    for row, indices in enumerate(itertools.product(*(range(len(nodes)) for nodes, _ in rules))):
        prior_log_weight = 0.0
        for i, index in enumerate(indices):
            nodes, weights = rules[i]
            phases[row, i] = nodes[index]
            prior_log_weight += math.log(float(weights[index]))
        u = np.exp(1j * phases[row])
        log_weights[row] = prior_log_weight + log_likelihood_phasors(graph, u)
    normalized = np.exp(log_weights - logsumexp(log_weights))
    u_all = np.exp(1j * phases)
    z = normalized @ u_all
    Z = np.einsum("s,si,sj->ij", normalized, u_all, u_all.conj(), optimize=True)
    Z = 0.5 * (Z + Z.conj().T)
    np.fill_diagonal(Z, 1.0)
    return MomentEstimate(
        z=z.astype(np.complex128),
        Z=Z.astype(np.complex128),
        method="tensor_gauss_legendre",
        status="DISCRETE_QUADRATURE_EXACT",
        sample_count=state_count,
        metadata={"order": int(order), "state_count": int(state_count)},
    )


def forest_sum_product(
    graph: PhaseGraph, order: int, edge_tolerance: float = 1e-12
) -> ForestQuadratureResult:
    """Run log-domain sum-product when the nonzero interaction graph is a forest.

    The result is exact for the selected finite quadrature rule.  It returns
    singleton moments and moments on retained edges; it does not claim an
    ``O(P Q^2)`` construction of every entry of the full second-moment matrix.
    """

    if edge_tolerance < 0.0:
        raise ValueError("edge_tolerance must be nonnegative")
    p = graph.group_size
    rules = [phase_quadrature(float(width), order) for width in graph.delta]
    nodes = [np.exp(1j * rule[0]) for rule in rules]
    log_unary = [
        np.log(rule[1]) + 2.0 * np.real(np.conj(nodes[i]) * graph.c[i])
        for i, rule in enumerate(rules)
    ]
    neighbors: list[list[int]] = [[] for _ in range(p)]
    edges: list[tuple[int, int]] = []
    pruned_edge_count = 0
    parent = list(range(p))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> bool:
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        parent[rj] = ri
        return True

    for i in range(p):
        for j in range(i + 1, p):
            magnitude = abs(graph.G[i, j])
            if magnitude == 0.0:
                continue
            if magnitude <= float(edge_tolerance):
                pruned_edge_count += 1
                continue
            if not union(i, j):
                raise ValueError("the phase interaction graph contains a cycle")
            edges.append((i, j))
            neighbors[i].append(j)
            neighbors[j].append(i)

    pair_cache: dict[tuple[int, int], FloatArray] = {}

    def log_pair(i: int, j: int) -> FloatArray:
        key = (i, j)
        if key not in pair_cache:
            pair_cache[key] = -2.0 * np.real(
                np.conj(nodes[i])[:, np.newaxis]
                * graph.G[i, j]
                * nodes[j][np.newaxis, :]
            )
        return pair_cache[key]

    message_cache: dict[tuple[int, int], FloatArray] = {}

    def message(i: int, j: int) -> FloatArray:
        key = (i, j)
        if key in message_cache:
            return message_cache[key]
        base = log_unary[i].copy()
        for k in neighbors[i]:
            if k != j:
                base += message(k, i)
        pair = log_pair(i, j)
        out = logsumexp(base[:, np.newaxis] + pair, axis=0)
        out -= logsumexp(out)
        message_cache[key] = out
        return out

    z = np.empty(p, dtype=np.complex128)
    for i in range(p):
        belief = log_unary[i].copy()
        for k in neighbors[i]:
            belief += message(k, i)
        probability = np.exp(belief - logsumexp(belief))
        z[i] = probability @ nodes[i]

    edge_moments: dict[tuple[int, int], complex] = {}
    for i, j in edges:
        left = log_unary[i].copy()
        right = log_unary[j].copy()
        for k in neighbors[i]:
            if k != j:
                left += message(k, i)
        for k in neighbors[j]:
            if k != i:
                right += message(k, j)
        joint_log = left[:, np.newaxis] + right[np.newaxis, :] + log_pair(i, j)
        joint = np.exp(joint_log - logsumexp(joint_log))
        edge_moments[(i, j)] = complex(
            np.sum(joint * nodes[i][:, np.newaxis] * np.conj(nodes[j])[np.newaxis, :])
        )
    return ForestQuadratureResult(
        z=z,
        edge_second_moments=edge_moments,
        order=int(order),
        edge_count=len(edges),
        pruned_edge_count=pruned_edge_count,
        status=(
            "DISCRETE_FOREST_SUM_PRODUCT_EXACT"
            if pruned_edge_count == 0
            else "DISCRETE_PRUNED_FOREST_APPROXIMATION"
        ),
    )


def _arc_mode_cosine(mu: float, delta: float) -> float:
    lower, upper = -delta, delta
    two_pi = 2.0 * np.pi
    first = math.ceil((lower - mu) / two_pi)
    last = math.floor((upper - mu) / two_pi)
    if first <= last:
        return 1.0
    return max(math.cos(lower - mu), math.cos(upper - mu))


def sample_truncated_von_mises(
    q: complex,
    delta: float,
    rng: np.random.Generator,
    maximum_proposals: int = 100_000,
    batch_size: int = 256,
) -> tuple[float, int]:
    """Exact rejection draw from a bounded-uniform-prior one-site conditional.

    The proposal is uniform on the declared arc.  The return value includes the
    number of proposals so numerical failure cannot be hidden.
    """

    value, logical_proposals, _ = _sample_truncated_von_mises_accounted(
        q,
        delta,
        rng,
        maximum_proposals=maximum_proposals,
        batch_size=batch_size,
    )
    return value, logical_proposals


def _sample_truncated_von_mises_accounted(
    q: complex,
    delta: float,
    rng: np.random.Generator,
    maximum_proposals: int = 100_000,
    batch_size: int = 256,
) -> tuple[float, int, int]:
    """Return a draw, logical proposal prefix, and generated candidate count."""

    width = float(delta)
    if int(maximum_proposals) < 1 or int(batch_size) < 1:
        raise ValueError("proposal cap and batch_size must be positive")
    if width == 0.0:
        return 0.0, 0, 0
    if not 0.0 < width <= np.pi:
        raise ValueError("delta must lie in (0, pi]")
    kappa = 2.0 * abs(q)
    if kappa == 0.0:
        return float(rng.uniform(-width, width)), 1, 1
    mu = float(np.angle(q))
    maximum_cosine = _arc_mode_cosine(mu, width)
    proposals = 0
    generated = 0
    cap = int(maximum_proposals)
    while proposals < cap:
        count = min(int(batch_size), cap - proposals)
        candidates = rng.uniform(-width, width, size=count)
        generated += count
        log_accept = kappa * (np.cos(candidates - mu) - maximum_cosine)
        uniforms = rng.random(count)
        accepted = np.flatnonzero(np.log(uniforms) <= log_accept)
        if accepted.size:
            index = int(accepted[0])
            return float(candidates[index]), proposals + index + 1, generated
        proposals += count
    raise ConditionalSamplingError(
        f"truncated-von-Mises rejection exhausted {cap} proposals",
        logical_proposals=cap,
        generated_candidate_values=generated,
    )


def _random_scan_sweep(
    graph: PhaseGraph,
    phases: FloatArray,
    residual: ComplexArray,
    rng: np.random.Generator,
    maximum_conditional_proposals: int,
    conditional_batch_size: int,
) -> tuple[ComplexArray, int, int]:
    u = np.exp(1j * phases)
    proposals = 0
    generated = 0
    for _ in range(graph.group_size):
        i = int(rng.integers(0, graph.group_size))
        column = graph.A[:, i]
        q_i = np.vdot(column, residual + column * u[i])
        try:
            new_phase, used, created = _sample_truncated_von_mises_accounted(
                q_i,
                float(graph.delta[i]),
                rng,
                maximum_conditional_proposals,
                conditional_batch_size,
            )
        except ConditionalSamplingError as error:
            raise ConditionalSamplingError(
                str(error),
                logical_proposals=proposals + int(error.logical_proposals or 0),
                generated_candidate_values=(
                    generated + int(error.generated_candidate_values or 0)
                ),
            ) from error
        proposals += used
        generated += created
        new_u = complex(np.exp(1j * new_phase))
        residual += column * (u[i] - new_u)
        phases[i] = new_phase
        u[i] = new_u
    return residual, proposals, generated


def heat_bath_gibbs(
    graph: PhaseGraph,
    *,
    seed: int,
    chains: int,
    burn_sweeps: int,
    retained_draws: int,
    thinning_sweeps: int = 1,
    maximum_conditional_proposals: int = 100_000,
    conditional_batch_size: int = 256,
    return_samples: bool = False,
) -> MomentEstimate:
    """Run independent exact-conditional random-scan chains.

    Retained draws within a chain are correlated.  R-hat is empirical and is
    never represented as a mathematical accuracy certificate.
    """

    if not isinstance(return_samples, (bool, np.bool_)):
        raise ValueError("return_samples must be Boolean")
    if chains < 2 or burn_sweeps < 0 or retained_draws < 2 or thinning_sweeps < 1:
        raise ValueError("invalid Gibbs chain budget")
    if (
        not isinstance(conditional_batch_size, int)
        or isinstance(conditional_batch_size, (bool, np.bool_))
        or conditional_batch_size < 1
    ):
        raise ValueError("conditional_batch_size must be a positive integer")
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(int(chains))
    samples = np.empty((chains, retained_draws, graph.group_size), dtype=np.complex128)
    proposal_count = 0
    generated_candidate_value_count = 0
    per_chain_proposal_count = np.zeros(chains, dtype=np.int64)
    per_chain_generated_candidate_value_count = np.zeros(chains, dtype=np.int64)
    for chain, child in enumerate(child_sequences):
        rng = np.random.default_rng(child)
        phases = rng.uniform(-graph.delta, graph.delta).astype(np.float64)
        phases = np.where(graph.delta == 0.0, 0.0, phases)
        residual = (graph.x - graph.A @ np.exp(1j * phases)).astype(np.complex128)
        phase_label = "burn"
        sweep_index = -1
        draw_index = -1
        try:
            for sweep_index in range(int(burn_sweeps)):
                residual, used, generated = _random_scan_sweep(
                    graph,
                    phases,
                    residual,
                    rng,
                    maximum_conditional_proposals,
                    conditional_batch_size,
                )
                proposal_count += used
                generated_candidate_value_count += generated
                per_chain_proposal_count[chain] += used
                per_chain_generated_candidate_value_count[chain] += generated
            phase_label = "retained"
            for draw_index in range(int(retained_draws)):
                for sweep_index in range(int(thinning_sweeps)):
                    residual, used, generated = _random_scan_sweep(
                        graph,
                        phases,
                        residual,
                        rng,
                        maximum_conditional_proposals,
                        conditional_batch_size,
                    )
                    proposal_count += used
                    generated_candidate_value_count += generated
                    per_chain_proposal_count[chain] += used
                    per_chain_generated_candidate_value_count[chain] += generated
                samples[chain, draw_index] = np.exp(1j * phases)
        except ConditionalSamplingError as error:
            failed_logical = int(error.logical_proposals or 0)
            failed_generated = int(error.generated_candidate_values or 0)
            proposal_count += failed_logical
            generated_candidate_value_count += failed_generated
            per_chain_proposal_count[chain] += failed_logical
            per_chain_generated_candidate_value_count[chain] += failed_generated
            raise GibbsSamplingError(
                "Gibbs conditional sampler failed with structured provenance",
                audit={
                    "chain_index": int(chain),
                    "phase": phase_label,
                    "burn_or_thinning_sweep_index": int(sweep_index),
                    "retained_draw_index": (
                        None if phase_label == "burn" else int(draw_index)
                    ),
                    "completed_chain_count": int(chain),
                    "logical_proposal_count": int(proposal_count),
                    "generated_candidate_value_count": int(
                        generated_candidate_value_count
                    ),
                    "per_chain_logical_proposal_count": per_chain_proposal_count.tolist(),
                    "per_chain_generated_candidate_value_count": (
                        per_chain_generated_candidate_value_count.tolist()
                    ),
                    "maximum_conditional_proposals": int(
                        maximum_conditional_proposals
                    ),
                    "conditional_batch_size": int(conditional_batch_size),
                    "cause": str(error),
                },
            ) from error
    flat = samples.reshape((-1, graph.group_size))
    z = np.mean(flat, axis=0)
    Z = np.einsum("si,sj->ij", flat, flat.conj(), optimize=True) / flat.shape[0]
    Z = 0.5 * (Z + Z.conj().T)
    np.fill_diagonal(Z, 1.0)
    diagnostics = summarize_chains(samples)
    eta = dobrushin_eta(graph)
    certified_burn = certified_random_scan_sweeps(graph, 0.01)
    retained = samples if bool(return_samples) else None
    if retained is not None:
        retained.setflags(write=False)
    return MomentEstimate(
        z=z.astype(np.complex128),
        Z=Z.astype(np.complex128),
        method="exact_conditional_random_scan_gibbs",
        status="EMPIRICAL_MOMENT_ESTIMATE",
        sample_count=int(flat.shape[0]),
        proposal_count=int(proposal_count),
        diagnostics=diagnostics,
        metadata={
            "seed": int(seed),
            "chains": int(chains),
            "burn_sweeps": int(burn_sweeps),
            "retained_draws": int(retained_draws),
            "thinning_sweeps": int(thinning_sweeps),
            "maximum_conditional_proposals": int(maximum_conditional_proposals),
            "conditional_batch_size": int(conditional_batch_size),
            "logical_proposal_count": int(proposal_count),
            "generated_candidate_value_count": int(
                generated_candidate_value_count
            ),
            "per_chain_logical_proposal_count": per_chain_proposal_count.tolist(),
            "per_chain_generated_candidate_value_count": (
                per_chain_generated_candidate_value_count.tolist()
            ),
            "dobrushin_eta": eta,
            "tv_0p01_sufficient_burn_sweeps": certified_burn,
            "retained_samples_returned": bool(return_samples),
        },
        retained_samples=retained,
    )


def prior_rejection_reference(
    graph: PhaseGraph,
    *,
    seed: int,
    accepted_samples: int,
    maximum_proposals: int,
    batch_size: int = 4096,
) -> MomentEstimate:
    """Draw exact i.i.d. posterior samples using the prior as proposal.

    This method is retained as a small/reference tool after its locked
    source-scale acceptance screen failed.  The conservative envelope uses
    ``exp(-||x-Au||^2) <= 1``.
    """

    if int(accepted_samples) < 1:
        raise ValueError("accepted_samples must be positive")
    if int(maximum_proposals) < 1 or int(batch_size) < 1:
        raise ValueError("proposal cap and batch_size must be positive")

    rng = np.random.default_rng(int(seed))
    accepted: list[ComplexArray] = []
    proposals = 0
    while proposals < int(maximum_proposals) and len(accepted) < int(accepted_samples):
        count = min(int(batch_size), int(maximum_proposals) - proposals)
        phases = rng.uniform(-graph.delta, graph.delta, size=(count, graph.group_size))
        phases[:, graph.delta == 0.0] = 0.0
        u = np.exp(1j * phases)
        residual = graph.x[np.newaxis, :] - u @ graph.A.T
        log_accept = -np.sum(np.abs(residual) ** 2, axis=1)
        keep = np.flatnonzero(np.log(rng.random(count)) <= log_accept)
        needed = int(accepted_samples) - len(accepted)
        if keep.size >= needed:
            accepted.extend(u[keep[:needed]])
            proposals += int(keep[needed - 1]) + 1
            break
        accepted.extend(u[keep])
        proposals += count
    if len(accepted) < int(accepted_samples):
        raise ConditionalSamplingError(
            f"prior rejection accepted {len(accepted)}/{accepted_samples} after {proposals} proposals"
        )
    samples = np.asarray(accepted, dtype=np.complex128)
    z = np.mean(samples, axis=0)
    Z = np.einsum("si,sj->ij", samples, samples.conj(), optimize=True) / samples.shape[0]
    Z = 0.5 * (Z + Z.conj().T)
    np.fill_diagonal(Z, 1.0)
    return MomentEstimate(
        z=z,
        Z=Z,
        method="prior_rejection_reference",
        status="EXACT_IID_SAMPLES_REFERENCE_ONLY",
        sample_count=int(samples.shape[0]),
        proposal_count=int(proposals),
        acceptance_rate=float(samples.shape[0] / proposals),
        metadata={"production_eligible": False, "locked_source_scale_screen": "FAIL"},
    )
