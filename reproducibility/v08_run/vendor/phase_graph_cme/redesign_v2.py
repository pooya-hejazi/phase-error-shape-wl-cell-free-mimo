"""Prospective numerical methods for the bounded R2C01 V2 redesign.

The physical model and posterior are unchanged.  The two methods here are
finite numerical approximations with explicit diagnostics:

* replica-exchange exact-conditional Gibbs (the proposed controller); and
* adaptive-tempering resample-move SMC with an independently coded
  reflective random-walk kernel (the cross-method reference).

Neither method is described as an exact oracle.  Exactness below refers only
to invariance of an individual transition kernel for the declared continuous
posterior, not to finite-run convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm, rankdata

from .model import PhaseGraph, PilotGroupModel
from .moments import channel_transfer


@dataclass(frozen=True)
class ReplicaExchangeResult:
    retained_phasors: np.ndarray
    inverse_temperatures: np.ndarray
    logical_proposal_count: int
    generated_candidate_value_count: int
    burn_swap_attempts_by_edge: np.ndarray
    burn_swap_accepts_by_edge: np.ndarray
    burn_swap_acceptance_rates_by_edge: np.ndarray
    post_burn_swap_attempts_by_edge: np.ndarray
    post_burn_swap_accepts_by_edge: np.ndarray
    post_burn_swap_acceptance_rates_by_edge: np.ndarray
    burn_round_trips_by_ensemble: np.ndarray
    post_burn_round_trips_by_ensemble: np.ndarray
    post_burn_initial_labels: np.ndarray
    post_burn_label_history: np.ndarray
    residual_drift_path: np.ndarray
    residual_audit_phasors: np.ndarray
    residual_audit_incremental_residuals: np.ndarray
    residual_audit_ensemble_indices: np.ndarray
    residual_audit_cycle_indices: np.ndarray
    residual_audit_temperature_indices: np.ndarray
    maximum_residual_drift: float
    residual_reconstruction_count: int
    metadata: dict[str, Any]

    @property
    def swap_attempts_by_edge(self) -> np.ndarray:
        """Qualification-facing alias for post-burn swap attempts."""

        return self.post_burn_swap_attempts_by_edge

    @property
    def swap_accepts_by_edge(self) -> np.ndarray:
        """Qualification-facing alias for post-burn swap accepts."""

        return self.post_burn_swap_accepts_by_edge

    @property
    def swap_acceptance_rates_by_edge(self) -> np.ndarray:
        """Qualification-facing alias for post-burn swap rates."""

        return self.post_burn_swap_acceptance_rates_by_edge

    @property
    def round_trips_by_ensemble(self) -> np.ndarray:
        """Qualification-facing alias for wholly post-burn round trips."""

        return self.post_burn_round_trips_by_ensemble


@dataclass(frozen=True)
class SMCResult:
    particles: np.ndarray
    inverse_temperature_path: np.ndarray
    conditional_ess_path: np.ndarray
    pre_resampling_log_likelihoods: np.ndarray
    systematic_resampling_offsets: np.ndarray
    local_acceptance_path: np.ndarray
    global_acceptance_path: np.ndarray
    unique_initial_ancestor_count: int
    final_initial_ancestor_labels: np.ndarray
    local_accepts_by_stage: np.ndarray
    local_attempts_by_stage: np.ndarray
    global_accepts_by_stage: np.ndarray
    global_attempts_by_stage: np.ndarray
    residual_drift_path: np.ndarray
    stage_audit_phasors: np.ndarray
    stage_audit_incremental_residuals: np.ndarray
    stage_audit_particle_indices: np.ndarray
    maximum_residual_drift: float
    final_residual_consistency_error: float
    final_incremental_residuals: np.ndarray
    residual_reconstruction_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FunctionalDiagnostics:
    maximum_rank_folded_split_rhat: float
    minimum_bulk_effective_sample_size: float
    maximum_prior_rms_normalized_mcse: float
    maximum_prior_rms_normalized_chain_disagreement: float
    all_finite: bool


def _validate_inverse_temperatures(values: Sequence[float]) -> np.ndarray:
    # ``np.asarray`` can alias a caller-owned ndarray.  The result is frozen
    # before it is returned, so take ownership here instead of accidentally
    # making the caller's array read-only.
    betas = np.array(values, dtype=np.float64, copy=True)
    if betas.shape == (1,) and betas[0] == 1.0:
        return betas
    if (
        betas.ndim != 1
        or betas.size < 3
        or not np.all(np.isfinite(betas))
        or betas[0] != 0.0
        or betas[-1] != 1.0
        or np.any(np.diff(betas) <= 0.0)
    ):
        raise ValueError("inverse temperatures must strictly increase from 0 to 1")
    return betas


def _safe_log_uniform(values: object) -> np.ndarray:
    """Return ``log(U)`` for U in [0, 1), including representable U=0."""

    uniforms = np.asarray(values, dtype=np.float64)
    if (
        not np.all(np.isfinite(uniforms))
        or np.any(uniforms < 0.0)
        or np.any(uniforms >= 1.0)
    ):
        raise ValueError("uniform variates must be finite and lie in [0, 1)")
    floor = np.nextafter(np.float64(0.0), np.float64(1.0))
    return np.log(np.maximum(uniforms, floor))


def _swap_acceptance_rates(accepts: np.ndarray, attempts: np.ndarray) -> np.ndarray:
    """Compute per-edge rates, using -1 for an edge with no attempts."""

    rates = np.full(attempts.shape, -1.0, dtype=np.float64)
    np.divide(accepts, attempts, out=rates, where=attempts > 0)
    return rates


def _new_round_trip_tracker(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a tracker at a phase boundary from the current ladder."""

    hot_seen = np.zeros(labels.size, dtype=bool)
    cold_after_hot = np.zeros(labels.size, dtype=bool)
    if labels.size > 1:
        hot_seen[int(labels[0])] = True
    return hot_seen, cold_after_hot


def _update_round_trip_tracker(
    labels: np.ndarray, hot_seen: np.ndarray, cold_after_hot: np.ndarray
) -> int:
    """Update label transport state and return newly completed round trips."""

    if labels.size <= 1:
        return 0
    hot_label = int(labels[0])
    cold_label = int(labels[-1])
    hot_seen[hot_label] = True
    if hot_seen[cold_label]:
        cold_after_hot[cold_label] = True
    if cold_after_hot[hot_label]:
        cold_after_hot[hot_label] = False
        return 1
    return 0


def reconstruct_post_burn_transport(
    initial_labels: np.ndarray,
    label_history: np.ndarray,
    *,
    burn_cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct post-burn swap attempts, accepts, and round trips.

    ``initial_labels`` has shape ``(ensembles, temperatures)`` at the exact
    burn boundary. ``label_history`` records the complete ladder after every
    subsequent post-burn cycle, including cycles skipped by thinning.  The
    global cycle parity therefore starts at ``burn_cycles % 2``.
    """

    initial = np.asarray(initial_labels)
    history = np.asarray(label_history)
    if initial.ndim != 2 or history.ndim != 3:
        raise ValueError("transport labels must have rank two and three")
    if history.shape[0] != initial.shape[0] or history.shape[2] != initial.shape[1]:
        raise ValueError("transport label shapes do not agree")
    if not np.issubdtype(initial.dtype, np.integer) or not np.issubdtype(
        history.dtype, np.integer
    ):
        raise ValueError("transport labels must have integer dtype")
    if int(burn_cycles) < 0:
        raise ValueError("burn_cycles must be nonnegative")
    ensemble_count, temperature_count = initial.shape
    expected_labels = np.arange(temperature_count, dtype=np.int64)
    attempts = np.zeros(max(temperature_count - 1, 0), dtype=np.int64)
    accepts = np.zeros_like(attempts)
    round_trips = np.zeros(ensemble_count, dtype=np.int64)
    if temperature_count == 1:
        round_trips.fill(-1)

    for ensemble in range(ensemble_count):
        previous = initial[ensemble].astype(np.int64, copy=True)
        if not np.array_equal(np.sort(previous), expected_labels):
            raise ValueError("initial ladder labels must be a permutation")
        hot_seen, cold_after_hot = _new_round_trip_tracker(previous)
        for post_cycle in range(history.shape[1]):
            current = history[ensemble, post_cycle].astype(np.int64, copy=False)
            if not np.array_equal(np.sort(current), expected_labels):
                raise ValueError("each ladder history row must be a permutation")
            expected = previous.copy()
            parity = (int(burn_cycles) + post_cycle) % 2
            for edge in range(parity, temperature_count - 1, 2):
                attempts[edge] += 1
                unchanged = (
                    current[edge] == previous[edge]
                    and current[edge + 1] == previous[edge + 1]
                )
                swapped = (
                    current[edge] == previous[edge + 1]
                    and current[edge + 1] == previous[edge]
                )
                if swapped:
                    accepts[edge] += 1
                    expected[edge], expected[edge + 1] = (
                        expected[edge + 1],
                        expected[edge],
                    )
                elif not unchanged:
                    raise ValueError("history contains a non-adjacent label transition")
            if not np.array_equal(current, expected):
                raise ValueError("history changed a label outside attempted swap edges")
            round_trips[ensemble] += _update_round_trip_tracker(
                current, hot_seen, cold_after_hot
            )
            previous = current.copy()
    return attempts, accepts, round_trips


def _arc_mode_cosine_vector(mu: np.ndarray, delta: float) -> np.ndarray:
    if delta == np.pi:
        return np.ones_like(mu)
    inside = np.abs(mu) <= delta
    boundary = np.maximum(np.cos(-delta - mu), np.cos(delta - mu))
    return np.where(inside, 1.0, boundary)


def _sample_truncated_von_mises_vectorized(
    q: np.ndarray,
    delta: float,
    rng: np.random.Generator,
    *,
    maximum_proposals: int,
    batch_size: int,
    resource_guard: Callable[[], None] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Exact rejection draws for a vector of bounded von-Mises conditionals."""

    values = np.asarray(q, dtype=np.complex128)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("q must be a finite vector")
    if not 0.0 <= float(delta) <= np.pi:
        raise ValueError("delta must lie in [0, pi]")
    if int(maximum_proposals) < 1 or int(batch_size) < 1:
        raise ValueError("proposal cap and batch size must be positive")
    count = values.size
    if delta == 0.0:
        return np.zeros(count, dtype=np.float64), 0, 0
    result = np.empty(count, dtype=np.float64)
    logical = np.zeros(count, dtype=np.int64)
    generated = np.zeros(count, dtype=np.int64)
    magnitude = 2.0 * np.abs(values)
    zero = magnitude == 0.0
    if np.any(zero):
        result[zero] = rng.uniform(-delta, delta, size=int(np.sum(zero)))
        logical[zero] = 1
        generated[zero] = 1
    active = ~zero
    mu = np.angle(values)
    maximum_cosine = _arc_mode_cosine_vector(mu, float(delta))
    cap = int(maximum_proposals)
    batch = int(batch_size)
    while np.any(active):
        if resource_guard is not None:
            resource_guard()
        indices = np.flatnonzero(active)
        remaining = cap - logical[indices]
        width = int(min(batch, int(np.min(remaining))))
        if width <= 0:
            raise RuntimeError("truncated-von-Mises proposal cap exhausted")
        candidates = rng.uniform(-delta, delta, size=(indices.size, width))
        uniforms = rng.random((indices.size, width))
        log_accept = magnitude[indices, np.newaxis] * (
            np.cos(candidates - mu[indices, np.newaxis])
            - maximum_cosine[indices, np.newaxis]
        )
        accepted_matrix = _safe_log_uniform(uniforms) <= log_accept
        accepted_any = np.any(accepted_matrix, axis=1)
        generated[indices] += width
        if np.any(accepted_any):
            accepted_rows = np.flatnonzero(accepted_any)
            first = np.argmax(accepted_matrix[accepted_rows], axis=1)
            accepted_indices = indices[accepted_rows]
            result[accepted_indices] = candidates[accepted_rows, first]
            logical[accepted_indices] += first + 1
            active[accepted_indices] = False
        rejected_indices = indices[~accepted_any]
        logical[rejected_indices] += width
        if np.any(active & (logical >= cap)):
            raise RuntimeError("truncated-von-Mises proposal cap exhausted")
    return result, int(np.sum(logical)), int(np.sum(generated))


def replica_exchange_gibbs(
    graph: PhaseGraph,
    *,
    seed: int,
    inverse_temperatures: Sequence[float],
    ensembles: int,
    burn_cycles: int,
    retained_draws: int,
    thinning_cycles: int = 1,
    sweeps_per_cycle: int = 1,
    maximum_conditional_proposals: int = 100_000,
    conditional_batch_size: int = 32,
    residual_recompute_interval_cycles: int = 32,
    resource_guard: Callable[[], None] | None = None,
) -> ReplicaExchangeResult:
    """Run independent replica-exchange ladders with exact heat-bath moves.

    Each fixed-beta coordinate transition is an exact conditional draw for
    ``prior(u) * likelihood(u)**beta``.  Adjacent swaps use the exact
    Metropolis ratio.  Finite output still requires convergence diagnostics.
    """

    betas = _validate_inverse_temperatures(inverse_temperatures)
    if (
        int(ensembles) < 2
        or int(burn_cycles) < 0
        or int(retained_draws) < 4
        or int(thinning_cycles) < 1
        or int(sweeps_per_cycle) < 1
        or int(residual_recompute_interval_cycles) < 1
    ):
        raise ValueError("invalid replica-exchange budget")
    if not np.all(np.isfinite(graph.A)) or not np.all(np.isfinite(graph.x)):
        raise ValueError("phase graph must be finite")
    ensemble_count = int(ensembles)
    temperature_count = int(betas.size)
    variable_count = int(graph.group_size)
    child_sequences = np.random.SeedSequence(int(seed)).spawn(ensemble_count)
    samples = np.empty(
        (ensemble_count, int(retained_draws), variable_count), dtype=np.complex128
    )
    burn_swap_attempts = np.zeros(temperature_count - 1, dtype=np.int64)
    burn_swap_accepts = np.zeros(temperature_count - 1, dtype=np.int64)
    post_burn_swap_attempts = np.zeros(temperature_count - 1, dtype=np.int64)
    post_burn_swap_accepts = np.zeros(temperature_count - 1, dtype=np.int64)
    burn_round_trips = np.zeros(ensemble_count, dtype=np.int64)
    post_burn_round_trips = np.zeros(ensemble_count, dtype=np.int64)
    total_logical = 0
    total_generated = 0
    maximum_residual_drift = 0.0
    residual_drift_path: list[float] = []
    residual_audit_phasors: list[np.ndarray] = []
    residual_audit_incremental_residuals: list[np.ndarray] = []
    residual_audit_ensemble_indices: list[int] = []
    residual_audit_cycle_indices: list[int] = []
    residual_audit_temperature_indices: list[int] = []
    total_cycles = int(burn_cycles) + int(retained_draws) * int(thinning_cycles)
    post_burn_cycle_count = int(retained_draws) * int(thinning_cycles)
    post_burn_initial_labels = np.full(
        (ensemble_count, temperature_count), -1, dtype=np.int32
    )
    post_burn_label_history = np.full(
        (ensemble_count, post_burn_cycle_count, temperature_count),
        -1,
        dtype=np.int32,
    )

    for ensemble, child in enumerate(child_sequences):
        rng = np.random.default_rng(child)
        phases = rng.uniform(
            -graph.delta[np.newaxis, :],
            graph.delta[np.newaxis, :],
            size=(temperature_count, variable_count),
        ).astype(np.float64)
        phases[:, graph.delta == 0.0] = 0.0
        phasors = np.exp(1j * phases)
        residual = graph.x[np.newaxis, :] - phasors @ graph.A.T
        labels = np.arange(temperature_count, dtype=np.int64)
        burn_hot_seen: np.ndarray | None = None
        burn_cold_after_hot: np.ndarray | None = None
        post_burn_hot_seen: np.ndarray | None = None
        post_burn_cold_after_hot: np.ndarray | None = None
        if int(burn_cycles) > 0:
            burn_hot_seen, burn_cold_after_hot = _new_round_trip_tracker(labels)
        else:
            post_burn_hot_seen, post_burn_cold_after_hot = _new_round_trip_tracker(
                labels
            )
            post_burn_initial_labels[ensemble] = labels
        retained_index = 0
        for cycle in range(total_cycles):
            if resource_guard is not None:
                resource_guard()
            # Reset at the exact burn boundary.  A qualification round trip
            # may use the label's boundary position, but no earlier hot/cold
            # visit or partial trip.
            if cycle == int(burn_cycles) and post_burn_hot_seen is None:
                post_burn_hot_seen, post_burn_cold_after_hot = _new_round_trip_tracker(
                    labels
                )
                post_burn_initial_labels[ensemble] = labels
            is_post_burn = cycle >= int(burn_cycles)
            for sweep in range(int(sweeps_per_cycle)):
                # A rotating deterministic scan updates every coordinate and
                # avoids the with-replacement omissions present in V1.
                offset = (cycle * int(sweeps_per_cycle) + sweep) % variable_count
                for step in range(variable_count):
                    index = (offset + step) % variable_count
                    column = graph.A[:, index]
                    old = phasors[:, index].copy()
                    q = residual @ column.conj() + np.vdot(column, column).real * old
                    new_phase, logical, generated = (
                        _sample_truncated_von_mises_vectorized(
                            betas * q,
                            float(graph.delta[index]),
                            rng,
                            maximum_proposals=int(maximum_conditional_proposals),
                            batch_size=int(conditional_batch_size),
                            resource_guard=resource_guard,
                        )
                    )
                    total_logical += logical
                    total_generated += generated
                    new = np.exp(1j * new_phase)
                    residual += (old - new)[:, np.newaxis] * column[np.newaxis, :]
                    phases[:, index] = new_phase
                    phasors[:, index] = new

            if (cycle + 1) % int(residual_recompute_interval_cycles) == 0:
                exact_residual = graph.x[np.newaxis, :] - phasors @ graph.A.T
                drift_by_temperature = np.max(
                    np.abs(exact_residual - residual), axis=1, initial=0.0
                )
                audit_temperature = int(np.argmax(drift_by_temperature))
                event_drift = float(drift_by_temperature[audit_temperature])
                if not np.isfinite(event_drift):
                    raise FloatingPointError(
                        "replica-exchange residual drift is nonfinite"
                    )
                residual_drift_path.append(event_drift)
                residual_audit_phasors.append(phasors[audit_temperature].copy())
                residual_audit_incremental_residuals.append(
                    residual[audit_temperature].copy()
                )
                residual_audit_ensemble_indices.append(ensemble)
                residual_audit_cycle_indices.append(cycle)
                residual_audit_temperature_indices.append(audit_temperature)
                maximum_residual_drift = max(maximum_residual_drift, event_drift)
                residual = exact_residual

            log_likelihood = -np.sum(np.abs(residual) ** 2, axis=1)
            # One alternating parity per cycle avoids a deterministic
            # two-edge ping-pong when adjacent swaps have high acceptance.
            parity = cycle % 2
            for edge in range(parity, temperature_count - 1, 2):
                if is_post_burn:
                    post_burn_swap_attempts[edge] += 1
                else:
                    burn_swap_attempts[edge] += 1
                log_ratio = (betas[edge + 1] - betas[edge]) * (
                    log_likelihood[edge] - log_likelihood[edge + 1]
                )
                if float(_safe_log_uniform(rng.random())) <= min(0.0, float(log_ratio)):
                    for array in (phases, phasors, residual):
                        temporary = array[edge].copy()
                        array[edge] = array[edge + 1]
                        array[edge + 1] = temporary
                    log_likelihood[edge], log_likelihood[edge + 1] = (
                        log_likelihood[edge + 1],
                        log_likelihood[edge],
                    )
                    labels[edge], labels[edge + 1] = labels[edge + 1], labels[edge]
                    if is_post_burn:
                        post_burn_swap_accepts[edge] += 1
                    else:
                        burn_swap_accepts[edge] += 1

            if temperature_count > 1:
                if is_post_burn:
                    if post_burn_hot_seen is None or post_burn_cold_after_hot is None:
                        raise RuntimeError("post-burn round-trip tracker is absent")
                    post_burn_round_trips[ensemble] += _update_round_trip_tracker(
                        labels, post_burn_hot_seen, post_burn_cold_after_hot
                    )
                else:
                    if burn_hot_seen is None or burn_cold_after_hot is None:
                        raise RuntimeError("burn round-trip tracker is absent")
                    burn_round_trips[ensemble] += _update_round_trip_tracker(
                        labels, burn_hot_seen, burn_cold_after_hot
                    )

            if is_post_burn:
                post_burn_index = cycle - int(burn_cycles)
                post_burn_label_history[ensemble, post_burn_index] = labels

            if cycle >= int(burn_cycles) and (
                (cycle - int(burn_cycles) + 1) % int(thinning_cycles) == 0
            ):
                samples[ensemble, retained_index] = phasors[-1]
                retained_index += 1
        if retained_index != int(retained_draws):
            raise RuntimeError("replica-exchange retention count mismatch")

    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("replica-exchange returned nonfinite samples")
    if np.any(post_burn_initial_labels < 0) or np.any(post_burn_label_history < 0):
        raise RuntimeError("post-burn label history is incomplete")
    if temperature_count == 1:
        burn_round_trips.fill(-1)
        post_burn_round_trips.fill(-1)
    reconstructed_attempts, reconstructed_accepts, reconstructed_round_trips = (
        reconstruct_post_burn_transport(
            post_burn_initial_labels,
            post_burn_label_history,
            burn_cycles=int(burn_cycles),
        )
    )
    if (
        not np.array_equal(reconstructed_attempts, post_burn_swap_attempts)
        or not np.array_equal(reconstructed_accepts, post_burn_swap_accepts)
        or not np.array_equal(reconstructed_round_trips, post_burn_round_trips)
    ):
        raise RuntimeError("post-burn transport reconstruction mismatch")
    burn_swap_rates = _swap_acceptance_rates(burn_swap_accepts, burn_swap_attempts)
    post_burn_swap_rates = _swap_acceptance_rates(
        post_burn_swap_accepts, post_burn_swap_attempts
    )
    residual_drift_array = np.asarray(residual_drift_path, dtype=np.float64)
    residual_audit_phasor_array = np.asarray(
        residual_audit_phasors, dtype=np.complex128
    ).reshape(-1, variable_count)
    residual_audit_incremental_residual_array = np.asarray(
        residual_audit_incremental_residuals, dtype=np.complex128
    ).reshape(-1, graph.x.size)
    residual_audit_ensemble_index_array = np.asarray(
        residual_audit_ensemble_indices, dtype=np.int32
    )
    residual_audit_cycle_index_array = np.asarray(
        residual_audit_cycle_indices, dtype=np.int32
    )
    residual_audit_temperature_index_array = np.asarray(
        residual_audit_temperature_indices, dtype=np.int32
    )
    samples.setflags(write=False)
    betas.setflags(write=False)
    for array in (
        burn_swap_attempts,
        burn_swap_accepts,
        burn_swap_rates,
        post_burn_swap_attempts,
        post_burn_swap_accepts,
        post_burn_swap_rates,
        burn_round_trips,
        post_burn_round_trips,
        post_burn_initial_labels,
        post_burn_label_history,
        residual_drift_array,
        residual_audit_phasor_array,
        residual_audit_incremental_residual_array,
        residual_audit_ensemble_index_array,
        residual_audit_cycle_index_array,
        residual_audit_temperature_index_array,
    ):
        array.setflags(write=False)
    return ReplicaExchangeResult(
        retained_phasors=samples,
        inverse_temperatures=betas,
        logical_proposal_count=int(total_logical),
        generated_candidate_value_count=int(total_generated),
        burn_swap_attempts_by_edge=burn_swap_attempts,
        burn_swap_accepts_by_edge=burn_swap_accepts,
        burn_swap_acceptance_rates_by_edge=burn_swap_rates,
        post_burn_swap_attempts_by_edge=post_burn_swap_attempts,
        post_burn_swap_accepts_by_edge=post_burn_swap_accepts,
        post_burn_swap_acceptance_rates_by_edge=post_burn_swap_rates,
        burn_round_trips_by_ensemble=burn_round_trips,
        post_burn_round_trips_by_ensemble=post_burn_round_trips,
        post_burn_initial_labels=post_burn_initial_labels,
        post_burn_label_history=post_burn_label_history,
        residual_drift_path=residual_drift_array,
        residual_audit_phasors=residual_audit_phasor_array,
        residual_audit_incremental_residuals=(
            residual_audit_incremental_residual_array
        ),
        residual_audit_ensemble_indices=residual_audit_ensemble_index_array,
        residual_audit_cycle_indices=residual_audit_cycle_index_array,
        residual_audit_temperature_indices=(residual_audit_temperature_index_array),
        maximum_residual_drift=maximum_residual_drift,
        residual_reconstruction_count=len(residual_drift_array),
        metadata={
            "method": "replica_exchange_exact_conditional_gibbs",
            "seed": int(seed),
            "ensembles": ensemble_count,
            "temperatures": temperature_count,
            "burn_cycles": int(burn_cycles),
            "retained_draws": int(retained_draws),
            "thinning_cycles": int(thinning_cycles),
            "sweeps_per_cycle": int(sweeps_per_cycle),
            "maximum_conditional_proposals": int(maximum_conditional_proposals),
            "conditional_batch_size": int(conditional_batch_size),
            "residual_recompute_interval_cycles": int(
                residual_recompute_interval_cycles
            ),
            "maximum_residual_drift": maximum_residual_drift,
            "residual_reconstruction_count": len(residual_drift_array),
            "residual_audit_selection": (
                "FIRST_TEMPERATURE_WITH_EVENT_MAXIMUM_ABSOLUTE_RESIDUAL_DRIFT"
            ),
            "qualification_swap_accounting": "POST_BURN_ONLY",
            "burn_swap_accounting": "SEPARATE_NOT_QUALIFICATION_FACING",
            "post_burn_label_history_timing": "AFTER_EVERY_POST_BURN_SWAP_CYCLE",
            "swap_rate_no_attempt_sentinel": -1.0,
            "round_trip_tracking_reset_at_burn_boundary": True,
            "transition_invariance": "EXACT_PER_KERNEL_FINITE_RUN_NOT_CERTIFIED",
        },
    )


def _systematic_resample(
    weights: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    count = weights.size
    offset = float(rng.random())
    positions = (offset + np.arange(count, dtype=np.float64)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right"), offset


def _reflect_to_bounds(values: np.ndarray, half_width: float) -> np.ndarray:
    if half_width == 0.0:
        return np.zeros_like(values)
    period = 4.0 * half_width
    shifted = np.mod(values + half_width, period)
    reflected = np.where(
        shifted <= 2.0 * half_width,
        shifted - half_width,
        3.0 * half_width - shifted,
    )
    return reflected.astype(np.float64)


def _conditional_ess(log_increment: np.ndarray) -> float:
    shifted = log_increment - float(np.max(log_increment))
    weights = np.exp(shifted)
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def adaptive_tempering_smc(
    graph: PhaseGraph,
    *,
    seed: int,
    particles: int,
    conditional_ess_fraction: float,
    local_sweeps_per_stage: int,
    proposal_scale_fraction: float,
    global_prior_proposals_per_stage: int = 1,
    maximum_stages: int = 256,
    resource_guard: Callable[[], None] | None = None,
) -> SMCResult:
    """Adaptive-tempering SMC with independent reflective-MH rejuvenation.

    This implementation intentionally does not call the heat-bath conditional
    sampler.  Its local proposals are symmetric reflected Gaussian random
    walks and its global proposals are independent prior draws, both corrected
    by the exact likelihood-ratio Metropolis rule at the current beta.
    """

    count = int(particles)
    if count < 64:
        raise ValueError("SMC requires at least 64 particles")
    if not 0.1 <= float(conditional_ess_fraction) < 1.0:
        raise ValueError("conditional ESS fraction must lie in [0.1, 1)")
    if (
        int(local_sweeps_per_stage) < 1
        or not 0.0 < float(proposal_scale_fraction) <= 2.0
    ):
        raise ValueError("invalid SMC local-move configuration")
    if int(global_prior_proposals_per_stage) < 0 or int(maximum_stages) < 2:
        raise ValueError("invalid SMC stage configuration")
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(-graph.delta, graph.delta, size=(count, graph.group_size))
    phases[:, graph.delta == 0.0] = 0.0
    phasors = np.exp(1j * phases)
    residual = graph.x[np.newaxis, :] - phasors @ graph.A.T
    log_likelihood = -np.sum(np.abs(residual) ** 2, axis=1)
    ancestors = np.arange(count, dtype=np.int64)
    beta = 0.0
    beta_path = [beta]
    cess_path: list[float] = []
    pre_resampling_log_likelihoods: list[np.ndarray] = []
    systematic_resampling_offsets: list[float] = []
    local_path: list[float] = []
    global_path: list[float] = []
    local_accepts_path: list[int] = []
    local_attempts_path: list[int] = []
    global_accepts_path: list[int] = []
    global_attempts_path: list[int] = []
    residual_drift_path: list[float] = []
    stage_audit_phasors: list[np.ndarray] = []
    stage_audit_incremental_residuals: list[np.ndarray] = []
    stage_audit_particle_indices: list[int] = []
    residual_reconstruction_count = 0

    while beta < 1.0:
        if resource_guard is not None:
            resource_guard()
        if len(cess_path) >= int(maximum_stages):
            raise RuntimeError("SMC maximum stage count exhausted")
        pre_resampling_log_likelihoods.append(log_likelihood.copy())
        target = float(conditional_ess_fraction) * count
        if _conditional_ess((1.0 - beta) * log_likelihood) >= target:
            next_beta = 1.0
        else:
            lower, upper = beta, 1.0
            for _ in range(60):
                middle = 0.5 * (lower + upper)
                if _conditional_ess((middle - beta) * log_likelihood) < target:
                    upper = middle
                else:
                    lower = middle
            next_beta = lower
        if next_beta <= beta + 10.0 * np.finfo(np.float64).eps:
            raise RuntimeError("SMC inverse-temperature path stalled")
        log_weight = (next_beta - beta) * log_likelihood
        shifted = log_weight - float(np.max(log_weight))
        weight = np.exp(shifted)
        weight /= np.sum(weight)
        cess = float(1.0 / np.sum(weight**2))
        indices, resampling_offset = _systematic_resample(weight, rng)
        systematic_resampling_offsets.append(resampling_offset)
        phases = phases[indices].copy()
        phasors = phasors[indices].copy()
        residual = residual[indices].copy()
        log_likelihood = log_likelihood[indices].copy()
        ancestors = ancestors[indices].copy()
        beta = float(next_beta)

        global_accepts = 0
        global_attempts = count * int(global_prior_proposals_per_stage)
        for _ in range(int(global_prior_proposals_per_stage)):
            if resource_guard is not None:
                resource_guard()
            proposed_phases = rng.uniform(
                -graph.delta, graph.delta, size=(count, graph.group_size)
            )
            proposed_phases[:, graph.delta == 0.0] = 0.0
            proposed_phasors = np.exp(1j * proposed_phases)
            proposed_residual = graph.x[np.newaxis, :] - proposed_phasors @ graph.A.T
            proposed_log_likelihood = -np.sum(np.abs(proposed_residual) ** 2, axis=1)
            accept = _safe_log_uniform(rng.random(count)) <= np.minimum(
                0.0, beta * (proposed_log_likelihood - log_likelihood)
            )
            global_accepts += int(np.sum(accept))
            phases[accept] = proposed_phases[accept]
            phasors[accept] = proposed_phasors[accept]
            residual[accept] = proposed_residual[accept]
            log_likelihood[accept] = proposed_log_likelihood[accept]

        local_accepts = 0
        local_attempts = count * graph.group_size * int(local_sweeps_per_stage)
        for sweep in range(int(local_sweeps_per_stage)):
            if resource_guard is not None:
                resource_guard()
            offset = (
                len(cess_path) * int(local_sweeps_per_stage) + sweep
            ) % graph.group_size
            for step in range(graph.group_size):
                if resource_guard is not None:
                    resource_guard()
                index = (offset + step) % graph.group_size
                delta = float(graph.delta[index])
                proposed_phase = _reflect_to_bounds(
                    phases[:, index]
                    + rng.normal(
                        0.0, float(proposal_scale_fraction) * delta, size=count
                    ),
                    delta,
                )
                proposed_phasor = np.exp(1j * proposed_phase)
                column = graph.A[:, index]
                proposed_residual = (
                    residual
                    + (phasors[:, index] - proposed_phasor)[:, np.newaxis]
                    * column[np.newaxis, :]
                )
                proposed_log_likelihood = -np.sum(
                    np.abs(proposed_residual) ** 2, axis=1
                )
                accept = _safe_log_uniform(rng.random(count)) <= np.minimum(
                    0.0, beta * (proposed_log_likelihood - log_likelihood)
                )
                local_accepts += int(np.sum(accept))
                phases[accept, index] = proposed_phase[accept]
                phasors[accept, index] = proposed_phasor[accept]
                residual[accept] = proposed_residual[accept]
                log_likelihood[accept] = proposed_log_likelihood[accept]

        exact_residual = graph.x[np.newaxis, :] - phasors @ graph.A.T
        drift_by_particle = np.max(
            np.abs(exact_residual - residual), axis=1, initial=0.0
        )
        audit_particle_index = int(np.argmax(drift_by_particle))
        stage_residual_drift = float(drift_by_particle[audit_particle_index])
        if not np.isfinite(stage_residual_drift):
            raise FloatingPointError("SMC residual drift is nonfinite")
        residual_drift_path.append(stage_residual_drift)
        stage_audit_phasors.append(phasors[audit_particle_index].copy())
        stage_audit_incremental_residuals.append(residual[audit_particle_index].copy())
        stage_audit_particle_indices.append(audit_particle_index)
        residual = exact_residual
        log_likelihood = -np.sum(np.abs(residual) ** 2, axis=1)
        if not np.all(np.isfinite(log_likelihood)):
            raise FloatingPointError("SMC exact log likelihood is nonfinite")
        residual_reconstruction_count += 1

        beta_path.append(beta)
        cess_path.append(cess)
        local_path.append(float(local_accepts / local_attempts))
        global_path.append(
            float(global_accepts / global_attempts) if global_attempts else 0.0
        )
        local_accepts_path.append(local_accepts)
        local_attempts_path.append(local_attempts)
        global_accepts_path.append(global_accepts)
        global_attempts_path.append(global_attempts)

    final_exact_residual = graph.x[np.newaxis, :] - phasors @ graph.A.T
    final_residual_consistency_error = float(
        np.max(np.abs(final_exact_residual - residual), initial=0.0)
    )
    if not np.isfinite(final_residual_consistency_error):
        raise FloatingPointError("SMC final residual consistency error is nonfinite")
    if not np.all(np.isfinite(phasors)) or not np.allclose(
        np.abs(phasors), 1.0, atol=1e-12
    ):
        raise FloatingPointError("SMC returned invalid phasors")
    residual_drift_array = np.asarray(residual_drift_path, dtype=np.float64)
    stage_audit_phasor_array = np.asarray(
        stage_audit_phasors, dtype=np.complex128
    ).reshape(-1, graph.group_size)
    stage_audit_incremental_residual_array = np.asarray(
        stage_audit_incremental_residuals, dtype=np.complex128
    ).reshape(-1, graph.x.size)
    stage_audit_particle_index_array = np.asarray(
        stage_audit_particle_indices, dtype=np.int64
    )
    maximum_residual_drift = float(np.max(residual_drift_array, initial=0.0))
    final_initial_ancestor_labels = np.asarray(ancestors, dtype=np.int64)
    local_accepts_array = np.asarray(local_accepts_path, dtype=np.int64)
    local_attempts_array = np.asarray(local_attempts_path, dtype=np.int64)
    global_accepts_array = np.asarray(global_accepts_path, dtype=np.int64)
    global_attempts_array = np.asarray(global_attempts_path, dtype=np.int64)
    final_incremental_residual_array = np.asarray(residual, dtype=np.complex128)
    inverse_temperature_array = np.asarray(beta_path, dtype=np.float64)
    conditional_ess_array = np.asarray(cess_path, dtype=np.float64)
    pre_resampling_log_likelihood_array = np.asarray(
        pre_resampling_log_likelihoods, dtype=np.float64
    ).reshape(-1, count)
    systematic_resampling_offset_array = np.asarray(
        systematic_resampling_offsets, dtype=np.float64
    )
    local_acceptance_array = np.asarray(local_path, dtype=np.float64)
    global_acceptance_array = np.asarray(global_path, dtype=np.float64)
    phasors.setflags(write=False)
    for array in (
        inverse_temperature_array,
        conditional_ess_array,
        pre_resampling_log_likelihood_array,
        systematic_resampling_offset_array,
        local_acceptance_array,
        global_acceptance_array,
        final_initial_ancestor_labels,
        local_accepts_array,
        local_attempts_array,
        global_accepts_array,
        global_attempts_array,
        residual_drift_array,
        stage_audit_phasor_array,
        stage_audit_incremental_residual_array,
        stage_audit_particle_index_array,
        final_incremental_residual_array,
    ):
        array.setflags(write=False)
    return SMCResult(
        particles=phasors,
        inverse_temperature_path=inverse_temperature_array,
        conditional_ess_path=conditional_ess_array,
        pre_resampling_log_likelihoods=pre_resampling_log_likelihood_array,
        systematic_resampling_offsets=systematic_resampling_offset_array,
        local_acceptance_path=local_acceptance_array,
        global_acceptance_path=global_acceptance_array,
        unique_initial_ancestor_count=int(np.unique(ancestors).size),
        final_initial_ancestor_labels=final_initial_ancestor_labels,
        local_accepts_by_stage=local_accepts_array,
        local_attempts_by_stage=local_attempts_array,
        global_accepts_by_stage=global_accepts_array,
        global_attempts_by_stage=global_attempts_array,
        residual_drift_path=residual_drift_array,
        stage_audit_phasors=stage_audit_phasor_array,
        stage_audit_incremental_residuals=stage_audit_incremental_residual_array,
        stage_audit_particle_indices=stage_audit_particle_index_array,
        maximum_residual_drift=maximum_residual_drift,
        final_residual_consistency_error=final_residual_consistency_error,
        final_incremental_residuals=final_incremental_residual_array,
        residual_reconstruction_count=residual_reconstruction_count,
        metadata={
            "method": "adaptive_tempering_resample_move_smc_reflective_mh",
            "seed": int(seed),
            "particles": count,
            "conditional_ess_fraction": float(conditional_ess_fraction),
            "local_sweeps_per_stage": int(local_sweeps_per_stage),
            "proposal_scale_fraction": float(proposal_scale_fraction),
            "global_prior_proposals_per_stage": int(global_prior_proposals_per_stage),
            "maximum_stages": int(maximum_stages),
            "maximum_residual_drift": maximum_residual_drift,
            "final_residual_consistency_error": final_residual_consistency_error,
            "residual_reconstruction_count": residual_reconstruction_count,
            "residual_reconstruction_schedule": "END_OF_EVERY_TEMPERING_STAGE",
            "tempering_and_resampling_audit": (
                "RAW_PRE_RESAMPLING_LOG_LIKELIHOODS_AND_SYSTEMATIC_OFFSETS"
            ),
            "stage_audit_selection": (
                "FIRST_PARTICLE_WITH_STAGE_MAXIMUM_ABSOLUTE_RESIDUAL_DRIFT"
            ),
            "transition_invariance": "EXACT_PER_MH_KERNEL_FINITE_SMC_ERROR_UNCERTIFIED",
        },
    )


def rao_blackwellized_phase_functions(
    graph: PhaseGraph, samples: np.ndarray, *, quadrature_order: int = 32
) -> np.ndarray:
    """The same finite conditional integral, in at most 64-row work buffers."""
    values = np.asarray(samples, dtype=np.complex128)
    if values.ndim < 2 or values.shape[-1] != graph.group_size:
        raise ValueError("samples must end with the phase-variable dimension")
    if int(quadrature_order) < 8:
        raise ValueError("Rao-Blackwell quadrature order must be at least eight")
    flat = values.reshape(-1, graph.group_size)
    result = np.empty_like(flat)
    base_nodes, base_weights = leggauss(int(quadrature_order))
    for start in range(0, len(flat), 64):
        stop = min(start + 64, len(flat))
        block = flat[start:stop]
        residual = graph.x[np.newaxis, :] - block @ graph.A.T
        for index in range(graph.group_size):
            delta = float(graph.delta[index])
            if delta == 0.0:
                result[start:stop, index] = 1.0 + 0.0j
                continue
            column = graph.A[:, index]
            q = (
                residual @ column.conj()
                + np.vdot(column, column).real * block[:, index]
            )
            angles = delta * base_nodes
            nodes = np.exp(1j * angles)
            log_density = 2.0 * (
                q.real[:, None] * nodes.real[None, :]
                + q.imag[:, None] * nodes.imag[None, :]
            )
            log_density -= np.max(log_density, axis=1, keepdims=True)
            weights = base_weights[None, :] * np.exp(log_density)
            numerator = weights @ nodes.real + 1j * (weights @ nodes.imag)
            result[start:stop, index] = numerator / np.sum(weights, axis=1)
    return result.reshape(values.shape)


def prior_rms_channel_scale(model: PilotGroupModel) -> np.ndarray:
    power = np.stack(
        [
            np.real(np.diag(model.covariances[target]))
            + np.abs(model.los[:, target]) ** 2
            for target in range(model.group_size)
        ]
    )
    return np.sqrt(np.maximum(power, np.finfo(np.float64).tiny))


def channel_functional_samples(
    model: PilotGroupModel, observation: np.ndarray, phase_functions: np.ndarray
) -> np.ndarray:
    values = np.asarray(phase_functions, dtype=np.complex128)
    if values.ndim != 3 or values.shape[2] != model.group_size:
        raise ValueError("phase functions must have shape (chains, draws, P)")
    transfer = np.stack(
        [
            channel_transfer(model, observation, target)[1]
            for target in range(model.group_size)
        ]
    )
    return np.einsum("knp,cdp->cdkn", transfer, values, optimize=True)


def channel_second_functional_samples(
    model: PilotGroupModel, observation: np.ndarray, phase_samples: np.ndarray
) -> np.ndarray:
    """Return draw-wise ``(B_k u)(B_k u)^H`` for covariance diagnostics."""

    first = channel_functional_samples(model, observation, phase_samples)
    return np.einsum("cdkn,cdkm->cdknm", first, first.conj(), optimize=True)


def _classic_rhat(values: np.ndarray) -> float:
    chains, draws = values.shape
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1))
    if within <= np.finfo(np.float64).eps:
        return 1.0 if between <= np.finfo(np.float64).eps else float("inf")
    variance = ((draws - 1.0) / draws) * within + between / draws
    return float(np.sqrt(max(variance / within, 0.0)))


def _split(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    if values.shape[0] < 2 or half < 4:
        raise ValueError("diagnostics require at least two chains and eight draws")
    return np.concatenate((values[:, :half], values[:, half : 2 * half]), axis=0)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Apply pooled Blom rank normalization to a chain-by-draw array."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("rank normalization requires a finite chain-by-draw array")
    total = source.size
    ranks = rankdata(source.reshape(-1), method="average")
    probabilities = (ranks - 3.0 / 8.0) / (total + 1.0 / 4.0)
    return norm.ppf(probabilities).reshape(source.shape)


def _rank_folded_rhat(values: np.ndarray) -> float:
    split = _split(values)
    bulk = _classic_rhat(_rank_normalize(split))
    folded = _classic_rhat(_rank_normalize(np.abs(split - np.median(split))))
    return max(bulk, folded)


def _autocovariance(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    draws = values.size
    transform = np.fft.rfft(centered, n=2 * draws)
    covariance = np.fft.irfft(transform * np.conj(transform), n=2 * draws)[:draws]
    return covariance / np.arange(draws, 0, -1)


def _effective_sample_size_from_split(split: np.ndarray) -> float:
    """Geyer initial-positive/monotone ESS for an already split trace."""

    chains, draws = split.shape
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(draws * np.var(np.mean(split, axis=1), ddof=1))
    variance_plus = ((draws - 1.0) / draws) * within + between / draws
    if variance_plus <= np.finfo(np.float64).eps:
        return float(chains * draws)
    autocov = np.mean(np.stack([_autocovariance(chain) for chain in split]), axis=0)
    rho = np.empty(draws, dtype=np.float64)
    rho[0] = 1.0
    rho[1:] = 1.0 - (within - autocov[1:]) / variance_plus
    positive_pairs: list[float] = []
    for index in range(0, draws - 1, 2):
        pair = float(rho[index] + rho[index + 1])
        if pair < 0.0:
            break
        if positive_pairs:
            pair = min(pair, positive_pairs[-1])
        positive_pairs.append(pair)
    tau = max(-1.0 + 2.0 * sum(positive_pairs), 1.0)
    return float(min(chains * draws / tau, chains * draws))


def _raw_split_effective_sample_size(values: np.ndarray) -> float:
    """Return split ESS on raw values; this is deliberately not called bulk ESS."""

    return _effective_sample_size_from_split(_split(values))


def _rank_normalized_split_bulk_ess(values: np.ndarray) -> float:
    """Return genuinely rank-normalized split bulk ESS."""

    split = _split(values)
    return _effective_sample_size_from_split(_rank_normalize(split))


def functional_diagnostics(
    functional: np.ndarray, scale: np.ndarray
) -> FunctionalDiagnostics:
    values = np.asarray(functional, dtype=np.complex128)
    channel_scale = np.asarray(scale, dtype=np.float64)
    if values.ndim < 3 or channel_scale.shape != values.shape[2:]:
        raise ValueError("functional or scale has the wrong shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(channel_scale)):
        return FunctionalDiagnostics(
            float("inf"), 0.0, float("inf"), float("inf"), False
        )
    flattened = values.reshape(values.shape[0], values.shape[1], -1)
    rhats: list[float] = []
    ess_values: list[float] = []
    mcse_real = np.empty(flattened.shape[2], dtype=np.float64)
    mcse_imag = np.empty(flattened.shape[2], dtype=np.float64)
    for output_index in range(flattened.shape[2]):
        for component, destination in (
            (flattened[:, :, output_index].real, mcse_real),
            (flattened[:, :, output_index].imag, mcse_imag),
        ):
            rhats.append(_rank_folded_rhat(component))
            effective = _rank_normalized_split_bulk_ess(component)
            ess_values.append(effective)
            variance = float(np.var(component, ddof=1))
            destination[output_index] = math.sqrt(max(variance, 0.0) / effective)
    complex_mcse = np.hypot(mcse_real, mcse_imag).reshape(channel_scale.shape)
    chain_means = np.mean(values, axis=1)
    pooled = np.mean(chain_means, axis=0)
    disagreement = np.max(
        np.abs(chain_means - pooled[np.newaxis, ...]) / channel_scale[np.newaxis, ...]
    )
    return FunctionalDiagnostics(
        maximum_rank_folded_split_rhat=float(max(rhats)),
        minimum_bulk_effective_sample_size=float(min(ess_values)),
        maximum_prior_rms_normalized_mcse=float(np.max(complex_mcse / channel_scale)),
        maximum_prior_rms_normalized_chain_disagreement=float(disagreement),
        all_finite=True,
    )


def phase_moments(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.complex128)
    if values.ndim < 2:
        raise ValueError("phase samples must have at least two dimensions")
    flat = values.reshape(-1, values.shape[-1])
    z = np.mean(flat, axis=0)
    second = np.einsum("si,sj->ij", flat, flat.conj(), optimize=True) / flat.shape[0]
    second = 0.5 * (second + second.conj().T)
    np.fill_diagonal(second, 1.0)
    return z.astype(np.complex128), second.astype(np.complex128)


def phase_moment_distance(
    model: PilotGroupModel,
    observation: np.ndarray,
    first_z: np.ndarray,
    first_Z: np.ndarray,
    second_z: np.ndarray,
    second_Z: np.ndarray,
) -> dict[str, float]:
    """Compare phase moments and their channel mean/covariance images."""

    transfer = np.stack(
        [
            channel_transfer(model, observation, target)[1]
            for target in range(model.group_size)
        ]
    )
    total_scale = np.sqrt(
        np.maximum(
            np.asarray(
                [
                    np.trace(model.covariances[target]).real
                    + np.vdot(model.los[:, target], model.los[:, target]).real
                    for target in range(model.group_size)
                ]
            ),
            np.finfo(np.float64).tiny,
        )
    )
    mean_delta = np.einsum("knp,p->kn", transfer, first_z - second_z, optimize=True)
    first_covariance = first_Z - np.outer(first_z, first_z.conj())
    second_covariance = second_Z - np.outer(second_z, second_z.conj())
    covariance_delta = np.einsum(
        "knp,pq,kmq->knm",
        transfer,
        first_covariance - second_covariance,
        transfer.conj(),
        optimize=True,
    )
    return {
        "maximum_absolute_z": float(np.max(np.abs(first_z - second_z))),
        "maximum_absolute_Z": float(np.max(np.abs(first_Z - second_Z))),
        "d_mu": float(np.max(np.linalg.norm(mean_delta, axis=1) / total_scale)),
        "d_C": float(
            np.max(np.linalg.norm(covariance_delta, axis=(1, 2)) / total_scale**2)
        ),
    }
