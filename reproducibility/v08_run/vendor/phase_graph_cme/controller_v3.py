"""Mean-only adapter for the independently qualified R2C01 V3 controller.

This module intentionally does not replace :mod:`phase_graph_cme.controller`.
It ports the frozen A4_RE16_PRECISION decision rule used by the integrated V3
qualification into a small reusable group-level interface.  The nonlinear
output is a finite-run numerical approximation; empirical diagnostics are not
a convergence certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from . import redesign_v2 as redesign_v2_module
from .baselines import clmmse
from .model import PilotGroupModel, build_phase_graph
from .moments import channel_moments
from .redesign_v2 import (
    ReplicaExchangeResult,
    channel_functional_samples,
    channel_second_functional_samples,
    functional_diagnostics,
    phase_moments,
    prior_rms_channel_scale,
    rao_blackwellized_phase_functions,
    replica_exchange_gibbs,
)

ComplexArray = NDArray[np.complex128]


QUALIFIED_REDESIGN_V2_SHA256 = (
    "b7a15cd268cfaa18222d9afc8f4418b2c559fac3d90f756b6fea4ab9d4035db3"
)
CONTROLLER_DIAGNOSTICS_SCHEMA_VERSION = "r2c01-v3-controller-diagnostics/1.0"


@dataclass(frozen=True, slots=True)
class V3A4RE16PrecisionConfig:
    """Immutable operational configuration from integrated qualification."""

    configuration_id: str = "A4_RE16_PRECISION"
    inverse_temperatures: tuple[float, ...] = (
        0.0,
        0.01,
        0.025,
        0.05,
        0.09,
        0.14,
        0.20,
        0.28,
        0.37,
        0.47,
        0.58,
        0.69,
        0.79,
        0.88,
        0.95,
        1.0,
    )
    ensembles: int = 4
    burn_cycles: int = 256
    retained_draws: int = 1024
    thinning_cycles: int = 1
    sweeps_per_cycle: int = 1
    maximum_conditional_proposals: int = 100_000
    conditional_batch_size: int = 16
    residual_recompute_interval_cycles: int = 32
    rao_blackwell_check_order: int = 128
    rao_blackwell_selected_order: int = 192

    def __post_init__(self) -> None:
        observed = (
            self.configuration_id,
            self.inverse_temperatures,
            self.ensembles,
            self.burn_cycles,
            self.retained_draws,
            self.thinning_cycles,
            self.sweeps_per_cycle,
            self.maximum_conditional_proposals,
            self.conditional_batch_size,
            self.residual_recompute_interval_cycles,
            self.rao_blackwell_check_order,
            self.rao_blackwell_selected_order,
        )
        canonical = (
            "A4_RE16_PRECISION",
            (
                0.0,
                0.01,
                0.025,
                0.05,
                0.09,
                0.14,
                0.20,
                0.28,
                0.37,
                0.47,
                0.58,
                0.69,
                0.79,
                0.88,
                0.95,
                1.0,
            ),
            4,
            256,
            1024,
            1,
            1,
            100_000,
            16,
            32,
            128,
            192,
        )
        if observed != canonical:
            raise ValueError(
                "the A4_RE16_PRECISION identity cannot be modified in place"
            )


@dataclass(frozen=True, slots=True)
class V3AuditTrace:
    """Engineering-only reconstruction witness, never a receiver input."""

    retained_phasors: ComplexArray
    post_burn_initial_labels: NDArray[np.int32]
    post_burn_label_history: NDArray[np.int32]
    residual_drift_path: NDArray[np.float64]
    residual_audit_phasors: ComplexArray
    residual_audit_incremental_residuals: ComplexArray
    residual_audit_ensemble_indices: NDArray[np.int32]
    residual_audit_cycle_indices: NDArray[np.int32]
    residual_audit_temperature_indices: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class V3GroupControllerResult:
    """One inference reused for every target in one AP/pilot group."""

    estimates: ComplexArray
    mode: str
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    phase_mean_diagnostic: ComplexArray | None = None
    audit_trace: V3AuditTrace | None = None


class V3FatalControllerError(RuntimeError):
    """Base class for failures that must not dispatch a scientific fallback."""


class V3ResourceLimitError(V3FatalControllerError):
    """Raised by a live resource guard; always fatal to the current stage."""


class V3PostSamplerError(V3FatalControllerError):
    """A completed sampler could not be evaluated or transferred safely."""


_THRESHOLDS: dict[str, float | int] = {
    "rb_maximum_order_disagreement": 1.0e-9,
    "mean_maximum_rank_folded_split_rhat": 1.05,
    "mean_minimum_bulk_ess": 400.0,
    "mean_maximum_normalized_mcse": 0.012,
    "mean_maximum_normalized_chain_disagreement": 0.05,
    "second_maximum_rank_folded_split_rhat": 1.05,
    "second_minimum_bulk_ess": 400.0,
    "second_maximum_normalized_mcse": 0.010,
    "second_maximum_normalized_chain_disagreement": 0.05,
    "energy_maximum_rank_folded_split_rhat": 1.05,
    "energy_minimum_bulk_ess": 400.0,
    "phase_maximum_rank_folded_split_rhat": 1.05,
    "phase_minimum_bulk_ess": 400.0,
    "phase_maximum_normalized_mcse": 0.025,
    "phase_maximum_normalized_chain_disagreement": 0.10,
    "minimum_adjacent_swap_acceptance": 0.10,
    "minimum_round_trips_per_ensemble": 1,
    "minimum_aggregate_round_trips": 20,
    "maximum_absolute_residual_drift": 1.0e-10,
}

_TOLERANCES = {
    "summary_atol": 1.0e-13,
    "summary_rtol": 1.0e-11,
    "threshold_atol": 5.0e-13,
    "threshold_rtol": 5.0e-12,
    "near_threshold_absolute_margin": 1.0e-13,
    "near_threshold_relative_margin": 2.0e-10,
}


def _threshold_margin(threshold: float) -> float:
    return max(
        _TOLERANCES["threshold_atol"]
        + _TOLERANCES["threshold_rtol"] * abs(float(threshold)),
        _TOLERANCES["near_threshold_absolute_margin"],
        _TOLERANCES["near_threshold_relative_margin"] * abs(float(threshold)),
    )


def _upper_pass(observed: float, threshold: float) -> bool:
    values = np.asarray([observed, threshold], dtype=np.float64)
    return bool(
        np.all(np.isfinite(values))
        and float(observed) <= float(threshold) - _threshold_margin(threshold)
    )


def _lower_pass(observed: float, threshold: float) -> bool:
    values = np.asarray([observed, threshold], dtype=np.float64)
    return bool(
        np.all(np.isfinite(values))
        and float(observed) >= float(threshold) + _threshold_margin(threshold)
    )


def _summary_agrees(left: object, right: object) -> bool:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or not (
        np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    ):
        return False
    bound = _TOLERANCES["summary_atol"] + _TOLERANCES["summary_rtol"] * np.abs(
        second
    )
    return bool(np.all(np.abs(first - second) <= bound))


def _raw_array_checks(
    sampler: ReplicaExchangeResult, graph: Any, config: V3A4RE16PrecisionConfig
) -> dict[str, bool]:
    samples = sampler.retained_phasors
    angles = np.angle(samples)
    temperature_count = len(config.inverse_temperatures)
    return {
        "exact_trace_shape": samples.shape
        == (config.ensembles, config.retained_draws, graph.group_size),
        "dtype_complex128": samples.dtype == np.complex128,
        "all_finite": bool(np.all(np.isfinite(samples))),
        "unit_modulus": bool(
            np.max(np.abs(np.abs(samples) - 1.0), initial=0.0) <= 1.0e-12
        ),
        "within_support": bool(
            np.all(
                np.abs(angles)
                <= graph.delta[np.newaxis, np.newaxis, :] + 1.0e-12
            )
        ),
        "exact_transport_initial_shape": sampler.post_burn_initial_labels.shape
        == (config.ensembles, temperature_count),
        "exact_transport_history_shape": sampler.post_burn_label_history.shape
        == (
            config.ensembles,
            config.retained_draws * config.thinning_cycles,
            temperature_count,
        ),
        "transport_label_dtype_int32": (
            sampler.post_burn_initial_labels.dtype == np.int32
            and sampler.post_burn_label_history.dtype == np.int32
        ),
        "burn_cycle_parity_registered": config.burn_cycles >= 0,
    }


def _residual_audit_checks(
    sampler: ReplicaExchangeResult, graph: Any, config: V3A4RE16PrecisionConfig
) -> dict[str, bool]:
    cycles = config.burn_cycles + config.retained_draws * config.thinning_cycles
    interval = config.residual_recompute_interval_cycles
    per_ensemble_cycles = np.arange(interval - 1, cycles, interval, dtype=np.int32)
    expected_count = config.ensembles * per_ensemble_cycles.size
    expected_ensembles = np.repeat(
        np.arange(config.ensembles, dtype=np.int32), per_ensemble_cycles.size
    )
    expected_cycles = np.tile(per_ensemble_cycles, config.ensembles)
    reconstructed = np.max(
        np.abs(
            graph.x[np.newaxis, :]
            - sampler.residual_audit_phasors @ graph.A.T
            - sampler.residual_audit_incremental_residuals
        ),
        axis=1,
        initial=0.0,
    )
    return {
        "residual_audit_count": (
            sampler.residual_reconstruction_count
            == expected_count
            == sampler.residual_drift_path.size
        ),
        "residual_audit_shapes": (
            sampler.residual_audit_phasors.shape
            == (expected_count, graph.group_size)
            and sampler.residual_audit_incremental_residuals.shape
            == (expected_count, graph.x.size)
            and sampler.residual_audit_ensemble_indices.shape == (expected_count,)
            and sampler.residual_audit_cycle_indices.shape == (expected_count,)
            and sampler.residual_audit_temperature_indices.shape == (expected_count,)
        ),
        "residual_audit_dtypes": (
            sampler.residual_drift_path.dtype == np.float64
            and sampler.residual_audit_phasors.dtype == np.complex128
            and sampler.residual_audit_incremental_residuals.dtype == np.complex128
            and sampler.residual_audit_ensemble_indices.dtype == np.int32
            and sampler.residual_audit_cycle_indices.dtype == np.int32
            and sampler.residual_audit_temperature_indices.dtype == np.int32
        ),
        "residual_audit_schedule": (
            np.array_equal(sampler.residual_audit_ensemble_indices, expected_ensembles)
            and np.array_equal(sampler.residual_audit_cycle_indices, expected_cycles)
            and np.all(sampler.residual_audit_temperature_indices >= 0)
            and np.all(
                sampler.residual_audit_temperature_indices
                < len(config.inverse_temperatures)
            )
        ),
        "residual_audit_path_reconstructed": _summary_agrees(
            reconstructed, sampler.residual_drift_path
        ),
        "residual_audit_maximum_reconstructed": _summary_agrees(
            float(np.max(reconstructed, initial=0.0)),
            float(sampler.maximum_residual_drift),
        ),
    }


def _diagnostic_bundle(
    model: PilotGroupModel,
    observation: ComplexArray,
    graph: Any,
    sampler: ReplicaExchangeResult,
    config: V3A4RE16PrecisionConfig,
) -> tuple[ComplexArray, ComplexArray, dict[str, Any]]:
    samples = sampler.retained_phasors
    rb_low = rao_blackwellized_phase_functions(
        graph, samples, quadrature_order=config.rao_blackwell_check_order
    )
    rb_high = rao_blackwellized_phase_functions(
        graph, samples, quadrature_order=config.rao_blackwell_selected_order
    )
    scale = prior_rms_channel_scale(model)
    mean = functional_diagnostics(
        channel_functional_samples(model, observation, rb_high), scale
    )
    second_scale = np.einsum("kn,km->knm", scale, scale)
    second = functional_diagnostics(
        channel_second_functional_samples(model, observation, samples), second_scale
    )
    residual = graph.x[np.newaxis, np.newaxis, :] - np.einsum(
        "cdp,np->cdn", samples, graph.A, optimize=True
    )
    energy = np.sum(np.abs(residual) ** 2, axis=2)
    energy_scale = np.asarray(
        [max(float(np.std(energy)), np.finfo(np.float64).eps)]
    )
    energy_diagnostic = functional_diagnostics(
        energy[:, :, np.newaxis], energy_scale
    )
    phase_diagnostic = functional_diagnostics(samples, np.ones(graph.group_size))
    z_rb, _ = phase_moments(rb_high)
    _, Z_raw = phase_moments(samples)
    bundle = {
        "rb_quadrature_orders": [
            config.rao_blackwell_check_order,
            config.rao_blackwell_selected_order,
        ],
        "rb_maximum_order_disagreement": float(np.max(np.abs(rb_low - rb_high))),
        "channel_mean": asdict(mean),
        "channel_second_functional": asdict(second),
        "energy": asdict(energy_diagnostic),
        "phase_sentinel": asdict(phase_diagnostic),
    }
    return z_rb.astype(np.complex128), Z_raw.astype(np.complex128), bundle


def _diagnostic_checks(
    bundle: dict[str, Any], sampler: ReplicaExchangeResult
) -> dict[str, bool]:
    mean = bundle["channel_mean"]
    second = bundle["channel_second_functional"]
    energy = bundle["energy"]
    phase = bundle["phase_sentinel"]
    rates = np.divide(
        sampler.swap_accepts_by_edge,
        sampler.swap_attempts_by_edge,
        out=np.full(sampler.swap_attempts_by_edge.shape, np.nan),
        where=sampler.swap_attempts_by_edge != 0,
    )
    return {
        "all_diagnostics_finite": all(
            bool(item["all_finite"]) for item in (mean, second, energy, phase)
        ),
        "rb_order_agreement": _upper_pass(
            bundle["rb_maximum_order_disagreement"],
            float(_THRESHOLDS["rb_maximum_order_disagreement"]),
        ),
        "mean_rhat": _upper_pass(
            mean["maximum_rank_folded_split_rhat"],
            float(_THRESHOLDS["mean_maximum_rank_folded_split_rhat"]),
        ),
        "mean_ess": _lower_pass(
            mean["minimum_bulk_effective_sample_size"],
            float(_THRESHOLDS["mean_minimum_bulk_ess"]),
        ),
        "mean_mcse": _upper_pass(
            mean["maximum_prior_rms_normalized_mcse"],
            float(_THRESHOLDS["mean_maximum_normalized_mcse"]),
        ),
        "mean_chain_disagreement": _upper_pass(
            mean["maximum_prior_rms_normalized_chain_disagreement"],
            float(_THRESHOLDS["mean_maximum_normalized_chain_disagreement"]),
        ),
        "second_rhat": _upper_pass(
            second["maximum_rank_folded_split_rhat"],
            float(_THRESHOLDS["second_maximum_rank_folded_split_rhat"]),
        ),
        "second_ess": _lower_pass(
            second["minimum_bulk_effective_sample_size"],
            float(_THRESHOLDS["second_minimum_bulk_ess"]),
        ),
        "second_mcse": _upper_pass(
            second["maximum_prior_rms_normalized_mcse"],
            float(_THRESHOLDS["second_maximum_normalized_mcse"]),
        ),
        "second_chain_disagreement": _upper_pass(
            second["maximum_prior_rms_normalized_chain_disagreement"],
            float(_THRESHOLDS["second_maximum_normalized_chain_disagreement"]),
        ),
        "energy_rhat": _upper_pass(
            energy["maximum_rank_folded_split_rhat"],
            float(_THRESHOLDS["energy_maximum_rank_folded_split_rhat"]),
        ),
        "energy_ess": _lower_pass(
            energy["minimum_bulk_effective_sample_size"],
            float(_THRESHOLDS["energy_minimum_bulk_ess"]),
        ),
        "phase_rhat": _upper_pass(
            phase["maximum_rank_folded_split_rhat"],
            float(_THRESHOLDS["phase_maximum_rank_folded_split_rhat"]),
        ),
        "phase_ess": _lower_pass(
            phase["minimum_bulk_effective_sample_size"],
            float(_THRESHOLDS["phase_minimum_bulk_ess"]),
        ),
        "phase_mcse": _upper_pass(
            phase["maximum_prior_rms_normalized_mcse"],
            float(_THRESHOLDS["phase_maximum_normalized_mcse"]),
        ),
        "phase_chain_disagreement": _upper_pass(
            phase["maximum_prior_rms_normalized_chain_disagreement"],
            float(_THRESHOLDS["phase_maximum_normalized_chain_disagreement"]),
        ),
        "minimum_swap_acceptance": bool(
            rates.size
            and _lower_pass(
                float(np.min(rates)),
                float(_THRESHOLDS["minimum_adjacent_swap_acceptance"]),
            )
        ),
        "round_trip_each_ensemble": bool(
            np.all(
                sampler.round_trips_by_ensemble
                >= int(_THRESHOLDS["minimum_round_trips_per_ensemble"])
            )
        ),
        "aggregate_round_trips": bool(
            np.sum(sampler.round_trips_by_ensemble)
            >= int(_THRESHOLDS["minimum_aggregate_round_trips"])
        ),
        "residual_drift": _upper_pass(
            float(sampler.maximum_residual_drift),
            float(_THRESHOLDS["maximum_absolute_residual_drift"]),
        ),
    }


def _copy_audit_trace(sampler: ReplicaExchangeResult) -> V3AuditTrace:
    return V3AuditTrace(
        retained_phasors=np.array(sampler.retained_phasors, copy=True),
        post_burn_initial_labels=np.array(
            sampler.post_burn_initial_labels, dtype=np.int32, copy=True
        ),
        post_burn_label_history=np.array(
            sampler.post_burn_label_history, dtype=np.int32, copy=True
        ),
        residual_drift_path=np.array(
            sampler.residual_drift_path, dtype=np.float64, copy=True
        ),
        residual_audit_phasors=np.array(
            sampler.residual_audit_phasors, dtype=np.complex128, copy=True
        ),
        residual_audit_incremental_residuals=np.array(
            sampler.residual_audit_incremental_residuals,
            dtype=np.complex128,
            copy=True,
        ),
        residual_audit_ensemble_indices=np.array(
            sampler.residual_audit_ensemble_indices, dtype=np.int32, copy=True
        ),
        residual_audit_cycle_indices=np.array(
            sampler.residual_audit_cycle_indices, dtype=np.int32, copy=True
        ),
        residual_audit_temperature_indices=np.array(
            sampler.residual_audit_temperature_indices, dtype=np.int32, copy=True
        ),
    )


def _candidate_upper_bound(
    config: V3A4RE16PrecisionConfig, group_size: int
) -> int:
    cycles = config.burn_cycles + config.retained_draws * config.thinning_cycles
    return (
        config.ensembles
        * cycles
        * config.sweeps_per_cycle
        * int(group_size)
        * len(config.inverse_temperatures)
        * config.maximum_conditional_proposals
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_qualified_redesign_source() -> str:
    source = Path(redesign_v2_module.__file__).resolve(strict=True)
    observed = _sha256_file(source)
    if observed.lower() != QUALIFIED_REDESIGN_V2_SHA256.lower():
        raise V3FatalControllerError(
            "active redesign_v2.py does not match the independently qualified source"
        )
    return observed


def _strict_json_safe(value: object) -> object:
    """Return a strict JSON value while preserving explicit nonfinite evidence."""

    if value is None or type(value) in {bool, str, int}:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if type(value) is float or isinstance(value, np.floating):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        if math.isnan(numeric):
            return "NONFINITE_NAN"
        return "NONFINITE_POSITIVE_INFINITY" if numeric > 0 else "NONFINITE_NEGATIVE_INFINITY"
    if type(value) is complex or isinstance(value, np.complexfloating):
        numeric = complex(value)
        return {
            "real": _strict_json_safe(numeric.real),
            "imag": _strict_json_safe(numeric.imag),
        }
    if isinstance(value, np.ndarray):
        return _strict_json_safe(value.tolist())
    if type(value) in {list, tuple}:
        return [_strict_json_safe(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("public diagnostic mappings require string keys")
        return {key: _strict_json_safe(item) for key, item in value.items()}
    raise TypeError(
        f"public diagnostics contain unsupported type {type(value).__name__}"
    )


def _public_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    normalized = _strict_json_safe(value)
    if type(normalized) is not dict:
        raise TypeError("controller diagnostics must remain a mapping")
    json.dumps(normalized, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return normalized


class V3FiniteRunController:
    """Frozen group-level V3 mean controller with an immutable B2 fallback."""

    def __init__(self, config: V3A4RE16PrecisionConfig | None = None) -> None:
        if config is not None and type(config) is not V3A4RE16PrecisionConfig:
            raise TypeError("config must be the exact frozen V3 configuration type")
        self.config = config if config is not None else V3A4RE16PrecisionConfig()
        self.qualified_redesign_v2_sha256 = _verify_qualified_redesign_source()

    def estimate_all(
        self,
        model: PilotGroupModel,
        observation: object,
        *,
        seed: int,
        resource_guard: Callable[[], None] | None = None,
        retain_audit_trace: bool = False,
    ) -> V3GroupControllerResult:
        """Estimate all co-pilot means after exactly one phase inference."""

        if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)):
            raise ValueError("seed must be a nonnegative integer")
        if int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(retain_audit_trace, (bool, np.bool_)):
            raise ValueError("retain_audit_trace must be Boolean")
        y = np.asarray(observation, dtype=np.complex128)
        graph = build_phase_graph(model, y)
        cfg = self.config

        def guarded_resource_check() -> None:
            if resource_guard is None:
                return
            try:
                resource_guard()
            except V3FatalControllerError:
                raise
            except Exception as error:
                raise V3ResourceLimitError("resource guard failed") from error

        direct_before = np.column_stack(
            [clmmse(model, y, target) for target in range(model.group_size)]
        ).astype(np.complex128)
        started = time.perf_counter()
        try:
            guarded_resource_check()
            sampler = replica_exchange_gibbs(
                graph,
                seed=int(seed),
                inverse_temperatures=cfg.inverse_temperatures,
                ensembles=cfg.ensembles,
                burn_cycles=cfg.burn_cycles,
                retained_draws=cfg.retained_draws,
                thinning_cycles=cfg.thinning_cycles,
                sweeps_per_cycle=cfg.sweeps_per_cycle,
                maximum_conditional_proposals=cfg.maximum_conditional_proposals,
                conditional_batch_size=cfg.conditional_batch_size,
                residual_recompute_interval_cycles=(
                    cfg.residual_recompute_interval_cycles
                ),
                resource_guard=(guarded_resource_check if resource_guard else None),
            )
        except V3FatalControllerError:
            raise
        except Exception as error:
            direct_after = np.column_stack(
                [clmmse(model, y, target) for target in range(model.group_size)]
            ).astype(np.complex128)
            equal = bool(np.array_equal(direct_before, direct_after))
            difference = float(
                np.max(np.abs(direct_before - direct_after), initial=0.0)
            )
            if not equal or difference != 0.0:
                raise V3FatalControllerError(
                    "independent CLMMSE fallback recomputation changed"
                ) from error
            fallback_diagnostics = _public_diagnostics(
                {
                    "diagnostics_schema_version": (
                        CONTROLLER_DIAGNOSTICS_SCHEMA_VERSION
                    ),
                    "configuration_id": cfg.configuration_id,
                    "fallback_reason": "ALLOWED_PRECOMPLETION_SAMPLER_EXCEPTION",
                    "structured_failure": {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    "fallback_independent_recomputation": True,
                    "fallback_array_equal": equal,
                    "fallback_maximum_absolute_difference": difference,
                    "cost_accounting": {
                        "logical_conditional_proposals": None,
                        "generated_candidate_values": None,
                        "coordinate_updates": None,
                        "swap_attempts": None,
                        "swap_accepts": None,
                        "residual_reconstructions": None,
                        "failed_path_generated_candidate_upper_bound": (
                            _candidate_upper_bound(cfg, graph.group_size)
                        ),
                        "accounting_complete": False,
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                    "finite_run_convergence_certified": False,
                    "audit_trace_retained": False,
                }
            )
            return V3GroupControllerResult(
                estimates=direct_after,
                mode="CLMMSE_FALLBACK",
                status="BASELINE_FALLBACK",
                diagnostics=fallback_diagnostics,
            )

        try:
            z_rb, Z_raw, bundle = _diagnostic_bundle(
                model, y, graph, sampler, cfg
            )
            if not np.all(np.isfinite(z_rb)) or not np.all(np.isfinite(Z_raw)):
                raise FloatingPointError("nonfinite post-sampler phase moments")
            checks = {
                **_raw_array_checks(sampler, graph, cfg),
                **_residual_audit_checks(sampler, graph, cfg),
                **_diagnostic_checks(bundle, sampler),
            }
            if any(not isinstance(value, (bool, np.bool_)) for value in checks.values()):
                raise TypeError("controller check is not Boolean")
            checks = {key: bool(value) for key, value in checks.items()}
            nonlinear = all(checks.values())
            selected: ComplexArray | None = None
            if nonlinear:
                selected = np.column_stack(
                    [
                        channel_moments(model, y, target, z_rb).mean
                        for target in range(model.group_size)
                    ]
                ).astype(np.complex128)
                if not np.all(np.isfinite(selected)):
                    raise FloatingPointError("nonfinite V3 channel-mean output")
        except Exception as error:
            raise V3PostSamplerError(
                "post-sampler diagnostics or mean transfer failed"
            ) from error

        rates = np.divide(
            sampler.swap_accepts_by_edge,
            sampler.swap_attempts_by_edge,
            out=np.full(sampler.swap_attempts_by_edge.shape, np.nan),
            where=sampler.swap_attempts_by_edge != 0,
        )
        cycles = cfg.burn_cycles + cfg.retained_draws * cfg.thinning_cycles
        coordinate_updates = (
            cfg.ensembles
            * cycles
            * cfg.sweeps_per_cycle
            * graph.group_size
            * len(cfg.inverse_temperatures)
        )
        burn_attempts = np.asarray(
            sampler.burn_swap_attempts_by_edge, dtype=np.int64
        )
        burn_accepts = np.asarray(
            sampler.burn_swap_accepts_by_edge, dtype=np.int64
        )
        diagnostics: dict[str, Any] = {
            "diagnostics_schema_version": CONTROLLER_DIAGNOSTICS_SCHEMA_VERSION,
            "configuration_id": cfg.configuration_id,
            "qualified_redesign_v2_sha256": self.qualified_redesign_v2_sha256,
            "checks": checks,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "diagnostic_bundle": bundle,
            "phase_second_moment_role": "DIAGNOSTIC_ONLY_NOT_RECEIVER_INPUT",
            "phase_second_moment_diagonal_maximum_error": float(
                np.max(np.abs(np.diag(Z_raw) - 1.0), initial=0.0)
            ),
            "logical_proposal_count": int(sampler.logical_proposal_count),
            "generated_candidate_value_count": int(
                sampler.generated_candidate_value_count
            ),
            "residual_reconstruction_count": int(
                sampler.residual_reconstruction_count
            ),
            "cost_accounting": {
                "logical_conditional_proposals": int(
                    sampler.logical_proposal_count
                ),
                "generated_candidate_values": int(
                    sampler.generated_candidate_value_count
                ),
                "coordinate_updates": int(coordinate_updates),
                "swap_attempts": int(
                    np.sum(burn_attempts) + np.sum(sampler.swap_attempts_by_edge)
                ),
                "swap_accepts": int(
                    np.sum(burn_accepts) + np.sum(sampler.swap_accepts_by_edge)
                ),
                "residual_reconstructions": int(
                    sampler.residual_reconstruction_count
                ),
                "failed_path_generated_candidate_upper_bound": None,
                "accounting_complete": True,
            },
            "minimum_adjacent_swap_acceptance": float(np.min(rates)),
            "round_trips_by_ensemble": sampler.round_trips_by_ensemble.tolist(),
            "aggregate_round_trips": int(np.sum(sampler.round_trips_by_ensemble)),
            "maximum_absolute_residual_drift": float(
                sampler.maximum_residual_drift
            ),
            "sampler_metadata": dict(sampler.metadata),
            "elapsed_seconds": time.perf_counter() - started,
            "finite_run_convergence_certified": False,
            "audit_trace_retained": bool(retain_audit_trace),
        }
        diagnostics = _public_diagnostics(diagnostics)
        audit_trace = _copy_audit_trace(sampler) if retain_audit_trace else None

        if nonlinear:
            if selected is None:
                raise V3PostSamplerError("accepted controller has no selected mean")
            return V3GroupControllerResult(
                estimates=selected,
                mode="REPLICA_EXCHANGE_FINITE_RUN_APPROXIMATION",
                status=(
                    "EMPIRICALLY_ACCEPTED_FINITE_RUN_APPROXIMATION_NOT_CERTIFIED"
                ),
                diagnostics=diagnostics,
                phase_mean_diagnostic=np.array(z_rb, copy=True),
                audit_trace=audit_trace,
            )

        direct_after = np.column_stack(
            [clmmse(model, y, target) for target in range(model.group_size)]
        ).astype(np.complex128)
        equal = bool(np.array_equal(direct_before, direct_after))
        difference = float(np.max(np.abs(direct_before - direct_after), initial=0.0))
        if not equal or difference != 0.0:
            raise V3FatalControllerError(
                "independent CLMMSE fallback recomputation changed"
            )
        diagnostics.update(
            {
                "fallback_reason": "DIAGNOSTIC_ACCEPTANCE_BUNDLE_REJECTED",
                "fallback_independent_recomputation": True,
                "fallback_array_equal": equal,
                "fallback_maximum_absolute_difference": difference,
            }
        )
        diagnostics = _public_diagnostics(diagnostics)
        return V3GroupControllerResult(
            estimates=direct_after,
            mode="CLMMSE_FALLBACK",
            status="BASELINE_FALLBACK",
            diagnostics=diagnostics,
            phase_mean_diagnostic=np.array(z_rb, copy=True),
            audit_trace=audit_trace,
        )
