"""Locked-stage core for the 144-case small-reference qualification.

This module writes no artifacts and does not claim that a final preregistration
lock exists. A future locked runner can call this deterministic core after
separately verifying its configuration and source hashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from .baselines import clmmse
from .controller import ControllerConfig, InferenceController
from .model import build_phase_graph
from .moments import channel_moments
from .production_registry import (
    SmallReferenceCase,
    build_small_reference_registry,
    derived_seed,
    preflight_small_reference_registry,
)
from .reference_runner import (
    compare_moments,
    maximum_absolute_error,
    normalized_frobenius_error,
    normalized_squared_error,
    small_reference_orders,
    streaming_tensor_moments,
)
from .scenario import (
    ScenarioConfig,
    generate_channel_frame,
    generate_topology,
    pilot_group,
)


CAMPAIGN_ID = "R2C01_SMALL_REFERENCE_QUALIFICATION"
EXPECTED_CASE_COUNT = 144


def _require_integer(value: object, *, label: str, expected: int) -> None:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) != expected
    ):
        raise ValueError(f"{label} must equal the fixed integer {expected}")


def _require_real(value: object, *, label: str, expected: float) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"{label} must equal the fixed value {expected!r}")


def _require_boolean(value: object, *, label: str, expected: bool) -> None:
    if not isinstance(value, (bool, np.bool_)) or bool(value) is not expected:
        raise ValueError(f"{label} must be Boolean {expected}")


@dataclass(frozen=True, slots=True)
class SmallReferenceCampaignConfig:
    """Strict, non-retunable contract for the fixed small-reference stage."""

    campaign_id: str = CAMPAIGN_ID
    expected_case_count: int = EXPECTED_CASE_COUNT
    access_points: int = 1
    antennas_per_ap: int = 4
    pilot_symbols: int = 1
    coherence_symbols: int = 200
    area_side: float = 1.0
    minimum_distance: float = 0.03
    pathloss_exponent: float = 3.7
    pilot_power: float = 1.0
    payload_power: float = 1.0
    pilot_noise_variance: float = 1.0
    payload_noise_variance: float = 1.0
    spatial_correlation: float = 0.4
    pilot_assignment: str = "cyclic"
    phase_error_model: str = "matched_uniform"
    reference_maximum_states: int = 1_100_000
    reference_chunk_size: int = 8_192
    reference_maximum_abs_z_disagreement: float = 1e-6
    reference_maximum_abs_Z_disagreement: float = 1e-6
    controller_quadrature_order: int = 8
    controller_quadrature_check_order: int = 12
    controller_quadrature_maximum_states: int = 250_000
    controller_quadrature_moment_tolerance: float = 0.001
    controller_quadrature_second_moment_tolerance: float = 0.001
    controller_gibbs_chains: int = 4
    controller_gibbs_burn_sweeps: int = 96
    controller_gibbs_retained_draws: int = 192
    controller_gibbs_thinning_sweeps: int = 1
    controller_maximum_conditional_proposals: int = 50_000
    controller_maximum_split_rhat: float = 1.05
    controller_maximum_chain_mean_disagreement: float = 0.05
    controller_accept_empirical_gibbs: bool = True
    controller_retain_gibbs_samples: bool = True
    p95_quantile: float = 0.95
    quantile_method: str = "linear"
    p95_maximum_absolute_z_error: float = 0.005
    p95_maximum_absolute_Z_error: float = 0.01
    maximum_channel_mean_nse: float = 0.05
    maximum_channel_covariance_normalized_frobenius_error: float = 0.10

    def __post_init__(self) -> None:
        if self.campaign_id != CAMPAIGN_ID:
            raise ValueError(f"campaign_id must equal {CAMPAIGN_ID}")
        integers = {
            "expected_case_count": 144,
            "access_points": 1,
            "antennas_per_ap": 4,
            "pilot_symbols": 1,
            "coherence_symbols": 200,
            "reference_maximum_states": 1_100_000,
            "reference_chunk_size": 8_192,
            "controller_quadrature_order": 8,
            "controller_quadrature_check_order": 12,
            "controller_quadrature_maximum_states": 250_000,
            "controller_gibbs_chains": 4,
            "controller_gibbs_burn_sweeps": 96,
            "controller_gibbs_retained_draws": 192,
            "controller_gibbs_thinning_sweeps": 1,
            "controller_maximum_conditional_proposals": 50_000,
        }
        for name, expected in integers.items():
            _require_integer(getattr(self, name), label=name, expected=expected)
        reals = {
            "area_side": 1.0,
            "minimum_distance": 0.03,
            "pathloss_exponent": 3.7,
            "pilot_power": 1.0,
            "payload_power": 1.0,
            "pilot_noise_variance": 1.0,
            "payload_noise_variance": 1.0,
            "spatial_correlation": 0.4,
            "reference_maximum_abs_z_disagreement": 1e-6,
            "reference_maximum_abs_Z_disagreement": 1e-6,
            "controller_quadrature_moment_tolerance": 0.001,
            "controller_quadrature_second_moment_tolerance": 0.001,
            "controller_maximum_split_rhat": 1.05,
            "controller_maximum_chain_mean_disagreement": 0.05,
            "p95_quantile": 0.95,
            "p95_maximum_absolute_z_error": 0.005,
            "p95_maximum_absolute_Z_error": 0.01,
            "maximum_channel_mean_nse": 0.05,
            "maximum_channel_covariance_normalized_frobenius_error": 0.10,
        }
        for name, expected in reals.items():
            _require_real(getattr(self, name), label=name, expected=expected)
        if self.pilot_assignment != "cyclic":
            raise ValueError("pilot_assignment must equal cyclic")
        if self.phase_error_model != "matched_uniform":
            raise ValueError("phase_error_model must equal matched_uniform")
        if self.quantile_method != "linear":
            raise ValueError("quantile_method must equal linear")
        _require_boolean(
            self.controller_accept_empirical_gibbs,
            label="controller_accept_empirical_gibbs",
            expected=True,
        )
        _require_boolean(
            self.controller_retain_gibbs_samples,
            label="controller_retain_gibbs_samples",
            expected=True,
        )

    def controller_config(self) -> ControllerConfig:
        return ControllerConfig(
            quadrature_order=self.controller_quadrature_order,
            quadrature_check_order=self.controller_quadrature_check_order,
            quadrature_maximum_states=self.controller_quadrature_maximum_states,
            quadrature_moment_tolerance=self.controller_quadrature_moment_tolerance,
            quadrature_second_moment_tolerance=(
                self.controller_quadrature_second_moment_tolerance
            ),
            gibbs_chains=self.controller_gibbs_chains,
            gibbs_burn_sweeps=self.controller_gibbs_burn_sweeps,
            gibbs_retained_draws=self.controller_gibbs_retained_draws,
            gibbs_thinning_sweeps=self.controller_gibbs_thinning_sweeps,
            maximum_conditional_proposals=(
                self.controller_maximum_conditional_proposals
            ),
            maximum_split_rhat=self.controller_maximum_split_rhat,
            maximum_chain_mean_disagreement=(
                self.controller_maximum_chain_mean_disagreement
            ),
            accept_empirical_gibbs=self.controller_accept_empirical_gibbs,
            retain_gibbs_samples=self.controller_retain_gibbs_samples,
        )


@dataclass(frozen=True, slots=True)
class SmallReferenceCaseResult:
    case_index: int
    group_size: int
    kappa: int
    delta_deg: int
    case_seed: int
    topology_seed: int
    frame_seed: int
    controller_seed: int
    coarse_order: int
    fine_order: int
    coarse_fine_maximum_absolute_z: float
    coarse_fine_maximum_absolute_Z: float
    reference_resolved: bool
    controller_mode: str
    controller_status: str
    controller_moment_source: str
    maximum_absolute_z_error: float | None
    normalized_squared_z_error: float | None
    maximum_absolute_Z_error: float | None
    normalized_frobenius_Z_error: float | None
    maximum_channel_mean_nse: float | None
    maximum_channel_covariance_normalized_frobenius_error: float | None
    fallback_array_equal: bool | None
    fallback_maximum_absolute_difference: float | None
    all_numeric_outputs_finite: bool

    @property
    def identity_tuple(self) -> tuple[int, int, int, int]:
        return (self.group_size, self.kappa, self.delta_deg, self.case_seed)


@dataclass(frozen=True, slots=True)
class SmallReferenceCampaignResult:
    cases: tuple[SmallReferenceCaseResult, ...]
    verdict: dict[str, Any]


def scenario_for_case(
    case: SmallReferenceCase,
    config: SmallReferenceCampaignConfig,
) -> ScenarioConfig:
    if type(case) is not SmallReferenceCase:
        raise TypeError("case must be a SmallReferenceCase")
    if not isinstance(config, SmallReferenceCampaignConfig):
        raise TypeError("config must be SmallReferenceCampaignConfig")
    return ScenarioConfig(
        access_points=config.access_points,
        antennas_per_ap=config.antennas_per_ap,
        users=case.group_size,
        pilot_symbols=config.pilot_symbols,
        coherence_symbols=config.coherence_symbols,
        area_side=config.area_side,
        minimum_distance=config.minimum_distance,
        pathloss_exponent=config.pathloss_exponent,
        rician_kappa=float(case.kappa),
        residual_phase_half_width_deg=float(case.delta_deg),
        pilot_power=config.pilot_power,
        payload_power=config.payload_power,
        pilot_noise_variance=config.pilot_noise_variance,
        payload_noise_variance=config.payload_noise_variance,
        spatial_correlation=config.spatial_correlation,
        pilot_assignment=config.pilot_assignment,
        phase_error_model=config.phase_error_model,
    )


def small_reference_frame_seed(case: SmallReferenceCase) -> int:
    if type(case) is not SmallReferenceCase:
        raise TypeError("case must be a SmallReferenceCase")
    return derived_seed(
        case.case_seed,
        CAMPAIGN_ID,
        case.group_size,
        case.kappa,
        case.delta_deg,
        "frame",
    )


def small_reference_controller_seed(case: SmallReferenceCase) -> int:
    if type(case) is not SmallReferenceCase:
        raise TypeError("case must be a SmallReferenceCase")
    return derived_seed(
        case.case_seed,
        CAMPAIGN_ID,
        case.group_size,
        case.kappa,
        case.delta_deg,
        "controller",
    )


def _unresolved_case_result(
    case: SmallReferenceCase,
    *,
    frame_seed: int,
    controller_seed: int,
    coarse_order: int,
    fine_order: int,
    coarse_fine_z: float,
    coarse_fine_Z: float,
    finite: bool,
) -> SmallReferenceCaseResult:
    return SmallReferenceCaseResult(
        case_index=case.case_index,
        group_size=case.group_size,
        kappa=case.kappa,
        delta_deg=case.delta_deg,
        case_seed=case.case_seed,
        topology_seed=case.case_seed,
        frame_seed=frame_seed,
        controller_seed=controller_seed,
        coarse_order=coarse_order,
        fine_order=fine_order,
        coarse_fine_maximum_absolute_z=coarse_fine_z,
        coarse_fine_maximum_absolute_Z=coarse_fine_Z,
        reference_resolved=False,
        controller_mode="NOT_RUN_UNRESOLVED_REFERENCE",
        controller_status="UNRESOLVED_NOT_TRUTH",
        controller_moment_source="NONE",
        maximum_absolute_z_error=None,
        normalized_squared_z_error=None,
        maximum_absolute_Z_error=None,
        normalized_frobenius_Z_error=None,
        maximum_channel_mean_nse=None,
        maximum_channel_covariance_normalized_frobenius_error=None,
        fallback_array_equal=None,
        fallback_maximum_absolute_difference=None,
        all_numeric_outputs_finite=finite,
    )


def execute_small_reference_case(
    case: SmallReferenceCase,
    config: SmallReferenceCampaignConfig | None = None,
) -> SmallReferenceCaseResult:
    """Execute one deterministic case without writing or mutating global state."""

    if type(case) is not SmallReferenceCase:
        raise TypeError("case must be a SmallReferenceCase")
    contract = config or SmallReferenceCampaignConfig()
    if not isinstance(contract, SmallReferenceCampaignConfig):
        raise TypeError("config must be SmallReferenceCampaignConfig")
    scenario = scenario_for_case(case, contract)
    topology = generate_topology(scenario, seed=case.case_seed)
    frame_seed = small_reference_frame_seed(case)
    controller_seed = small_reference_controller_seed(case)
    frame = generate_channel_frame(topology, scenario, seed=frame_seed)
    model, observation, users, _ = pilot_group(
        topology, frame, scenario, access_point=0, pilot=0
    )
    if model.group_size != case.group_size or users.size != case.group_size:
        raise RuntimeError("single-pilot case did not contain every registered user")

    graph = build_phase_graph(model, observation)
    coarse_order, fine_order = small_reference_orders(case.group_size)
    coarse = streaming_tensor_moments(
        graph,
        coarse_order,
        maximum_states=contract.reference_maximum_states,
        chunk_size=contract.reference_chunk_size,
    )
    fine = streaming_tensor_moments(
        graph,
        fine_order,
        maximum_states=contract.reference_maximum_states,
        chunk_size=contract.reference_chunk_size,
    )
    coarse_fine_z = maximum_absolute_error(coarse.z, fine.z)
    coarse_fine_Z = maximum_absolute_error(coarse.Z, fine.Z)
    reference_arrays_finite = bool(
        np.all(np.isfinite(coarse.z))
        and np.all(np.isfinite(coarse.Z))
        and np.all(np.isfinite(fine.z))
        and np.all(np.isfinite(fine.Z))
        and math.isfinite(coarse_fine_z)
        and math.isfinite(coarse_fine_Z)
    )
    reference_resolved = bool(
        reference_arrays_finite
        and coarse_fine_z <= contract.reference_maximum_abs_z_disagreement
        and coarse_fine_Z <= contract.reference_maximum_abs_Z_disagreement
    )
    if not reference_resolved:
        return _unresolved_case_result(
            case,
            frame_seed=frame_seed,
            controller_seed=controller_seed,
            coarse_order=coarse_order,
            fine_order=fine_order,
            coarse_fine_z=coarse_fine_z,
            coarse_fine_Z=coarse_fine_Z,
            finite=reference_arrays_finite,
        )

    controller = InferenceController(contract.controller_config())
    result = controller.estimate_all(model, observation, seed=controller_seed)
    inferred = result.phase_moments or result.attempted_phase_moments
    if inferred is None:
        # No phase moments means the numerical engine cannot be compared with
        # the resolved reference. Infinite sentinels fail the numerical gate.
        moment_errors = (math.inf, math.inf, math.inf, math.inf)
        channel_mean_error = math.inf
        channel_covariance_error = math.inf
        moment_source = "NONE"
    else:
        discrepancy = compare_moments(inferred.z, inferred.Z, fine.z, fine.Z)
        moment_errors = (
            discrepancy.maximum_absolute_z,
            discrepancy.normalized_squared_z,
            discrepancy.maximum_absolute_Z,
            discrepancy.normalized_frobenius_Z,
        )
        mean_errors: list[float] = []
        covariance_errors: list[float] = []
        for target in range(model.group_size):
            approximate = channel_moments(
                model, observation, target, inferred.z, inferred.Z
            )
            reference = channel_moments(model, observation, target, fine.z, fine.Z)
            if approximate.covariance is None or reference.covariance is None:
                raise RuntimeError("full phase moments did not produce channel covariance")
            mean_errors.append(
                normalized_squared_error(approximate.mean, reference.mean)
            )
            covariance_errors.append(
                normalized_frobenius_error(
                    approximate.covariance, reference.covariance
                )
            )
        channel_mean_error = max(mean_errors)
        channel_covariance_error = max(covariance_errors)
        moment_source = (
            "ACCEPTED_PHASE_MOMENTS"
            if result.phase_moments is not None
            else "DIAGNOSTIC_REJECTED_ATTEMPTED_PHASE_MOMENTS"
        )

    fallback = result.mode == "CLMMSE_FALLBACK"
    direct_clmmse = np.column_stack(
        [clmmse(model, observation, target) for target in range(model.group_size)]
    ).astype(np.complex128)
    fallback_array_equal = (
        bool(np.array_equal(result.estimates, direct_clmmse)) if fallback else None
    )
    fallback_difference = (
        float(np.max(np.abs(result.estimates - direct_clmmse), initial=0.0))
        if fallback
        else None
    )
    numeric_values = (
        coarse_fine_z,
        coarse_fine_Z,
        *moment_errors,
        channel_mean_error,
        channel_covariance_error,
    )
    all_finite = bool(
        reference_arrays_finite
        and np.all(np.isfinite(result.estimates))
        and np.all(np.isfinite(direct_clmmse))
        and all(math.isfinite(float(value)) for value in numeric_values)
    )
    return SmallReferenceCaseResult(
        case_index=case.case_index,
        group_size=case.group_size,
        kappa=case.kappa,
        delta_deg=case.delta_deg,
        case_seed=case.case_seed,
        topology_seed=case.case_seed,
        frame_seed=frame_seed,
        controller_seed=controller_seed,
        coarse_order=coarse_order,
        fine_order=fine_order,
        coarse_fine_maximum_absolute_z=coarse_fine_z,
        coarse_fine_maximum_absolute_Z=coarse_fine_Z,
        reference_resolved=True,
        controller_mode=result.mode,
        controller_status=result.status,
        controller_moment_source=moment_source,
        maximum_absolute_z_error=moment_errors[0],
        normalized_squared_z_error=moment_errors[1],
        maximum_absolute_Z_error=moment_errors[2],
        normalized_frobenius_Z_error=moment_errors[3],
        maximum_channel_mean_nse=channel_mean_error,
        maximum_channel_covariance_normalized_frobenius_error=(
            channel_covariance_error
        ),
        fallback_array_equal=fallback_array_equal,
        fallback_maximum_absolute_difference=fallback_difference,
        all_numeric_outputs_finite=all_finite,
    )


def _linear_p95(
    values: Sequence[float], config: SmallReferenceCampaignConfig
) -> float:
    if not values or not all(math.isfinite(float(value)) for value in values):
        return math.inf
    return float(
        np.quantile(
            np.asarray(values, dtype=np.float64),
            config.p95_quantile,
            method=config.quantile_method,
        )
    )


def evaluate_small_reference_verdict(
    config: SmallReferenceCampaignConfig,
    rows: Sequence[SmallReferenceCaseResult],
) -> dict[str, Any]:
    """Apply the frozen gate as a pure conjunction with linear p95."""

    if not isinstance(config, SmallReferenceCampaignConfig):
        raise TypeError("config must be SmallReferenceCampaignConfig")
    observed = tuple(rows)
    if any(type(row) is not SmallReferenceCaseResult for row in observed):
        raise TypeError("rows must contain only SmallReferenceCaseResult values")
    expected = build_small_reference_registry()
    exact_registry = bool(
        len(observed) == config.expected_case_count == len(expected)
        and all(
            row.case_index == case.case_index
            and row.identity_tuple == case.identity_tuple
            and row.topology_seed == case.case_seed
            for row, case in zip(observed, expected)
        )
    )
    unresolved_count = sum(not row.reference_resolved for row in observed)

    required_metric_names = (
        "maximum_absolute_z_error",
        "normalized_squared_z_error",
        "maximum_absolute_Z_error",
        "normalized_frobenius_Z_error",
        "maximum_channel_mean_nse",
        "maximum_channel_covariance_normalized_frobenius_error",
    )
    nonfinite_count = 0
    for row in observed:
        finite = bool(
            row.all_numeric_outputs_finite
            and math.isfinite(row.coarse_fine_maximum_absolute_z)
            and math.isfinite(row.coarse_fine_maximum_absolute_Z)
        )
        if row.reference_resolved:
            finite = finite and all(
                getattr(row, name) is not None
                and math.isfinite(float(getattr(row, name)))
                for name in required_metric_names
            )
        if not finite:
            nonfinite_count += 1

    resolved = [row for row in observed if row.reference_resolved]
    p95_z = _linear_p95(
        [float(row.maximum_absolute_z_error) for row in resolved], config
    )
    p95_Z = _linear_p95(
        [float(row.maximum_absolute_Z_error) for row in resolved], config
    )
    maximum_mean_nse = (
        max(float(row.maximum_channel_mean_nse) for row in resolved)
        if resolved
        else math.inf
    )
    maximum_covariance_error = (
        max(
            float(row.maximum_channel_covariance_normalized_frobenius_error)
            for row in resolved
        )
        if resolved
        else math.inf
    )
    fallback_rows = [
        row for row in observed if row.controller_mode == "CLMMSE_FALLBACK"
    ]
    fallbacks_exact = all(
        row.fallback_array_equal is True
        and row.fallback_maximum_absolute_difference is not None
        and row.fallback_maximum_absolute_difference == 0.0
        for row in fallback_rows
    )
    checks = {
        "exact_144_case_registry_in_locked_order": exact_registry,
        "unresolved_reference_cases_zero": unresolved_count == 0,
        "nonfinite_cases_zero": nonfinite_count == 0,
        "p95_maximum_absolute_z_error": (
            p95_z <= config.p95_maximum_absolute_z_error
        ),
        "p95_maximum_absolute_Z_error": (
            p95_Z <= config.p95_maximum_absolute_Z_error
        ),
        "maximum_channel_mean_nse": (
            maximum_mean_nse <= config.maximum_channel_mean_nse
        ),
        "maximum_channel_covariance_normalized_frobenius_error": (
            maximum_covariance_error
            <= config.maximum_channel_covariance_normalized_frobenius_error
        ),
        "every_fallback_exactly_equals_clmmse": fallbacks_exact,
    }
    passed = all(checks.values())
    return {
        "decision": (
            "PASS_SMALL_REFERENCE_QUALIFICATION"
            if passed
            else "FAIL_SMALL_REFERENCE_QUALIFICATION"
        ),
        "all_checks_pass": passed,
        "checks": checks,
        "counts": {
            "observed_cases": len(observed),
            "expected_cases": config.expected_case_count,
            "resolved_cases": len(resolved),
            "unresolved_cases": unresolved_count,
            "nonfinite_cases": nonfinite_count,
            "fallback_cases": len(fallback_rows),
        },
        "metrics": {
            "p95_quantile": config.p95_quantile,
            "quantile_method": config.quantile_method,
            "p95_maximum_absolute_z_error": p95_z,
            "p95_maximum_absolute_Z_error": p95_Z,
            "maximum_channel_mean_nse": maximum_mean_nse,
            "maximum_channel_covariance_normalized_frobenius_error": (
                maximum_covariance_error
            ),
        },
    }


def run_small_reference_campaign(
    config: SmallReferenceCampaignConfig,
    *,
    acknowledge_locked_preregistration: bool = False,
) -> SmallReferenceCampaignResult:
    """Execute the canonical registry only after an explicit external lock ack."""

    if not isinstance(config, SmallReferenceCampaignConfig):
        raise TypeError("config must be SmallReferenceCampaignConfig")
    if acknowledge_locked_preregistration is not True:
        raise RuntimeError("small-reference stage requires a verified lock acknowledgement")
    registry = build_small_reference_registry()
    preflight = preflight_small_reference_registry(registry)
    if int(preflight["case_count"]) != config.expected_case_count:
        raise RuntimeError("small-reference registry count differs from the fixed contract")
    rows = tuple(execute_small_reference_case(case, config) for case in registry)
    return SmallReferenceCampaignResult(
        cases=rows,
        verdict=evaluate_small_reference_verdict(config, rows),
    )


__all__ = [
    "CAMPAIGN_ID",
    "EXPECTED_CASE_COUNT",
    "SmallReferenceCampaignConfig",
    "SmallReferenceCampaignResult",
    "SmallReferenceCaseResult",
    "evaluate_small_reference_verdict",
    "execute_small_reference_case",
    "run_small_reference_campaign",
    "scenario_for_case",
    "small_reference_controller_seed",
    "small_reference_frame_seed",
]
