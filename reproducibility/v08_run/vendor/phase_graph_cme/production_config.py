"""Strict configuration models for R2C01 production engineering.

The pre-lock qualification is deliberately a non-evidence activity.  Locked
scientific stages use a separate schema and cannot be enabled by changing a
Boolean in this document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PRELOCK_SCHEMA_VERSION = "r2c01-prelock-engineering/1.0"
PRELOCK_STAGE = "PRELOCK_ENGINEERING_NON_EVIDENCE"
PRELOCK_BOUNDARY = "PRELOCK_ENGINEERING_NON_EVIDENCE_NEVER_FINAL_SCIENTIFIC_EVIDENCE"
PRELOCK_LOCK_STATUS = "UNLOCKED_PRELOCK_NON_EVIDENCE"
PRELOCK_REQUIRED_OUTPUT_FILES = (
    "TASK_REGISTRY.jsonl",
    "SEED_REGISTRY.jsonl",
    "PROGRESS.jsonl",
    "controller_outcomes.jsonl",
    "controller_compact_traces.jsonl",
    "reference_metrics.jsonl",
    "resource_metrics.jsonl",
    "summary.json",
    "run_manifest.json",
    "run_manifest.sha256",
    "COMPLETE.json",
)


class ProductionConfigurationError(ValueError):
    """Raised when a production-engineering document fails closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ProductionConfigurationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProductionConfigurationError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProductionConfigurationError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionConfigurationError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) < minimum
    ):
        raise ProductionConfigurationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ProductionConfigurationError(f"{label} must be numeric and not Boolean")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProductionConfigurationError(f"{label} is outside its finite domain")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ProductionConfigurationError(f"{label} must be Boolean")
    return bool(value)


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProductionConfigurationError(f"{label} must be a sequence")
    result = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if not result or len(set(result)) != len(result):
        raise ProductionConfigurationError(f"{label} must be nonempty and unique")
    return result


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProductionConfigurationError(f"{label} must be a sequence")
    result = tuple(_integer(item, f"{label}[{index}]", 0) for index, item in enumerate(value))
    if not result or len(set(result)) != len(result):
        raise ProductionConfigurationError(f"{label} must be nonempty and unique")
    return result


def _relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    if "\\" in text:
        raise ProductionConfigurationError(f"{label} must use portable forward slashes")
    path = PurePosixPath(text)
    if path.is_absolute() or text.startswith("//") or ":" in path.parts[0]:
        raise ProductionConfigurationError(f"{label} must be a project-relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ProductionConfigurationError(f"{label} contains an unsafe component")
    return text


@dataclass(frozen=True)
class ProductionCondition:
    id: str
    kappa: float
    delta_deg: float
    role: str

    @property
    def is_core(self) -> bool:
        return self.role in {"core", "core_hard"}


@dataclass(frozen=True)
class SourceSystemSpec:
    access_points: int
    antennas_per_ap: int
    users: int
    pilot_symbols: int
    coherence_symbols: int
    area_side: float
    minimum_distance: float
    pathloss_exponent: float
    pilot_power: float
    payload_power: float
    pilot_noise_variance: float
    payload_noise_variance: float
    spatial_correlation: float
    pilot_assignment: str
    phase_error_model: str


@dataclass(frozen=True)
class ProductionControllerSpec:
    quadrature_order: int
    quadrature_check_order: int
    quadrature_maximum_states: int
    quadrature_moment_tolerance: float
    quadrature_second_moment_tolerance: float
    gibbs_chains: int
    gibbs_burn_sweeps: int
    gibbs_retained_draws: int
    gibbs_thinning_sweeps: int
    maximum_conditional_proposals: int
    conditional_batch_size: int
    maximum_split_rhat: float
    maximum_chain_mean_disagreement: float
    accept_empirical_gibbs: bool
    retain_gibbs_samples: bool


@dataclass(frozen=True)
class TaskSelectionSpec:
    tasks_per_frame: int
    references_per_condition: int
    ranking_serialization: str
    reference_pool: str


@dataclass(frozen=True)
class ReferenceResolutionSpec:
    maximum_split_rhat: float
    maximum_chain_mean_disagreement_z: float
    maximum_chain_mean_disagreement_Z: float
    independent_repeat_maximum_abs_z: float
    independent_repeat_maximum_abs_Z: float
    final_reference_rule: str


@dataclass(frozen=True)
class HighBudgetReferenceSpec:
    chains: int
    burn_sweeps: int
    retained_draws: int
    thinning_sweeps: int
    maximum_conditional_proposals: int
    conditional_batch_size: int
    independent_repeat_required: bool
    raw_complex_phasor_draws_required: bool
    resolution: ReferenceResolutionSpec


@dataclass(frozen=True)
class PrelockPassRules:
    expected_controller_tasks: int
    expected_reference_tasks: int
    minimum_resolved_reference_tasks: int
    every_core_condition_has_resolved_reference: bool
    minimum_core_nonlinear_acceptance_fraction: float
    nonfinite_outputs: int
    unrecorded_exceptions: int
    every_fallback_exactly_equals_clmmse: bool
    trace_reconstruction_required: bool
    deterministic_registry_required: bool


@dataclass(frozen=True)
class PrelockResourceLimits:
    maximum_wall_seconds: float
    maximum_process_rss_bytes: int
    maximum_output_bytes: int
    blas_threads_per_process: int
    process_workers: int


@dataclass(frozen=True)
class ReservedSeedNamespaces:
    strong_affine_base_seed: int
    strong_affine_topology_seeds: tuple[int, ...]
    strong_affine_bootstrap_seed: int
    small_reference_case_seeds: tuple[int, ...]
    source_scale_base_seed: int
    source_scale_topology_seeds: tuple[int, ...]
    heldout_base_seed: int
    heldout_topology_seeds: tuple[int, ...]
    robustness_base_seed: int
    robustness_topology_seeds: tuple[int, ...]
    future_bootstrap_seeds: tuple[int, ...]

    @property
    def explicit_values(self) -> tuple[int, ...]:
        return (
            self.strong_affine_base_seed,
            *self.strong_affine_topology_seeds,
            self.strong_affine_bootstrap_seed,
            *self.small_reference_case_seeds,
            self.source_scale_base_seed,
            *self.source_scale_topology_seeds,
            self.heldout_base_seed,
            *self.heldout_topology_seeds,
            self.robustness_base_seed,
            *self.robustness_topology_seeds,
            *self.future_bootstrap_seeds,
        )


@dataclass(frozen=True)
class PrelockQualificationConfig:
    schema_version: str
    candidate_id: str
    experiment_id: str
    stage_id: str
    evidence_role: str
    decision_eligible: bool
    lock_status: str
    executable: bool
    base_seed: int
    reference_repeat_base_seed: int
    topology_seeds: tuple[int, ...]
    conditions: tuple[ProductionCondition, ...]
    realizations_per_topology_condition: int
    system: SourceSystemSpec
    controller: ProductionControllerSpec
    task_selection: TaskSelectionSpec
    high_budget_reference: HighBudgetReferenceSpec
    pass_rules: PrelockPassRules
    resource_limits: PrelockResourceLimits
    reserved_seed_namespaces: ReservedSeedNamespaces
    output_directory: str
    required_output_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PRELOCK_SCHEMA_VERSION:
            raise ProductionConfigurationError("unsupported pre-lock schema version")
        if self.candidate_id != "R2C01_EXACT_PHASE_CME":
            raise ProductionConfigurationError("pre-lock config targets the wrong candidate")
        if self.stage_id != PRELOCK_STAGE or self.evidence_role != PRELOCK_BOUNDARY:
            raise ProductionConfigurationError("pre-lock scientific boundary is not exact")
        if self.decision_eligible or self.lock_status != PRELOCK_LOCK_STATUS:
            raise ProductionConfigurationError("pre-lock config cannot be evidence eligible or locked")
        if not self.executable:
            raise ProductionConfigurationError("pre-lock engineering config is not executable")
        if self.base_seed == self.reference_repeat_base_seed:
            raise ProductionConfigurationError("controller and reference-repeat bases must differ")
        if len(self.topology_seeds) != 4:
            raise ProductionConfigurationError("pre-lock qualification requires four topologies")
        if len(self.conditions) != 5 or len({item.id for item in self.conditions}) != 5:
            raise ProductionConfigurationError("pre-lock qualification requires five unique conditions")
        expected_roles = ("near_linear_control", "core", "core", "core_hard", "hard_negative")
        if tuple(item.role for item in self.conditions) != expected_roles:
            raise ProductionConfigurationError("condition roles or order differ from the fixed design")
        if self.realizations_per_topology_condition != 1:
            raise ProductionConfigurationError("pre-lock qualification fixes one frame per cell")
        if self.system.users % self.system.pilot_symbols != 0:
            raise ProductionConfigurationError("pre-lock pilot groups must be balanced")
        task_count = (
            len(self.topology_seeds)
            * len(self.conditions)
            * self.realizations_per_topology_condition
            * self.task_selection.tasks_per_frame
        )
        if task_count != self.pass_rules.expected_controller_tasks:
            raise ProductionConfigurationError("registered controller-task count is inconsistent")
        reference_count = len(self.conditions) * self.task_selection.references_per_condition
        if reference_count != self.pass_rules.expected_reference_tasks:
            raise ProductionConfigurationError("registered reference-task count is inconsistent")
        explicit = (
            self.base_seed,
            self.reference_repeat_base_seed,
            *self.topology_seeds,
            *self.reserved_seed_namespaces.explicit_values,
        )
        if len(set(explicit)) != len(explicit):
            raise ProductionConfigurationError("explicit pre-lock/future seed namespaces overlap")
        if self.required_output_files != PRELOCK_REQUIRED_OUTPUT_FILES:
            raise ProductionConfigurationError(
                "required_output_files differs from the fixed pre-lock v1 contract"
            )


def _parse_conditions(value: object) -> tuple[ProductionCondition, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProductionConfigurationError("conditions must be a sequence")
    result: list[ProductionCondition] = []
    for index, item in enumerate(value):
        row = _mapping(item, f"conditions[{index}]")
        _exact_keys(row, {"id", "kappa", "delta_deg", "role"}, f"conditions[{index}]")
        kappa = _number(row["kappa"], f"conditions[{index}].kappa", minimum=0.0)
        delta = _number(row["delta_deg"], f"conditions[{index}].delta_deg", minimum=0.0)
        if delta > 180.0:
            raise ProductionConfigurationError("condition delta_deg exceeds 180")
        result.append(
            ProductionCondition(
                id=_string(row["id"], f"conditions[{index}].id"),
                kappa=kappa,
                delta_deg=delta,
                role=_string(row["role"], f"conditions[{index}].role"),
            )
        )
    return tuple(result)


def _parse_system(value: object) -> SourceSystemSpec:
    row = _mapping(value, "system")
    fields = {
        "access_points", "antennas_per_ap", "users", "pilot_symbols",
        "coherence_symbols", "area_side", "minimum_distance", "pathloss_exponent",
        "pilot_power", "payload_power", "pilot_noise_variance",
        "payload_noise_variance", "spatial_correlation", "pilot_assignment",
        "phase_error_model",
    }
    _exact_keys(row, fields, "system")
    result = SourceSystemSpec(
        access_points=_integer(row["access_points"], "system.access_points", 1),
        antennas_per_ap=_integer(row["antennas_per_ap"], "system.antennas_per_ap", 1),
        users=_integer(row["users"], "system.users", 1),
        pilot_symbols=_integer(row["pilot_symbols"], "system.pilot_symbols", 1),
        coherence_symbols=_integer(row["coherence_symbols"], "system.coherence_symbols", 2),
        area_side=_number(row["area_side"], "system.area_side", minimum=np.finfo(float).tiny),
        minimum_distance=_number(row["minimum_distance"], "system.minimum_distance", minimum=np.finfo(float).tiny),
        pathloss_exponent=_number(row["pathloss_exponent"], "system.pathloss_exponent", minimum=np.finfo(float).tiny),
        pilot_power=_number(row["pilot_power"], "system.pilot_power", minimum=np.finfo(float).tiny),
        payload_power=_number(row["payload_power"], "system.payload_power", minimum=np.finfo(float).tiny),
        pilot_noise_variance=_number(row["pilot_noise_variance"], "system.pilot_noise_variance", minimum=np.finfo(float).tiny),
        payload_noise_variance=_number(row["payload_noise_variance"], "system.payload_noise_variance", minimum=np.finfo(float).tiny),
        spatial_correlation=_number(row["spatial_correlation"], "system.spatial_correlation", minimum=0.0),
        pilot_assignment=_string(row["pilot_assignment"], "system.pilot_assignment"),
        phase_error_model=_string(row["phase_error_model"], "system.phase_error_model"),
    )
    if result.pilot_symbols > result.users or result.pilot_symbols >= result.coherence_symbols:
        raise ProductionConfigurationError("system pilot dimensions are invalid")
    if not 0.0 <= result.spatial_correlation < 1.0:
        raise ProductionConfigurationError("system.spatial_correlation must lie in [0,1)")
    if result.pilot_assignment != "cyclic" or result.phase_error_model != "matched_uniform":
        raise ProductionConfigurationError("pre-lock source system fixes cyclic pilots and matched phases")
    return result


def _parse_controller(value: object) -> ProductionControllerSpec:
    row = _mapping(value, "controller")
    fields = {
        "quadrature_order", "quadrature_check_order", "quadrature_maximum_states",
        "quadrature_moment_tolerance", "quadrature_second_moment_tolerance",
        "gibbs_chains", "gibbs_burn_sweeps", "gibbs_retained_draws",
        "gibbs_thinning_sweeps", "maximum_conditional_proposals",
        "conditional_batch_size",
        "maximum_split_rhat", "maximum_chain_mean_disagreement",
        "accept_empirical_gibbs", "retain_gibbs_samples",
    }
    _exact_keys(row, fields, "controller")
    result = ProductionControllerSpec(
        quadrature_order=_integer(row["quadrature_order"], "controller.quadrature_order", 1),
        quadrature_check_order=_integer(row["quadrature_check_order"], "controller.quadrature_check_order", 2),
        quadrature_maximum_states=_integer(row["quadrature_maximum_states"], "controller.quadrature_maximum_states", 1),
        quadrature_moment_tolerance=_number(row["quadrature_moment_tolerance"], "controller.quadrature_moment_tolerance", minimum=0.0),
        quadrature_second_moment_tolerance=_number(row["quadrature_second_moment_tolerance"], "controller.quadrature_second_moment_tolerance", minimum=0.0),
        gibbs_chains=_integer(row["gibbs_chains"], "controller.gibbs_chains", 2),
        gibbs_burn_sweeps=_integer(row["gibbs_burn_sweeps"], "controller.gibbs_burn_sweeps", 0),
        gibbs_retained_draws=_integer(row["gibbs_retained_draws"], "controller.gibbs_retained_draws", 4),
        gibbs_thinning_sweeps=_integer(row["gibbs_thinning_sweeps"], "controller.gibbs_thinning_sweeps", 1),
        maximum_conditional_proposals=_integer(row["maximum_conditional_proposals"], "controller.maximum_conditional_proposals", 1),
        conditional_batch_size=_integer(
            row["conditional_batch_size"], "controller.conditional_batch_size", 1
        ),
        maximum_split_rhat=_number(row["maximum_split_rhat"], "controller.maximum_split_rhat", minimum=np.finfo(float).tiny),
        maximum_chain_mean_disagreement=_number(row["maximum_chain_mean_disagreement"], "controller.maximum_chain_mean_disagreement", minimum=0.0),
        accept_empirical_gibbs=_boolean(row["accept_empirical_gibbs"], "controller.accept_empirical_gibbs"),
        retain_gibbs_samples=_boolean(row["retain_gibbs_samples"], "controller.retain_gibbs_samples"),
    )
    if result.quadrature_check_order <= result.quadrature_order:
        raise ProductionConfigurationError("controller check order must exceed coarse order")
    if not result.accept_empirical_gibbs or not result.retain_gibbs_samples:
        raise ProductionConfigurationError("pre-lock qualification requires accepted Gibbs and trace opt-in")
    return result


def _parse_task_selection(value: object) -> TaskSelectionSpec:
    row = _mapping(value, "task_selection")
    fields = {"tasks_per_frame", "references_per_condition", "ranking_serialization", "reference_pool"}
    _exact_keys(row, fields, "task_selection")
    result = TaskSelectionSpec(
        tasks_per_frame=_integer(row["tasks_per_frame"], "task_selection.tasks_per_frame", 1),
        references_per_condition=_integer(row["references_per_condition"], "task_selection.references_per_condition", 1),
        ranking_serialization=_string(row["ranking_serialization"], "task_selection.ranking_serialization"),
        reference_pool=_string(row["reference_pool"], "task_selection.reference_pool"),
    )
    if result.ranking_serialization != "canonical_ascii_json_compact" or result.reference_pool != "selected_controller_tasks_within_condition":
        raise ProductionConfigurationError("task ranking or reference pool contract differs")
    return result


def _parse_high_reference(value: object) -> HighBudgetReferenceSpec:
    row = _mapping(value, "high_budget_reference")
    fields = {
        "chains", "burn_sweeps", "retained_draws", "thinning_sweeps",
        "maximum_conditional_proposals", "conditional_batch_size",
        "independent_repeat_required",
        "raw_complex_phasor_draws_required", "resolution",
    }
    _exact_keys(row, fields, "high_budget_reference")
    resolution_row = _mapping(row["resolution"], "high_budget_reference.resolution")
    resolution_fields = {
        "maximum_split_rhat", "maximum_chain_mean_disagreement_z",
        "maximum_chain_mean_disagreement_Z", "independent_repeat_maximum_abs_z",
        "independent_repeat_maximum_abs_Z", "final_reference_rule",
    }
    _exact_keys(resolution_row, resolution_fields, "high_budget_reference.resolution")
    resolution = ReferenceResolutionSpec(
        maximum_split_rhat=_number(resolution_row["maximum_split_rhat"], "resolution.maximum_split_rhat", minimum=np.finfo(float).tiny),
        maximum_chain_mean_disagreement_z=_number(resolution_row["maximum_chain_mean_disagreement_z"], "resolution.maximum_chain_mean_disagreement_z", minimum=0.0),
        maximum_chain_mean_disagreement_Z=_number(resolution_row["maximum_chain_mean_disagreement_Z"], "resolution.maximum_chain_mean_disagreement_Z", minimum=0.0),
        independent_repeat_maximum_abs_z=_number(resolution_row["independent_repeat_maximum_abs_z"], "resolution.independent_repeat_maximum_abs_z", minimum=0.0),
        independent_repeat_maximum_abs_Z=_number(resolution_row["independent_repeat_maximum_abs_Z"], "resolution.independent_repeat_maximum_abs_Z", minimum=0.0),
        final_reference_rule=_string(resolution_row["final_reference_rule"], "resolution.final_reference_rule"),
    )
    if resolution.final_reference_rule != "equal_sample_count_pooled_two_repeats":
        raise ProductionConfigurationError("high-budget final reference rule differs")
    return HighBudgetReferenceSpec(
        chains=_integer(row["chains"], "high_budget_reference.chains", 2),
        burn_sweeps=_integer(row["burn_sweeps"], "high_budget_reference.burn_sweeps", 0),
        retained_draws=_integer(row["retained_draws"], "high_budget_reference.retained_draws", 4),
        thinning_sweeps=_integer(row["thinning_sweeps"], "high_budget_reference.thinning_sweeps", 1),
        maximum_conditional_proposals=_integer(row["maximum_conditional_proposals"], "high_budget_reference.maximum_conditional_proposals", 1),
        conditional_batch_size=_integer(
            row["conditional_batch_size"],
            "high_budget_reference.conditional_batch_size",
            1,
        ),
        independent_repeat_required=_boolean(row["independent_repeat_required"], "high_budget_reference.independent_repeat_required"),
        raw_complex_phasor_draws_required=_boolean(row["raw_complex_phasor_draws_required"], "high_budget_reference.raw_complex_phasor_draws_required"),
        resolution=resolution,
    )


def _parse_pass_rules(value: object) -> PrelockPassRules:
    row = _mapping(value, "pass_rules")
    fields = {
        "expected_controller_tasks", "expected_reference_tasks",
        "minimum_resolved_reference_tasks", "every_core_condition_has_resolved_reference",
        "minimum_core_nonlinear_acceptance_fraction", "nonfinite_outputs",
        "unrecorded_exceptions", "every_fallback_exactly_equals_clmmse",
        "trace_reconstruction_required", "deterministic_registry_required",
    }
    _exact_keys(row, fields, "pass_rules")
    result = PrelockPassRules(
        expected_controller_tasks=_integer(row["expected_controller_tasks"], "pass_rules.expected_controller_tasks", 1),
        expected_reference_tasks=_integer(row["expected_reference_tasks"], "pass_rules.expected_reference_tasks", 1),
        minimum_resolved_reference_tasks=_integer(row["minimum_resolved_reference_tasks"], "pass_rules.minimum_resolved_reference_tasks", 1),
        every_core_condition_has_resolved_reference=_boolean(row["every_core_condition_has_resolved_reference"], "pass_rules.every_core_condition_has_resolved_reference"),
        minimum_core_nonlinear_acceptance_fraction=_number(row["minimum_core_nonlinear_acceptance_fraction"], "pass_rules.minimum_core_nonlinear_acceptance_fraction", minimum=0.0),
        nonfinite_outputs=_integer(row["nonfinite_outputs"], "pass_rules.nonfinite_outputs", 0),
        unrecorded_exceptions=_integer(row["unrecorded_exceptions"], "pass_rules.unrecorded_exceptions", 0),
        every_fallback_exactly_equals_clmmse=_boolean(row["every_fallback_exactly_equals_clmmse"], "pass_rules.every_fallback_exactly_equals_clmmse"),
        trace_reconstruction_required=_boolean(row["trace_reconstruction_required"], "pass_rules.trace_reconstruction_required"),
        deterministic_registry_required=_boolean(row["deterministic_registry_required"], "pass_rules.deterministic_registry_required"),
    )
    if not 0.0 <= result.minimum_core_nonlinear_acceptance_fraction <= 1.0:
        raise ProductionConfigurationError("core acceptance fraction must lie in [0,1]")
    if result.nonfinite_outputs != 0 or result.unrecorded_exceptions != 0:
        raise ProductionConfigurationError("pre-lock failure budgets must be zero")
    return result


def _parse_resource_limits(value: object) -> PrelockResourceLimits:
    row = _mapping(value, "resource_limits")
    fields = {
        "maximum_wall_seconds", "maximum_process_rss_bytes", "maximum_output_bytes",
        "blas_threads_per_process", "process_workers",
    }
    _exact_keys(row, fields, "resource_limits")
    result = PrelockResourceLimits(
        maximum_wall_seconds=_number(row["maximum_wall_seconds"], "resource_limits.maximum_wall_seconds", minimum=np.finfo(float).tiny),
        maximum_process_rss_bytes=_integer(row["maximum_process_rss_bytes"], "resource_limits.maximum_process_rss_bytes", 1),
        maximum_output_bytes=_integer(row["maximum_output_bytes"], "resource_limits.maximum_output_bytes", 1),
        blas_threads_per_process=_integer(row["blas_threads_per_process"], "resource_limits.blas_threads_per_process", 1),
        process_workers=_integer(row["process_workers"], "resource_limits.process_workers", 1),
    )
    if result.blas_threads_per_process != 1 or result.process_workers != 1:
        raise ProductionConfigurationError("pre-lock v1 fixes one process and one BLAS thread")
    return result


def _parse_reserved(value: object) -> ReservedSeedNamespaces:
    row = _mapping(value, "reserved_seed_namespaces")
    fields = {
        "strong_affine_base_seed", "strong_affine_topology_seeds",
        "strong_affine_bootstrap_seed", "small_reference_case_seeds",
        "source_scale_base_seed", "source_scale_topology_seeds",
        "heldout_base_seed", "heldout_topology_seeds", "robustness_base_seed",
        "robustness_topology_seeds", "future_bootstrap_seeds",
    }
    _exact_keys(row, fields, "reserved_seed_namespaces")
    return ReservedSeedNamespaces(
        strong_affine_base_seed=_integer(row["strong_affine_base_seed"], "reserved.strong_affine_base_seed"),
        strong_affine_topology_seeds=_integer_list(row["strong_affine_topology_seeds"], "reserved.strong_affine_topology_seeds"),
        strong_affine_bootstrap_seed=_integer(row["strong_affine_bootstrap_seed"], "reserved.strong_affine_bootstrap_seed"),
        small_reference_case_seeds=_integer_list(row["small_reference_case_seeds"], "reserved.small_reference_case_seeds"),
        source_scale_base_seed=_integer(row["source_scale_base_seed"], "reserved.source_scale_base_seed"),
        source_scale_topology_seeds=_integer_list(row["source_scale_topology_seeds"], "reserved.source_scale_topology_seeds"),
        heldout_base_seed=_integer(row["heldout_base_seed"], "reserved.heldout_base_seed"),
        heldout_topology_seeds=_integer_list(row["heldout_topology_seeds"], "reserved.heldout_topology_seeds"),
        robustness_base_seed=_integer(row["robustness_base_seed"], "reserved.robustness_base_seed"),
        robustness_topology_seeds=_integer_list(row["robustness_topology_seeds"], "reserved.robustness_topology_seeds"),
        future_bootstrap_seeds=_integer_list(row["future_bootstrap_seeds"], "reserved.future_bootstrap_seeds"),
    )


def load_prelock_config(path: str | Path) -> PrelockQualificationConfig:
    source = Path(path)
    try:
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ProductionConfigurationError(f"cannot read pre-lock configuration: {error}") from error
    root = _mapping(document, "root")
    fields = {
        "schema_version", "candidate_id", "experiment_id", "stage_id",
        "evidence_role", "decision_eligible", "lock_status", "executable",
        "base_seed", "reference_repeat_base_seed", "topology_seeds", "conditions",
        "realizations_per_topology_condition", "system", "controller",
        "task_selection", "high_budget_reference", "pass_rules", "resource_limits",
        "reserved_seed_namespaces", "output_directory", "required_output_files",
    }
    _exact_keys(root, fields, "root")
    result = PrelockQualificationConfig(
        schema_version=_string(root["schema_version"], "schema_version"),
        candidate_id=_string(root["candidate_id"], "candidate_id"),
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        stage_id=_string(root["stage_id"], "stage_id"),
        evidence_role=_string(root["evidence_role"], "evidence_role"),
        decision_eligible=_boolean(root["decision_eligible"], "decision_eligible"),
        lock_status=_string(root["lock_status"], "lock_status"),
        executable=_boolean(root["executable"], "executable"),
        base_seed=_integer(root["base_seed"], "base_seed"),
        reference_repeat_base_seed=_integer(root["reference_repeat_base_seed"], "reference_repeat_base_seed"),
        topology_seeds=_integer_list(root["topology_seeds"], "topology_seeds"),
        conditions=_parse_conditions(root["conditions"]),
        realizations_per_topology_condition=_integer(root["realizations_per_topology_condition"], "realizations_per_topology_condition", 1),
        system=_parse_system(root["system"]),
        controller=_parse_controller(root["controller"]),
        task_selection=_parse_task_selection(root["task_selection"]),
        high_budget_reference=_parse_high_reference(root["high_budget_reference"]),
        pass_rules=_parse_pass_rules(root["pass_rules"]),
        resource_limits=_parse_resource_limits(root["resource_limits"]),
        reserved_seed_namespaces=_parse_reserved(root["reserved_seed_namespaces"]),
        output_directory=_relative_path(root["output_directory"], "output_directory"),
        required_output_files=_string_list(root["required_output_files"], "required_output_files"),
    )
    if any(_relative_path(name, "required_output_files item") != name for name in result.required_output_files):
        raise ProductionConfigurationError("required output file path is not portable")
    return result
