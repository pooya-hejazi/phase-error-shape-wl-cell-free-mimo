"""Excluded-seed engineering qualification for the R2C01 production stack.

Nothing emitted by this module is final scientific evidence.  Its sole verdict
is whether the production preregistration may proceed to an independent lock
audit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baselines import clmmse
from .controller import ControllerConfig, InferenceController
from .inference import GibbsSamplingError, heat_bath_gibbs
from .model import build_phase_graph
from .production_artifacts import (
    AtomicRunDirectory,
    append_jsonl_record,
    canonical_json_bytes,
    output_tree_bytes,
    peak_process_rss_bytes,
    require_single_thread_runtime,
    sha256_array,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from .production_config import (
    PRELOCK_BOUNDARY,
    PrelockQualificationConfig,
    ProductionCondition,
    load_prelock_config,
)
from .production_registry import canonical_ascii_json, derived_seed
from .scenario import (
    ScenarioConfig,
    generate_channel_frame,
    generate_topology,
    pilot_group,
)
from .trace_io import (
    build_compact_gibbs_statistics,
    reconstruct_compact_diagnostics,
    reconstruct_compact_moments,
    write_gibbs_trace_bundle,
)


CONTROLLER_OUTCOME_FIELDS = (
    "sequence_index",
    "task_id",
    "condition_id",
    "topology_seed",
    "realization_index",
    "access_point",
    "pilot",
    "global_user_ids",
    "mode",
    "status",
    "fallback_reason",
    "quadrature_rejection_reason",
    "maximum_split_rhat",
    "maximum_chain_mean_disagreement",
    "dobrushin_eta",
    "logical_proposal_count",
    "generated_candidate_value_count",
    "controller_estimates",
    "direct_clmmse_estimates",
    "controller_estimates_sha256",
    "direct_clmmse_estimates_sha256",
    "fallback_array_equal",
    "fallback_maximum_absolute_difference",
    "all_numeric_outputs_finite",
    "compact_trace_sha256",
    "compact_trace_reconstructs",
    "structured_gibbs_failure",
)

COMPACT_TRACE_FIELDS = (
    "sequence_index",
    "task_id",
    "condition_id",
    "topology_seed",
    "realization_index",
    "access_point",
    "pilot",
    "global_user_ids",
    "trace_kind",
    "compact_sha256",
    "document",
    "structured_failure",
)

PROGRESS_FIELDS = (
    "sequence_index",
    "work_kind",
    "task_id",
    "repeat_id",
    "status",
    "condition_id",
    "topology_seed",
    "realization_index",
    "access_point",
    "pilot",
    "cumulative_controller_tasks",
    "cumulative_reference_repeats",
    "cumulative_exception_count",
    "peak_process_rss_bytes",
    "completed_unix",
)

RESOURCE_FIELDS = (
    "sequence_index",
    "work_kind",
    "task_id",
    "repeat_id",
    "elapsed_seconds",
    "peak_process_rss_bytes",
    "logical_proposal_count",
    "generated_candidate_value_count",
    "output_bytes_after_work",
)

REFERENCE_FIELDS = (
    "sequence_index",
    "task_id",
    "condition_id",
    "topology_seed",
    "realization_index",
    "access_point",
    "pilot",
    "primary_seed",
    "repeat_seed",
    "primary_manifest",
    "repeat_manifest",
    "primary_payload_sha256",
    "repeat_payload_sha256",
    "primary_maximum_split_rhat",
    "repeat_maximum_split_rhat",
    "primary_maximum_chain_mean_disagreement_z",
    "repeat_maximum_chain_mean_disagreement_z",
    "primary_maximum_chain_mean_disagreement_Z",
    "repeat_maximum_chain_mean_disagreement_Z",
    "repeat_maximum_abs_z",
    "repeat_maximum_abs_Z",
    "all_diagnostics_finite",
    "resolved",
    "resolution_failures",
    "pooled_reference_z",
    "pooled_reference_Z",
    "controller_mode",
    "controller_maximum_abs_z_error",
    "controller_maximum_abs_Z_error",
)

RUNTIME_SOURCE_PATHS = (
    "src/phase_graph_cme/__init__.py",
    "src/phase_graph_cme/prelock_qualification.py",
    "src/phase_graph_cme/production_config.py",
    "src/phase_graph_cme/production_registry.py",
    "src/phase_graph_cme/reference_runner.py",
    "src/phase_graph_cme/production_artifacts.py",
    "src/phase_graph_cme/controller.py",
    "src/phase_graph_cme/inference.py",
    "src/phase_graph_cme/trace_io.py",
    "src/phase_graph_cme/baselines.py",
    "src/phase_graph_cme/moments.py",
    "src/phase_graph_cme/diagnostics.py",
    "src/phase_graph_cme/posterior.py",
    "src/phase_graph_cme/model.py",
    "src/phase_graph_cme/scenario.py",
    "scripts/validate_r2c01_prelock_output.py",
)

RUNTIME_TEST_PATHS = (
    "tests/unit/test_phase_graph_production_config.py",
    "tests/unit/test_phase_graph_production_artifacts.py",
    "tests/unit/test_phase_graph_production_registry.py",
    "tests/unit/test_phase_graph_reference_runner.py",
    "tests/unit/test_phase_graph_controller_trace_optin.py",
    "tests/unit/test_phase_graph_sampler_accounting.py",
    "tests/unit/test_phase_graph_prelock_qualification.py",
    "tests/unit/test_r2c01_prelock_output_validator.py",
)


def _frame_id(task: Any) -> str:
    return (
        f"f{int(task.topology_index):02d}c{int(task.condition_index):02d}"
        f"r{int(task.realization_index):02d}"
    )


def _task_id(task: Any) -> str:
    return (
        f"t{int(task.topology_index):02d}c{int(task.condition_index):02d}"
        f"r{int(task.realization_index):02d}a{int(task.ap_index):03d}"
        f"p{int(task.pilot_index):02d}"
    )


@dataclass(frozen=True)
class _PrelockReferenceTask:
    task: Any
    primary_seed: int
    repeat_seed: int
    primary_base_seed: int
    repeat_base_seed: int
    primary_seed_tags_json: str
    repeat_seed_tags_json: str

    @property
    def task_id(self) -> str:
        return _task_id(self.task)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)


def _complex_json(value: object) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("cannot serialize nonfinite complex values")
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _finite_float_or_none(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _runtime_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise RuntimeError("runtime summary requires a nonempty finite vector")
    if np.any(array < 0.0):
        raise RuntimeError("runtime measurements must be nonnegative")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "maximum": float(np.max(array)),
        "sum": float(np.sum(array)),
        "quantile_method": "numpy_linear",
    }


def _structured_gibbs_failure_is_complete(
    value: object, *, chains: int, proposal_cap: int
) -> bool:
    required = {
        "chain_index",
        "phase",
        "burn_or_thinning_sweep_index",
        "retained_draw_index",
        "completed_chain_count",
        "logical_proposal_count",
        "generated_candidate_value_count",
        "per_chain_logical_proposal_count",
        "per_chain_generated_candidate_value_count",
        "maximum_conditional_proposals",
        "conditional_batch_size",
        "cause",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    integers = (
        value["chain_index"],
        value["burn_or_thinning_sweep_index"],
        value["completed_chain_count"],
        value["logical_proposal_count"],
        value["generated_candidate_value_count"],
        value["maximum_conditional_proposals"],
        value["conditional_batch_size"],
    )
    if any(type(item) is not int for item in integers):
        return False
    chain_index = value["chain_index"]
    if not 0 <= chain_index < chains or value["completed_chain_count"] != chain_index:
        return False
    if value["phase"] not in {"burn", "retained"}:
        return False
    retained_index = value["retained_draw_index"]
    if value["phase"] == "burn":
        if retained_index is not None:
            return False
    elif type(retained_index) is not int or retained_index < 0:
        return False
    logical = value["per_chain_logical_proposal_count"]
    generated = value["per_chain_generated_candidate_value_count"]
    if (
        not isinstance(logical, list)
        or not isinstance(generated, list)
        or len(logical) != chains
        or len(generated) != chains
        or any(type(item) is not int or item < 0 for item in (*logical, *generated))
    ):
        return False
    return bool(
        value["logical_proposal_count"] == sum(logical)
        and value["generated_candidate_value_count"] == sum(generated)
        and value["generated_candidate_value_count"]
        >= value["logical_proposal_count"]
        and value["maximum_conditional_proposals"] == proposal_cap
        and value["conditional_batch_size"] >= 1
        and isinstance(value["cause"], str)
        and bool(value["cause"])
    )


def _hash_paths(root: Path, paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file():
            raise RuntimeError(f"runtime closure path is absent or unsafe: {relative}")
        result[relative] = sha256_file(source)
    return result


def _scenario(
    config: PrelockQualificationConfig, condition: ProductionCondition
) -> ScenarioConfig:
    system = config.system
    return ScenarioConfig(
        access_points=system.access_points,
        antennas_per_ap=system.antennas_per_ap,
        users=system.users,
        pilot_symbols=system.pilot_symbols,
        coherence_symbols=system.coherence_symbols,
        area_side=system.area_side,
        minimum_distance=system.minimum_distance,
        pathloss_exponent=system.pathloss_exponent,
        rician_kappa=condition.kappa,
        residual_phase_half_width_deg=condition.delta_deg,
        pilot_power=system.pilot_power,
        payload_power=system.payload_power,
        pilot_noise_variance=system.pilot_noise_variance,
        payload_noise_variance=system.payload_noise_variance,
        spatial_correlation=system.spatial_correlation,
        pilot_assignment=system.pilot_assignment,
        phase_error_model=system.phase_error_model,
    )


def _controller(config: PrelockQualificationConfig) -> InferenceController:
    specification = config.controller
    return InferenceController(
        ControllerConfig(**asdict(specification))
    )


def _high_reference_kwargs(config: PrelockQualificationConfig) -> dict[str, Any]:
    specification = config.high_budget_reference
    return {
        "chains": specification.chains,
        "burn_sweeps": specification.burn_sweeps,
        "retained_draws": specification.retained_draws,
        "thinning_sweeps": specification.thinning_sweeps,
        "maximum_conditional_proposals": specification.maximum_conditional_proposals,
        "conditional_batch_size": specification.conditional_batch_size,
        "return_samples": True,
    }


def _reference_resolution(config: PrelockQualificationConfig, first: Any, second: Any) -> dict[str, Any]:
    # Imported lazily so the module remains importable while the independent
    # reference implementation is developed and audited separately.
    from .reference_runner import HighBudgetRepeatThresholds, resolve_high_budget_repeat

    rule = config.high_budget_reference.resolution
    thresholds = HighBudgetRepeatThresholds(
        maximum_split_rhat=rule.maximum_split_rhat,
        maximum_chain_mean_disagreement_z=rule.maximum_chain_mean_disagreement_z,
        maximum_chain_mean_disagreement_Z=rule.maximum_chain_mean_disagreement_Z,
        independent_repeat_maximum_abs_z=rule.independent_repeat_maximum_abs_z,
        independent_repeat_maximum_abs_Z=rule.independent_repeat_maximum_abs_Z,
    )
    if first.retained_samples is None or second.retained_samples is None:
        raise RuntimeError("high-budget reference did not retain its raw chains")
    result = resolve_high_budget_repeat(
        first.retained_samples, second.retained_samples, thresholds
    )
    check_names = (
        "primary_split_rhat_pass",
        "repeat_split_rhat_pass",
        "primary_z_chain_disagreement_pass",
        "repeat_z_chain_disagreement_pass",
        "primary_Z_chain_disagreement_pass",
        "repeat_Z_chain_disagreement_pass",
        "independent_repeat_z_pass",
        "independent_repeat_Z_pass",
    )
    return {
        "resolved": bool(result.resolved),
        "status": result.status,
        "primary_maximum_split_rhat": result.primary.maximum_split_rhat,
        "repeat_maximum_split_rhat": result.repeat.maximum_split_rhat,
        "primary_maximum_chain_mean_disagreement_z": (
            result.primary.maximum_chain_mean_disagreement_z
        ),
        "repeat_maximum_chain_mean_disagreement_z": (
            result.repeat.maximum_chain_mean_disagreement_z
        ),
        "primary_maximum_chain_mean_disagreement_Z": (
            result.primary.maximum_chain_mean_disagreement_Z
        ),
        "repeat_maximum_chain_mean_disagreement_Z": (
            result.repeat.maximum_chain_mean_disagreement_Z
        ),
        "repeat_maximum_abs_z": result.independent_repeat_maximum_abs_z,
        "repeat_maximum_abs_Z": result.independent_repeat_maximum_abs_Z,
        "all_diagnostics_finite": bool(
            result.primary.all_finite
            and result.repeat.all_finite
            and np.isfinite(result.independent_repeat_maximum_abs_z)
            and np.isfinite(result.independent_repeat_maximum_abs_Z)
        ),
        "failures": [name for name in check_names if not getattr(result, name)],
    }


def evaluate_prelock_verdict(
    config: PrelockQualificationConfig,
    controller_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int,
    elapsed_seconds: float,
    peak_rss_bytes: int,
    output_bytes: int,
) -> dict[str, Any]:
    """Apply the registered conjunctive engineering gate without retuning."""

    rules = config.pass_rules
    core_ids = {item.id for item in config.conditions if item.is_core}
    core = [row for row in controller_rows if row["condition_id"] in core_ids]
    nonlinear = [row for row in core if row["mode"] != "CLMMSE_FALLBACK"]
    core_fraction = float(len(nonlinear) / len(core)) if core else float("nan")
    resolved = [row for row in reference_rows if row["resolved"] is True]
    resolved_core_ids = {
        str(row["condition_id"])
        for row in resolved
        if str(row["condition_id"]) in core_ids
    }
    controller_nonfinite_count = sum(
        int(row["all_numeric_outputs_finite"] is not True) for row in controller_rows
    )
    reference_nonfinite_count = sum(
        int(row.get("all_diagnostics_finite") is not True) for row in reference_rows
    )
    nonfinite_count = controller_nonfinite_count + reference_nonfinite_count
    fallback_rows = [row for row in controller_rows if row["mode"] == "CLMMSE_FALLBACK"]
    fallback_exact = all(
        row["fallback_array_equal"] is True
        and row["controller_estimates_sha256"] == row["direct_clmmse_estimates_sha256"]
        and float(row["fallback_maximum_absolute_difference"]) == 0.0
        for row in fallback_rows
    )
    trace_complete = all(
        (
            row["compact_trace_sha256"] is not None
            and row.get("compact_trace_reconstructs") is True
        )
        or _structured_gibbs_failure_is_complete(
            row["structured_gibbs_failure"],
            chains=config.controller.gibbs_chains,
            proposal_cap=config.controller.maximum_conditional_proposals,
        )
        or row["mode"] == "CALIBRATED_TENSOR_QUADRATURE"
        for row in controller_rows
    )
    checks = {
        "controller_task_count": len(controller_rows) == rules.expected_controller_tasks,
        "reference_task_count": len(reference_rows) == rules.expected_reference_tasks,
        "nonfinite_outputs": nonfinite_count == rules.nonfinite_outputs,
        "unrecorded_exceptions": exception_count == rules.unrecorded_exceptions,
        "fallback_exact_equality": fallback_exact,
        "trace_reconstruction_coverage": trace_complete,
        "minimum_resolved_references": len(resolved) >= rules.minimum_resolved_reference_tasks,
        "every_core_condition_resolved": resolved_core_ids == core_ids,
        "minimum_core_nonlinear_acceptance": (
            np.isfinite(core_fraction)
            and core_fraction >= rules.minimum_core_nonlinear_acceptance_fraction
        ),
        "wall_clock_budget": elapsed_seconds <= config.resource_limits.maximum_wall_seconds,
        "peak_rss_budget": peak_rss_bytes <= config.resource_limits.maximum_process_rss_bytes,
        "output_size_budget": output_bytes <= config.resource_limits.maximum_output_bytes,
    }
    passed = bool(all(checks.values()))
    return {
        "decision": (
            "PASS_PRELOCK_ENGINEERING_QUALIFICATION"
            if passed
            else "FAIL_PRELOCK_ENGINEERING_QUALIFICATION_LOCK_BLOCKED"
        ),
        "scientific_boundary": PRELOCK_BOUNDARY,
        "all_checks_pass": passed,
        "checks": checks,
        "observed": {
            "controller_task_count": len(controller_rows),
            "reference_task_count": len(reference_rows),
            "exception_count": exception_count,
            "nonfinite_output_count": nonfinite_count,
            "controller_nonfinite_output_count": controller_nonfinite_count,
            "reference_nonfinite_diagnostic_count": reference_nonfinite_count,
            "fallback_count": len(fallback_rows),
            "resolved_reference_count": len(resolved),
            "resolved_core_condition_ids": sorted(resolved_core_ids),
            "core_controller_task_count": len(core),
            "core_nonlinear_accept_count": len(nonlinear),
            "core_nonlinear_acceptance_fraction": core_fraction,
            "elapsed_seconds": elapsed_seconds,
            "peak_process_rss_bytes": peak_rss_bytes,
            "output_bytes_before_manifest_and_complete": output_bytes,
        },
    }


def _load_registry(config: PrelockQualificationConfig) -> tuple[list[Any], list[Any], list[Any], dict[str, Any]]:
    from .production_registry import (
        build_source_frame_registry,
        build_source_task_registry,
        preflight_seed_sets,
        preflight_source_registry,
    )

    frames = build_source_frame_registry(
        campaign_id=config.experiment_id,
        base_seed=config.base_seed,
        topology_seeds=config.topology_seeds,
        condition_ids=tuple(item.id for item in config.conditions),
        realizations_per_topology_condition=config.realizations_per_topology_condition,
    )
    all_tasks = build_source_task_registry(
        frames,
        access_points=config.system.access_points,
        pilot_symbols=config.system.pilot_symbols,
    )
    selected: list[Any] = []
    by_frame: dict[int, list[Any]] = defaultdict(list)
    for task in all_tasks:
        by_frame[int(task.frame_index)].append(task)
    for frame in frames:
        ranked = sorted(
            by_frame[frame.frame_index],
            key=lambda task: (task.ranking_sha256, task.identity_tuple),
        )
        selected.extend(ranked[: config.task_selection.tasks_per_frame])
    selected_by_condition: dict[str, list[Any]] = defaultdict(list)
    for task in selected:
        selected_by_condition[str(task.condition_id)].append(task)
    references: list[_PrelockReferenceTask] = []
    for condition in config.conditions:
        ranked = sorted(
            selected_by_condition[condition.id],
            key=lambda task: (task.ranking_sha256, task.identity_tuple),
        )[: config.task_selection.references_per_condition]
        for task in ranked:
            task_id = _task_id(task)
            primary_tags = [
                config.experiment_id,
                task_id,
                "reference",
                0,
            ]
            repeat_tags = [
                config.experiment_id,
                task_id,
                "reference",
                1,
            ]
            references.append(
                _PrelockReferenceTask(
                    task=task,
                    primary_seed=derived_seed(config.base_seed, *primary_tags),
                    repeat_seed=derived_seed(
                        config.reference_repeat_base_seed, *repeat_tags
                    ),
                    primary_base_seed=config.base_seed,
                    repeat_base_seed=config.reference_repeat_base_seed,
                    primary_seed_tags_json=canonical_ascii_json(
                        [config.base_seed, *primary_tags]
                    ).decode("ascii"),
                    repeat_seed_tags_json=canonical_ascii_json(
                        [config.reference_repeat_base_seed, *repeat_tags]
                    ).decode("ascii"),
                )
            )
    registry_preflight = preflight_source_registry(
        frames,
        all_tasks,
        reserved_seeds=config.reserved_seed_namespaces.explicit_values,
    )
    reference_seeds = tuple(
        seed
        for item in references
        for seed in (item.primary_seed, item.repeat_seed)
    )
    seed_preflight = preflight_seed_sets(
        {
            "prelock_explicit": (
                config.base_seed,
                config.reference_repeat_base_seed,
                *config.topology_seeds,
            ),
            "future_reserved_explicit": config.reserved_seed_namespaces.explicit_values,
            "frame_derived": tuple(frame.frame_seed for frame in frames),
            "controller_derived": tuple(task.controller_seed for task in selected),
            "reference_derived": reference_seeds,
        }
    )
    preflight = {
        "source_registry": asdict(registry_preflight),
        "selected_controller_task_count": len(selected),
        "selected_reference_task_count": len(references),
        "selected_controller_registry_sha256": sha256_bytes(
            canonical_ascii_json([list(task.identity_tuple) for task in selected])
        ),
        "selected_reference_registry_sha256": sha256_bytes(
            canonical_ascii_json([list(item.task.identity_tuple) for item in references])
        ),
        "all_prelock_seed_sets": asdict(seed_preflight),
    }
    return list(all_tasks), list(selected), list(references), preflight


def _task_record(task: Any, selected_ids: set[str], reference_ids: set[str], sequence_index: int) -> dict[str, Any]:
    task_id = _task_id(task)
    frame_seed = derived_seed(
        task.base_seed,
        task.campaign_id,
        task.condition_id,
        task.topology_seed,
        task.realization_index,
        "frame",
    )
    return {
        "sequence_index": sequence_index,
        "task_id": task_id,
        "experiment_id": task.campaign_id,
        "condition_id": task.condition_id,
        "topology_seed": task.topology_seed,
        "realization_index": task.realization_index,
        "access_point": task.ap_index,
        "pilot": task.pilot_index,
        "rank_sha256": task.ranking_sha256,
        "frame_seed": frame_seed,
        "controller_seed": task.controller_seed if task_id in selected_ids else None,
        "controller_selected": task_id in selected_ids,
        "reference_selected": task_id in reference_ids,
    }


def _seed_records(
    config: PrelockQualificationConfig,
    selected: Sequence[Any],
    references: Sequence[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append_explicit(kind: str, identity: str, seed: int) -> None:
        rows.append(
            {
                "sequence_index": len(rows),
                "kind": kind,
                "identity": identity,
                "seed": int(seed),
                "base_seed": None,
                "tags_json": canonical_ascii_json(
                    ["explicit", config.experiment_id, kind, identity, int(seed)]
                ).decode("ascii"),
            }
        )

    # Materialize the complete disjoint namespace, including future-reserved
    # values, rather than leaving the preflight proof as the only record of
    # those exclusions.
    append_explicit("prelock_base", "base_seed", config.base_seed)
    append_explicit(
        "prelock_reference_repeat_base",
        "reference_repeat_base_seed",
        config.reference_repeat_base_seed,
    )
    for index, seed in enumerate(config.topology_seeds):
        append_explicit("prelock_topology", f"topology_{index:02d}", seed)

    reserved = config.reserved_seed_namespaces
    reserved_groups = (
        ("reserved_strong_affine_base", (reserved.strong_affine_base_seed,)),
        ("reserved_strong_affine_topology", reserved.strong_affine_topology_seeds),
        ("reserved_strong_affine_bootstrap", (reserved.strong_affine_bootstrap_seed,)),
        ("reserved_small_reference_case", reserved.small_reference_case_seeds),
        ("reserved_source_scale_base", (reserved.source_scale_base_seed,)),
        ("reserved_source_scale_topology", reserved.source_scale_topology_seeds),
        ("reserved_heldout_base", (reserved.heldout_base_seed,)),
        ("reserved_heldout_topology", reserved.heldout_topology_seeds),
        ("reserved_robustness_base", (reserved.robustness_base_seed,)),
        ("reserved_robustness_topology", reserved.robustness_topology_seeds),
        ("reserved_future_bootstrap", reserved.future_bootstrap_seeds),
    )
    for kind, seeds in reserved_groups:
        for index, seed in enumerate(seeds):
            append_explicit(kind, f"{kind}_{index:03d}", seed)

    seen_frames: set[str] = set()
    for task in selected:
        frame_id = _frame_id(task)
        frame_seed = derived_seed(
            task.base_seed,
            task.campaign_id,
            task.condition_id,
            task.topology_seed,
            task.realization_index,
            "frame",
        )
        frame_tags = canonical_ascii_json(
            [
                task.base_seed,
                task.campaign_id,
                task.condition_id,
                task.topology_seed,
                task.realization_index,
                "frame",
            ]
        ).decode("ascii")
        if frame_id not in seen_frames:
            seen_frames.add(frame_id)
            rows.append(
                {
                    "sequence_index": len(rows),
                    "kind": "frame",
                    "identity": frame_id,
                    "seed": frame_seed,
                    "base_seed": task.base_seed,
                    "tags_json": frame_tags,
                }
            )
        rows.append(
            {
                "sequence_index": len(rows),
                "kind": "controller",
                "identity": _task_id(task),
                "seed": task.controller_seed,
                "base_seed": task.base_seed,
                "tags_json": canonical_ascii_json(
                    [
                        task.base_seed,
                        task.campaign_id,
                        task.condition_id,
                        task.topology_seed,
                        task.realization_index,
                        task.ap_index,
                        task.pilot_index,
                        "controller",
                    ]
                ).decode("ascii"),
            }
        )
    for reference in references:
        for repeat_id, seed, base_seed, tags in (
            (0, reference.primary_seed, reference.primary_base_seed, reference.primary_seed_tags_json),
            (1, reference.repeat_seed, reference.repeat_base_seed, reference.repeat_seed_tags_json),
        ):
            rows.append(
                {
                    "sequence_index": len(rows),
                    "kind": f"reference_repeat_{repeat_id}",
                    "identity": reference.task_id,
                    "seed": seed,
                    "base_seed": base_seed,
                    "tags_json": tags,
                }
            )
    return rows


def run_prelock_qualification(
    config: PrelockQualificationConfig,
    *,
    project_root: Path,
    config_path: Path,
    acknowledge_non_evidence: bool = False,
) -> dict[str, Any]:
    """Run the fixed excluded-seed qualification exactly once."""

    if acknowledge_non_evidence is not True:
        raise RuntimeError("pre-lock run requires explicit non-evidence acknowledgement")
    root = project_root.resolve(strict=True)
    source_config = config_path.resolve(strict=True)
    if root not in source_config.parents:
        raise RuntimeError("configuration path leaves project root")
    if load_prelock_config(source_config) != config:
        raise RuntimeError("parsed configuration differs from the current on-disk snapshot")
    thread_environment = require_single_thread_runtime()
    source_hashes = _hash_paths(root, RUNTIME_SOURCE_PATHS)
    test_hashes = _hash_paths(root, RUNTIME_TEST_PATHS)
    config_sha256 = sha256_file(source_config)
    all_tasks, selected, references, seed_preflight = _load_registry(config)
    selected_ids = {_task_id(item) for item in selected}
    reference_ids = {item.task_id for item in references}
    if len(selected_ids) != config.pass_rules.expected_controller_tasks:
        raise RuntimeError("controller registry count differs before output creation")
    if len(reference_ids) != config.pass_rules.expected_reference_tasks:
        raise RuntimeError("reference registry count differs before output creation")

    lifecycle = AtomicRunDirectory(root, config.output_directory, config.experiment_id)
    started_unix = time.time()
    staging = lifecycle.start(
        {
            "stage_id": config.stage_id,
            "scientific_boundary": PRELOCK_BOUNDARY,
            "started_unix": started_unix,
            "config_sha256": config_sha256,
        }
    )
    controller_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    task_context: dict[str, tuple[Any, Any, np.ndarray]] = {}
    attempted_moment_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    progress_count = 0
    resource_count = 0
    reference_repeat_count = 0
    peak_rss = peak_process_rss_bytes()
    open_handles: list[Any] = []
    try:
        write_jsonl_exclusive(
            staging / "TASK_REGISTRY.jsonl",
            (
                _task_record(task, selected_ids, reference_ids, index)
                for index, task in enumerate(all_tasks)
            ),
        )
        seed_rows = _seed_records(config, selected, references)
        expected_seed_rows = int(
            seed_preflight["all_prelock_seed_sets"]["total_seed_count"]
        )
        if len(seed_rows) != expected_seed_rows:
            raise RuntimeError("materialized seed registry differs from preflight count")
        write_jsonl_exclusive(staging / "SEED_REGISTRY.jsonl", seed_rows)

        condition_lookup = {item.id: item for item in config.conditions}
        selected_index = {int(task.task_index): index for index, task in enumerate(selected)}
        selected_by_frame: dict[str, list[Any]] = defaultdict(list)
        for task in selected:
            selected_by_frame[_frame_id(task)].append(task)
        controller = _controller(config)
        outcome_handle = (staging / "controller_outcomes.jsonl").open("x", encoding="utf-8", newline="\n")
        compact_handle = (staging / "controller_compact_traces.jsonl").open("x", encoding="utf-8", newline="\n")
        progress_handle = (staging / "PROGRESS.jsonl").open("x", encoding="utf-8", newline="\n")
        resource_handle = (staging / "resource_metrics.jsonl").open("x", encoding="utf-8", newline="\n")
        open_handles.extend(
            [outcome_handle, compact_handle, progress_handle, resource_handle]
        )
        try:
            for frame_id in sorted(
                selected_by_frame,
                key=lambda key: selected_by_frame[key][0].frame_index,
            ):
                frame_tasks = sorted(
                    selected_by_frame[frame_id],
                    key=lambda item: selected_index[item.task_index],
                )
                first_task = frame_tasks[0]
                condition = condition_lookup[first_task.condition_id]
                scenario = _scenario(config, condition)
                topology = generate_topology(scenario, seed=first_task.topology_seed)
                frame_seed = derived_seed(
                    first_task.base_seed,
                    first_task.campaign_id,
                    first_task.condition_id,
                    first_task.topology_seed,
                    first_task.realization_index,
                    "frame",
                )
                frame = generate_channel_frame(topology, scenario, seed=frame_seed)
                for task in frame_tasks:
                    task_id = _task_id(task)
                    started_task = time.perf_counter()
                    model, observation, users, _ = pilot_group(
                        topology,
                        frame,
                        scenario,
                        access_point=task.ap_index,
                        pilot=task.pilot_index,
                    )
                    task_context[task_id] = (model, observation.copy(), users.copy())
                    structured_failure: dict[str, Any] | None = None
                    try:
                        result = controller.estimate_all(
                            model, observation, seed=task.controller_seed
                        )
                    except GibbsSamplingError as error:
                        # Controller normally converts this into a fallback.  This
                        # branch protects against an uncaught structured failure.
                        structured_failure = dict(error.audit)
                        exception_rows.append(
                            {
                                "task_id": task_id,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "structured": structured_failure,
                            }
                        )
                        raise
                    direct = np.column_stack(
                        [clmmse(model, observation, target) for target in range(model.group_size)]
                    ).astype(np.complex128)
                    fallback = result.mode == "CLMMSE_FALLBACK"
                    array_equal = bool(np.array_equal(result.estimates, direct)) if fallback else None
                    maximum_difference = (
                        float(np.max(np.abs(result.estimates - direct), initial=0.0))
                        if fallback
                        else None
                    )
                    attempted = (
                        getattr(result, "attempted_phase_moments", None)
                        or result.phase_moments
                    )
                    compact_document: dict[str, Any] | None = None
                    compact_hash: str | None = None
                    compact_reconstructs: bool | None = None
                    if attempted is not None and attempted.retained_samples is not None:
                        attempted_moment_lookup[task_id] = (
                            np.asarray(attempted.z, dtype=np.complex128).copy(),
                            np.asarray(attempted.Z, dtype=np.complex128).copy(),
                        )
                        compact_document = build_compact_gibbs_statistics(
                            attempted.retained_samples,
                            task_id=task_id,
                            sampler_metadata=attempted.metadata,
                            method=attempted.method,
                            status=attempted.status,
                            proposal_count=int(attempted.proposal_count),
                        )
                        reconstructed_z, reconstructed_Z = reconstruct_compact_moments(
                            compact_document
                        )
                        reconstructed_diagnostic = reconstruct_compact_diagnostics(
                            compact_document
                        )
                        original_diagnostic = attempted.diagnostics
                        compact_reconstructs = bool(
                            original_diagnostic is not None
                            and np.allclose(
                                reconstructed_z,
                                attempted.z,
                                rtol=1e-13,
                                atol=1e-13,
                            )
                            and np.allclose(
                                reconstructed_Z,
                                attempted.Z,
                                rtol=1e-13,
                                atol=1e-13,
                            )
                            and np.allclose(
                                reconstructed_diagnostic.split_rhat_real,
                                original_diagnostic.split_rhat_real,
                                rtol=1e-13,
                                atol=1e-13,
                            )
                            and np.allclose(
                                reconstructed_diagnostic.split_rhat_imag,
                                original_diagnostic.split_rhat_imag,
                                rtol=1e-13,
                                atol=1e-13,
                            )
                            and reconstructed_diagnostic.maximum_split_rhat
                            == original_diagnostic.maximum_split_rhat
                            and reconstructed_diagnostic.maximum_chain_mean_disagreement
                            == original_diagnostic.maximum_chain_mean_disagreement
                            and reconstructed_diagnostic.all_finite
                            is original_diagnostic.all_finite
                        )
                        compact_hash = sha256_bytes(canonical_json_bytes(compact_document))
                    structured_failure = result.diagnostics.get("gibbs_failure_audit")
                    trace_kind = (
                        "GIBBS_COMPACT"
                        if compact_document is not None
                        else (
                            "STRUCTURED_GIBBS_FAILURE"
                            if structured_failure is not None
                            else "NO_GIBBS_ATTEMPT"
                        )
                    )
                    trace_row = {
                        "sequence_index": len(controller_rows),
                        "task_id": task_id,
                        "condition_id": task.condition_id,
                        "topology_seed": task.topology_seed,
                        "realization_index": task.realization_index,
                        "access_point": task.ap_index,
                        "pilot": task.pilot_index,
                        "global_user_ids": users.tolist(),
                        "trace_kind": trace_kind,
                        "compact_sha256": compact_hash,
                        "document": compact_document,
                        "structured_failure": structured_failure,
                    }
                    append_jsonl_record(compact_handle, trace_row, ordered_fields=COMPACT_TRACE_FIELDS)
                    logical = result.diagnostics.get("gibbs_proposal_count", 0)
                    generated = result.diagnostics.get("gibbs_generated_candidate_value_count", 0)
                    all_finite = bool(
                        np.all(np.isfinite(result.estimates)) and np.all(np.isfinite(direct))
                    )
                    outcome = {
                        "sequence_index": len(controller_rows),
                        "task_id": task_id,
                        "condition_id": task.condition_id,
                        "topology_seed": task.topology_seed,
                        "realization_index": task.realization_index,
                        "access_point": task.ap_index,
                        "pilot": task.pilot_index,
                        "global_user_ids": users.tolist(),
                        "mode": result.mode,
                        "status": result.status,
                        "fallback_reason": result.diagnostics.get("fallback_reason"),
                        "quadrature_rejection_reason": result.diagnostics.get("quadrature_rejection_reason"),
                        "maximum_split_rhat": result.diagnostics.get("maximum_split_rhat"),
                        "maximum_chain_mean_disagreement": result.diagnostics.get("maximum_chain_mean_disagreement"),
                        "dobrushin_eta": result.diagnostics.get("dobrushin_eta"),
                        "logical_proposal_count": int(logical),
                        "generated_candidate_value_count": int(generated),
                        "controller_estimates": _complex_json(result.estimates),
                        "direct_clmmse_estimates": _complex_json(direct),
                        "controller_estimates_sha256": sha256_array(result.estimates),
                        "direct_clmmse_estimates_sha256": sha256_array(direct),
                        "fallback_array_equal": array_equal,
                        "fallback_maximum_absolute_difference": maximum_difference,
                        "all_numeric_outputs_finite": all_finite,
                        "compact_trace_sha256": compact_hash,
                        "compact_trace_reconstructs": compact_reconstructs,
                        "structured_gibbs_failure": structured_failure,
                    }
                    append_jsonl_record(outcome_handle, outcome, ordered_fields=CONTROLLER_OUTCOME_FIELDS)
                    controller_rows.append(outcome)
                    elapsed_task = time.perf_counter() - started_task
                    peak_rss = max(peak_rss, peak_process_rss_bytes())
                    resource = {
                        "sequence_index": resource_count,
                        "work_kind": "controller",
                        "task_id": task_id,
                        "repeat_id": None,
                        "elapsed_seconds": elapsed_task,
                        "peak_process_rss_bytes": peak_rss,
                        "logical_proposal_count": int(logical),
                        "generated_candidate_value_count": int(generated),
                        "output_bytes_after_work": output_tree_bytes(staging),
                    }
                    append_jsonl_record(resource_handle, resource, ordered_fields=RESOURCE_FIELDS)
                    resource_rows.append(resource)
                    resource_count += 1
                    progress = {
                        "sequence_index": progress_count,
                        "work_kind": "controller",
                        "task_id": task_id,
                        "repeat_id": None,
                        "status": "COMPLETE" if all_finite else "NONFINITE",
                        "condition_id": task.condition_id,
                        "topology_seed": task.topology_seed,
                        "realization_index": task.realization_index,
                        "access_point": task.ap_index,
                        "pilot": task.pilot_index,
                        "cumulative_controller_tasks": len(controller_rows),
                        "cumulative_reference_repeats": reference_repeat_count,
                        "cumulative_exception_count": len(exception_rows),
                        "peak_process_rss_bytes": peak_rss,
                        "completed_unix": time.time(),
                    }
                    append_jsonl_record(progress_handle, progress, ordered_fields=PROGRESS_FIELDS)
                    progress_count += 1
        finally:
            outcome_handle.close()
            compact_handle.close()

        references_dir = staging / "references"
        references_dir.mkdir(exist_ok=False)
        reference_handle = (staging / "reference_metrics.jsonl").open("x", encoding="utf-8", newline="\n")
        open_handles.append(reference_handle)
        try:
            outcome_lookup = {row["task_id"]: row for row in controller_rows}
            for reference_index, reference in enumerate(references):
                model, observation, _ = task_context[reference.task_id]
                estimates: list[Any] = []
                manifests: list[Path] = []
                for repeat_id, seed in enumerate((reference.primary_seed, reference.repeat_seed)):
                    started_reference = time.perf_counter()
                    graph = build_phase_graph(model, observation)
                    try:
                        estimate = heat_bath_gibbs(
                            graph,
                            seed=seed,
                            **_high_reference_kwargs(config),
                        )
                    except GibbsSamplingError as error:
                        exception_rows.append(
                            {
                                "task_id": reference.task_id,
                                "repeat_id": repeat_id,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "structured": error.audit,
                            }
                        )
                        raise
                    trace_id = f"{reference.task_id}.repeat{repeat_id}"
                    _, manifest_path = write_gibbs_trace_bundle(
                        references_dir, trace_id, estimate
                    )
                    estimates.append(estimate)
                    manifests.append(manifest_path)
                    elapsed_reference = time.perf_counter() - started_reference
                    peak_rss = max(peak_rss, peak_process_rss_bytes())
                    resource = {
                        "sequence_index": resource_count,
                        "work_kind": "reference",
                        "task_id": reference.task_id,
                        "repeat_id": repeat_id,
                        "elapsed_seconds": elapsed_reference,
                        "peak_process_rss_bytes": peak_rss,
                        "logical_proposal_count": int(estimate.proposal_count),
                        "generated_candidate_value_count": int(
                            estimate.metadata.get("generated_candidate_value_count", 0)
                        ),
                        "output_bytes_after_work": output_tree_bytes(staging),
                    }
                    append_jsonl_record(resource_handle, resource, ordered_fields=RESOURCE_FIELDS)
                    resource_rows.append(resource)
                    resource_count += 1
                    reference_repeat_count += 1
                    progress = {
                        "sequence_index": progress_count,
                        "work_kind": "reference",
                        "task_id": reference.task_id,
                        "repeat_id": repeat_id,
                        "status": "COMPLETE",
                        "condition_id": reference.condition_id,
                        "topology_seed": reference.topology_seed,
                        "realization_index": reference.realization_index,
                        "access_point": reference.ap_index,
                        "pilot": reference.pilot_index,
                        "cumulative_controller_tasks": len(controller_rows),
                        "cumulative_reference_repeats": reference_repeat_count,
                        "cumulative_exception_count": len(exception_rows),
                        "peak_process_rss_bytes": peak_rss,
                        "completed_unix": time.time(),
                    }
                    append_jsonl_record(progress_handle, progress, ordered_fields=PROGRESS_FIELDS)
                    progress_count += 1
                resolution = _reference_resolution(config, estimates[0], estimates[1])
                pooled_z = 0.5 * (estimates[0].z + estimates[1].z)
                pooled_Z = 0.5 * (estimates[0].Z + estimates[1].Z)
                # Reference errors use the already executed controller attempt;
                # the controller is never re-run after task selection or result
                # inspection.
                controller_z_error = None
                controller_Z_error = None
                attempted_pair = attempted_moment_lookup.get(reference.task_id)
                if attempted_pair is not None:
                    attempted_z, attempted_Z = attempted_pair
                    controller_z_error = float(
                        np.max(np.abs(attempted_z - pooled_z), initial=0.0)
                    )
                    controller_Z_error = float(
                        np.max(np.abs(attempted_Z - pooled_Z), initial=0.0)
                    )
                primary_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                repeat_manifest = json.loads(manifests[1].read_text(encoding="utf-8"))
                row = {
                    "sequence_index": reference_index,
                    "task_id": reference.task_id,
                    "condition_id": reference.condition_id,
                    "topology_seed": reference.topology_seed,
                    "realization_index": reference.realization_index,
                    "access_point": reference.ap_index,
                    "pilot": reference.pilot_index,
                    "primary_seed": reference.primary_seed,
                    "repeat_seed": reference.repeat_seed,
                    "primary_manifest": str(manifests[0].relative_to(staging)).replace("\\", "/"),
                    "repeat_manifest": str(manifests[1].relative_to(staging)).replace("\\", "/"),
                    "primary_payload_sha256": primary_manifest["payload_sha256"],
                    "repeat_payload_sha256": repeat_manifest["payload_sha256"],
                    "primary_maximum_split_rhat": _finite_float_or_none(resolution["primary_maximum_split_rhat"]),
                    "repeat_maximum_split_rhat": _finite_float_or_none(resolution["repeat_maximum_split_rhat"]),
                    "primary_maximum_chain_mean_disagreement_z": _finite_float_or_none(resolution["primary_maximum_chain_mean_disagreement_z"]),
                    "repeat_maximum_chain_mean_disagreement_z": _finite_float_or_none(resolution["repeat_maximum_chain_mean_disagreement_z"]),
                    "primary_maximum_chain_mean_disagreement_Z": _finite_float_or_none(resolution["primary_maximum_chain_mean_disagreement_Z"]),
                    "repeat_maximum_chain_mean_disagreement_Z": _finite_float_or_none(resolution["repeat_maximum_chain_mean_disagreement_Z"]),
                    "repeat_maximum_abs_z": _finite_float_or_none(resolution["repeat_maximum_abs_z"]),
                    "repeat_maximum_abs_Z": _finite_float_or_none(resolution["repeat_maximum_abs_Z"]),
                    "all_diagnostics_finite": resolution["all_diagnostics_finite"],
                    "resolved": resolution["resolved"],
                    "resolution_failures": resolution["failures"],
                    "pooled_reference_z": _complex_json(pooled_z),
                    "pooled_reference_Z": _complex_json(pooled_Z),
                    "controller_mode": outcome_lookup[reference.task_id]["mode"],
                    "controller_maximum_abs_z_error": controller_z_error,
                    "controller_maximum_abs_Z_error": controller_Z_error,
                }
                append_jsonl_record(reference_handle, row, ordered_fields=REFERENCE_FIELDS)
                reference_rows.append(row)
                del estimates
                gc.collect()
        finally:
            reference_handle.close()
            progress_handle.close()
            resource_handle.close()

        if exception_rows:
            with (staging / "exceptions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
                for row in exception_rows:
                    append_jsonl_record(handle, row)
        elapsed_seconds = time.time() - started_unix
        peak_rss = max(peak_rss, peak_process_rss_bytes())
        bytes_before_manifest = output_tree_bytes(staging)
        verdict = evaluate_prelock_verdict(
            config,
            controller_rows,
            reference_rows,
            exception_count=len(exception_rows),
            elapsed_seconds=elapsed_seconds,
            peak_rss_bytes=peak_rss,
            output_bytes=bytes_before_manifest,
        )
        summary = {
            "experiment_id": config.experiment_id,
            "stage_id": config.stage_id,
            "scientific_boundary": PRELOCK_BOUNDARY,
            "decision_eligible": False,
            "seed_preflight": seed_preflight,
            "task_registry_count": len(all_tasks),
            "selected_controller_task_count": len(selected),
            "selected_reference_task_count": len(references),
            "progress_record_count": progress_count,
            "resource_record_count": resource_count,
            "controller_mode_counts": dict(sorted(Counter(row["mode"] for row in controller_rows).items())),
            "runtime_seconds": {
                "controller": _runtime_summary(
                    [
                        float(row["elapsed_seconds"])
                        for row in resource_rows
                        if row["work_kind"] == "controller"
                    ]
                ),
                "reference_repeat": _runtime_summary(
                    [
                        float(row["elapsed_seconds"])
                        for row in resource_rows
                        if row["work_kind"] == "reference"
                    ]
                ),
            },
            "engineering_verdict": verdict,
        }
        write_json_exclusive(staging / "summary.json", summary)
        if load_prelock_config(source_config) != config:
            raise RuntimeError("TOCTOU: configuration changed during execution")
        if config_sha256 != sha256_file(source_config):
            raise RuntimeError("TOCTOU: configuration hash changed during execution")
        if source_hashes != _hash_paths(root, RUNTIME_SOURCE_PATHS) or test_hashes != _hash_paths(root, RUNTIME_TEST_PATHS):
            raise RuntimeError("TOCTOU: source or test closure changed during execution")
        payload_files = sorted(
            path for path in staging.rglob("*") if path.is_file() and path.name != "RUNNING.json"
        )
        manifest = {
            "schema_version": "r2c01-prelock-run-manifest/1.0",
            "experiment_id": config.experiment_id,
            "stage_id": config.stage_id,
            "scientific_boundary": PRELOCK_BOUNDARY,
            "config_path": str(source_config.relative_to(root)).replace("\\", "/"),
            "config_sha256": config_sha256,
            "source_sha256": source_hashes,
            "test_sha256": test_hashes,
            "thread_environment": thread_environment,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "started_unix": started_unix,
            "completed_unix": time.time(),
            "seed_preflight": seed_preflight,
            "output_sha256": {
                str(path.relative_to(staging)).replace("\\", "/"): sha256_file(path)
                for path in payload_files
            },
        }
        write_json_exclusive(staging / "run_manifest.json", manifest)
        manifest_sha256 = sha256_file(staging / "run_manifest.json")
        with (staging / "run_manifest.sha256").open("x", encoding="ascii", newline="\n") as handle:
            handle.write(f"{manifest_sha256}  run_manifest.json\n")
            handle.flush()
            os.fsync(handle.fileno())
        missing_required = [
            relative
            for relative in config.required_output_files
            if relative != "COMPLETE.json" and not (staging / relative).is_file()
        ]
        if missing_required:
            raise RuntimeError(
                f"required pre-lock output files are absent: {missing_required}"
            )
        if output_tree_bytes(staging) > config.resource_limits.maximum_output_bytes:
            raise RuntimeError("output-size budget exceeded after manifest creation")
        output = lifecycle.complete(
            {
                "completed_unix": time.time(),
                "run_manifest_sha256": manifest_sha256,
                "engineering_decision": verdict["decision"],
                "all_checks_pass": verdict["all_checks_pass"],
                "production_lock_permitted": bool(verdict["all_checks_pass"]),
                "final_scientific_evidence": False,
            }
        )
        return {**summary, "output_directory": str(output)}
    except BaseException as error:
        for handle in open_handles:
            try:
                if not handle.closed:
                    handle.close()
            except BaseException:
                pass
        lifecycle.fail(
            error,
            {
                "failed_unix": time.time(),
                "scientific_boundary": PRELOCK_BOUNDARY,
                "completed_controller_task_count": len(controller_rows),
                "completed_reference_task_count": len(reference_rows),
                "recorded_exception_count": len(exception_rows),
                "peak_process_rss_bytes": peak_rss,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m phase_graph_cme.prelock_qualification")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--acknowledge-non-evidence", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    config_path = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    try:
        config = load_prelock_config(config_path)
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "status": "VALID_PRELOCK_NON_EVIDENCE_CONFIGURATION",
                        "experiment_id": config.experiment_id,
                        "decision_eligible": config.decision_eligible,
                    },
                    indent=2,
                )
            )
            return 0
        summary = run_prelock_qualification(
            config,
            project_root=root,
            config_path=config_path,
            acknowledge_non_evidence=args.acknowledge_non_evidence,
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
