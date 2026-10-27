"""Fail-closed, preregistration-ready P=2 strong-affine screen.

The decision-eligible stage cannot execute without a detached hash manifest and
an explicit acknowledgement.  A separately bounded smoke stage is permanently
non-evidence and cannot return a scientific pass decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np
import scipy
import yaml
from numpy.typing import NDArray

from .metrics import uatf_uplink
from .model import PilotGroupModel
from .receivers import mr_combiner, rzf_combiner
from .scenario import ScenarioConfig, generate_channel_frame, generate_topology, pilot_group

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

METHODS = ("exact_tensor_cme", "widely_linear_lmmse")
RECEIVERS = ("MR", "RZF")
SCREEN_STAGE = "STRONG_AFFINE_SCREEN"
SMOKE_STAGE = "NON_EVIDENCE_SMOKE"
LOCKED = "LOCKED_BEFORE_PROPOSED_RESULTS"
SMOKE_UNLOCKED = "UNLOCKED_NON_EVIDENCE_SMOKE"
RUNTIME_SOURCE_PATHS = (
    "src/phase_graph_cme/__init__.py",
    "src/phase_graph_cme/strong_affine_screen.py",
    "src/phase_graph_cme/controller.py",
    "src/phase_graph_cme/baselines.py",
    "src/phase_graph_cme/inference.py",
    "src/phase_graph_cme/moments.py",
    "src/phase_graph_cme/diagnostics.py",
    "src/phase_graph_cme/posterior.py",
    "src/phase_graph_cme/model.py",
    "src/phase_graph_cme/scenario.py",
    "src/phase_graph_cme/metrics.py",
    "src/phase_graph_cme/receivers.py",
    "scripts/validate_strong_affine_screen_output.py",
)
RUNTIME_TEST_PATHS = (
    "tests/unit/test_phase_graph_strong_affine_screen.py",
    "tests/unit/test_strong_affine_screen_output_validator.py",
)
THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PROGRESS_SCHEMA_VERSION = "strong-affine-progress/1.0"
PROGRESS_FIELDS = (
    "schema_version",
    "sequence_index",
    "experiment_id",
    "stage",
    "topology_index",
    "topology_seed",
    "condition_index",
    "condition_id",
    "expected_cell_task_count",
    "executed_cell_task_count",
    "failed_or_rejected_cell_task_count",
    "paired_cell_valid",
    "cumulative_executed_task_count",
    "cumulative_failure_event_count",
    "peak_process_rss_bytes",
    "rss_measurement_scope",
    "completed_unix",
)
FAILURE_CODES = {
    0: "OK",
    1: "QUADRATURE_ORDER_DISAGREEMENT",
    2: "NUMERICAL_EXCEPTION",
}


class ScreenConfigurationError(ValueError):
    """Raised when a screen configuration violates the exact schema."""


class StatisticalContractError(ValueError):
    """Raised when a registered statistic cannot be evaluated as specified."""


class DuplicateKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: DuplicateKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ScreenConfigurationError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ScreenConfigurationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class Condition:
    id: str
    kappa: float
    delta_deg: float


@dataclass(frozen=True)
class SystemSpec:
    access_points: int
    antennas_per_ap: int
    coherence_symbols: int
    area_side: float
    minimum_distance: float
    pathloss_exponent: float
    pilot_power: float
    payload_power: float
    pilot_noise_variance: float
    payload_noise_variance: float
    spatial_correlation: float

    def __post_init__(self) -> None:
        for label, value in (
            ("access_points", self.access_points),
            ("antennas_per_ap", self.antennas_per_ap),
            ("coherence_symbols", self.coherence_symbols),
        ):
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or int(value) < 1:
                raise ScreenConfigurationError(f"system.{label} must be a positive integer")
        if self.coherence_symbols <= 1:
            raise ScreenConfigurationError("coherence_symbols must exceed the one-symbol pilot")
        for label, value in (
            ("area_side", self.area_side),
            ("minimum_distance", self.minimum_distance),
            ("pathloss_exponent", self.pathloss_exponent),
            ("pilot_power", self.pilot_power),
            ("payload_power", self.payload_power),
            ("pilot_noise_variance", self.pilot_noise_variance),
            ("payload_noise_variance", self.payload_noise_variance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ScreenConfigurationError(f"system.{label} must be finite and positive")
        if not math.isfinite(self.spatial_correlation) or not 0.0 <= self.spatial_correlation < 1.0:
            raise ScreenConfigurationError("system.spatial_correlation must lie in [0,1)")

    def scenario(self, condition: Condition) -> ScenarioConfig:
        return ScenarioConfig(
            access_points=int(self.access_points),
            antennas_per_ap=int(self.antennas_per_ap),
            users=2,
            pilot_symbols=1,
            coherence_symbols=int(self.coherence_symbols),
            area_side=self.area_side,
            minimum_distance=self.minimum_distance,
            pathloss_exponent=self.pathloss_exponent,
            rician_kappa=condition.kappa,
            residual_phase_half_width_deg=condition.delta_deg,
            pilot_power=self.pilot_power,
            payload_power=self.payload_power,
            pilot_noise_variance=self.pilot_noise_variance,
            payload_noise_variance=self.payload_noise_variance,
            spatial_correlation=self.spatial_correlation,
            pilot_assignment="cyclic",
            phase_error_model="matched_uniform",
        )


@dataclass(frozen=True)
class QuadratureSpec:
    order: int
    check_order: int
    maximum_states: int
    task_batch_size: int
    first_moment_tolerance: float
    second_moment_tolerance: float


@dataclass(frozen=True)
class MonteCarloSpec:
    split_rule: str
    maximum_absolute_primary_sum_se_difference: float
    maximum_absolute_primary_relative_percentage_point_difference: float
    maximum_absolute_global_p05_user_se_difference: float


@dataclass(frozen=True)
class MaterialitySpec:
    minimum_relative_gain_pct: float
    minimum_absolute_gain: float
    minimum_p05_lower_bound: float


@dataclass(frozen=True)
class StrongAffineScreenConfig:
    schema_version: str
    experiment_id: str
    stage: str
    decision_eligible: bool
    lock_status: str
    base_seed: int
    topology_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    conditions: tuple[Condition, ...]
    realizations_per_topology_condition: int
    system: SystemSpec
    quadrature: QuadratureSpec
    monte_carlo: MonteCarloSpec
    receiver_ids: tuple[str, ...]
    rzf_regularization: float
    materiality: MaterialitySpec
    output_directory: str
    preregistration_path: str | None
    lock_manifest_path: str | None

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise ScreenConfigurationError("unsupported schema_version")
        if not self.experiment_id or any(char.isspace() for char in self.experiment_id):
            raise ScreenConfigurationError("experiment_id must be nonempty without whitespace")
        if self.stage not in {SCREEN_STAGE, SMOKE_STAGE}:
            raise ScreenConfigurationError("unsupported stage")
        if not isinstance(self.decision_eligible, bool):
            raise ScreenConfigurationError("decision_eligible must be Boolean")
        for label, value, minimum in (
            ("base_seed", self.base_seed, 0),
            ("bootstrap_seed", self.bootstrap_seed, 0),
            ("bootstrap_replicates", self.bootstrap_replicates, 1),
            ("realizations_per_topology_condition", self.realizations_per_topology_condition, 4),
            ("quadrature.order", self.quadrature.order, 1),
            ("quadrature.check_order", self.quadrature.check_order, 1),
            ("quadrature.maximum_states", self.quadrature.maximum_states, 1),
            ("quadrature.task_batch_size", self.quadrature.task_batch_size, 1),
        ):
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or int(value) < minimum:
                raise ScreenConfigurationError(f"{label} must be an integer >= {minimum}")
            if "seed" in label and int(value) >= 2**63:
                raise ScreenConfigurationError(f"{label} must be below 2^63")
        if self.realizations_per_topology_condition % 2:
            raise ScreenConfigurationError("realizations must be even for the even/odd split-half gate")
        if not self.topology_seeds or len(set(self.topology_seeds)) != len(self.topology_seeds):
            raise ScreenConfigurationError("topology_seeds must be nonempty and unique")
        for seed in self.topology_seeds:
            if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or not 0 <= int(seed) < 2**63:
                raise ScreenConfigurationError("every topology seed must be an integer in [0,2^63)")
        all_seed_values = [int(self.base_seed), int(self.bootstrap_seed), *map(int, self.topology_seeds)]
        if len(set(all_seed_values)) != len(all_seed_values):
            raise ScreenConfigurationError("base, bootstrap, and topology seeds must be mutually distinct")
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ScreenConfigurationError("confidence_level must lie in (0,1)")
        if not self.conditions or len({item.id for item in self.conditions}) != len(self.conditions):
            raise ScreenConfigurationError("conditions must be nonempty with unique ids")
        for item in self.conditions:
            if not item.id or any(char.isspace() for char in item.id):
                raise ScreenConfigurationError("condition ids must be nonempty without whitespace")
            if not math.isfinite(item.kappa) or item.kappa < 0.0:
                raise ScreenConfigurationError("condition kappa must be finite and nonnegative")
            if not math.isfinite(item.delta_deg) or not 0.0 <= item.delta_deg <= 180.0:
                raise ScreenConfigurationError("condition delta_deg must lie in [0,180]")
        if self.receiver_ids != RECEIVERS:
            raise ScreenConfigurationError("receiver_ids must be exactly [MR, RZF] in that order")
        if self.quadrature.order >= self.quadrature.check_order:
            raise ScreenConfigurationError("quadrature check_order must exceed order")
        if self.quadrature.check_order**2 > self.quadrature.maximum_states:
            raise ScreenConfigurationError("maximum_states is below the worst-case P=2 check rule")
        for label, value in (
            ("quadrature.first_moment_tolerance", self.quadrature.first_moment_tolerance),
            ("quadrature.second_moment_tolerance", self.quadrature.second_moment_tolerance),
            (
                "monte_carlo.maximum_absolute_primary_sum_se_difference",
                self.monte_carlo.maximum_absolute_primary_sum_se_difference,
            ),
            (
                "monte_carlo.maximum_absolute_primary_relative_percentage_point_difference",
                self.monte_carlo.maximum_absolute_primary_relative_percentage_point_difference,
            ),
            (
                "monte_carlo.maximum_absolute_global_p05_user_se_difference",
                self.monte_carlo.maximum_absolute_global_p05_user_se_difference,
            ),
            ("materiality.minimum_relative_gain_pct", self.materiality.minimum_relative_gain_pct),
            ("materiality.minimum_absolute_gain", self.materiality.minimum_absolute_gain),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ScreenConfigurationError(f"{label} must be finite and nonnegative")
        if self.monte_carlo.split_rule != "even_odd_realization_index":
            raise ScreenConfigurationError("only even_odd_realization_index split_rule is supported")
        if not math.isfinite(self.materiality.minimum_p05_lower_bound):
            raise ScreenConfigurationError("minimum_p05_lower_bound must be finite")
        if not math.isfinite(self.rzf_regularization) or self.rzf_regularization <= 0.0:
            raise ScreenConfigurationError("rzf_regularization must be finite and positive")
        _validate_relative_path(self.output_directory, "output_directory")
        if self.stage == SCREEN_STAGE:
            if not self.decision_eligible or self.lock_status != LOCKED:
                raise ScreenConfigurationError("scientific screen must be decision-eligible and locked")
            if not self.preregistration_path or not self.lock_manifest_path:
                raise ScreenConfigurationError("scientific screen requires preregistration and detached lock manifest")
            _validate_relative_path(self.preregistration_path, "preregistration_path")
            _validate_relative_path(self.lock_manifest_path, "lock_manifest_path")
            if Path(self.preregistration_path) == Path(self.lock_manifest_path):
                raise ScreenConfigurationError("preregistration and detached lock manifest must differ")
        else:
            if self.decision_eligible or self.lock_status != SMOKE_UNLOCKED:
                raise ScreenConfigurationError("smoke must be unlocked and decision-ineligible")
            if self.preregistration_path is not None or self.lock_manifest_path is not None:
                raise ScreenConfigurationError("smoke cannot name lock artifacts")
            if not self.experiment_id.startswith("NON_EVIDENCE_"):
                raise ScreenConfigurationError("smoke experiment_id must start with NON_EVIDENCE_")
            if (
                len(self.topology_seeds) > 2
                or len(self.conditions) > 2
                or self.realizations_per_topology_condition > 4
                or self.system.access_points > 4
                or self.bootstrap_replicates > 200
                or self.quadrature.check_order > 12
            ):
                raise ScreenConfigurationError("smoke budget exceeds the hard non-evidence envelope")


@dataclass(frozen=True)
class ModelBatch:
    los: ComplexArray
    covariances: ComplexArray
    beta: ComplexArray
    noise_covariance: ComplexArray
    delta: FloatArray
    observation: ComplexArray

    def slice(self, start: int, stop: int) -> ModelBatch:
        return ModelBatch(
            self.los[start:stop],
            self.covariances[start:stop],
            self.beta[start:stop],
            self.noise_covariance[start:stop],
            self.delta[start:stop],
            self.observation[start:stop],
        )


@dataclass(frozen=True)
class TensorBatchResult:
    estimates: ComplexArray
    z: ComplexArray
    Z: ComplexArray
    state_count: int


@dataclass(frozen=True)
class IntegritySnapshot:
    config_sha256: str
    preregistration_sha256: str | None
    lock_manifest_sha256: str | None
    source_sha256: dict[str, str]
    test_sha256: dict[str, str]
    runtime_fingerprint: dict[str, Any]


@dataclass(frozen=True)
class UatFSufficientStatistics:
    sinr: FloatArray
    spectral_efficiency: FloatArray
    coherent_gain: ComplexArray
    denominator: FloatArray
    numerator: FloatArray
    second_effective_power: FloatArray
    mean_combiner_norm: FloatArray


def _validate_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path == Path("."):
        raise ScreenConfigurationError(f"{label} must be a non-root project-relative path")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ScreenConfigurationError(f"{label} must be a string-keyed mapping")
    return value


def _keys(
    value: object, label: str, required: set[str], optional: set[str] | None = None
) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    optional = set() if optional is None else optional
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required - optional)
    if missing or unknown:
        raise ScreenConfigurationError(f"{label}: missing={missing}; unknown={unknown}")
    return mapping


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or int(value) < minimum:
        raise ScreenConfigurationError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ScreenConfigurationError(f"{label} must be numeric and not Boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ScreenConfigurationError(f"{label} must be finite")
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ScreenConfigurationError(f"{label} must be a string")
    return value


def _load_yaml_unique(path: Path, label: str) -> object:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=DuplicateKeySafeLoader)
    except (OSError, yaml.YAMLError, ScreenConfigurationError) as error:
        raise ScreenConfigurationError(f"cannot read {label}: {error}") from error


def load_screen_config(path: str | Path) -> StrongAffineScreenConfig:
    root_required = {
        "schema_version", "experiment_id", "stage", "decision_eligible", "lock_status",
        "base_seed", "topology_seeds", "bootstrap_seed", "bootstrap_replicates",
        "confidence_level", "conditions", "realizations_per_topology_condition", "system",
        "quadrature", "monte_carlo", "receiver_ids", "rzf_regularization", "materiality",
        "output_directory", "preregistration_path", "lock_manifest_path",
    }
    root = _keys(_load_yaml_unique(Path(path), "screen config"), "root", root_required)
    system_fields = {
        "access_points", "antennas_per_ap", "coherence_symbols", "area_side", "minimum_distance",
        "pathloss_exponent", "pilot_power", "payload_power", "pilot_noise_variance",
        "payload_noise_variance", "spatial_correlation",
    }
    raw_system = _keys(root["system"], "system", system_fields)
    system = SystemSpec(
        access_points=_integer(raw_system["access_points"], "system.access_points", 1),
        antennas_per_ap=_integer(raw_system["antennas_per_ap"], "system.antennas_per_ap", 1),
        coherence_symbols=_integer(raw_system["coherence_symbols"], "system.coherence_symbols", 2),
        **{
            key: _number(raw_system[key], f"system.{key}")
            for key in system_fields - {"access_points", "antennas_per_ap", "coherence_symbols"}
        },
    )
    raw_quadrature = _keys(
        root["quadrature"], "quadrature",
        {"order", "check_order", "maximum_states", "task_batch_size", "first_moment_tolerance", "second_moment_tolerance"},
    )
    quadrature = QuadratureSpec(
        order=_integer(raw_quadrature["order"], "quadrature.order", 1),
        check_order=_integer(raw_quadrature["check_order"], "quadrature.check_order", 1),
        maximum_states=_integer(raw_quadrature["maximum_states"], "quadrature.maximum_states", 1),
        task_batch_size=_integer(raw_quadrature["task_batch_size"], "quadrature.task_batch_size", 1),
        first_moment_tolerance=_number(raw_quadrature["first_moment_tolerance"], "quadrature.first_moment_tolerance"),
        second_moment_tolerance=_number(raw_quadrature["second_moment_tolerance"], "quadrature.second_moment_tolerance"),
    )
    raw_mc = _keys(
        root["monte_carlo"], "monte_carlo",
        {
            "split_rule",
            "maximum_absolute_primary_sum_se_difference",
            "maximum_absolute_primary_relative_percentage_point_difference",
            "maximum_absolute_global_p05_user_se_difference",
        },
    )
    monte_carlo = MonteCarloSpec(
        split_rule=_string(raw_mc["split_rule"], "monte_carlo.split_rule"),
        maximum_absolute_primary_sum_se_difference=_number(
            raw_mc["maximum_absolute_primary_sum_se_difference"],
            "monte_carlo.maximum_absolute_primary_sum_se_difference",
        ),
        maximum_absolute_primary_relative_percentage_point_difference=_number(
            raw_mc["maximum_absolute_primary_relative_percentage_point_difference"],
            "monte_carlo.maximum_absolute_primary_relative_percentage_point_difference",
        ),
        maximum_absolute_global_p05_user_se_difference=_number(
            raw_mc["maximum_absolute_global_p05_user_se_difference"],
            "monte_carlo.maximum_absolute_global_p05_user_se_difference",
        ),
    )
    raw_materiality = _keys(
        root["materiality"], "materiality",
        {"minimum_relative_gain_pct", "minimum_absolute_gain", "minimum_p05_lower_bound"},
    )
    materiality = MaterialitySpec(
        minimum_relative_gain_pct=_number(raw_materiality["minimum_relative_gain_pct"], "materiality.minimum_relative_gain_pct"),
        minimum_absolute_gain=_number(raw_materiality["minimum_absolute_gain"], "materiality.minimum_absolute_gain"),
        minimum_p05_lower_bound=_number(raw_materiality["minimum_p05_lower_bound"], "materiality.minimum_p05_lower_bound"),
    )
    raw_conditions = root["conditions"]
    if not isinstance(raw_conditions, list):
        raise ScreenConfigurationError("conditions must be a list")
    conditions: list[Condition] = []
    for index, raw in enumerate(raw_conditions):
        item = _keys(raw, f"conditions[{index}]", {"id", "kappa", "delta_deg"})
        if not isinstance(item["id"], str):
            raise ScreenConfigurationError(f"conditions[{index}].id must be a string")
        conditions.append(
            Condition(
                item["id"],
                _number(item["kappa"], f"conditions[{index}].kappa"),
                _number(item["delta_deg"], f"conditions[{index}].delta_deg"),
            )
        )
    if not isinstance(root["topology_seeds"], list) or not isinstance(root["receiver_ids"], list):
        raise ScreenConfigurationError("topology_seeds and receiver_ids must be lists")
    if not isinstance(root["decision_eligible"], bool):
        raise ScreenConfigurationError("decision_eligible must be Boolean")
    return StrongAffineScreenConfig(
        schema_version=_string(root["schema_version"], "schema_version"),
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        stage=_string(root["stage"], "stage"),
        decision_eligible=root["decision_eligible"],
        lock_status=_string(root["lock_status"], "lock_status"),
        base_seed=_integer(root["base_seed"], "base_seed"),
        topology_seeds=tuple(_integer(value, "topology_seed") for value in root["topology_seeds"]),
        bootstrap_seed=_integer(root["bootstrap_seed"], "bootstrap_seed"),
        bootstrap_replicates=_integer(root["bootstrap_replicates"], "bootstrap_replicates", 1),
        confidence_level=_number(root["confidence_level"], "confidence_level"),
        conditions=tuple(conditions),
        realizations_per_topology_condition=_integer(
            root["realizations_per_topology_condition"], "realizations_per_topology_condition", 4
        ),
        system=system,
        quadrature=quadrature,
        monte_carlo=monte_carlo,
        receiver_ids=tuple(_string(value, "receiver_id") for value in root["receiver_ids"]),
        rzf_regularization=_number(root["rzf_regularization"], "rzf_regularization"),
        materiality=materiality,
        output_directory=_string(root["output_directory"], "output_directory"),
        preregistration_path=None if root["preregistration_path"] is None else _string(root["preregistration_path"], "preregistration_path"),
        lock_manifest_path=None if root["lock_manifest_path"] is None else _string(root["lock_manifest_path"], "lock_manifest_path"),
    )


def seed_tag_json(base_seed: int, *tags: object) -> str:
    _integer(base_seed, "base_seed")
    if any(not isinstance(tag, (str, Integral)) or isinstance(tag, (bool, np.bool_)) for tag in tags):
        raise ValueError("seed tags must be strings or non-Boolean integers")
    normalized = [int(tag) if isinstance(tag, Integral) else tag for tag in tags]
    return json.dumps([int(base_seed), *normalized], ensure_ascii=True, separators=(",", ":"))


def derived_seed(base_seed: int, *tags: object) -> int:
    payload = seed_tag_json(base_seed, *tags).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _frame_seed_metadata(
    values: object,
    config: StrongAffineScreenConfig,
    *,
    label: str,
) -> dict[str, Any]:
    expected_shape = (
        len(config.topology_seeds),
        len(config.conditions),
        config.realizations_per_topology_condition,
    )
    registry = np.asarray(values)
    if registry.shape != expected_shape or not np.issubdtype(registry.dtype, np.integer):
        raise RuntimeError(f"{label} has the wrong shape or dtype")
    canonical = np.asarray(registry, dtype=np.int64)
    flat = canonical.reshape(-1)
    unique = np.unique(flat)
    if unique.size != flat.size:
        raise RuntimeError(f"{label} contains a derived frame-seed collision")
    reserved = np.asarray(
        [config.base_seed, config.bootstrap_seed, *config.topology_seeds],
        dtype=np.int64,
    )
    overlap = np.intersect1d(unique, reserved, assume_unique=False)
    if overlap.size:
        raise RuntimeError(
            f"{label} overlaps a registered base/bootstrap/topology seed: "
            f"{int(overlap[0])}"
        )
    big_endian = canonical.astype(">i8", copy=False)
    return {
        "frame_seed_count": int(flat.size),
        "unique_frame_seed_count": int(unique.size),
        "reserved_seed_count": int(reserved.size),
        "overlap_with_registered_seed_count": 0,
        "frame_seed_registry_sha256_big_endian_int64": hashlib.sha256(
            big_endian.tobytes(order="C")
        ).hexdigest(),
    }


def preflight_frame_seed_registry(config: StrongAffineScreenConfig) -> dict[str, Any]:
    """Materialize and certify every frame seed before any output is created."""

    values = np.empty(
        (
            len(config.topology_seeds),
            len(config.conditions),
            config.realizations_per_topology_condition,
        ),
        dtype=np.int64,
    )
    for topology_index, topology_seed in enumerate(config.topology_seeds):
        for condition_index, condition in enumerate(config.conditions):
            for realization in range(config.realizations_per_topology_condition):
                values[topology_index, condition_index, realization] = derived_seed(
                    config.base_seed,
                    config.stage,
                    condition.id,
                    int(topology_seed),
                    realization,
                    "frame",
                )
    return _frame_seed_metadata(values, config, label="preflight frame-seed registry")


def stack_models(models: Sequence[PilotGroupModel], observations: Sequence[object]) -> ModelBatch:
    if not models or len(models) != len(observations):
        raise ValueError("models and observations must have equal nonzero length")
    if any(model.group_size != 2 for model in models):
        raise ValueError("strong-affine screen requires P=2")
    antenna_count = models[0].antenna_count
    if any(model.antenna_count != antenna_count for model in models):
        raise ValueError("all batched models must have the same antenna count")
    observation = np.asarray(observations, dtype=np.complex128)
    if observation.shape != (len(models), antenna_count) or not np.all(np.isfinite(observation)):
        raise ValueError("observations must be finite with shape (B,N)")
    return ModelBatch(
        np.stack([model.los for model in models]),
        np.stack([model.covariances for model in models]),
        np.stack([model.beta for model in models]),
        np.stack([model.noise_covariance for model in models]),
        np.stack([model.delta for model in models]),
        observation,
    )


def widely_linear_lmmse_vectorized(batch: ModelBatch) -> ComplexArray:
    """Independent batched bounded-phase widely-linear LMMSE formula."""

    rho = np.sinc(batch.delta / np.pi).astype(np.complex128)
    nu = np.sinc(2.0 * batch.delta / np.pi).astype(np.complex128)
    means = batch.los * rho[:, None, :]
    los_covariance = np.einsum("bnp,bmp->bpnm", batch.los, batch.los.conj(), optimize=True)
    los_pseudo = np.einsum("bnp,bmp->bpnm", batch.los, batch.los, optimize=True)
    covariance = batch.covariances + (1.0 - np.abs(rho) ** 2)[:, :, None, None] * los_covariance
    pseudo = (nu - rho**2)[:, :, None, None] * los_pseudo
    mean_y = np.einsum("bnp,bp->bn", means, batch.beta, optimize=True)
    covariance_y = batch.noise_covariance + np.einsum(
        "bp,bpnm->bnm", np.abs(batch.beta) ** 2, covariance, optimize=True
    )
    pseudo_y = np.einsum("bp,bpnm->bnm", batch.beta**2, pseudo, optimize=True)
    augmented = np.concatenate(
        (
            np.concatenate((covariance_y, pseudo_y), axis=2),
            np.concatenate((pseudo_y.conj(), covariance_y.conj()), axis=2),
        ),
        axis=1,
    )
    innovation = batch.observation - mean_y
    augmented_innovation = np.concatenate((innovation, innovation.conj()), axis=1)
    solved = np.linalg.solve(augmented, augmented_innovation[..., None])[..., 0]
    cross = np.concatenate(
        (
            batch.beta.conj()[:, :, None, None] * covariance,
            batch.beta[:, :, None, None] * pseudo,
        ),
        axis=3,
    )
    correction = np.einsum("bpnm,bm->bpn", cross, solved, optimize=True).transpose(0, 2, 1)
    result = means + correction
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite vectorized widely-linear estimate")
    return result.astype(np.complex128)


@lru_cache(maxsize=64)
def _cached_quadrature_grid(
    delta_0: float, delta_1: float, order: int
) -> tuple[ComplexArray, FloatArray]:
    rules: list[tuple[FloatArray, FloatArray]] = []
    for width in (delta_0, delta_1):
        if width == 0.0:
            rules.append((np.zeros(1), np.ones(1)))
        else:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            rules.append(((width * nodes).astype(np.float64), (0.5 * weights).astype(np.float64)))
    e0, e1 = np.meshgrid(rules[0][0], rules[1][0], indexing="ij")
    w0, w1 = np.meshgrid(rules[0][1], rules[1][1], indexing="ij")
    phasors = np.stack((np.exp(1j * e0.ravel()), np.exp(1j * e1.ravel())), axis=1)
    log_prior = np.log((w0 * w1).ravel())
    phasors = phasors.astype(np.complex128, copy=False)
    log_prior = log_prior.astype(np.float64, copy=False)
    phasors.setflags(write=False)
    log_prior.setflags(write=False)
    return phasors, log_prior


def exact_tensor_cme_vectorized(
    batch: ModelBatch, order: int, maximum_states: int
) -> TensorBatchResult:
    """Return the CME exact for one declared finite P=2 tensor rule."""

    if not isinstance(order, Integral) or isinstance(order, (bool, np.bool_)) or order < 1:
        raise ValueError("order must be a positive integer")
    if not isinstance(maximum_states, Integral) or maximum_states < 1:
        raise ValueError("maximum_states must be a positive integer")
    if not np.all(batch.delta == batch.delta[0]):
        raise ValueError("one vectorized batch must share exactly the same phase support")
    u, log_prior = _cached_quadrature_grid(
        float(batch.delta[0, 0]), float(batch.delta[0, 1]), int(order)
    )
    state_count = int(u.shape[0])
    if state_count > int(maximum_states):
        raise ValueError(f"tensor rule has {state_count} states above the declared cap")
    V = batch.noise_covariance + np.einsum(
        "bp,bpnm->bnm", np.abs(batch.beta) ** 2, batch.covariances, optimize=True
    )
    D = batch.los * batch.beta[:, None, :]
    residual = batch.observation[:, None, :] - np.einsum("bnp,sp->bsn", D, u, optimize=True)
    solved = np.linalg.solve(V, residual.transpose(0, 2, 1)).transpose(0, 2, 1)
    quadratic = np.real(np.einsum("bsn,bsn->bs", residual.conj(), solved, optimize=True))
    log_weight = -quadratic + log_prior[None, :]
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    normalizer = np.sum(weight, axis=1, keepdims=True)
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise FloatingPointError("nonfinite tensor posterior normalizer")
    weight /= normalizer
    z = np.einsum("bs,sp->bp", weight, u, optimize=True)
    Z = np.einsum("bs,sp,sq->bpq", weight, u, u.conj(), optimize=True)
    Z = 0.5 * (Z + Z.conj().transpose(0, 2, 1))
    Z[:, 0, 0] = 1.0
    Z[:, 1, 1] = 1.0
    posterior_residual = batch.observation - np.einsum("bnp,bp->bn", D, z, optimize=True)
    diffuse_solve = np.linalg.solve(V, posterior_residual[..., None])[..., 0]
    correction = (
        np.einsum("bpnm,bm->bpn", batch.covariances, diffuse_solve, optimize=True)
        * batch.beta.conj()[:, :, None]
    ).transpose(0, 2, 1)
    estimates = batch.los * z[:, None, :] + correction
    if not all(np.all(np.isfinite(value)) for value in (z, Z, estimates)):
        raise FloatingPointError("nonfinite tensor CME result")
    return TensorBatchResult(estimates, z, Z, state_count)


def _uatf_statistics(
    channel: ComplexArray,
    combiner: ComplexArray,
    payload_power: float,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
) -> UatFSufficientStatistics:
    powers = np.full(channel.shape[2], payload_power, dtype=np.float64)
    result = uatf_uplink(channel, combiner, powers, noise_variance, tau_p, tau_c)
    inner = np.einsum("smk,smi->ski", combiner.conj(), channel, optimize=True)
    coherent = np.mean(np.diagonal(inner, axis1=1, axis2=2), axis=0)
    second = np.mean(np.abs(inner) ** 2, axis=0)
    mean_norm = np.mean(np.sum(np.abs(combiner) ** 2, axis=1), axis=0)
    numerator = powers * np.abs(coherent) ** 2
    denominator = second @ powers - numerator + noise_variance * mean_norm
    if not np.allclose(coherent, result.coherent_gain, rtol=1e-12, atol=1e-12):
        raise FloatingPointError("UatF coherent-gain sufficient-statistic mismatch")
    if not np.allclose(denominator, result.denominator, rtol=1e-12, atol=1e-12):
        raise FloatingPointError("UatF denominator sufficient-statistic mismatch")
    arrays = (result.sinr, result.spectral_efficiency, coherent, denominator, numerator, second, mean_norm)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("nonfinite UatF sufficient statistic")
    return UatFSufficientStatistics(
        result.sinr,
        result.spectral_efficiency,
        coherent.astype(np.complex128),
        denominator.astype(np.float64),
        numerator.astype(np.float64),
        second.astype(np.float64),
        mean_norm.astype(np.float64),
    )


def paired_cluster_bootstrap(
    spectral_efficiency: FloatArray,
    valid: BoolArray,
    *,
    receiver_index: int,
    bootstrap_seed: int,
    replicates: int,
    confidence_level: float,
    materiality: MaterialitySpec,
) -> dict[str, Any]:
    """Cluster bootstrap; p05 is global within each cluster-resampled replicate."""

    values = np.asarray(spectral_efficiency, dtype=np.float64)
    complete = np.asarray(valid, dtype=bool)
    if values.ndim != 5 or values.shape[2:] != (2, 2, 2) or complete.shape != values.shape[:2]:
        raise StatisticalContractError("expected SE[T,C,method,receiver,user] and valid[T,C]")
    if not np.all(complete):
        return {"status": "INCOMPLETE_PAIRED_INPUT", "decision": "FAIL_STRONG_AFFINE_SCREEN"}
    if not np.all(np.isfinite(values)):
        return {"status": "NONFINITE_PAIRED_INPUT", "decision": "FAIL_STRONG_AFFINE_SCREEN"}
    proposed = values[:, :, 0, receiver_index, :]
    baseline = values[:, :, 1, receiver_index, :]
    baseline_sum = np.sum(baseline, axis=2)
    if np.any(baseline_sum <= 0.0):
        return {"status": "ZERO_BASELINE_DENOMINATOR", "decision": "FAIL_STRONG_AFFINE_SCREEN"}
    user_difference = proposed - baseline
    absolute_cluster = np.mean(np.sum(user_difference, axis=2), axis=1)
    relative_cluster = np.mean(100.0 * np.sum(user_difference, axis=2) / baseline_sum, axis=1)

    def statistic(indices: NDArray[np.int64]) -> FloatArray:
        return np.array(
            [
                np.median(absolute_cluster[indices]),
                np.median(relative_cluster[indices]),
                np.quantile(user_difference[indices].reshape(-1), 0.05, method="linear"),
            ],
            dtype=np.float64,
        )

    point = statistic(np.arange(values.shape[0], dtype=np.int64))
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = np.empty((int(replicates), 3), dtype=np.float64)
    for repeat in range(int(replicates)):
        selected = rng.integers(0, values.shape[0], size=values.shape[0], dtype=np.int64)
        bootstrap[repeat] = statistic(selected)
    alpha = 1.0 - confidence_level
    interval = np.quantile(
        bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0, method="linear"
    )
    passed = bool(
        interval[0, 0] > 0.0
        and point[1] >= materiality.minimum_relative_gain_pct
        and point[0] >= materiality.minimum_absolute_gain
        and interval[0, 2] >= materiality.minimum_p05_lower_bound
    )
    return {
        "status": "COMPLETE",
        "decision": "PASS_STRONG_AFFINE_SCREEN" if passed else "FAIL_STRONG_AFFINE_SCREEN",
        "topology_cluster_count": int(values.shape[0]),
        "condition_count": int(values.shape[1]),
        "statistic_definition": "topology-median absolute/relative; global p05 recomputed after topology-cluster resampling",
        "quantile_method": "linear",
        "bootstrap_unit": "topology",
        "bootstrap_replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "absolute_sum_se_difference": {"point": float(point[0]), "ci": interval[:, 0].tolist()},
        "relative_sum_se_gain_pct": {"point": float(point[1]), "ci": interval[:, 1].tolist()},
        "global_p05_user_se_difference": {"point": float(point[2]), "ci": interval[:, 2].tolist()},
    }


def split_half_primary_gate(
    split_spectral_efficiency: FloatArray,
    valid: BoolArray,
    *,
    receiver_index: int,
    specification: MonteCarloSpec,
) -> dict[str, Any]:
    """Check MC precision on the same three primary point statistics.

    The two halves are fixed by even versus odd realization index.  Each half
    independently reconstructs the topology-median absolute and relative RZF
    sum-SE effects and the global user-level p05.  This is an adequacy gate,
    not a confidence interval and not an opportunity to extend the run.
    """

    values = np.asarray(split_spectral_efficiency, dtype=np.float64)
    complete = np.asarray(valid, dtype=bool)
    if values.ndim != 6 or values.shape[0] != 2 or values.shape[3:] != (2, 2, 2):
        raise StatisticalContractError(
            "expected split SE[half,T,C,method,receiver,user]"
        )
    if complete.shape != values.shape[1:3]:
        raise StatisticalContractError("split-half valid mask has the wrong shape")
    if not np.all(complete):
        return {
            "status": "INCOMPLETE_PAIRED_INPUT",
            "pass": False,
            "decision_if_scientific": "FAIL_STRONG_AFFINE_SCREEN_SPLIT_HALF_MC_GATE",
        }
    if not np.all(np.isfinite(values)):
        return {
            "status": "NONFINITE_PAIRED_INPUT",
            "pass": False,
            "decision_if_scientific": "FAIL_STRONG_AFFINE_SCREEN_SPLIT_HALF_MC_GATE",
        }

    points = np.empty((2, 3), dtype=np.float64)
    for half in range(2):
        proposed = values[half, :, :, 0, receiver_index, :]
        baseline = values[half, :, :, 1, receiver_index, :]
        baseline_sum = np.sum(baseline, axis=2)
        if np.any(baseline_sum <= 0.0):
            return {
                "status": "ZERO_BASELINE_DENOMINATOR",
                "pass": False,
                "decision_if_scientific": "FAIL_STRONG_AFFINE_SCREEN_SPLIT_HALF_MC_GATE",
            }
        user_difference = proposed - baseline
        absolute_cluster = np.mean(np.sum(user_difference, axis=2), axis=1)
        relative_cluster = np.mean(
            100.0 * np.sum(user_difference, axis=2) / baseline_sum, axis=1
        )
        points[half] = (
            np.median(absolute_cluster),
            np.median(relative_cluster),
            np.quantile(user_difference.reshape(-1), 0.05, method="linear"),
        )

    difference = np.abs(points[0] - points[1])
    thresholds = np.array(
        [
            specification.maximum_absolute_primary_sum_se_difference,
            specification.maximum_absolute_primary_relative_percentage_point_difference,
            specification.maximum_absolute_global_p05_user_se_difference,
        ],
        dtype=np.float64,
    )
    passed = bool(np.all(difference <= thresholds))
    return {
        "status": "COMPLETE",
        "pass": passed,
        "split_rule": specification.split_rule,
        "half_point_statistics": {
            "even_realization_index": {
                "primary_absolute_sum_se_difference": float(points[0, 0]),
                "primary_relative_sum_se_gain_pct": float(points[0, 1]),
                "global_p05_user_se_difference": float(points[0, 2]),
            },
            "odd_realization_index": {
                "primary_absolute_sum_se_difference": float(points[1, 0]),
                "primary_relative_sum_se_gain_pct": float(points[1, 1]),
                "global_p05_user_se_difference": float(points[1, 2]),
            },
        },
        "absolute_between_half_differences": {
            "primary_absolute_sum_se_difference": float(difference[0]),
            "primary_relative_percentage_points": float(difference[1]),
            "global_p05_user_se_difference": float(difference[2]),
        },
        "maximum_allowed_absolute_between_half_differences": {
            "primary_absolute_sum_se_difference": float(thresholds[0]),
            "primary_relative_percentage_points": float(thresholds[1]),
            "global_p05_user_se_difference": float(thresholds[2]),
        },
        "decision_if_scientific": (
            "MC_PRECISION_ADEQUATE"
            if passed
            else "FAIL_STRONG_AFFINE_SCREEN_SPLIT_HALF_MC_GATE"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root == path or root not in path.parents:
        raise RuntimeError("resolved path leaves project root or equals it")
    return path


def _hash_paths(root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = _inside(root, relative)
        if not path.is_file():
            raise RuntimeError(f"locked input is absent: {relative}")
        result[relative] = _sha256(path)
    return result


def _scientific_thread_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}


def _require_scientific_runtime_policy() -> None:
    observed = _scientific_thread_environment()
    invalid = {name: value for name, value in observed.items() if value != "1"}
    if invalid:
        raise RuntimeError(
            "scientific execution requires all BLAS/thread environment variables "
            f"to equal the literal '1'; invalid={invalid}"
        )


def runtime_fingerprint() -> dict[str, Any]:
    configuration = getattr(np.__config__, "CONFIG", {})
    build = configuration.get("Build Dependencies", {}) if isinstance(configuration, dict) else {}
    blas_relevant = {
        "blas": build.get("blas"),
        "lapack": build.get("lapack"),
        "simd": configuration.get("SIMD Extensions") if isinstance(configuration, dict) else None,
    }
    canonical = json.dumps(blas_relevant, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    result = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyyaml_version": yaml.__version__,
        "numpy_default_rng_bit_generator": type(
            np.random.default_rng(0).bit_generator
        ).__name__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "thread_environment": _scientific_thread_environment(),
        "blas_relevant_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "blas_relevant": blas_relevant,
    }
    serialized = json.dumps(
        result, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":")
    )
    result["runtime_fingerprint_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return result


def _current_integrity(
    root: Path,
    config_path: Path,
    preregistration: Path | None,
    lock_manifest: Path | None,
) -> IntegritySnapshot:
    return IntegritySnapshot(
        config_sha256=_sha256(config_path),
        preregistration_sha256=None if preregistration is None else _sha256(preregistration),
        lock_manifest_sha256=None if lock_manifest is None else _sha256(lock_manifest),
        source_sha256=_hash_paths(root, RUNTIME_SOURCE_PATHS),
        test_sha256=_hash_paths(root, RUNTIME_TEST_PATHS),
        runtime_fingerprint=runtime_fingerprint(),
    )


def _verify_locked_preregistration(
    config: StrongAffineScreenConfig, root: Path, config_path: Path
) -> tuple[Path | None, Path | None, IntegritySnapshot]:
    if config.stage == SMOKE_STAGE:
        snapshot = _current_integrity(root, config_path, None, None)
        if load_screen_config(config_path) != config:
            raise RuntimeError(
                "parsed screen configuration differs from the snapshotted file"
            )
        _verify_toctou(snapshot, root, config_path, None, None)
        return None, None, snapshot
    _require_scientific_runtime_policy()
    expected_module = (root / "src" / "phase_graph_cme" / "strong_affine_screen.py").resolve()
    if Path(__file__).resolve() != expected_module:
        raise RuntimeError("executed strong_affine_screen module is not the project-root source")
    preregistration = _inside(root, str(config.preregistration_path))
    lock_manifest = _inside(root, str(config.lock_manifest_path))
    if not preregistration.is_file() or not lock_manifest.is_file():
        raise RuntimeError("preregistration or detached lock manifest is absent")
    config_relative = str(config_path.relative_to(root)).replace("\\", "/")
    prereg_document = _mapping(_load_yaml_unique(preregistration, "preregistration"), "preregistration")
    if (
        prereg_document.get("schema_version") != "strong-affine-prereg/1.0"
        or prereg_document.get("candidate_id") != "R2C01_EXACT_PHASE_CME"
        or prereg_document.get("experiment_id") != config.experiment_id
        or prereg_document.get("lock_status") != LOCKED
    ):
        raise RuntimeError("preregistration identity, version, or lock status is invalid")
    prereg_config = _mapping(
        prereg_document.get("configuration"), "preregistration.configuration"
    )
    if prereg_config != {
        "path": config_relative,
        "sha256": _sha256(config_path),
    }:
        raise RuntimeError("preregistration does not bind this exact configuration")
    manifest_required = {
        "schema_version", "lock_status", "results_existed_at_lock", "preregistration_path",
        "preregistration_sha256", "authorized_experiment_ids", "authorized_config_sha256",
        "authorized_output_directories", "authorized_source_sha256", "authorized_test_sha256",
        "authorized_runtime_fingerprint",
    }
    document = _keys(
        _load_yaml_unique(lock_manifest, "detached lock manifest"),
        "detached lock manifest",
        manifest_required,
        {"created_utc"},
    )
    if document["schema_version"] != "strong-affine-lock/1.0" or document["lock_status"] != LOCKED:
        raise RuntimeError("detached lock manifest has invalid version or status")
    if document["results_existed_at_lock"] is not False:
        raise RuntimeError("detached lock manifest must declare no pre-existing results")
    if document["preregistration_path"] != config.preregistration_path:
        raise RuntimeError("detached manifest names a different preregistration")
    if document["preregistration_sha256"] != _sha256(preregistration):
        raise RuntimeError("preregistration hash mismatch")
    authorized_ids = document["authorized_experiment_ids"]
    if authorized_ids != [config.experiment_id]:
        raise RuntimeError("detached manifest must authorize exactly this experiment_id")
    authorized_configs = document["authorized_config_sha256"]
    if authorized_configs != {config_relative: _sha256(config_path)}:
        raise RuntimeError("detached manifest must authorize exactly this configuration")
    authorized_outputs = document["authorized_output_directories"]
    if authorized_outputs != [config.output_directory]:
        raise RuntimeError(
            "detached manifest must authorize exactly the configured output directory"
        )
    snapshot = _current_integrity(root, config_path, preregistration, lock_manifest)
    if document["authorized_source_sha256"] != snapshot.source_sha256:
        raise RuntimeError("source hash set is not exactly authorized")
    if document["authorized_test_sha256"] != snapshot.test_sha256:
        raise RuntimeError("test hash set is not exactly authorized")
    if document["authorized_runtime_fingerprint"] != snapshot.runtime_fingerprint:
        raise RuntimeError("runtime/BLAS fingerprint is not exactly authorized")
    if preregistration.stat().st_mtime_ns > time.time_ns() or lock_manifest.stat().st_mtime_ns > time.time_ns():
        raise RuntimeError("lock artifact timestamp is in the future")
    if load_screen_config(config_path) != config:
        raise RuntimeError(
            "parsed screen configuration differs from the snapshotted file"
        )
    _verify_toctou(snapshot, root, config_path, preregistration, lock_manifest)
    return preregistration, lock_manifest, snapshot


def validate_locked_screen(
    config: StrongAffineScreenConfig, *, project_root: Path, config_path: Path
) -> dict[str, Any]:
    """Validate every scientific lock input without starting a run."""

    if config.stage != SCREEN_STAGE:
        raise RuntimeError("validate-lock is only defined for the scientific screen")
    root = project_root.resolve()
    resolved_config = config_path.resolve()
    if root not in resolved_config.parents or not resolved_config.is_file():
        raise RuntimeError("config path is absent or leaves project root")
    preregistration, lock_manifest, snapshot = _verify_locked_preregistration(
        config, root, resolved_config
    )
    frame_seed_preflight = preflight_frame_seed_registry(config)
    output = _inside(root, config.output_directory)
    staging = output.with_name(f".{output.name}.STAGING")
    failed = output.with_name(f"{output.name}.FAILED")
    if output.exists() or staging.exists() or failed.exists():
        raise RuntimeError(
            "locked output, staging, or failure directory already exists"
        )
    _verify_toctou(
        snapshot, root, resolved_config, preregistration, lock_manifest
    )
    return {
        "status": "VALID_LOCK_READY_FOR_SINGLE_EXECUTION",
        "experiment_id": config.experiment_id,
        "output_directory": config.output_directory,
        "config_sha256": snapshot.config_sha256,
        "preregistration_sha256": snapshot.preregistration_sha256,
        "lock_manifest_sha256": snapshot.lock_manifest_sha256,
        "source_file_count": len(snapshot.source_sha256),
        "test_file_count": len(snapshot.test_sha256),
        "frame_seed_preflight": frame_seed_preflight,
        "runtime_fingerprint_sha256": snapshot.runtime_fingerprint[
            "runtime_fingerprint_sha256"
        ],
    }


def _verify_toctou(
    before: IntegritySnapshot,
    root: Path,
    config_path: Path,
    preregistration: Path | None,
    lock_manifest: Path | None,
) -> None:
    after = _current_integrity(root, config_path, preregistration, lock_manifest)
    if after != before:
        raise RuntimeError("TOCTOU integrity failure: a locked input or runtime changed during execution")


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, value: object, *, exclusive: bool = True) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False, default=_json_default) + "\n"
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty JSONL: {path}")
    schema = set(rows[0])
    if any(set(row) != schema for row in rows):
        raise RuntimeError("inconsistent JSONL row schema")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, allow_nan=False, default=_json_default, separators=(",", ":")) + "\n"
            )


def _write_complete_marker_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish COMPLETE.json atomically within staging; not a power-loss guarantee."""

    temporary = path.with_name("COMPLETE.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_progress_record(handle: TextIO, row: Mapping[str, Any]) -> None:
    if tuple(row) != PROGRESS_FIELDS:
        raise RuntimeError("progress record does not match the exact ordered schema")
    handle.write(
        json.dumps(
            row,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
            separators=(",", ":"),
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def _peak_process_rss_bytes() -> int:
    """Return low-overhead peak process RSS since launch on supported hosts."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_memory.restype = wintypes.BOOL
        if not get_memory(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            code = ctypes.get_last_error()
            raise OSError(code, "GetProcessMemoryInfo failed")
        result = int(counters.PeakWorkingSetSize)
    else:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result = maximum if sys.platform == "darwin" else maximum * 1024
    if result <= 0:
        raise RuntimeError("peak process RSS measurement is unavailable")
    return result


def _combiners(estimates: ComplexArray, receiver: str, regularization: float) -> ComplexArray:
    if receiver == "MR":
        return np.asarray([mr_combiner(frame) for frame in estimates])
    if receiver == "RZF":
        return np.asarray([rzf_combiner(frame, regularization) for frame in estimates])
    raise ValueError(f"unsupported receiver: {receiver}")


def _channel_nmse(estimates: ComplexArray, channels: ComplexArray) -> FloatArray:
    numerator = np.mean(np.sum(np.abs(estimates - channels) ** 2, axis=1), axis=0)
    denominator = np.mean(np.sum(np.abs(channels) ** 2, axis=1), axis=0)
    if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise FloatingPointError("nonfinite or zero channel-NMSE denominator")
    return (numerator / denominator).astype(np.float64)


def _finalize_primary_decision(
    stage: str, analysis: Mapping[str, Any], *, split_half_pass: bool
) -> dict[str, Any]:
    """Apply split precision only after complete upstream paired analysis."""

    result = dict(analysis)
    if stage == SMOKE_STAGE:
        result["decision"] = "NOT_ELIGIBLE_NON_EVIDENCE_SMOKE"
    elif result.get("status") == "COMPLETE" and not split_half_pass:
        result["decision"] = "FAIL_STRONG_AFFINE_SCREEN_SPLIT_HALF_MC_GATE"
    return result


def _execute(
    config: StrongAffineScreenConfig,
    *,
    progress_sink: Callable[[Mapping[str, Any]], None],
    expected_frame_seed_preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray[Any]]]:
    topology_count = len(config.topology_seeds)
    condition_count = len(config.conditions)
    realization_count = config.realizations_per_topology_condition
    access_points = config.system.access_points
    users = 2
    shape = (topology_count, condition_count, len(METHODS), len(RECEIVERS), users)
    se = np.full(shape, np.nan, dtype=np.float64)
    sinr = np.full(shape, np.nan, dtype=np.float64)
    denominator = np.full(shape, np.nan, dtype=np.float64)
    numerator = np.full(shape, np.nan, dtype=np.float64)
    coherent = np.full(shape, np.nan + 0j, dtype=np.complex128)
    mean_norm = np.full(shape, np.nan, dtype=np.float64)
    second_power = np.full(shape + (users,), np.nan, dtype=np.float64)
    nmse = np.full((topology_count, condition_count, len(METHODS), users), np.nan)
    split_spectral_efficiency = np.full((2,) + shape, np.nan, dtype=np.float64)
    split_user_difference = np.full(shape, np.nan, dtype=np.float64)
    split_sum_difference = np.full(shape[:-1], np.nan, dtype=np.float64)
    paired_valid = np.zeros((topology_count, condition_count), dtype=bool)
    task_shape = (topology_count, condition_count, realization_count, access_points)
    first_difference = np.full(task_shape, np.nan, dtype=np.float64)
    second_difference = np.full(task_shape, np.nan, dtype=np.float64)
    failure_code = np.full(task_shape, 2, dtype=np.int8)
    frame_seed = np.empty((topology_count, condition_count, realization_count), dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []
    executed_tasks = 0
    progress_record_count = 0
    peak_process_rss_bytes = _peak_process_rss_bytes()

    for topology_index, topology_seed in enumerate(config.topology_seeds):
        for condition_index, condition in enumerate(config.conditions):
            scenario = config.system.scenario(condition)
            topology = generate_topology(scenario, seed=int(topology_seed))
            channels: list[ComplexArray] = []
            models: list[PilotGroupModel] = []
            observations: list[ComplexArray] = []
            for realization in range(realization_count):
                tags = (config.stage, condition.id, int(topology_seed), realization, "frame")
                generated_seed = derived_seed(config.base_seed, *tags)
                frame_seed[topology_index, condition_index, realization] = generated_seed
                frame = generate_channel_frame(topology, scenario, seed=generated_seed)
                channels.append(
                    frame.channels.reshape(access_points * config.system.antennas_per_ap, users)
                )
                for access_point in range(access_points):
                    model, observation, group, _ = pilot_group(
                        topology, frame, scenario, access_point=access_point, pilot=0
                    )
                    if model.group_size != 2 or not np.array_equal(group, np.array([0, 1])):
                        raise RuntimeError("P=2 screen received a non-P=2 pilot group")
                    models.append(model)
                    observations.append(observation)
            model_batch = stack_models(models, observations)
            task_count = realization_count * access_points
            antenna_count = config.system.antennas_per_ap
            exact_flat = np.full((task_count, antenna_count, users), np.nan + 0j)
            affine_flat = np.full_like(exact_flat, np.nan + 0j)
            dz_flat = np.full(task_count, np.nan)
            dZ_flat = np.full(task_count, np.nan)
            code_flat = np.full(task_count, 2, dtype=np.int8)
            for start in range(0, task_count, config.quadrature.task_batch_size):
                stop = min(start + config.quadrature.task_batch_size, task_count)
                executed_tasks += stop - start
                try:
                    portion = model_batch.slice(start, stop)
                    coarse = exact_tensor_cme_vectorized(
                        portion, config.quadrature.order, config.quadrature.maximum_states
                    )
                    fine = exact_tensor_cme_vectorized(
                        portion, config.quadrature.check_order, config.quadrature.maximum_states
                    )
                    affine = widely_linear_lmmse_vectorized(portion)
                    dz = np.max(np.abs(coarse.z - fine.z), axis=1)
                    dZ = np.max(np.abs(coarse.Z - fine.Z), axis=(1, 2))
                    rejected = (
                        (dz > config.quadrature.first_moment_tolerance)
                        | (dZ > config.quadrature.second_moment_tolerance)
                    )
                    exact_flat[start:stop] = fine.estimates
                    affine_flat[start:stop] = affine
                    dz_flat[start:stop] = dz
                    dZ_flat[start:stop] = dZ
                    code_flat[start:stop] = rejected.astype(np.int8)
                except (FloatingPointError, MemoryError, np.linalg.LinAlgError, ValueError) as error:
                    failure_events.append(
                        {
                            "condition_id": condition.id,
                            "topology_seed": int(topology_seed),
                            "task_start": start,
                            "task_stop_exclusive": stop,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            first_difference[topology_index, condition_index] = dz_flat.reshape(realization_count, access_points)
            second_difference[topology_index, condition_index] = dZ_flat.reshape(realization_count, access_points)
            failure_code[topology_index, condition_index] = code_flat.reshape(realization_count, access_points)
            channel_samples = np.asarray(channels)
            estimate_samples = {
                METHODS[0]: exact_flat.reshape(
                    realization_count, access_points * antenna_count, users
                ),
                METHODS[1]: affine_flat.reshape(
                    realization_count, access_points * antenna_count, users
                ),
            }
            task_complete = bool(np.all(code_flat == 0))
            metric_complete = task_complete
            even = np.arange(realization_count) % 2 == 0
            condition_stats: dict[tuple[int, int], UatFSufficientStatistics] = {}
            split_stats: dict[tuple[int, int, int], UatFSufficientStatistics] = {}
            if task_complete:
                try:
                    for method_index, method in enumerate(METHODS):
                        nmse[topology_index, condition_index, method_index] = _channel_nmse(
                            estimate_samples[method], channel_samples
                        )
                        for receiver_index, receiver in enumerate(RECEIVERS):
                            combiner = _combiners(
                                estimate_samples[method], receiver, config.rzf_regularization
                            )
                            full = _uatf_statistics(
                                channel_samples,
                                combiner,
                                scenario.payload_power,
                                scenario.payload_noise_variance,
                                scenario.pilot_symbols,
                                scenario.coherence_symbols,
                            )
                            condition_stats[(method_index, receiver_index)] = full
                            for half_index, mask in enumerate((even, ~even)):
                                split_stats[(method_index, receiver_index, half_index)] = _uatf_statistics(
                                    channel_samples[mask],
                                    combiner[mask],
                                    scenario.payload_power,
                                    scenario.payload_noise_variance,
                                    scenario.pilot_symbols,
                                    scenario.coherence_symbols,
                                )
                                split_spectral_efficiency[
                                    half_index,
                                    topology_index,
                                    condition_index,
                                    method_index,
                                    receiver_index,
                                ] = split_stats[
                                    (method_index, receiver_index, half_index)
                                ].spectral_efficiency
                            first = split_stats[(method_index, receiver_index, 0)].spectral_efficiency
                            second = split_stats[(method_index, receiver_index, 1)].spectral_efficiency
                            split_user_difference[topology_index, condition_index, method_index, receiver_index] = np.abs(first - second)
                            split_sum_difference[topology_index, condition_index, method_index, receiver_index] = abs(float(np.sum(first) - np.sum(second)))
                except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                    metric_complete = False
                    failure_events.append(
                        {
                            "condition_id": condition.id,
                            "topology_seed": int(topology_seed),
                            "task_start": 0,
                            "task_stop_exclusive": task_count,
                            "error_type": type(error).__name__,
                            "error": f"METRIC_FAILURE: {error}",
                        }
                    )
            paired_valid[topology_index, condition_index] = metric_complete
            if metric_complete:
                for method_index in range(len(METHODS)):
                    for receiver_index in range(len(RECEIVERS)):
                        stats = condition_stats[(method_index, receiver_index)]
                        se[topology_index, condition_index, method_index, receiver_index] = stats.spectral_efficiency
                        sinr[topology_index, condition_index, method_index, receiver_index] = stats.sinr
                        denominator[topology_index, condition_index, method_index, receiver_index] = stats.denominator
                        numerator[topology_index, condition_index, method_index, receiver_index] = stats.numerator
                        coherent[topology_index, condition_index, method_index, receiver_index] = stats.coherent_gain
                        mean_norm[topology_index, condition_index, method_index, receiver_index] = stats.mean_combiner_norm
                        second_power[topology_index, condition_index, method_index, receiver_index] = stats.second_effective_power
            for method_index, method in enumerate(METHODS):
                for receiver_index, receiver in enumerate(RECEIVERS):
                    for user in range(users):
                        good = bool(metric_complete)
                        metric_rows.append(
                            {
                                "experiment_id": config.experiment_id,
                                "stage": config.stage,
                                "condition_id": condition.id,
                                "kappa": condition.kappa,
                                "delta_deg": condition.delta_deg,
                                "topology_seed": int(topology_seed),
                                "method": method,
                                "receiver": receiver,
                                "user": user,
                                "sinr": float(sinr[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "spectral_efficiency": float(se[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "uatf_numerator": float(numerator[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "uatf_denominator": float(denominator[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "coherent_gain_real": float(coherent[topology_index, condition_index, method_index, receiver_index, user].real) if good else None,
                                "coherent_gain_imag": float(coherent[topology_index, condition_index, method_index, receiver_index, user].imag) if good else None,
                                "mean_combiner_norm": float(mean_norm[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "second_effective_power_by_interferer": second_power[topology_index, condition_index, method_index, receiver_index, user].tolist() if good else None,
                                "channel_nmse": float(nmse[topology_index, condition_index, method_index, user]) if good else None,
                                "split_half_absolute_user_se_difference": float(split_user_difference[topology_index, condition_index, method_index, receiver_index, user]) if good else None,
                                "split_half_absolute_sum_se_difference": float(split_sum_difference[topology_index, condition_index, method_index, receiver_index]) if good else None,
                                "split_half_gate_pass": None,
                                "failure": "" if good else "INCOMPLETE_TOPOLOGY_CONDITION",
                            }
                        )
            peak_process_rss_bytes = max(
                peak_process_rss_bytes, _peak_process_rss_bytes()
            )
            progress_sink(
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "sequence_index": progress_record_count,
                    "experiment_id": config.experiment_id,
                    "stage": config.stage,
                    "topology_index": topology_index,
                    "topology_seed": int(topology_seed),
                    "condition_index": condition_index,
                    "condition_id": condition.id,
                    "expected_cell_task_count": task_count,
                    "executed_cell_task_count": task_count,
                    "failed_or_rejected_cell_task_count": int(
                        np.sum(code_flat != 0)
                    ),
                    "paired_cell_valid": bool(metric_complete),
                    "cumulative_executed_task_count": executed_tasks,
                    "cumulative_failure_event_count": len(failure_events),
                    "peak_process_rss_bytes": peak_process_rss_bytes,
                    "rss_measurement_scope": "PROCESS_PEAK_SINCE_LAUNCH",
                    "completed_unix": time.time(),
                }
            )
            progress_record_count += 1

    expected_tasks = topology_count * condition_count * realization_count * access_points
    expected_metric_rows = topology_count * condition_count * len(METHODS) * len(RECEIVERS) * users
    expected_progress_records = topology_count * condition_count
    if executed_tasks != expected_tasks:
        raise RuntimeError(f"task-count mismatch: expected {expected_tasks}, observed {executed_tasks}")
    if len(metric_rows) != expected_metric_rows:
        raise RuntimeError(f"metric-row mismatch: expected {expected_metric_rows}, observed {len(metric_rows)}")
    if progress_record_count != expected_progress_records:
        raise RuntimeError(
            "progress-record mismatch: "
            f"expected {expected_progress_records}, observed {progress_record_count}"
        )
    observed_frame_seed_preflight = _frame_seed_metadata(
        frame_seed, config, label="executed frame-seed registry"
    )
    if observed_frame_seed_preflight != dict(expected_frame_seed_preflight):
        raise RuntimeError(
            "executed frame-seed registry differs from the pre-output preflight"
        )
    analysis = paired_cluster_bootstrap(
        se,
        paired_valid,
        receiver_index=RECEIVERS.index("RZF"),
        bootstrap_seed=config.bootstrap_seed,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        materiality=config.materiality,
    )
    split_half_gate = split_half_primary_gate(
        split_spectral_efficiency,
        paired_valid,
        receiver_index=RECEIVERS.index("RZF"),
        specification=config.monte_carlo,
    )
    all_split_pass = bool(split_half_gate["pass"])
    for row in metric_rows:
        row["split_half_gate_pass"] = all_split_pass
    analysis = _finalize_primary_decision(
        config.stage, analysis, split_half_pass=all_split_pass
    )
    peak_process_rss_bytes = max(
        peak_process_rss_bytes, _peak_process_rss_bytes()
    )
    summary = {
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "decision_eligible": config.decision_eligible,
        "scientific_boundary": (
            "PREREGISTERED_DEVELOPMENT_GATE_NOT_FINAL_MANUSCRIPT_EVIDENCE"
            if config.stage == SCREEN_STAGE
            else "NON_EVIDENCE_SYNTHETIC_SMOKE_NO_SCIENTIFIC_DECISION"
        ),
        "methods": list(METHODS),
        "receivers": list(RECEIVERS),
        "topology_count": topology_count,
        "condition_count": condition_count,
        "realizations_per_topology_condition": realization_count,
        "expected_task_count": expected_tasks,
        "executed_task_count": executed_tasks,
        "expected_metric_row_count": expected_metric_rows,
        "observed_metric_row_count": len(metric_rows),
        "expected_progress_record_count": expected_progress_records,
        "observed_progress_record_count": progress_record_count,
        "peak_process_rss_bytes": peak_process_rss_bytes,
        "rss_measurement_scope": "PROCESS_PEAK_SINCE_LAUNCH",
        "frame_seed_preflight": observed_frame_seed_preflight,
        "failed_or_rejected_task_count": int(np.sum(failure_code != 0)),
        "failure_code_legend": {str(key): value for key, value in FAILURE_CODES.items()},
        "failure_event_count": len(failure_events),
        "all_paired_cells_valid": bool(np.all(paired_valid)),
        "split_half_mc_gate_pass": all_split_pass,
        "split_half_mc_gate": split_half_gate,
        "primary_analysis": analysis,
    }
    arrays: dict[str, NDArray[Any]] = {
        "spectral_efficiency": se,
        "sinr": sinr,
        "uatf_denominator": denominator,
        "uatf_numerator": numerator,
        "coherent_gain": coherent,
        "mean_combiner_norm": mean_norm,
        "second_effective_power": second_power,
        "channel_nmse": nmse,
        "split_half_spectral_efficiency": split_spectral_efficiency,
        "split_half_absolute_user_se_difference": split_user_difference,
        "split_half_absolute_sum_se_difference": split_sum_difference,
        "paired_valid": paired_valid,
        "quadrature_first_moment_difference": first_difference,
        "quadrature_second_moment_difference": second_difference,
        "task_failure_code": failure_code,
        "frame_seed": frame_seed,
        "topology_seeds": np.asarray(config.topology_seeds, dtype=np.int64),
        "condition_ids": np.asarray([item.id for item in config.conditions]),
        "method_ids": np.asarray(METHODS),
        "receiver_ids": np.asarray(RECEIVERS),
    }
    if failure_events:
        summary["failure_events"] = failure_events
    return summary, metric_rows, arrays


def run_screen(
    config: StrongAffineScreenConfig,
    *,
    project_root: Path,
    config_path: Path,
    acknowledge_locked_preregistration: bool = False,
) -> dict[str, Any]:
    """Run one immutable screen or bounded smoke with atomic state transitions."""

    root = project_root.resolve()
    config_path = config_path.resolve()
    if root not in config_path.parents or not config_path.is_file():
        raise RuntimeError("config path is absent or leaves project root")
    if config.stage == SCREEN_STAGE and acknowledge_locked_preregistration is not True:
        raise RuntimeError("scientific run requires explicit locked-preregistration acknowledgement")
    if config.stage == SMOKE_STAGE and acknowledge_locked_preregistration is not False:
        raise RuntimeError("smoke must not use the scientific acknowledgement")
    preregistration, lock_manifest, integrity = _verify_locked_preregistration(
        config, root, config_path
    )
    frame_seed_preflight = preflight_frame_seed_registry(config)
    _verify_toctou(integrity, root, config_path, preregistration, lock_manifest)
    output = _inside(root, config.output_directory)
    staging = output.with_name(f".{output.name}.STAGING")
    failed = output.with_name(f"{output.name}.FAILED")
    if output.exists() or staging.exists() or failed.exists():
        raise FileExistsError("refusing overwrite: output, staging, or failure directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    started = time.time()
    try:
        _write_json(
            staging / "RUNNING.json",
            {
                "status": "RUNNING",
                "experiment_id": config.experiment_id,
                "stage": config.stage,
                "started_unix": started,
                "config_sha256": integrity.config_sha256,
            },
        )
        with (staging / "PROGRESS.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as progress_handle:
            summary, metric_rows, arrays = _execute(
                config,
                progress_sink=lambda row: _append_progress_record(
                    progress_handle, row
                ),
                expected_frame_seed_preflight=frame_seed_preflight,
            )
        summary["elapsed_seconds"] = time.time() - started
        _write_jsonl(staging / "raw_metrics.jsonl", metric_rows)
        np.savez_compressed(staging / "screen_arrays.npz", **arrays)
        _write_json(staging / "summary.json", summary)
        _verify_toctou(integrity, root, config_path, preregistration, lock_manifest)
        payload = sorted(
            path for path in staging.iterdir() if path.is_file() and path.name != "RUNNING.json"
        )
        manifest = {
            "experiment_id": config.experiment_id,
            "stage": config.stage,
            "config_path": str(config_path.relative_to(root)).replace("\\", "/"),
            "config_sha256": integrity.config_sha256,
            "preregistration_path": None if preregistration is None else str(preregistration.relative_to(root)).replace("\\", "/"),
            "preregistration_sha256": integrity.preregistration_sha256,
            "lock_manifest_path": None if lock_manifest is None else str(lock_manifest.relative_to(root)).replace("\\", "/"),
            "lock_manifest_sha256": integrity.lock_manifest_sha256,
            "source_sha256": integrity.source_sha256,
            "test_sha256": integrity.test_sha256,
            "runtime_fingerprint": integrity.runtime_fingerprint,
            "frame_seed_preflight": frame_seed_preflight,
            "seed_serialization": "compact ASCII JSON [base_seed,*tags], SHA256 first 8 bytes big-endian masked to 63 bits",
            "task_index_order": "topology, condition, realization, access_point; pilot=0; P=2",
            "started_unix": started,
            "completed_unix": time.time(),
            "output_sha256": {path.name: _sha256(path) for path in payload},
        }
        _write_json(staging / "run_manifest.json", manifest)
        manifest_hash = _sha256(staging / "run_manifest.json")
        (staging / "run_manifest.sha256").write_text(
            f"{manifest_hash}  run_manifest.json\n", encoding="ascii"
        )
        _verify_toctou(integrity, root, config_path, preregistration, lock_manifest)
        _write_complete_marker_atomic(
            staging / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "experiment_id": config.experiment_id,
                "completed_unix": time.time(),
                "run_manifest_sha256": manifest_hash,
                "primary_analysis_status": summary["primary_analysis"]["status"],
                "scientific_decision": (
                    summary["primary_analysis"]["decision"]
                    if config.stage == SCREEN_STAGE
                    else None
                ),
                "computationally_complete_not_scientific_pass": bool(
                    config.stage == SMOKE_STAGE
                ),
                "split_half_mc_gate_pass": summary["split_half_mc_gate_pass"],
            },
        )
        (staging / "RUNNING.json").unlink()
        os.replace(staging, output)
        return summary
    except BaseException as error:
        try:
            for marker in (staging / "COMPLETE.tmp", staging / "COMPLETE.json"):
                if marker.exists():
                    marker.unlink()
            _write_json(
                staging / "FAILED.json",
                {
                    "status": "FAILED",
                    "experiment_id": config.experiment_id,
                    "failed_unix": time.time(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            os.replace(staging, failed)
        except BaseException:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m phase_graph_cme.strong_affine_screen")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    validate_lock = commands.add_parser("validate-lock")
    validate_lock.add_argument("config", type=Path)
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--acknowledge-locked-preregistration", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    config_path = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    try:
        config = load_screen_config(config_path)
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "experiment_id": config.experiment_id,
                        "stage": config.stage,
                        "decision_eligible": config.decision_eligible,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "validate-lock":
            print(
                json.dumps(
                    validate_locked_screen(
                        config, project_root=root, config_path=config_path
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        summary = run_screen(
            config,
            project_root=root,
            config_path=config_path,
            acknowledge_locked_preregistration=args.acknowledge_locked_preregistration,
        )
        print(json.dumps(summary, indent=2, allow_nan=False, default=_json_default))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
