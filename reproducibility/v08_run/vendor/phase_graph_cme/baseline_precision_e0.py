"""Bounded baseline-only precision engineering cell for R2C01 V3.

No controller or proposed-method module is imported here.  This cell evaluates
only B1, receiver interfaces, and streaming-versus-array UatF reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from .baselines import widely_linear_mmse
from .metrics import uatf_uplink
from .model import PilotGroupModel
from .production_registry import derived_seed
from .receivers import mr_combiner, rzf_combiner
from .scenario import ScenarioConfig, generate_channel_frame, generate_topology, pilot_group

EXPERIMENT_ID = "R2C01_V3_BASELINE_ONLY_PRECISION_E0_01"
CONFIG_RELATIVE = f"configs/{EXPERIMENT_ID}.yaml"
OUTPUT_RELATIVE = f"results/engineering/{EXPERIMENT_ID}"
VALIDATION_RELATIVE = f"{OUTPUT_RELATIVE}.independent_validation.json"
SCHEMA_VERSION = "r2c01-v3-baseline-precision-e0/0.1-draft"
SCIENTIFIC_BOUNDARY = "NON_SCIENTIFIC_BASELINE_ONLY_ENGINEERING"
CONDITION_SPECS = (
    ("CORE_K5_D45", 5.0, 45.0),
    ("CORE_K5_D90", 5.0, 90.0),
    ("CORE_K20_D90", 20.0, 90.0),
)
RECEIVER_SPECS = (
    ("MR", None), ("RZF_L0P25", 0.25), ("RZF_L0P5", 0.5),
    ("RZF_L1", 1.0), ("RZF_L2", 2.0), ("RZF_L4", 4.0),
)
EXPECTED_REQUIRED_PAYLOADS = (
    "CONFIG_SNAPSHOT.yaml", "SEED_REGISTRY.jsonl", "FRAME_REGISTRY.jsonl",
    "PROGRESS.jsonl", "resource_metrics.jsonl", "baseline_arrays.npz",
    "baseline_arrays_manifest.json", "scalar_equivalence.json",
    "uatf_sufficient_statistics.json", "summary.json", "run_provenance.json",
    "run_manifest.json", "run_manifest.sha256", "COMPLETE.json",
)
SOURCE_PATHS: Mapping[str, str] = MappingProxyType({
    "baseline_precision_e0_sha256": "src/phase_graph_cme/baseline_precision_e0.py",
    "scenario_sha256": "src/phase_graph_cme/scenario.py",
    "model_sha256": "src/phase_graph_cme/model.py",
    "baselines_sha256": "src/phase_graph_cme/baselines.py",
    "receivers_sha256": "src/phase_graph_cme/receivers.py",
    "metrics_sha256": "src/phase_graph_cme/metrics.py",
    "production_registry_sha256": "src/phase_graph_cme/production_registry.py",
    "production_artifacts_sha256": "src/phase_graph_cme/production_artifacts.py",
    "runner_sha256": "scripts/run_r2c01_v3_baseline_precision_e0.py",
    "validator_sha256": "scripts/validate_r2c01_v3_baseline_precision_e0.py",
})
RUNTIME_SOURCE_KEYS = frozenset(SOURCE_PATHS) - {"validator_sha256"}

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class BaselinePrecisionProtocolError(RuntimeError):
    """The immutable baseline-only engineering contract was violated."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise BaselinePrecisionProtocolError("YAML mapping key is unhashable") from error
        if duplicate:
            raise BaselinePrecisionProtocolError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class Condition:
    id: str
    rician_kappa: float
    residual_phase_half_width_deg: float


@dataclass(frozen=True)
class Receiver:
    id: str
    regularization: float | None


@dataclass(frozen=True)
class BaselinePrecisionConfig:
    raw: Mapping[str, Any]
    source_identity: Mapping[str, str]
    conditions: tuple[Condition, ...]
    receivers: tuple[Receiver, ...]
    base_seed: int
    topology_seed: int
    realizations: int
    output_directory: str
    independent_validation_sibling: str
    wall_cap_seconds: float
    rss_cap_bytes: int
    output_cap_bytes: int
    resource_sample_interval_seconds: float

    @property
    def frame_count(self) -> int:
        return len(self.conditions) * self.realizations

    def scenario(self, condition: Condition) -> ScenarioConfig:
        system = _mapping(self.raw["system"], "system")
        return ScenarioConfig(
            access_points=int(system["access_points"]), antennas_per_ap=int(system["antennas_per_ap"]),
            users=int(system["users"]), pilot_symbols=int(system["pilot_symbols"]),
            coherence_symbols=int(system["coherence_symbols"]), area_side=float(system["area_side"]),
            minimum_distance=float(system["minimum_distance"]), pathloss_exponent=float(system["pathloss_exponent"]),
            rician_kappa=condition.rician_kappa,
            residual_phase_half_width_deg=condition.residual_phase_half_width_deg,
            pilot_power=float(system["pilot_power"]), payload_power=float(system["payload_power"]),
            pilot_noise_variance=float(system["pilot_noise_variance"]),
            payload_noise_variance=float(system["payload_noise_variance"]),
            spatial_correlation=float(system["spatial_correlation"]),
            pilot_assignment=str(system["pilot_assignment"]), phase_error_model=str(system["phase_error_model"]),
        )


@dataclass(frozen=True)
class Registries:
    seed_rows: tuple[Mapping[str, Any], ...]
    frame_rows: tuple[Mapping[str, Any], ...]
    frame_seeds: NDArray[np.int64]
    scalar_witness_keys: frozenset[tuple[int, int, int]]


@dataclass(frozen=True)
class ModelBatch:
    los: ComplexArray
    covariances: ComplexArray
    beta: ComplexArray
    noise_covariance: ComplexArray
    delta: FloatArray
    observation: ComplexArray


@dataclass(frozen=True)
class BaselinePrecisionProducts:
    arrays: Mapping[str, np.ndarray]
    scalar_equivalence: Mapping[str, Any]
    uatf_statistics: Mapping[str, Any]
    summary: Mapping[str, Any]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise BaselinePrecisionProtocolError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BaselinePrecisionProtocolError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise BaselinePrecisionProtocolError(f"{label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise BaselinePrecisionProtocolError(f"{label} is below {minimum}")
    return result


def _number(value: object, label: str, minimum: float = 0.0) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise BaselinePrecisionProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise BaselinePrecisionProtocolError(f"{label} must be finite and >= {minimum}")
    return result


def _literal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise BaselinePrecisionProtocolError(f"{label} must equal {expected!r}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(yaml.load(handle, Loader=_UniqueKeyLoader), "configuration root")


def load_baseline_precision_config(path: str | Path) -> BaselinePrecisionConfig:
    """Load the one registered configuration with strict keys and types."""
    raw = _load_yaml(Path(path))
    _keys(raw, {
        "schema_version", "candidate_id", "experiment_id", "stage_id", "evidence_role",
        "executable", "lock_status", "execution_authorization", "execution_status",
        "results_existed_at_lock", "final_partition", "scientific_boundary", "system",
        "conditions", "sampling", "baseline", "receivers", "uatf_validation",
        "resource_policy", "artifacts", "source_identity", "preexecution_gates",
    }, "configuration root")
    _literal(raw["schema_version"], SCHEMA_VERSION, "schema_version")
    _literal(raw["candidate_id"], "R2C01_EXACT_PHASE_CME", "candidate_id")
    _literal(raw["experiment_id"], EXPERIMENT_ID, "experiment_id")
    _literal(raw["stage_id"], "BASELINE_ONLY_ENGINEERING_E0", "stage_id")
    _literal(raw["evidence_role"], "NON_SCIENTIFIC_ENGINEERING_THROUGHPUT_AND_EQUIVALENCE_ONLY", "evidence_role")
    if type(raw["executable"]) is not bool:
        raise BaselinePrecisionProtocolError("executable must be boolean")
    _literal(raw["execution_status"], "NOT_RUN", "execution_status")
    if raw["results_existed_at_lock"] is not None:
        _literal(raw["results_existed_at_lock"], False, "results_existed_at_lock")
    _literal(raw["final_partition"], "PROHIBITED", "final_partition")

    boundary = _mapping(raw["scientific_boundary"], "scientific_boundary")
    expected_boundary = {
        "proposed_method_import_or_dispatch": "PROHIBITED",
        "proposed_vs_baseline_effect_metrics": "PROHIBITED",
        "lambda_selection": "PROHIBITED_IN_THIS_CELL",
        "realization_count_selection": "PROHIBITED_IN_THIS_CELL",
        "manuscript_evidence": "PROHIBITED", "automatic_advance": False,
    }
    _keys(boundary, set(expected_boundary), "scientific_boundary")
    for key, expected in expected_boundary.items():
        _literal(boundary[key], expected, f"scientific_boundary.{key}")

    system = _mapping(raw["system"], "system")
    expected_system = {
        "model_label": "CONTROLLED_SOURCE_INSPIRED_SYNTHETIC_CF_RICIAN_MODEL",
        "access_points": 32, "antennas_per_ap": 4, "users": 16,
        "pilot_symbols": 4, "pilot_group_size": 4, "coherence_symbols": 200,
        "area_side": 1.0, "minimum_distance": 0.03, "pathloss_exponent": 3.7,
        "spatial_correlation": 0.4, "pilot_assignment": "balanced_random",
        "phase_error_model": "matched_uniform", "pilot_power": 1.0,
        "payload_power": 1.0, "pilot_noise_variance": 1.0, "payload_noise_variance": 1.0,
    }
    _keys(system, set(expected_system), "system")
    for key, expected in expected_system.items():
        _literal(system[key], expected, f"system.{key}")

    raw_conditions = raw["conditions"]
    if type(raw_conditions) is not list or len(raw_conditions) != 3:
        raise BaselinePrecisionProtocolError("conditions must contain exactly three rows")
    conditions = []
    for index, (row_value, expected) in enumerate(zip(raw_conditions, CONDITION_SPECS)):
        row = _mapping(row_value, f"conditions[{index}]")
        _keys(row, {"id", "rician_kappa", "residual_phase_half_width_deg"}, f"conditions[{index}]")
        _literal(row["id"], expected[0], f"conditions[{index}].id")
        if _number(row["rician_kappa"], f"conditions[{index}].rician_kappa") != expected[1]:
            raise BaselinePrecisionProtocolError("condition kappa changed")
        if _number(row["residual_phase_half_width_deg"], f"conditions[{index}].width") != expected[2]:
            raise BaselinePrecisionProtocolError("condition phase width changed")
        conditions.append(Condition(*expected))

    sampling = _mapping(raw["sampling"], "sampling")
    _keys(sampling, {
        "base_seed", "topology_seeds", "realizations_per_topology_condition",
        "frame_seed_tags", "replacement_seeds", "adaptive_extension",
        "historical_union_collision_status",
    }, "sampling")
    base_seed = _integer(sampling["base_seed"], "sampling.base_seed", 1)
    _literal(base_seed, 202609520101, "sampling.base_seed")
    if type(sampling["topology_seeds"]) is not list or len(sampling["topology_seeds"]) != 1:
        raise BaselinePrecisionProtocolError("exactly one topology seed is required")
    topology_seed = _integer(sampling["topology_seeds"][0], "sampling.topology_seeds[0]", 1)
    _literal(topology_seed, 954102, "sampling.topology_seeds[0]")
    realizations = _integer(sampling["realizations_per_topology_condition"], "sampling.realizations_per_topology_condition", 1)
    _literal(realizations, 64, "sampling.realizations_per_topology_condition")
    _literal(sampling["frame_seed_tags"], ["experiment_id", "condition_id", "topology_seed", "realization_index", "frame"], "sampling.frame_seed_tags")
    _literal(sampling["replacement_seeds"], "PROHIBITED", "sampling.replacement_seeds")
    _literal(sampling["adaptive_extension"], "PROHIBITED", "sampling.adaptive_extension")

    baseline = _mapping(raw["baseline"], "baseline")
    expected_baseline = {
        "estimator_id": "B1", "implementation": "bounded_phase_widely_linear_lmmse",
        "batched_group_size": 4,
        "scalar_equivalence_selection": "LOWEST_EIGHT_SHA256_RANKED_LOCAL_GROUPS_PER_CONDITION",
        "scalar_equivalence_rtol": 5.0e-12, "scalar_equivalence_atol": 5.0e-12,
    }
    _keys(baseline, set(expected_baseline), "baseline")
    for key, expected in expected_baseline.items():
        _literal(baseline[key], expected, f"baseline.{key}")

    receiver_root = _mapping(raw["receivers"], "receivers")
    _keys(receiver_root, {"common_information_interface", "profiles"}, "receivers")
    _literal(receiver_root["common_information_interface"], True, "receivers.common_information_interface")
    profiles = receiver_root["profiles"]
    if type(profiles) is not list or len(profiles) != 6:
        raise BaselinePrecisionProtocolError("receiver grid must have exactly six profiles")
    receivers = []
    for index, (row_value, expected) in enumerate(zip(profiles, RECEIVER_SPECS)):
        row = _mapping(row_value, f"receivers.profiles[{index}]")
        _keys(row, {"id", "regularization"}, f"receivers.profiles[{index}]")
        _literal(row["id"], expected[0], f"receivers.profiles[{index}].id")
        _literal(row["regularization"], expected[1], f"receivers.profiles[{index}].regularization")
        receivers.append(Receiver(*expected))

    validation = _mapping(raw["uatf_validation"], "uatf_validation")
    expected_validation = {
        "streaming_accumulator": "required", "array_reconstruction": "required",
        "compare_every_condition_receiver_user": True, "rtol": 1.0e-12,
        "atol": 1.0e-12, "split_rule": "even_odd_realization_index",
        "quantile_method": "linear", "effect_statistics": "PROHIBITED",
    }
    _keys(validation, set(expected_validation), "uatf_validation")
    for key, expected in expected_validation.items():
        _literal(validation[key], expected, f"uatf_validation.{key}")

    resource = _mapping(raw["resource_policy"], "resource_policy")
    _keys(resource, {
        "worker_count", "blas_threads_per_worker", "wall_cap_seconds",
        "process_peak_rss_cap_bytes", "output_cap_bytes",
        "resource_sample_interval_seconds", "failed_path_cost_charged",
    }, "resource_policy")
    _literal(resource["worker_count"], 1, "resource_policy.worker_count")
    _literal(resource["blas_threads_per_worker"], 1, "resource_policy.blas_threads_per_worker")
    _literal(resource["failed_path_cost_charged"], True, "resource_policy.failed_path_cost_charged")
    for key, expected in {
        "wall_cap_seconds": 300,
        "process_peak_rss_cap_bytes": 1073741824,
        "output_cap_bytes": 134217728,
        "resource_sample_interval_seconds": 0.25,
    }.items():
        _literal(resource[key], expected, f"resource_policy.{key}")

    artifacts = _mapping(raw["artifacts"], "artifacts")
    _keys(artifacts, {"output_directory", "lifecycle", "required_payloads", "independent_validation_sibling"}, "artifacts")
    _literal(artifacts["output_directory"], OUTPUT_RELATIVE, "artifacts.output_directory")
    _literal(artifacts["lifecycle"], "ATOMIC_STAGING_THEN_IMMUTABLE_COMPLETE", "artifacts.lifecycle")
    _literal(artifacts["required_payloads"], list(EXPECTED_REQUIRED_PAYLOADS), "artifacts.required_payloads")
    _literal(artifacts["independent_validation_sibling"], VALIDATION_RELATIVE, "artifacts.independent_validation_sibling")

    identity = _mapping(raw["source_identity"], "source_identity")
    if set(identity) != set(SOURCE_PATHS):
        raise BaselinePrecisionProtocolError("source_identity keys do not close the source map")
    for key, value in identity.items():
        if type(value) is not str or (value != "PENDING_AFTER_IMPLEMENTATION" and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        )):
            raise BaselinePrecisionProtocolError(f"source_identity.{key} is not SHA-256")

    gates = _mapping(raw["preexecution_gates"], "preexecution_gates")
    expected_gates = {
        "strict_config_and_source_identity", "fresh_seed_namespace_and_union_check",
        "scalar_batched_b1_unit_equivalence", "streaming_array_uatf_unit_equivalence",
        "runner_lifecycle_and_manifest_tests", "independent_validator_tests",
    }
    _keys(gates, expected_gates, "preexecution_gates")
    if any(value not in {"NOT_RUN", "PASS"} for value in gates.values()):
        raise BaselinePrecisionProtocolError("preexecution gate has invalid status")

    return BaselinePrecisionConfig(
        raw=MappingProxyType(raw), source_identity=MappingProxyType(dict(identity)),
        conditions=tuple(conditions), receivers=tuple(receivers), base_seed=base_seed,
        topology_seed=topology_seed, realizations=realizations,
        output_directory=str(artifacts["output_directory"]),
        independent_validation_sibling=str(artifacts["independent_validation_sibling"]),
        wall_cap_seconds=_number(resource["wall_cap_seconds"], "resource_policy.wall_cap_seconds", 1.0),
        rss_cap_bytes=_integer(resource["process_peak_rss_cap_bytes"], "resource_policy.process_peak_rss_cap_bytes", 1),
        output_cap_bytes=_integer(resource["output_cap_bytes"], "resource_policy.output_cap_bytes", 1),
        resource_sample_interval_seconds=_number(resource["resource_sample_interval_seconds"], "resource_policy.resource_sample_interval_seconds", 0.001),
    )


def require_execution_authorization(config: BaselinePrecisionConfig) -> None:
    required = {
        "executable": True,
        "lock_status": "LOCKED_FOR_BASELINE_ONLY_PRECISION_E0",
        "execution_authorization": "AUTHORIZED_BASELINE_ONLY_PRECISION_E0",
        "execution_status": "NOT_RUN", "results_existed_at_lock": False,
    }
    for key, expected in required.items():
        _literal(config.raw[key], expected, key)
    if any(value != "PASS" for value in _mapping(config.raw["preexecution_gates"], "preexecution_gates").values()):
        raise BaselinePrecisionProtocolError("every preexecution gate must be PASS")
    _literal(
        _mapping(config.raw["sampling"], "sampling")["historical_union_collision_status"],
        "PASS",
        "sampling.historical_union_collision_status",
    )
    if any(value == "PENDING_AFTER_IMPLEMENTATION" for value in config.source_identity.values()):
        raise BaselinePrecisionProtocolError("source identity is not frozen")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _rank_group(condition_id: str, realization: int, access_point: int, pilot: int) -> str:
    return hashlib.sha256(_canonical_json([EXPERIMENT_ID, condition_id, realization, access_point, pilot])).hexdigest()


def build_registries(config: BaselinePrecisionConfig) -> Registries:
    seed_rows, frame_rows = [], []
    frame_seeds = np.empty((3, 64), dtype=np.int64)
    witness_keys: set[tuple[int, int, int]] = set()
    frame_index = 0
    for condition_index, condition in enumerate(config.conditions):
        candidates = [
            (_rank_group(condition.id, realization, ap, pilot), realization, ap, pilot)
            for realization in range(64) for ap in range(32) for pilot in range(4)
        ]
        for _, realization, ap, pilot in sorted(candidates)[:8]:
            witness_keys.add((condition_index, realization, ap * 4 + pilot))
        for realization in range(64):
            tags = (EXPERIMENT_ID, condition.id, config.topology_seed, realization, "frame")
            seed = derived_seed(config.base_seed, *tags)
            frame_seeds[condition_index, realization] = seed
            tag_hash = hashlib.sha256(_canonical_json([config.base_seed, *tags])).hexdigest()
            seed_rows.append(MappingProxyType({
                "experiment_id": EXPERIMENT_ID, "seed_role": "frame",
                "condition_id": condition.id, "topology_seed": config.topology_seed,
                "realization_index": realization, "seed": seed, "tags_sha256": tag_hash,
            }))
            frame_rows.append(MappingProxyType({
                "experiment_id": EXPERIMENT_ID, "frame_index": frame_index,
                "condition_index": condition_index, "condition_id": condition.id,
                "topology_seed": config.topology_seed, "realization_index": realization,
                "frame_seed": seed,
            }))
            frame_index += 1
    if np.unique(frame_seeds).size != 192 or len(witness_keys) != 24:
        raise BaselinePrecisionProtocolError("registry uniqueness/completeness failure")
    return Registries(tuple(seed_rows), tuple(frame_rows), frame_seeds, frozenset(witness_keys))


def stack_models(models: Sequence[PilotGroupModel], observations: Sequence[object]) -> ModelBatch:
    if not models or len(models) != len(observations):
        raise ValueError("models and observations must have equal nonzero length")
    if any(model.group_size != 4 for model in models):
        raise ValueError("baseline precision E0 requires P=4 local groups")
    n = models[0].antenna_count
    if any(model.antenna_count != n for model in models):
        raise ValueError("all models must have the same antenna count")
    observation = np.asarray(observations, dtype=np.complex128)
    if observation.shape != (len(models), n) or not np.all(np.isfinite(observation)):
        raise ValueError("observations must be finite with shape (B,N)")
    return ModelBatch(
        np.stack([model.los for model in models]), np.stack([model.covariances for model in models]),
        np.stack([model.beta for model in models]), np.stack([model.noise_covariance for model in models]),
        np.stack([model.delta for model in models]), observation,
    )


def batched_b1(batch: ModelBatch) -> ComplexArray:
    """Vectorized P=4 bounded-phase widely-linear affine estimator."""
    if batch.los.ndim != 3 or batch.los.shape[2] != 4:
        raise ValueError("batched B1 requires shape (B,N,4)")
    rho = np.sinc(batch.delta / np.pi).astype(np.complex128)
    nu = np.sinc(2.0 * batch.delta / np.pi).astype(np.complex128)
    means = batch.los * rho[:, None, :]
    los_covariance = np.einsum("bnp,bmp->bpnm", batch.los, batch.los.conj(), optimize=True)
    los_pseudo = np.einsum("bnp,bmp->bpnm", batch.los, batch.los, optimize=True)
    covariance = batch.covariances + (1.0 - np.abs(rho) ** 2)[:, :, None, None] * los_covariance
    pseudo = (nu - rho**2)[:, :, None, None] * los_pseudo
    mean_y = np.einsum("bnp,bp->bn", means, batch.beta, optimize=True)
    covariance_y = batch.noise_covariance + np.einsum("bp,bpnm->bnm", np.abs(batch.beta) ** 2, covariance, optimize=True)
    pseudo_y = np.einsum("bp,bpnm->bnm", batch.beta**2, pseudo, optimize=True)
    augmented = np.concatenate((
        np.concatenate((covariance_y, pseudo_y), axis=2),
        np.concatenate((pseudo_y.conj(), covariance_y.conj()), axis=2),
    ), axis=1)
    innovation = batch.observation - mean_y
    solved = np.linalg.solve(augmented, np.concatenate((innovation, innovation.conj()), axis=1)[..., None])[..., 0]
    cross = np.concatenate((
        batch.beta.conj()[:, :, None, None] * covariance,
        batch.beta[:, :, None, None] * pseudo,
    ), axis=3)
    result = means + np.einsum("bpnm,bm->bpn", cross, solved, optimize=True).transpose(0, 2, 1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite batched B1 estimate")
    return result.astype(np.complex128)


class UatFAccumulator:
    """Online sums sufficient to reconstruct every registered UatF quantity."""
    def __init__(self, users: int) -> None:
        self.count = 0
        self.coherent_sum = np.zeros(users, dtype=np.complex128)
        self.second_sum = np.zeros((users, users), dtype=np.float64)
        self.norm_sum = np.zeros(users, dtype=np.float64)

    def update(self, channel: object, combiner: object) -> None:
        h, v = np.asarray(channel, dtype=np.complex128), np.asarray(combiner, dtype=np.complex128)
        if h.ndim != 2 or v.shape != h.shape or not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
            raise ValueError("one UatF sample must contain finite matched (M,K) arrays")
        inner = v.conj().T @ h
        self.coherent_sum += np.diag(inner)
        self.second_sum += np.abs(inner) ** 2
        self.norm_sum += np.sum(np.abs(v) ** 2, axis=0)
        self.count += 1

    def finalize(self, powers: object, noise_variance: float, tau_p: int, tau_c: int):
        if self.count < 1:
            raise BaselinePrecisionProtocolError("cannot finalize an empty UatF accumulator")
        p = np.asarray(powers, dtype=np.float64)
        coherent, second, norm = self.coherent_sum / self.count, self.second_sum / self.count, self.norm_sum / self.count
        numerator = p * np.abs(coherent) ** 2
        denominator = second @ p - numerator + float(noise_variance) * norm
        if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
            raise FloatingPointError("streaming UatF denominator is invalid")
        sinr = numerator / denominator
        se = (1.0 - float(tau_p) / float(tau_c)) * np.log2(1.0 + sinr)
        return {
            "count": self.count, "coherent_gain": coherent, "second_effective_power": second,
            "combiner_norm": norm, "numerator": numerator, "denominator": denominator,
            "sinr": sinr, "spectral_efficiency": se,
        }


def receiver_combiner(estimates: ComplexArray, receiver: Receiver) -> ComplexArray:
    return mr_combiner(estimates) if receiver.regularization is None else rzf_combiner(estimates, receiver.regularization)


def assemble_frame_estimates(
    local_estimates: object,
    local_users: Sequence[object],
    local_coordinates: Sequence[tuple[int, int]],
) -> ComplexArray:
    """Place 128 local ``(N,P)`` estimates into one ``(L,N,K)`` tensor."""
    values = np.asarray(local_estimates, dtype=np.complex128)
    if values.shape != (128, 4, 4) or len(local_users) != 128 or len(local_coordinates) != 128:
        raise BaselinePrecisionProtocolError("local estimate assembly dimensions differ")
    result = np.empty((32, 4, 16), dtype=np.complex128)
    coverage = np.zeros((32, 16), dtype=np.int8)
    for local_index, ((access_point, pilot), raw_users) in enumerate(zip(local_coordinates, local_users)):
        users = np.asarray(raw_users, dtype=np.int64)
        if users.shape != (4,) or np.unique(users).size != 4:
            raise BaselinePrecisionProtocolError("local user group is not four unique users")
        if not 0 <= access_point < 32 or not 0 <= pilot < 4 or np.any(users < 0) or np.any(users >= 16):
            raise BaselinePrecisionProtocolError("local estimate coordinate is outside the frame")
        result[access_point][:, users] = values[local_index]
        coverage[access_point, users] += 1
    if not np.all(coverage == 1) or not np.all(np.isfinite(result)):
        raise BaselinePrecisionProtocolError("local estimate assembly does not cover each AP/user once")
    return result


def _complex_json(value: object) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.complex128)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _statistics_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": int(value["count"]), "coherent_gain": _complex_json(value["coherent_gain"]),
        "second_effective_power": np.asarray(value["second_effective_power"]).tolist(),
        "combiner_norm": np.asarray(value["combiner_norm"]).tolist(),
        "numerator": np.asarray(value["numerator"]).tolist(),
        "denominator": np.asarray(value["denominator"]).tolist(),
        "sinr": np.asarray(value["sinr"]).tolist(),
        "spectral_efficiency": np.asarray(value["spectral_efficiency"]).tolist(),
    }


def _direct_statistics(channels, combiners, scenario):
    powers = np.full(channels.shape[2], scenario.payload_power)
    result = uatf_uplink(channels, combiners, powers, scenario.payload_noise_variance, scenario.pilot_symbols, scenario.coherence_symbols)
    inner = np.einsum("smk,smi->ski", combiners.conj(), channels, optimize=True)
    second = np.mean(np.abs(inner) ** 2, axis=0)
    norm = np.mean(np.sum(np.abs(combiners) ** 2, axis=1), axis=0)
    return {
        "count": channels.shape[0], "coherent_gain": result.coherent_gain,
        "second_effective_power": second, "combiner_norm": norm,
        "numerator": powers * np.abs(result.coherent_gain) ** 2,
        "denominator": result.denominator, "sinr": result.sinr,
        "spectral_efficiency": result.spectral_efficiency,
    }


def _compare_statistics(streaming, direct):
    if int(streaming["count"]) != int(direct["count"]):
        raise BaselinePrecisionProtocolError("streaming/direct UatF counts differ")
    differences = {}
    for key in ("coherent_gain", "second_effective_power", "combiner_norm", "numerator", "denominator", "sinr", "spectral_efficiency"):
        first, second = np.asarray(streaming[key]), np.asarray(direct[key])
        differences[key] = float(np.max(np.abs(first - second)))
        if not np.allclose(first, second, rtol=1e-12, atol=1e-12):
            raise BaselinePrecisionProtocolError(f"streaming/direct UatF mismatch for {key}")
    return differences


def execute_baseline_precision_e0(config, registries, *, progress_sink: Callable, resource_guard: Callable):
    """Execute the bounded B1-only pipeline after external authorization."""
    if registries.frame_seeds.shape != (3, 64):
        raise BaselinePrecisionProtocolError("registry shape differs from the fixed cell")
    channels = np.empty((3, 64, 128, 16), dtype=np.complex128)
    estimates = np.empty_like(channels)
    witness_identity, witness_users, witness_models = [], [], []
    witness_observations, witness_batched, witness_scalar = [], [], []
    accumulators = {(c, r, split): UatFAccumulator(16) for c in range(3) for r in range(6) for split in ("full", "even", "odd")}
    completed = 0
    for condition_index, condition in enumerate(config.conditions):
        scenario = config.scenario(condition)
        topology = generate_topology(scenario, seed=config.topology_seed)
        for realization in range(64):
            resource_guard("BEFORE_FRAME")
            frame = generate_channel_frame(topology, scenario, seed=int(registries.frame_seeds[condition_index, realization]))
            models, observations, groups, coordinates = [], [], [], []
            for ap in range(32):
                for pilot in range(4):
                    model, observation, users, _ = pilot_group(topology, frame, scenario, access_point=ap, pilot=pilot)
                    if model.group_size != 4 or users.size != 4:
                        raise BaselinePrecisionProtocolError("balanced P=4 group invariant failed")
                    models.append(model)
                    observations.append(observation)
                    groups.append(users)
                    coordinates.append((ap, pilot))
            batched = batched_b1(stack_models(models, observations))
            for local_index, ((ap, pilot), users) in enumerate(zip(coordinates, groups)):
                if (condition_index, realization, local_index) in registries.scalar_witness_keys:
                    scalar = np.column_stack([widely_linear_mmse(models[local_index], observations[local_index], target) for target in range(4)])
                    if not np.allclose(batched[local_index], scalar, rtol=5e-12, atol=5e-12):
                        raise BaselinePrecisionProtocolError("batched/scalar B1 equivalence failed")
                    witness_identity.append((condition_index, realization, ap, pilot))
                    witness_users.append(np.asarray(users, dtype=np.int64))
                    witness_models.append(models[local_index])
                    witness_observations.append(observations[local_index])
                    witness_batched.append(batched[local_index])
                    witness_scalar.append(scalar)
            estimate_lnk = assemble_frame_estimates(batched, groups, coordinates)
            h, e = frame.channels.reshape(128, 16), estimate_lnk.reshape(128, 16)
            channels[condition_index, realization], estimates[condition_index, realization] = h, e
            split = "even" if realization % 2 == 0 else "odd"
            for receiver_index, receiver in enumerate(config.receivers):
                combiner = receiver_combiner(e, receiver)
                accumulators[(condition_index, receiver_index, "full")].update(h, combiner)
                accumulators[(condition_index, receiver_index, split)].update(h, combiner)
            completed += 1
            progress_sink({
                "frame_index": completed - 1, "condition_index": condition_index,
                "condition_id": condition.id, "realization_index": realization,
                "frame_seed": int(registries.frame_seeds[condition_index, realization]),
                "completed_frames": completed, "expected_frames": config.frame_count, "status": "COMPLETE",
            })
            resource_guard("AFTER_FRAME")
    if completed != 192 or len(witness_identity) != 24:
        raise BaselinePrecisionProtocolError("pipeline completion or witness count is wrong")
    identity_array = np.asarray(witness_identity, dtype=np.int64)
    order = np.lexsort((identity_array[:, 3], identity_array[:, 2], identity_array[:, 1], identity_array[:, 0]))
    identity_array = identity_array[order]
    witness_users = np.asarray(witness_users, dtype=np.int64)[order]
    witness_models = [witness_models[index] for index in order]
    witness_observations = np.asarray(witness_observations)[order]
    witness_batched = np.asarray(witness_batched)[order]
    witness_scalar = np.asarray(witness_scalar)[order]
    scalar_errors = np.max(np.abs(witness_batched - witness_scalar), axis=(1, 2))

    rows, maximum_difference = [], 0.0
    for condition_index, condition in enumerate(config.conditions):
        scenario = config.scenario(condition)
        for receiver_index, receiver in enumerate(config.receivers):
            combiners = np.asarray([receiver_combiner(value, receiver) for value in estimates[condition_index]])
            for split, mask in (("full", np.ones(64, dtype=bool)), ("even", np.arange(64) % 2 == 0), ("odd", np.arange(64) % 2 == 1)):
                streaming = accumulators[(condition_index, receiver_index, split)].finalize(
                    np.full(16, scenario.payload_power), scenario.payload_noise_variance,
                    scenario.pilot_symbols, scenario.coherence_symbols,
                )
                direct = _direct_statistics(channels[condition_index, mask], combiners[mask], scenario)
                differences = _compare_statistics(streaming, direct)
                maximum_difference = max(maximum_difference, max(differences.values()))
                rows.append({
                    "condition_index": condition_index, "condition_id": condition.id,
                    "receiver_index": receiver_index, "receiver_id": receiver.id, "split": split,
                    "streaming": _statistics_json(streaming),
                    "array_reconstruction": _statistics_json(direct),
                    "maximum_absolute_differences": differences, "equivalence": "PASS",
                })
    arrays = {
        "channels": channels, "b1_estimates": estimates, "frame_seeds": registries.frame_seeds,
        "witness_identity": identity_array,
        "witness_users": witness_users,
        "witness_los": np.stack([model.los for model in witness_models]),
        "witness_covariances": np.stack([model.covariances for model in witness_models]),
        "witness_beta": np.stack([model.beta for model in witness_models]),
        "witness_noise_covariance": np.stack([model.noise_covariance for model in witness_models]),
        "witness_delta": np.stack([model.delta for model in witness_models]),
        "witness_observation": witness_observations,
        "witness_batched_estimate": witness_batched,
        "witness_scalar_estimate": witness_scalar,
    }
    scalar_document = {
        "schema_version": "r2c01-v3-baseline-precision-scalar-equivalence/1.0",
        "experiment_id": EXPERIMENT_ID, "estimator_id": "B1",
        "selection_rule": "LOWEST_EIGHT_SHA256_RANKED_LOCAL_GROUPS_PER_CONDITION",
        "witness_count": 24, "targets_per_witness": 4, "rtol": 5e-12, "atol": 5e-12,
        "maximum_absolute_error": float(np.max(scalar_errors)),
        "per_witness_maximum_absolute_error": scalar_errors.tolist(), "equivalence": "PASS",
    }
    uatf_document = {
        "schema_version": "r2c01-v3-baseline-precision-uatf-equivalence/1.0",
        "experiment_id": EXPERIMENT_ID, "estimator_id": "B1",
        "receiver_ids": [receiver.id for receiver in config.receivers],
        "splits": ["full", "even", "odd"], "rtol": 1e-12, "atol": 1e-12,
        "row_count": len(rows), "maximum_absolute_reconstruction_difference": maximum_difference,
        "rows": rows, "effect_statistics": "ABSENT_BY_CONTRACT", "equivalence": "PASS",
    }
    summary = {
        "schema_version": "r2c01-v3-baseline-precision-e0-summary/1.0",
        "experiment_id": EXPERIMENT_ID, "execution_status": "COMPLETE",
        "scientific_boundary": SCIENTIFIC_BOUNDARY, "topology_count": 1,
        "condition_count": 3, "frames_per_condition": 64, "frame_count": 192,
        "local_groups_per_frame": 128, "batched_group_size": 4,
        "estimator_ids": ["B1"], "receiver_ids": [receiver.id for receiver in config.receivers],
        "scalar_witness_count": 24, "scalar_equivalence": "PASS",
        "uatf_streaming_array_equivalence": "PASS",
        "proposed_method_import_or_dispatch": "ABSENT_BY_CONTRACT",
        "scientific_effect_metrics": "ABSENT_BY_CONTRACT", "lambda_selection": "NOT_PERFORMED",
        "realization_count_selection": "NOT_PERFORMED", "final_partition": "SEALED_UNINSPECTED",
        "automatic_advance": "PROHIBITED", "independent_validation": "PENDING",
    }
    return BaselinePrecisionProducts(MappingProxyType(arrays), MappingProxyType(scalar_document), MappingProxyType(uatf_document), MappingProxyType(summary))
