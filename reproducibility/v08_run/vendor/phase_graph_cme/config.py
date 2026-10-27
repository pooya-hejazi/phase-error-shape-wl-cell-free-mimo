"""Strict YAML configuration for reproducible phase-graph experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from .controller import ControllerConfig
from .scenario import ScenarioConfig

T = TypeVar("T")


class ConfigurationError(ValueError):
    """Raised when a run configuration violates the frozen schema."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    return value


def _strict_dataclass(cls: type[T], value: object, label: str) -> T:
    mapping = _mapping(value, label)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {label} fields: {unknown}")
    try:
        return cls(**dict(mapping))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid {label}: {error}") from error


@dataclass(frozen=True)
class RunConfig:
    schema_version: str
    experiment_id: str
    stage: str
    evidence_eligible: bool
    lock_status: str
    base_seed: int
    topology_seeds: tuple[int, ...]
    realizations_per_topology: int
    scenario: ScenarioConfig
    controller: ControllerConfig
    estimator_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    rzf_regularization: float
    outage_threshold_se: float
    output_directory: str
    preregistration_path: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ConfigurationError("unsupported schema_version")
        if not self.experiment_id or any(char.isspace() for char in self.experiment_id):
            raise ConfigurationError("experiment_id must be nonempty and contain no whitespace")
        if self.stage not in {"SOFTWARE_SMOKE", "BASELINE_REPRODUCTION", "FINAL_CAMPAIGN"}:
            raise ConfigurationError("unsupported stage")
        if not isinstance(self.evidence_eligible, bool):
            raise ConfigurationError("evidence_eligible must be Boolean")
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise ConfigurationError("base_seed must be an integer")
        if not self.topology_seeds or len(set(self.topology_seeds)) != len(self.topology_seeds):
            raise ConfigurationError("topology_seeds must be nonempty and unique")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.topology_seeds):
            raise ConfigurationError("every topology seed must be an integer")
        if self.realizations_per_topology < 1:
            raise ConfigurationError("realizations_per_topology must be positive")
        allowed_estimators = {
            "phase_graph_controller",
            "bounded_phase_clmmse",
            "widely_linear_lmmse",
            "phase_unaware_lmmse",
            "collision_blind_ls",
            "perfect_phase_oracle",
        }
        if not self.estimator_ids or set(self.estimator_ids) - allowed_estimators:
            raise ConfigurationError("unsupported or empty estimator_ids")
        if "bounded_phase_clmmse" not in self.estimator_ids:
            raise ConfigurationError("bounded_phase_clmmse is required for paired reporting")
        if len(set(self.estimator_ids)) != len(self.estimator_ids):
            raise ConfigurationError("estimator_ids must be unique")
        if len(self.estimator_ids) < 2:
            raise ConfigurationError("paired reporting requires at least one comparator")
        required = {
            "phase_graph_controller",
            "bounded_phase_clmmse",
            "widely_linear_lmmse",
            "phase_unaware_lmmse",
            "collision_blind_ls",
            "perfect_phase_oracle",
        }
        if self.stage == "FINAL_CAMPAIGN" and not required.issubset(self.estimator_ids):
            raise ConfigurationError("final campaign must retain all mandatory estimator families")
        if not self.receiver_ids or set(self.receiver_ids) - {"MR", "RZF"}:
            raise ConfigurationError("receiver_ids must contain only MR and/or RZF")
        if len(set(self.receiver_ids)) != len(self.receiver_ids):
            raise ConfigurationError("receiver_ids must be unique")
        if not math.isfinite(self.rzf_regularization) or self.rzf_regularization <= 0.0:
            raise ConfigurationError("rzf_regularization must be finite and positive")
        if not math.isfinite(self.outage_threshold_se) or self.outage_threshold_se < 0.0:
            raise ConfigurationError("outage_threshold_se must be finite and nonnegative")
        output = Path(self.output_directory)
        if output.is_absolute() or ".." in output.parts:
            raise ConfigurationError("output_directory must be a safe project-relative path")
        if self.stage == "BASELINE_REPRODUCTION" and "phase_graph_controller" in self.estimator_ids:
            raise ConfigurationError(
                "baseline reproduction cannot execute the proposed controller"
            )
        if self.stage == "FINAL_CAMPAIGN":
            if not self.evidence_eligible:
                raise ConfigurationError("final campaign must be marked evidence_eligible")
            if self.lock_status != "LOCKED_BEFORE_PROPOSED_RESULTS":
                raise ConfigurationError("final campaign requires locked preregistration status")
            if not self.preregistration_path:
                raise ConfigurationError("final campaign requires preregistration_path")
            preregistration = Path(self.preregistration_path)
            if preregistration.is_absolute() or ".." in preregistration.parts:
                raise ConfigurationError("preregistration_path must be project-relative")
        elif self.evidence_eligible:
            raise ConfigurationError("smoke/baseline runs cannot be marked final-evidence eligible")


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot read YAML configuration: {error}") from error
    root = _mapping(document, "root")
    required = {
        "schema_version",
        "experiment_id",
        "stage",
        "evidence_eligible",
        "lock_status",
        "base_seed",
        "topology_seeds",
        "realizations_per_topology",
        "scenario",
        "controller",
        "estimator_ids",
        "receiver_ids",
        "rzf_regularization",
        "outage_threshold_se",
        "output_directory",
    }
    optional = {"preregistration_path"}
    missing = sorted(required - set(root))
    unknown = sorted(set(root) - required - optional)
    if missing or unknown:
        raise ConfigurationError(f"missing fields={missing}; unknown fields={unknown}")
    controller = _strict_dataclass(ControllerConfig, root["controller"], "controller")
    if not isinstance(controller.accept_empirical_gibbs, bool):
        raise ConfigurationError("controller.accept_empirical_gibbs must be Boolean")
    scenario = _strict_dataclass(ScenarioConfig, root["scenario"], "scenario")
    realizations = root["realizations_per_topology"]
    if (
        not isinstance(realizations, int)
        or isinstance(realizations, bool)
        or realizations < 1
    ):
        raise ConfigurationError("realizations_per_topology must be a positive integer")
    return RunConfig(
        schema_version=str(root["schema_version"]),
        experiment_id=str(root["experiment_id"]),
        stage=str(root["stage"]),
        evidence_eligible=root["evidence_eligible"],
        lock_status=str(root["lock_status"]),
        base_seed=root["base_seed"],
        topology_seeds=tuple(root["topology_seeds"]),
        realizations_per_topology=realizations,
        scenario=scenario,
        controller=controller,
        estimator_ids=tuple(str(value) for value in root["estimator_ids"]),
        receiver_ids=tuple(str(value) for value in root["receiver_ids"]),
        rzf_regularization=float(root["rzf_regularization"]),
        outage_threshold_se=float(root["outage_threshold_se"]),
        output_directory=str(root["output_directory"]),
        preregistration_path=(
            None
            if root.get("preregistration_path") is None
            else str(root["preregistration_path"])
        ),
    )
