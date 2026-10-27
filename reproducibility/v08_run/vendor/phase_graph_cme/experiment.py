"""Paired end-to-end experiment runner with explicit controller provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .baselines import (
    clmmse,
    collision_blind_ls,
    perfect_phase_oracle,
    phase_unaware_lmmse,
    widely_linear_mmse,
)
from .config import RunConfig
from .controller import InferenceController
from .metrics import uatf_uplink
from .receivers import mr_combiner, rzf_combiner
from .scenario import generate_channel_frame, generate_topology, pilot_group


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(base_seed: int, *tags: object) -> int:
    """Stable 63-bit seed independent of Python's randomized hash."""

    payload = json.dumps(
        [int(base_seed), *tags], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    fieldnames = list(rows[0])
    normalized = [name.casefold() for name in fieldnames]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"case-insensitive duplicate CSV fields: {fieldnames}")
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise RuntimeError(
                f"inconsistent CSV schema at row {index}: "
                f"expected={sorted(expected)} observed={sorted(row)}"
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)}")


def _frame_estimates(
    config: RunConfig,
    topology: Any,
    frame: Any,
    *,
    topology_seed: int,
    realization_index: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, float]]:
    scenario = config.scenario
    estimates = {
        estimator: np.empty_like(frame.channels) for estimator in config.estimator_ids
    }
    controller = InferenceController(config.controller)
    controller_rows: list[dict[str, Any]] = []
    runtime = {estimator: 0.0 for estimator in config.estimator_ids}

    for access_point in range(scenario.access_points):
        for pilot in range(scenario.pilot_symbols):
            model, observation, users, oracle_phasors = pilot_group(
                topology,
                frame,
                scenario,
                access_point=access_point,
                pilot=pilot,
            )
            if "phase_graph_controller" in estimates:
                started = time.perf_counter()
                result = controller.estimate_all(
                    model,
                    observation,
                    seed=derived_seed(
                        config.base_seed,
                        "controller",
                        topology_seed,
                        realization_index,
                        access_point,
                        pilot,
                    ),
                )
                runtime["phase_graph_controller"] += time.perf_counter() - started
                estimates["phase_graph_controller"][access_point][:, users] = (
                    result.estimates
                )
                controller_rows.append(
                    {
                        "topology_seed": topology_seed,
                        "realization_index": realization_index,
                        "access_point": access_point,
                        "pilot": pilot,
                        "pilot_group_size": int(users.size),
                        "mode": result.mode,
                        "status": result.status,
                        "fallback_reason": result.diagnostics.get("fallback_reason", ""),
                        "quadrature_rejection_reason": result.diagnostics.get(
                            "quadrature_rejection_reason", ""
                        ),
                        "quadrature_coarse_states": result.diagnostics.get(
                            "quadrature_coarse_states", ""
                        ),
                        "quadrature_fine_states": result.diagnostics.get(
                            "quadrature_fine_states", ""
                        ),
                        "quadrature_coarse_order": result.diagnostics.get(
                            "coarse_order", ""
                        ),
                        "quadrature_fine_order": result.diagnostics.get(
                            "fine_order", ""
                        ),
                        "order_comparison_max_abs_first_moment": result.diagnostics.get(
                            "order_comparison_max_abs_first_moment", ""
                        ),
                        "order_comparison_max_abs_second_moment": result.diagnostics.get(
                            "order_comparison_max_abs_second_moment", ""
                        ),
                        "maximum_split_rhat": result.diagnostics.get(
                            "maximum_split_rhat", ""
                        ),
                        "maximum_chain_mean_disagreement": result.diagnostics.get(
                            "maximum_chain_mean_disagreement", ""
                        ),
                        "dobrushin_eta": result.diagnostics.get("dobrushin_eta", ""),
                        "gibbs_sample_count": result.diagnostics.get(
                            "gibbs_sample_count", 0
                        ),
                        "gibbs_proposal_count": result.diagnostics.get(
                            "gibbs_proposal_count", 0
                        ),
                    }
                )

            for local_target, global_user in enumerate(users):
                baseline_calls = {
                    "bounded_phase_clmmse": lambda: clmmse(
                        model, observation, local_target
                    ),
                    "phase_unaware_lmmse": lambda: phase_unaware_lmmse(
                        model, observation, local_target
                    ),
                    "widely_linear_lmmse": lambda: widely_linear_mmse(
                        model, observation, local_target
                    ),
                    "collision_blind_ls": lambda: collision_blind_ls(
                        model, observation, local_target
                    ),
                    "perfect_phase_oracle": lambda: perfect_phase_oracle(
                        model, observation, local_target, oracle_phasors
                    ),
                }
                for estimator, call in baseline_calls.items():
                    if estimator not in estimates:
                        continue
                    started = time.perf_counter()
                    value = call()
                    runtime[estimator] += time.perf_counter() - started
                    estimates[estimator][access_point, :, global_user] = value
    return estimates, controller_rows, runtime


def _stack_channels(values: list[np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    realizations, access_points, antennas, users = array.shape
    return array.reshape(realizations, access_points * antennas, users)


def _verify_preregistration(
    config: RunConfig, *, project_root: Path, config_path: Path
) -> Path | None:
    if config.stage != "FINAL_CAMPAIGN":
        return None
    if config.preregistration_path is None:
        raise RuntimeError("final campaign has no preregistration path")
    preregistration = (project_root / config.preregistration_path).resolve()
    if project_root not in preregistration.parents or not preregistration.is_file():
        raise RuntimeError("preregistration path is absent or leaves project root")
    try:
        document = yaml.safe_load(preregistration.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read preregistration: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError("preregistration root must be a mapping")
    if document.get("lock_status") != "LOCKED_BEFORE_PROPOSED_RESULTS":
        raise RuntimeError("preregistration document is not locked")
    if document.get("results_existed_at_lock") is not False:
        raise RuntimeError("preregistration must declare no pre-existing final results")
    authorized = document.get("authorized_experiment_ids", [])
    if not isinstance(authorized, list) or config.experiment_id not in authorized:
        raise RuntimeError("experiment_id is not authorized by preregistration")
    hashes = document.get("authorized_config_sha256", {})
    if not isinstance(hashes, dict):
        raise RuntimeError("preregistration has no configuration-hash mapping")
    relative = str(config_path.resolve().relative_to(project_root)).replace("\\", "/")
    expected = hashes.get(relative)
    actual = sha256_file(config_path)
    if not isinstance(expected, str) or expected.lower() != actual.lower():
        raise RuntimeError("configuration hash does not match locked preregistration")
    if preregistration.stat().st_mtime_ns > time.time_ns():
        raise RuntimeError("preregistration timestamp is in the future")
    return preregistration


def run_experiment(
    config: RunConfig,
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Execute one immutable configuration and write raw/aggregate artifacts."""

    root = project_root.resolve()
    config_path = config_path.resolve()
    if root not in config_path.parents or not config_path.is_file():
        raise RuntimeError("configuration path is absent or leaves project root")
    preregistration = _verify_preregistration(
        config, project_root=root, config_path=config_path
    )
    output = (root / config.output_directory).resolve()
    if root not in output.parents:
        raise RuntimeError("resolved output directory leaves project root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_unix = time.time()
    per_user_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    all_mode_counts: Counter[str] = Counter()

    for topology_seed in config.topology_seeds:
        topology = generate_topology(config.scenario, seed=topology_seed)
        channel_frames: list[np.ndarray] = []
        estimator_frames: dict[str, list[np.ndarray]] = {
            estimator: [] for estimator in config.estimator_ids
        }
        for realization in range(config.realizations_per_topology):
            frame = generate_channel_frame(
                topology,
                config.scenario,
                seed=derived_seed(
                    config.base_seed, "channel_frame", topology_seed, realization
                ),
            )
            frame_estimates, frame_controller_rows, frame_runtime = _frame_estimates(
                config,
                topology,
                frame,
                topology_seed=topology_seed,
                realization_index=realization,
            )
            channel_frames.append(frame.channels)
            for estimator, value in frame_estimates.items():
                estimator_frames[estimator].append(value)
                runtime_rows.append(
                    {
                        "topology_seed": topology_seed,
                        "realization_index": realization,
                        "estimator": estimator,
                        "wall_clock_seconds": frame_runtime[estimator],
                    }
                )
            controller_rows.extend(frame_controller_rows)
            all_mode_counts.update(row["mode"] for row in frame_controller_rows)

        channel_samples = _stack_channels(channel_frames)
        payload_powers = np.full(config.scenario.users, config.scenario.payload_power)
        for estimator, frames in estimator_frames.items():
            estimate_samples = _stack_channels(frames)
            error_power = np.mean(np.abs(estimate_samples - channel_samples) ** 2)
            channel_power = np.mean(np.abs(channel_samples) ** 2)
            nmse = float(error_power / channel_power)
            for receiver in config.receiver_ids:
                if receiver == "MR":
                    combiner_samples = np.asarray(
                        [mr_combiner(value) for value in estimate_samples]
                    )
                else:
                    combiner_samples = np.asarray(
                        [
                            rzf_combiner(value, config.rzf_regularization)
                            for value in estimate_samples
                        ]
                    )
                result = uatf_uplink(
                    channel_samples,
                    combiner_samples,
                    payload_powers,
                    config.scenario.payload_noise_variance,
                    config.scenario.pilot_symbols,
                    config.scenario.coherence_symbols,
                )
                spectral_efficiency = result.spectral_efficiency
                for user, (sinr, se) in enumerate(
                    zip(result.sinr, spectral_efficiency, strict=True)
                ):
                    per_user_rows.append(
                        {
                            "topology_seed": topology_seed,
                            "estimator": estimator,
                            "receiver": receiver,
                            "user": user,
                            "sinr": float(sinr),
                            "spectral_efficiency": float(se),
                            "outage": int(se < config.outage_threshold_se),
                        }
                    )
                topology_rows.append(
                    {
                        "topology_seed": topology_seed,
                        "estimator": estimator,
                        "receiver": receiver,
                        "sum_spectral_efficiency": float(np.sum(spectral_efficiency)),
                        "mean_spectral_efficiency": float(np.mean(spectral_efficiency)),
                        "p05_spectral_efficiency": float(
                            np.quantile(spectral_efficiency, 0.05)
                        ),
                        "outage_fraction": float(
                            np.mean(spectral_efficiency < config.outage_threshold_se)
                        ),
                        "channel_estimation_nmse": nmse,
                    }
                )

    _write_csv(output / "per_user_metrics.csv", per_user_rows)
    _write_csv(output / "topology_metrics.csv", topology_rows)
    _write_csv(output / "runtime.csv", runtime_rows)
    if controller_rows:
        _write_csv(output / "controller_diagnostics.csv", controller_rows)

    baseline_key = "bounded_phase_clmmse"
    paired_rows: list[dict[str, Any]] = []
    lookup = {
        (row["topology_seed"], row["estimator"], row["receiver"]): row
        for row in topology_rows
    }
    for row in topology_rows:
        if row["estimator"] == baseline_key:
            continue
        baseline = lookup[(row["topology_seed"], baseline_key, row["receiver"])]
        baseline_sum = float(baseline["sum_spectral_efficiency"])
        paired_rows.append(
            {
                "topology_seed": row["topology_seed"],
                "estimator": row["estimator"],
                "receiver": row["receiver"],
                "sum_se_difference": float(row["sum_spectral_efficiency"])
                - baseline_sum,
                "sum_se_relative_change_pct": (
                    ""
                    if baseline_sum == 0.0
                    else 100.0
                    * (float(row["sum_spectral_efficiency"]) - baseline_sum)
                    / baseline_sum
                ),
                "p05_se_difference": float(row["p05_spectral_efficiency"])
                - float(baseline["p05_spectral_efficiency"]),
                "outage_fraction_difference": float(row["outage_fraction"])
                - float(baseline["outage_fraction"]),
            }
        )
    _write_csv(output / "paired_vs_clmmse.csv", paired_rows)

    numeric_outputs: list[float] = []
    for row in per_user_rows:
        numeric_outputs.extend(
            [float(row["sinr"]), float(row["spectral_efficiency"])]
        )
    for row in topology_rows:
        numeric_outputs.extend(
            [
                float(row["sum_spectral_efficiency"]),
                float(row["mean_spectral_efficiency"]),
                float(row["p05_spectral_efficiency"]),
                float(row["outage_fraction"]),
                float(row["channel_estimation_nmse"]),
            ]
        )
    for row in runtime_rows:
        numeric_outputs.append(float(row["wall_clock_seconds"]))
    nonfinite_output_count = sum(not np.isfinite(value) for value in numeric_outputs)
    if nonfinite_output_count:
        raise FloatingPointError(
            f"run produced {nonfinite_output_count} nonfinite numeric outputs"
        )

    summary = {
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "evidence_eligible": config.evidence_eligible,
        "topology_count": len(config.topology_seeds),
        "realizations_per_topology": config.realizations_per_topology,
        "controller_group_task_count": len(controller_rows),
        "controller_mode_counts": dict(sorted(all_mode_counts.items())),
        "failed_controller_tasks": sum(
            int(row["mode"] == "CLMMSE_FALLBACK") for row in controller_rows
        ),
        "nonfinite_output_count": nonfinite_output_count,
        "scientific_boundary": {
            "SOFTWARE_SMOKE": "SOFTWARE_SMOKE_NOT_SCIENTIFIC_EVIDENCE",
            "BASELINE_REPRODUCTION": "BASELINE_SOURCE_TREND_ONLY_NOT_PROPOSED_METHOD_EVIDENCE",
            "FINAL_CAMPAIGN": "USE_ONLY_WITH_THE_LOCKED_PREREGISTRATION_AND_FULL_MANIFEST",
        }[config.stage],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    package_root = root / "src" / "phase_graph_cme"
    code_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(package_root.glob("*.py"))
    }
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "evidence_eligible": config.evidence_eligible,
        "command_config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "preregistration_path": (
            None if preregistration is None else str(preregistration)
        ),
        "preregistration_sha256": (
            None if preregistration is None else sha256_file(preregistration)
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "started_unix": started_unix,
        "completed_unix": time.time(),
        "code_sha256": code_hashes,
        "output_sha256": {
            path.name: sha256_file(path) for path in output_files
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary
