"""Expanded phase-error-shape campaign (R2C01_DIRA_SHAPE_EXP_v01).

This driver compares B1 (widely-linear, uses m1 and m2) with B2 (complex-affine,
uses m1 only) at fixed m1 across uniform, wrapped-Gaussian and binary phase
laws, plus a c = 0 control. It is EXPLORATORY and NON-CONFIRMATORY.

The scientific core is in DERIVATION.md and the design is in SCOPE.md. The
driver is packet-local: it does not modify src/phase_graph_cme. It imports the
library's topology generator and receivers, and carries its own shape-aware
frame generator, model and estimators.

Refusal versus execution: every check before the first numpy import is a
pre-import check. A refused run prints one JSON line and writes no file; it may
create an empty run/output directory. Once the checks pass, the execution marker
is written (the one execution is consumed) and the first scientific import
follows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve()
PKT = HERE.parents[1]
ROOT = HERE.parents[3]
PKT_REL = "reproducibility/expanded_run_v02"
CHECKSUMS_REL = f"{PKT_REL}/run/E10_shape_inputs.sha256"
CONFIG_REL = f"{PKT_REL}/run/E10_shape_config.json"
MANIFEST_REL = f"{PKT_REL}/ARTIFACT_MANIFEST.json"
ENVIRONMENT_LOCK_REL = f"{PKT_REL}/run/ENVIRONMENT_LOCK.json"
OUTPUT_REL = f"{PKT_REL}/output/E10_shape_output.json"
MARKER_REL = f"{PKT_REL}/output/E10_shape_output.json.started"
PYCACHE_REL = f"{PKT_REL}/run/__pycache__"
TEMP_REL = f"{PKT_REL}/run/temp"
GATE_TOKEN = "DIRA_SHAPE_EXP_V02_GATE_PASSED"
IDENTITY = "R2C01_DIRA_SHAPE_EXP_v02"
PINNED = {"numpy": "2.5.2", "scipy": "1.18.1", "PyYAML": "6.0.3"}
SHAPE_NAMES = {"uniform", "wrapped_gaussian", "binary", "zero_c", "fixed", "proper"}
REQUIRED_INPUTS = (
    f"{PKT_REL}/run/E10_shape_run.py",
    f"{PKT_REL}/run/E10_shape_config.json",
    f"{PKT_REL}/run/shape_core.py",
    f"{PKT_REL}/run/shape_model.py",
    f"{PKT_REL}/run/shape_estimators.py",
    f"{PKT_REL}/run/shape_stats.py",
    f"{PKT_REL}/run/shape_rows.py",
    f"{PKT_REL}/run/shape_ordering.py",
    f"{PKT_REL}/run/shape_run_support.py",
)


class GateError(RuntimeError):
    """Raised when a pre-import gate or an execution-time invariant fails."""


def json_safe(value):
    """Replace non-finite floats with None so the output is strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict:
    manifest_path = ROOT / MANIFEST_REL
    if not manifest_path.is_file():
        raise GateError(f"freeze manifest is missing: {MANIFEST_REL}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != IDENTITY:
        raise GateError("freeze manifest identity mismatch")
    expected_list_hash = str(manifest.get("inputs_checksum", {}).get("sha256", ""))
    observed_list_hash = sha256_file(ROOT / CHECKSUMS_REL)
    if expected_list_hash != observed_list_hash:
        raise GateError(
            "checksum list hash does not match the freeze manifest: "
            f"{observed_list_hash} != {expected_list_hash}"
        )
    listing = (ROOT / CHECKSUMS_REL).read_text(encoding="utf-8").splitlines()
    if len(listing) < len(REQUIRED_INPUTS):
        raise GateError("checksum list is shorter than the required input set")
    if int(manifest.get("inputs_checksum", {}).get("entries", -1)) != len(listing):
        raise GateError("checksum list entry count does not match the freeze manifest")
    verified = {}
    for line in listing:
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise GateError(f"missing input file {relative}")
        observed = sha256_file(path)
        if observed != digest:
            raise GateError(f"checksum mismatch for {relative}")
        verified[relative] = observed
    missing = [relative for relative in REQUIRED_INPUTS if relative not in verified]
    if missing:
        raise GateError(f"checksum list does not bind required inputs: {missing}")
    return {
        "checksum_file": CHECKSUMS_REL,
        "checksum_file_sha256": sha256_file(ROOT / CHECKSUMS_REL),
        "files_verified": len(verified),
        "required_inputs": list(REQUIRED_INPUTS),
    }


def load_config() -> dict:
    config = json.loads((ROOT / CONFIG_REL).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("identity") != IDENTITY:
        raise GateError("config identity mismatch")
    if config.get("namespace") != "R2C01_DIRA_SHAPE_EXP_02":
        raise GateError("config namespace mismatch")
    if config.get("base_seed") != 202610280001:
        raise GateError("config base seed mismatch")
    system = config.get("system", {})
    if system.get("phase_error_model") != "packet_local_shape":
        raise GateError("config system.phase_error_model must be packet_local_shape")
    expected_system = {
        "access_points": 32,
        "users": 16,
        "pilot_symbols": 4,
        "coherence_symbols": 200,
        "area_side": 1.0,
        "minimum_distance": 0.03,
        "pathloss_exponent": 3.7,
        "spatial_correlation": 0.4,
        "pilot_assignment": "balanced_random",
        "pilot_power": 1.0,
        "payload_power": 1.0,
        "pilot_noise_variance": 1.0,
        "payload_noise_variance": 1.0,
    }
    for key, value in expected_system.items():
        if system.get(key) != value:
            raise GateError(f"config system.{key} mismatch")

    cells = config.get("cells", {})
    if len(cells) < 18:
        raise GateError("config must define at least 18 cells")
    for cid, cell in cells.items():
        if cell.get("shape") not in SHAPE_NAMES:
            raise GateError(f"cell {cid} has an unknown shape")
        if not 0.0 <= float(cell.get("m1", -1.0)) <= 1.0:
            raise GateError(f"cell {cid} has m1 outside [0, 1]")
        if float(cell.get("kappa", -1.0)) < 0.0:
            raise GateError(f"cell {cid} has a negative kappa")
        if cell.get("role") not in {"SHAPE", "SECONDARY", "STRUCTURAL_EQUALITY_CONTROL", "CALIBRATION"}:
            raise GateError(f"cell {cid} has an unknown role")
        if cell.get("calibration") is not None:
            if cell.get("role") != "CALIBRATION":
                raise GateError(f"cell {cid} calibrates without the CALIBRATION role")
            if float(cell["calibration"].get("sigma_c", 0.0)) <= 0.0:
                raise GateError(f"cell {cid} has a non-positive calibration sigma")

    cases = config.get("antenna_cases", [])
    expected_cells = {"N4_PRIMARY": 18, "N1_SUPPLEMENTARY": 7}
    if [case.get("id") for case in cases] != list(expected_cells):
        raise GateError("config antenna cases or order mismatch")
    for case in cases:
        cid_list = case.get("cells", [])
        if len(cid_list) < expected_cells[case["id"]]:
            raise GateError(f"antenna case {case['id']} has the wrong cell count")
        if len(set(cid_list)) != len(cid_list):
            raise GateError(f"antenna case {case['id']} has duplicate cells")
        for cid in cid_list:
            if cid not in cells:
                raise GateError(f"antenna case {case['id']} references unknown cell {cid}")
        if int(case.get("topologies", 0)) != 64:
            raise GateError(f"antenna case {case['id']} must use 64 topologies")
        if int(case.get("realizations_per_topology", 0)) not in {32, 64}:
            raise GateError(f"antenna case {case['id']} must use 64 realizations")
        if int(case.get("antennas_per_ap", 0)) not in {1, 4}:
            raise GateError(f"antenna case {case['id']} has an unsupported antenna count")

    stats = config.get("statistics", {})
    family = stats.get("primary_family", [])
    if len(family) != 3 or any(cid not in cells for cid in family):
        raise GateError("primary family must reference three defined cells")
    if [cells[cid]["shape"] for cid in family] != [
        "uniform",
        "wrapped_gaussian",
        "binary",
    ]:
        raise GateError("primary family must be ordered uniform, wrapped_gaussian, binary")
    if len(stats.get("primary_pairwise", [])) != 3:
        raise GateError("primary family must declare exactly three pairwise contrasts")
    if len(stats.get("primary_predicted_signs", [])) != 3:
        raise GateError("primary family must declare exactly three predicted signs")
    if stats.get("primary_metric") != "per_topology_mse_ratio_B2_over_B1":
        raise GateError("primary metric mismatch")
    if float(stats.get("alpha", -1.0)) != 0.05:
        raise GateError("alpha mismatch")
    if int(stats.get("bootstrap_resamples", 0)) != 10000:
        raise GateError("bootstrap resamples mismatch")
    collapse = stats.get("collapse_cells", [])
    if len(collapse) != 3 or any(cid not in cells for cid in collapse):
        raise GateError("collapse cells must reference three defined cells")

    controls = config.get("controls", {})
    if float(controls.get("witness_rtol", 0.0)) != 5e-12:
        raise GateError("witness relative tolerance mismatch")
    if float(controls.get("witness_atol", 0.0)) != 5e-12:
        raise GateError("witness absolute tolerance mismatch")
    if list(controls.get("witness_local_groups", [])) != [0, 37, 74, 111]:
        raise GateError("witness local groups mismatch")
    if list(controls.get("witness_realizations", [])) != [0, 1]:
        raise GateError("witness realizations mismatch")
    if float(controls.get("b2_moment_rtol", 0.0)) != 1e-12:
        raise GateError("B2 moment relative tolerance mismatch")
    if float(controls.get("b2_moment_atol", 0.0)) != 1e-12:
        raise GateError("B2 moment absolute tolerance mismatch")
    if int(controls.get("empirical_moment_sigma", 0)) != 5:
        raise GateError("empirical moment sigma mismatch")
    if float(controls.get("empirical_moment_atol", -1.0)) != 1e-12:
        raise GateError("empirical moment atol mismatch")
    if int(controls.get("b2_equality_sigma", 0)) != 4:
        raise GateError("B2 equality sigma mismatch")
    if float(controls.get("equality_estimate_relative_tolerance", -1.0)) != 1e-8:
        raise GateError("structural equality relative tolerance mismatch")
    if float(controls.get("equality_se_absolute_tolerance", -1.0)) != 1e-8:
        raise GateError("structural equality SE tolerance mismatch")
    if int(controls.get("witness_topology_index", -1)) != 0:
        raise GateError("witness topology index mismatch")
    expected_pairs = [
        ["B_K20_M090", "U_K20_M090"],
        ["U_K20_M090", "G_K20_M090"],
        ["B_K20_M090", "G_K20_M090"],
    ]
    if stats.get("primary_pairwise") != expected_pairs:
        raise GateError("primary pairwise contrasts mismatch")
    if [cells[cid]["shape"] for cid in stats.get("collapse_cells", [])] != [
        "uniform",
        "wrapped_gaussian",
        "binary",
    ]:
        raise GateError("collapse cells must be ordered uniform, wrapped_gaussian, binary")
    secondary = stats.get("secondary_descriptive", {})
    if secondary.get("role") != "DESCRIPTIVE_ONLY_NOT_A_FORMAL_ORDERING_TEST":
        raise GateError("secondary descriptive role mismatch")
    required_descriptive = {
        "K5_M099_N4",
        "K5_M090_N4",
        "K20_M070_N4",
        "K20_M099_N1",
        "K20_M090_N1",
    }
    if not required_descriptive.issubset(set(secondary.get("blocks", []))):
        raise GateError("secondary descriptive blocks are incomplete")

    resources = config.get("resources", {})
    if int(resources.get("internal_wall_cap_seconds", 0)) != 7200:
        raise GateError("internal wall cap mismatch")
    if int(resources.get("external_timeout_seconds", 0)) != 7800:
        raise GateError("external timeout mismatch")
    if int(resources.get("blas_threads", 0)) != 1:
        raise GateError("BLAS thread count mismatch")
    if resources.get("packet_local_temp") != TEMP_REL:
        raise GateError("packet-local temp path mismatch")
    if resources.get("execution_marker") != MARKER_REL:
        raise GateError("execution marker path mismatch")
    if resources.get("refuse_existing_pycache") is not True:
        raise GateError("pycache refusal policy mismatch")


def environment_checks() -> dict:
    lock = json.loads((ROOT / ENVIRONMENT_LOCK_REL).read_text(encoding="utf-8"))
    if lock.get("identity") != IDENTITY:
        raise GateError("environment lock identity mismatch")
    if sys.version_info[:2] != (3, 12):
        raise GateError("the frozen environment requires Python 3.12")
    if sys.version.split()[0] != str(lock.get("python_version")):
        raise GateError(
            f"Python version is {sys.version.split()[0]}, "
            f"the lock pins {lock.get('python_version')}"
        )
    if platform.platform() != str(lock.get("platform")):
        raise GateError(
            f"platform is {platform.platform()}, "
            f"the lock pins {lock.get('platform')}"
        )
    expected_prefix = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise GateError("the interpreter is not the project-local .venv")
    if Path(str(lock.get("prefix", ""))).resolve() != expected_prefix:
        raise GateError("the environment lock prefix is not the project-local .venv")
    if sys.flags.dont_write_bytecode != 1:
        raise GateError("the driver must run with -B (no bytecode writes)")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise GateError("PYTHONDONTWRITEBYTECODE must be 1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise GateError(f"{name} must be 1")
    temp_root = (ROOT / TEMP_REL).resolve()
    if not temp_root.is_dir():
        raise GateError(f"packet-local TEMP directory is missing: {TEMP_REL}")
    for name in ("TEMP", "TMP"):
        value = os.environ.get(name)
        if not value:
            raise GateError(f"{name} must be set to the packet-local temp directory")
        if not Path(value).resolve().is_relative_to(temp_root):
            raise GateError(
                f"{name}={value} is outside the packet-local temp directory "
                f"{temp_root}"
            )
    versions = {}
    for distribution, expected in PINNED.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GateError(f"{distribution} is not installed") from exc
        if observed != expected:
            raise GateError(f"{distribution} is {observed}, expected {expected}")
        versions[distribution] = observed
    return {
        "python": sys.version.split()[0],
        "prefix": str(Path(sys.prefix).resolve()),
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "pinned_versions": versions,
        "blas_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "platform": platform.platform(),
        "tmpdir": os.environ.get("TMPDIR"),
        "temp": os.environ.get("TEMP"),
        "tmp": os.environ.get("TMP"),
    }


def refuse(reason: str) -> int:
    print(
        json.dumps(
            {
                "refused": True,
                "reason": reason,
                "execution": "NOT_STARTED",
                "writes": "NONE",
            },
            ensure_ascii=False,
        )
    )
    return 2


def write_uncaught_failure(exc: BaseException) -> None:
    """Write a minimal status record if an uncaught error follows the marker.

    The marker means the one execution has started.  A failure in a path that
    the inner execution ``try`` does not cover (for example the seed registry or
    the analysis stage) must still leave a record; otherwise the operator has
    only a traceback and no status JSON.
    """

    marker_path = ROOT / MARKER_REL
    if not marker_path.exists():
        return
    output_path = ROOT / OUTPUT_REL
    fallback = output_path.with_name(output_path.name + ".exception.json")
    try:
        output_path.parent.mkdir(parents=False, exist_ok=True)
        target = output_path if not output_path.exists() else fallback
        payload = {
            "identity": IDENTITY,
            "status": "FAIL_INTEGRITY",
            "status_detail": f"UNCAUGHT_AFTER_MARKER: {type(exc).__name__}: {exc}",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(target.relative_to(ROOT)),
            "marker": MARKER_REL,
            "note": (
                "post-marker uncaught failure; the one execution is consumed. "
                "This record is JSON-safe and is not a scientific result."
            ),
        }
        with target.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=1,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        print(
            json.dumps(
                {
                    "written": True,
                    "status": "FAIL_INTEGRITY",
                    "output": str(target.relative_to(ROOT)),
                },
                ensure_ascii=False,
            )
        )
    except Exception as write_exc:  # noqa: BLE001 - report the write failure
        print(
            json.dumps(
                {
                    "written": False,
                    "status": "FAIL_INTEGRITY",
                    "error": f"{type(write_exc).__name__}: {write_exc}",
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-confirmation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.gate_confirmation != GATE_TOKEN:
        return refuse("gate token mismatch")
    if Path(args.output).as_posix() != OUTPUT_REL:
        return refuse("output path is not the frozen path")
    output_path = ROOT / OUTPUT_REL
    partial_path = output_path.with_name(output_path.name + ".partial")
    marker_path = ROOT / MARKER_REL
    if output_path.exists() or partial_path.exists():
        return refuse("output or partial output already exists")
    if marker_path.exists():
        return refuse("execution marker already exists; the one execution was consumed")
    pycache = ROOT / PYCACHE_REL
    if pycache.exists():
        if not pycache.is_dir():
            return refuse(
                "packet run/__pycache__ exists and is not a directory"
            )
        if any(pycache.iterdir()):
            return refuse(
                "packet run/__pycache__ is not empty; remove it before the frozen run"
            )
    src_pycache = HERE.parent / "phase_graph_cme" / "__pycache__"
    if src_pycache.exists():
        if not src_pycache.is_dir():
            return refuse(
                "vendored phase_graph_cme/__pycache__ exists and is not a directory"
            )
        if any(src_pycache.iterdir()):
            return refuse(
                "vendored phase_graph_cme/__pycache__ is not empty; remove it before the frozen run"
            )

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    try:
        environment = environment_checks()
        inputs = verify_inputs()
        config = load_config()
    except Exception as exc:  # noqa: BLE001 - refusal must be total
        return refuse(f"{type(exc).__name__}: {exc}")

    # The marker is written only after every pre-import gate has passed and
    # before the first scientific import. A hard kill after this point leaves
    # the marker behind, so a second invocation refuses and the one execution
    # is not silently repeated. A pre-import refusal writes no marker.
    marker_tmp = marker_path.with_name(marker_path.name + ".tmp")
    try:
        output_path.parent.mkdir(parents=False, exist_ok=True)
        with marker_tmp.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "identity": IDENTITY,
                        "started_utc": started_utc,
                        "pid": os.getpid(),
                        "gate_token": GATE_TOKEN,
                        "checksum_file_sha256": inputs["checksum_file_sha256"],
                        "output": OUTPUT_REL,
                        "note": "one execution; this marker is not removed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        os.replace(marker_tmp, marker_path)
    except Exception as exc:  # noqa: BLE001 - refusal must be total
        if marker_tmp.exists():
            marker_tmp.unlink()
        return refuse(f"could not create the execution marker: {type(exc).__name__}: {exc}")

    # ---- execution begins: every import below is after the gates and the marker ----
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE.parent))
    import numpy as np
    import phase_graph_cme
    from phase_graph_cme.baselines import clmmse, widely_linear_mmse
    from phase_graph_cme.metrics import uatf_uplink
    from phase_graph_cme.production_registry import derived_seed
    from phase_graph_cme.scenario import generate_topology
    from shape_core import (
        moment_check,
        moment_unit_checks,
        shape_spec,
    )
    from shape_estimators import (
        batched_b1_shape,
        batched_b2_shape,
        clmmse_shape,
        stack_shape_models,
        to_library_model,
        widely_linear_mmse_shape,
    )
    from shape_model import generate_shape_frame, shape_pilot_group
    from shape_ordering import (
        b2_equality_summary,
        collapse_summary,
        ordering_summary,
    )
    from shape_rows import mse_row
    from shape_run_support import assemble, combiner, scenario_for
    from shape_stats import complete_topologies

    system = config["system"]
    ns = config["namespace"]
    base = config["base_seed"]
    cap = float(config["resources"]["internal_wall_cap_seconds"])
    ctl = config["controls"]
    stats = config["statistics"]
    cases = config["antenna_cases"]
    cells = config["cells"]
    receivers = config["receivers"]
    L = int(system["access_points"])
    K = int(system["users"])
    TP = int(system["pilot_symbols"])
    TC = int(system["coherence_symbols"])
    powers = np.full(K, float(system["payload_power"]), dtype=np.float64)

    registry = {}
    for case in cases:
        for t in range(int(case["topologies"])):
            for kappa in sorted({float(cells[cid]["kappa"]) for cid in case["cells"]}):
                registry[("topology", t, f"{kappa:.6f}")] = derived_seed(
                    base, ns, "topology", t, f"{kappa:.6f}"
                )
    for case in cases:
        for t in range(int(case["topologies"])):
            for r in range(int(case["realizations_per_topology"])):
                registry[("frame", t, r)] = derived_seed(base, ns, "frame", t, r)
    if len(set(registry.values())) != len(registry):
        raise GateError("derived seed collision")
    draw_registry = {
        "namespace": ns,
        "base_seed": base,
        "derivation": "derived_seed(base, namespace, tags) from production_registry",
        "topology_entries": sum(
            1 for key in registry if key[0] == "topology"
        ),
        "frame_entries": sum(1 for key in registry if key[0] == "frame"),
        "total": len(registry),
        "unique": True,
        "common_random_numbers": (
            "the frame seed is shared across cells and shapes at a given "
            "(topology, realization); each shape maps the same uniform field, "
            "so the diffuse and noise draws are common random numbers"
        ),
    }

    status = "COMPLETE"
    status_detail = None
    integrity_failures = []
    records = {}
    witness_log = []
    control_log = []
    b2_moment_log = []
    moment_unit = []
    timing = {"cases": {}, "cells": []}
    try:
        expected_library = (HERE.parent / "phase_graph_cme").resolve()
        if Path(phase_graph_cme.__file__).resolve().parent != expected_library:
            raise GateError("phase_graph_cme is not imported from this project's src/")
        moment_unit = moment_unit_checks()
        if not all(item["pass"] for item in moment_unit):
            raise AssertionError("shape moment unit check failed")

        for case in cases:
            case_id = str(case["id"])
            n_ant = int(case["antennas_per_ap"])
            T = int(case["topologies"])
            F = int(case["realizations_per_topology"])
            case_start = time.monotonic()
            for t in range(T):
                for cid in case["cells"]:
                    if time.monotonic() - started > cap:
                        raise TimeoutError(
                            f"internal cap reached before {case_id}/{t}/{cid}"
                        )
                    cell_start = time.monotonic()
                    cell = cells[cid]
                    spec = shape_spec(cell["shape"], float(cell["m1"]))
                    cal = cell.get("calibration")
                    if cal is None:
                        m2_offset = np.zeros(L, dtype=np.float64)
                    else:
                        sigma_c = float(cal["sigma_c"])
                        m2_offset = np.array(
                            [
                                sigma_c
                                * np.random.default_rng(
                                    derived_seed(
                                        base, ns, "calibration", cid, t, ap
                                    )
                                ).standard_normal()
                                for ap in range(L)
                            ],
                            dtype=np.float64,
                        )
                    scenario = scenario_for(case, cell, system, spec)
                    topology = generate_topology(
                        scenario,
                        seed=registry[
                            ("topology", t, f"{float(cell['kappa']):.6f}")
                        ],
                    )
                    H = np.empty((F, L * n_ant, K), dtype=np.complex128)
                    E = {
                        "B1": np.empty_like(H),
                        "B2": np.empty_like(H),
                    }
                    max_rel_equal = 0.0
                    phase_blocks = []
                    for r in range(F):
                        frame = generate_shape_frame(
                            topology,
                            scenario,
                            spec,
                            seed=registry[("frame", t, r)],
                        )
                        models, observations, groups, coords = [], [], [], []
                        for ap in range(L):
                            for pilot in range(TP):
                                model, y, users, _ = shape_pilot_group(
                                    topology,
                                    frame,
                                    scenario,
                                    spec,
                                    access_point=ap,
                                    pilot=pilot,
                                    m2_est=float(spec.m2) + float(m2_offset[ap]),
                                )
                                models.append(model)
                                observations.append(y)
                                groups.append(users)
                                coords.append((ap, pilot))
                        batch = stack_shape_models(models, observations)
                        local = {
                            "B1": batched_b1_shape(batch),
                            "B2": batched_b2_shape(batch),
                        }

                        if t == int(ctl["witness_topology_index"]) and r in ctl[
                            "witness_realizations"
                        ]:
                            for gi in ctl["witness_local_groups"]:
                                for label, fn in (
                                    ("B1", widely_linear_mmse_shape),
                                    ("B2", clmmse_shape),
                                ):
                                    scalar = np.column_stack(
                                        [
                                            fn(models[gi], observations[gi], k)
                                            for k in range(4)
                                        ]
                                    )
                                    error = float(
                                        np.max(np.abs(local[label][gi] - scalar))
                                    )
                                    scale = max(
                                        float(np.max(np.abs(scalar))), 1.0
                                    )
                                    tolerance = (
                                        float(ctl["witness_rtol"]) * scale
                                        + float(ctl["witness_atol"])
                                    )
                                    passed = error <= tolerance
                                    witness_log.append(
                                        {
                                            "antenna_case": case_id,
                                            "cell": cid,
                                            "topology": t,
                                            "realization": r,
                                            "local_group": int(gi),
                                            "estimator": label,
                                            "kind": "shape_scalar",
                                            "max_abs_error": error,
                                            "tolerance": tolerance,
                                            "pass": bool(passed),
                                        }
                                    )
                                    if not passed:
                                        raise AssertionError(
                                            f"shape scalar witness failed at "
                                            f"{case_id}/{cid}/{t}/{r}/{gi}/{label}"
                                        )
                                if cell["shape"] == "uniform" and cal is None:
                                    delta_equiv = float(spec.parameter["delta_rad"])
                                    lib_model = to_library_model(
                                        models[gi], delta_equiv
                                    )
                                    for label, fn in (
                                        ("B1", widely_linear_mmse),
                                        ("B2", clmmse),
                                    ):
                                        scalar = np.column_stack(
                                            [
                                                fn(lib_model, observations[gi], k)
                                                for k in range(4)
                                            ]
                                        )
                                        error = float(
                                            np.max(
                                                np.abs(local[label][gi] - scalar)
                                            )
                                        )
                                        scale = max(
                                            float(np.max(np.abs(scalar))), 1.0
                                        )
                                        tolerance = (
                                            float(ctl["witness_rtol"]) * scale
                                            + float(ctl["witness_atol"])
                                        )
                                        passed = error <= tolerance
                                        witness_log.append(
                                            {
                                                "antenna_case": case_id,
                                                "cell": cid,
                                                "topology": t,
                                                "realization": r,
                                                "local_group": int(gi),
                                                "estimator": label,
                                                "kind": "library_uniform",
                                                "max_abs_error": error,
                                                "tolerance": tolerance,
                                                "pass": bool(passed),
                                            }
                                        )
                                        if not passed:
                                            raise AssertionError(
                                                f"library uniform witness failed at "
                                                f"{case_id}/{cid}/{t}/{r}/{gi}/{label}"
                                            )

                        if (
                            t == int(ctl["witness_topology_index"])
                            and r == 0
                            and cell["shape"]
                            in {"uniform", "wrapped_gaussian", "binary", "zero_c"}
                        ):
                            m1_value = float(spec.m1)
                            m2_variants = {
                                "uniform": float(shape_spec("uniform", m1_value).m2),
                                "wrapped_gaussian": float(
                                    shape_spec("wrapped_gaussian", m1_value).m2
                                ),
                                "binary": float(
                                    shape_spec("binary", m1_value).m2
                                ),
                            }
                            variant_values = []
                            for m2_value in m2_variants.values():
                                variant_model = replace(
                                    models[0],
                                    m2=np.full(
                                        models[0].group_size, m2_value, dtype=np.float64
                                    ),
                                )
                                variant_batch = stack_shape_models(
                                    [variant_model], [observations[0]]
                                )
                                variant_values.append(
                                    batched_b2_shape(variant_batch)[0]
                                )
                            reference = variant_values[0]
                            error = max(
                                float(np.max(np.abs(value - reference)))
                                for value in variant_values
                            )
                            scale = max(float(np.max(np.abs(reference))), 1.0)
                            tolerance = (
                                float(ctl["b2_moment_rtol"]) * scale
                                + float(ctl["b2_moment_atol"])
                            )
                            passed = error <= tolerance
                            b2_moment_log.append(
                                {
                                    "antenna_case": case_id,
                                    "cell": cid,
                                    "topology": t,
                                    "realization": r,
                                    "m1": m1_value,
                                    "m2_variants": m2_variants,
                                    "max_abs_error": error,
                                    "tolerance": tolerance,
                                    "pass": bool(passed),
                                }
                            )
                            if not passed:
                                raise AssertionError(
                                    f"B2 moment plumbing failed at {case_id}/{cid}"
                                )

                        H[r] = frame.channels.reshape(L * n_ant, K)
                        for label in ("B1", "B2"):
                            E[label][r] = assemble(
                                local[label], groups, coords, n_ant, L, K
                            )
                        if cell["role"] == "STRUCTURAL_EQUALITY_CONTROL":
                            scale = max(
                                float(np.max(np.abs(E["B1"][r]))),
                                float(np.max(np.abs(E["B2"][r]))),
                                1.0,
                            )
                            max_rel_equal = max(
                                max_rel_equal,
                                float(
                                    np.max(
                                        np.abs(E["B1"][r] - E["B2"][r])
                                    )
                                )
                                / scale,
                            )
                        phase_blocks.append(frame.residual_phases.reshape(-1))

                    phases = np.concatenate(phase_blocks)
                    if cell["shape"] == "proper":
                        moment = {
                            "m1_emp": 1.0,
                            "m2_emp": 1.0,
                            "m1_se": 0.0,
                            "m2_se": 0.0,
                            "m1_se_effective": 0.0,
                            "m2_se_effective": 0.0,
                            "moment_z1": 0.0,
                            "moment_z2": 0.0,
                            "moment_atol": float(ctl["empirical_moment_atol"]),
                            "moment_sigma": float(ctl["empirical_moment_sigma"]),
                            "moment_pass": True,
                        }
                        moment_note = "NO_LOS_PHASE_IRRELEVANT"
                    else:
                        moment = moment_check(
                            phases,
                            spec,
                            float(ctl["empirical_moment_sigma"]),
                            float(ctl["empirical_moment_atol"]),
                        )
                        moment_note = None
                    if not moment["moment_pass"]:
                        raise AssertionError(
                            f"empirical moment check failed at {case_id}/{cid}: "
                            f"z1={moment['moment_z1']}, z2={moment['moment_z2']}"
                        )

                    rec = {
                        "mse": {},
                        "se": {},
                        "m1_emp": float(moment["m1_emp"]),
                        "m2_emp": float(moment["m2_emp"]),
                        "m1_se": float(moment["m1_se"]),
                        "m2_se": float(moment["m2_se"]),
                        "m1_se_effective": float(moment["m1_se_effective"]),
                        "m2_se_effective": float(moment["m2_se_effective"]),
                        "moment_z1": float(moment["moment_z1"]),
                        "moment_z2": float(moment["moment_z2"]),
                        "moment_atol": float(moment["moment_atol"]),
                        "moment_pass": bool(moment["moment_pass"]),
                        "moment_note": moment_note,
                        "max_rel_equal": float(max_rel_equal),
                    }
                    for label in ("B1", "B2"):
                        rec["mse"][label] = float(
                            np.sum(np.abs(H - E[label]) ** 2)
                        )
                    rec["channel_energy"] = float(np.sum(np.abs(H) ** 2))
                    for receiver in receivers:
                        per = {}
                        for label in ("B1", "B2"):
                            V = np.stack(
                                [
                                    combiner(E[label][rr], receiver)
                                    for rr in range(F)
                                ]
                            )
                            full = uatf_uplink(
                                H,
                                V,
                                powers,
                                float(system["payload_noise_variance"]),
                                TP,
                                TC,
                            )
                            even = uatf_uplink(
                                H[0::2],
                                V[0::2],
                                powers,
                                float(system["payload_noise_variance"]),
                                TP,
                                TC,
                            )
                            odd = uatf_uplink(
                                H[1::2],
                                V[1::2],
                                powers,
                                float(system["payload_noise_variance"]),
                                TP,
                                TC,
                            )
                            per[label] = {
                                "full": full.spectral_efficiency.tolist(),
                                "even_mean": float(
                                    np.mean(even.spectral_efficiency)
                                ),
                                "odd_mean": float(
                                    np.mean(odd.spectral_efficiency)
                                ),
                            }
                        rec["se"][receiver["id"]] = per
                    if cell["role"] == "STRUCTURAL_EQUALITY_CONTROL":
                        se_gap = max(
                            abs(
                                float(np.mean(rec["se"][rv["id"]]["B1"]["full"]))
                                - float(
                                    np.mean(rec["se"][rv["id"]]["B2"]["full"])
                                )
                            )
                            for rv in receivers
                        )
                        passed = bool(
                            max_rel_equal
                            <= float(ctl["equality_estimate_relative_tolerance"])
                            and se_gap
                            <= float(ctl["equality_se_absolute_tolerance"])
                        )
                        control_log.append(
                            {
                                "antenna_case": case_id,
                                "cell": cid,
                                "topology": t,
                                "max_relative_estimate_gap": float(max_rel_equal),
                                "max_absolute_se_gap": float(se_gap),
                                "pass": passed,
                            }
                        )
                        if not passed:
                            raise AssertionError(
                                f"structural equality failed at {case_id}/{cid}/{t}"
                            )
                    records[(case_id, cid, t)] = rec
                    timing["cells"].append(
                        {
                            "antenna_case": case_id,
                            "cell": cid,
                            "topology": t,
                            "seconds": time.monotonic() - cell_start,
                        }
                    )
            timing["cases"][case_id] = time.monotonic() - case_start
    except TimeoutError as exc:
        status = "INCOMPLETE_RESOURCE"
        status_detail = f"TimeoutError: {exc}"
    except Exception as exc:  # noqa: BLE001 - the failure must be recorded
        status = "FAIL_INTEGRITY"
        status_detail = f"{type(exc).__name__}: {exc}"

    if status == "COMPLETE":
        bad_records = []
        for (case_id, cid, topology), rec in records.items():
            b1 = float(rec["mse"]["B1"])
            b2 = float(rec["mse"]["B2"])
            if (
                not np.isfinite(b1)
                or not np.isfinite(b2)
                or b1 <= 0.0
                or b2 <= 0.0
                or not np.isfinite(b2 / b1)
            ):
                bad_records.append(
                    {
                        "antenna_case": case_id,
                        "cell": cid,
                        "topology": topology,
                        "mse_B1": b1,
                        "mse_B2": b2,
                    }
                )
        if bad_records:
            integrity_failures = bad_records
            status = "FAIL_INTEGRITY"
            status_detail = (
                "nonpositive or nonfinite B1/B2 MSE in "
                f"{len(bad_records)} records; see integrity_failures"
            )

    complete_by_case = {
        case["id"]: complete_topologies(case, records) for case in cases
    }
    b2_equality = b2_equality_summary(
        cases,
        cells,
        records,
        complete_by_case,
        float(ctl["b2_equality_sigma"]),
    )
    if status == "COMPLETE" and b2_equality["all_pass"] is False:
        status = "FAIL_INTEGRITY"
        status_detail = (
            "B2 equality check failed: a shape-dependent difference in a "
            "B2-only statistic exceeded the pre-registered tolerance"
        )
    if status == "COMPLETE" and b2_equality["all_pass"] is None:
        status = "FAIL_INTEGRITY"
        status_detail = (
            "B2 equality check was not evaluated; the packet requires complete "
            "shape groups"
        )

    empirical_moment_checks = []
    for case in cases:
        complete = complete_by_case[case["id"]]
        for cid in case["cells"]:
            cell = cells[cid]
            if not complete:
                empirical_moment_checks.append(
                    {
                        "antenna_case": case["id"],
                        "cell": cid,
                        "shape": cell["shape"],
                        "topologies": 0,
                        "max_abs_z1": None,
                        "max_abs_z2": None,
                        "pass": None,
                        "note": "NO_COMPLETE_TOPOLOGIES",
                    }
                )
                continue
            z1 = [
                abs(float(records[(case["id"], cid, t)]["moment_z1"]))
                for t in complete
            ]
            z2 = [
                abs(float(records[(case["id"], cid, t)]["moment_z2"]))
                for t in complete
            ]
            passed = bool(
                all(
                    bool(records[(case["id"], cid, t)]["moment_pass"])
                    for t in complete
                )
            )
            notes = [
                str(records[(case["id"], cid, t)].get("moment_note"))
                for t in complete
                if records[(case["id"], cid, t)].get("moment_note")
            ]
            empirical_moment_checks.append(
                {
                    "antenna_case": case["id"],
                    "cell": cid,
                    "shape": cell["shape"],
                    "topologies": len(complete),
                    "max_abs_z1": float(max(z1)),
                    "max_abs_z2": float(max(z2)),
                    "pass": passed,
                    "note": notes[0] if notes else None,
                }
            )

    summaries = []
    for case in cases:
        complete = complete_by_case[case["id"]]
        for cid in case["cells"]:
            cell = cells[cid]
            spec = shape_spec(cell["shape"], float(cell["m1"]))
            for receiver in receivers:
                rng = np.random.default_rng(
                    derived_seed(
                        base,
                        ns,
                        "row_bootstrap",
                        case["id"],
                        cid,
                        receiver["id"],
                    )
                )
                summaries.append(
                    mse_row(
                        case,
                        cell,
                        cid,
                        receiver,
                        spec,
                        records,
                        complete,
                        status,
                        stats,
                        rng,
                    )
                )

    ordering = []
    primary_case = next(
        case
        for case in cases
        if case["id"] == stats["primary_family_antenna_case"]
    )
    primary_receiver = next(
        receiver
        for receiver in receivers
        if receiver["id"] == stats["primary_family_receiver"]
    )
    ordering_rng = np.random.default_rng(
        derived_seed(
            base, ns, "ordering", primary_case["id"], primary_receiver["id"]
        )
    )
    ordering.append(
        ordering_summary(
            primary_case,
            primary_receiver,
            records,
            complete_by_case[primary_case["id"]],
            stats,
            status,
            ordering_rng,
        )
    )

    collapse = []
    collapse_case = next(
        case for case in cases if case["id"] == stats["collapse_antenna_case"]
    )
    collapse_receiver = next(
        receiver
        for receiver in receivers
        if receiver["id"] == stats["collapse_receiver"]
    )
    collapse_rng = np.random.default_rng(
        derived_seed(
            base, ns, "collapse", collapse_case["id"], collapse_receiver["id"]
        )
    )
    collapse.append(
        collapse_summary(
            collapse_case,
            collapse_receiver,
            records,
            complete_by_case[collapse_case["id"]],
            stats,
            status,
            collapse_rng,
        )
    )

    finished = time.monotonic()
    document = {
        "identity": IDENTITY,
        "evidence_role": config["evidence_role"],
        "status": status,
        "status_detail": status_detail,
        "integrity_failures": integrity_failures,
        "status_meaning": {
            "COMPLETE": (
                "all pre-registered checks passed and every topology has a "
                "record for every cell of its antenna case"
            ),
            "INCOMPLETE_RESOURCE": (
                "the internal wall cap stopped the run; labels are not "
                "interpreted"
            ),
            "FAIL_INTEGRITY": (
                "a pre-registered integrity check failed; labels are not "
                "interpreted"
            ),
        },
        "interpretation_rules": (
            "Primary: per-topology MSE ratio r_t = MSE_B2(t)/MSE_B1(t). "
            "ORDER_CONFIRMED requires all three predicted signs and all three "
            "Bonferroni intervals excluding zero; effect_size_label is "
            "separate. The K5, N1 and m1=0.70 blocks are DESCRIPTIVE ONLY and "
            "are not a formal secondary ordering test. SE labels are secondary "
            "and descriptive. A c = 0, fixed or proper row with B1 != B2 is an "
            "integrity failure. A FAIL_INTEGRITY or INCOMPLETE_RESOURCE status "
            "makes every ordering/collapse/row label non-interpreted. The B2 "
            "moment check is a plumbing check, not a result."
        ),
        "labels": {
            "ordering": [
                "ORDER_CONFIRMED",
                "ORDER_NOT_CONFIRMED",
                "ORDER_CONFIRMED_EFFECT_MET",
                "ORDER_CONFIRMED_EFFECT_BELOW_THRESHOLD",
                "ORDER_CONFIRMED_EFFECT_UNDEFINED",
                "INSUFFICIENT_TOPOLOGIES",
                "NOT_INTERPRETED",
                "NOT_INTERPRETED_FAIL_INTEGRITY",
                "INCOMPLETE_RESOURCE",
            ],
            "effect_size": [
                "EFFECT_SIZE_MET",
                "EFFECT_SIZE_NOT_MET",
                "EFFECT_SIZE_UNDEFINED",
            ],
            "collapse": [
                "COLLAPSE_CONSISTENT",
                "COLLAPSE_NOT_CONSISTENT",
                "COLLAPSE_EFFECT_UNDEFINED",
                "INSUFFICIENT_TOPOLOGIES",
                "NOT_INTERPRETED",
                "NOT_INTERPRETED_FAIL_INTEGRITY",
                "INCOMPLETE_RESOURCE",
            ],
            "se": [
                "DETECTED_POSITIVE",
                "DETECTED_NEGATIVE",
                "INCONCLUSIVE",
                "CONTROL_EQUALITY_CHECKED",
                "CONTROL_NOT_EVALUATED",
                "INSUFFICIENT_TOPOLOGIES",
                "NOT_INTERPRETED_FAIL_INTEGRITY",
                "INCOMPLETE_RESOURCE",
            ],
        },
        "execution": {
            "marker": MARKER_REL,
            "marker_exists": bool(marker_path.exists()),
            "marker_is_never_removed": True,
            "pycache_refusal": True,
            "one_execution": True,
        },
        "inputs": inputs,
        "config": config,
        "environment": environment,
        "library_path": str(Path(phase_graph_cme.__file__).resolve()),
        "draw_registry": draw_registry,
        "witness_checks": witness_log,
        "equality_controls": control_log,
        "moment_unit_checks": moment_unit,
        "b2_moment_checks": b2_moment_log,
        "empirical_moment_checks": empirical_moment_checks,
        "b2_equality_checks": {
            "sigma": b2_equality["sigma"],
            "evaluated_pairs": b2_equality["evaluated_pairs"],
            "failures": b2_equality["failures"],
            "all_pass": b2_equality["all_pass"],
            "groups": b2_equality["groups"],
        },
        "ordering": ordering,
        "collapse": collapse,
        "summaries": summaries,
        "records": [
            {
                "antenna_case": case_id,
                "cell": cid,
                "topology": t,
                "mse_B1": float(rec["mse"]["B1"]),
                "mse_B2": float(rec["mse"]["B2"]),
                "channel_energy": float(rec["channel_energy"]),
                "m1_emp": float(rec["m1_emp"]),
                "m2_emp": float(rec["m2_emp"]),
                "m1_se": float(rec["m1_se"]),
                "m2_se": float(rec["m2_se"]),
                "m1_se_effective": float(rec["m1_se_effective"]),
                "m2_se_effective": float(rec["m2_se_effective"]),
                "moment_z1": float(rec["moment_z1"]),
                "moment_z2": float(rec["moment_z2"]),
                "moment_atol": float(rec["moment_atol"]),
                "moment_pass": bool(rec["moment_pass"]),
            }
            for (case_id, cid, t), rec in sorted(records.items())
        ],
        "timing_seconds": {
            "total": finished - started,
            "by_case": timing["cases"],
            "by_cell_topology": timing["cells"],
        },
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.parent.mkdir(parents=False, exist_ok=True)
    payload = (
        json.dumps(
            json_safe(document),
            indent=1,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with partial_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(partial_path, output_path)
    except Exception as exc:  # noqa: BLE001 - a failed write must be reported
        if partial_path.exists():
            partial_path.unlink()
        print(
            json.dumps(
                {
                    "written": False,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 1
    digest = sha256_file(output_path)
    print(
        json.dumps(
            {
                "written": True,
                "identity": IDENTITY,
                "status": status,
                "status_detail": status_detail,
                "output": OUTPUT_REL,
                "output_sha256": digest,
                "ordering_label": ordering[0].get("label") if ordering else None,
                "collapse_label": collapse[0].get("label") if collapse else None,
                "summaries": len(summaries),
                "elapsed_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:  # noqa: BLE001 - a post-marker failure must be recorded
        write_uncaught_failure(exc)
        exit_code = 1
    raise SystemExit(exit_code)



