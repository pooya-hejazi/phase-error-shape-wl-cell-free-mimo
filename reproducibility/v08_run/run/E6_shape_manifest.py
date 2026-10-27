"""Build the v08 freeze receipt and input checksum list (stdlib only).

Run once, from the repository root or any directory:

    .venv\\Scripts\\python.exe -B docs\\claude_review\\R2C01_DIRA_SHAPE_v08\\run\\E6_shape_manifest.py

The script writes:

1. ``run/E10_shape_inputs.sha256``: the 23 inputs the driver reads (9 run
   modules/config, requirements.lock, ENVIRONMENT_LOCK.json, and the 12
   imported ``src/phase_graph_cme`` files).
2. ``ARTIFACT_MANIFEST.json``: the freeze receipt for every packet file and the
   12 imported library files. The manifest excludes itself and every runtime
   artifact (``run/output/**``, ``run/__pycache__/**``, ``run/temp/**``).

The manifest is written with exclusive creation. Re-running the builder after a
freeze fails rather than silently changing the receipt.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKT = HERE.parents[1]
ROOT = HERE.parents[4]
PKT_REL = "docs/claude_review/R2C01_DIRA_SHAPE_v08"
IDENTITY = "R2C01_DIRA_SHAPE_v08"
INPUTS_REL = f"{PKT_REL}/run/E10_shape_inputs.sha256"
MANIFEST_REL = f"{PKT_REL}/ARTIFACT_MANIFEST.json"

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
ENV_INPUTS = (
    f"{PKT_REL}/run/requirements.lock",
    f"{PKT_REL}/run/ENVIRONMENT_LOCK.json",
)
SRC_FILES = (
    "src/phase_graph_cme/__init__.py",
    "src/phase_graph_cme/baselines.py",
    "src/phase_graph_cme/controller.py",
    "src/phase_graph_cme/diagnostics.py",
    "src/phase_graph_cme/inference.py",
    "src/phase_graph_cme/metrics.py",
    "src/phase_graph_cme/model.py",
    "src/phase_graph_cme/moments.py",
    "src/phase_graph_cme/posterior.py",
    "src/phase_graph_cme/production_registry.py",
    "src/phase_graph_cme/receivers.py",
    "src/phase_graph_cme/scenario.py",
)



def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_record(path: Path, role: str) -> dict:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
        utf8 = True
        crlf = text.count("\r\n")
        lf = text.count("\n")
        bom = data.startswith(b"\xef\xbb\xbf")
    except UnicodeDecodeError:
        utf8 = False
        crlf = 0
        lf = 0
        bom = False
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "role": role,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "crlf_lines": int(crlf),
        "lf_count": int(lf),
        "utf8": bool(utf8),
        "bom": bool(bom),
    }


def packet_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(PKT.glob("*.md")):
        files.append((path, "document"))
    for path in sorted((PKT / "run").iterdir()):
        if not path.is_file():
            continue
        if path.name in {"E10_shape_inputs.sha256"}:
            role = "input_checksums"
        elif path.suffix == ".py":
            if path.name == "E6_shape_manifest.py":
                role = "builder"
            elif path.name == "test_shape_controls.py":
                role = "test"
            else:
                role = "code"
        elif path.name == "E10_shape_config.json":
            role = "config"
        elif path.name == "ENVIRONMENT_LOCK.json":
            role = "environment_receipt"
        elif path.suffix == ".lock":
            role = "environment_lock"
        else:
            role = "packet_file"
        files.append((path, role))
    return files


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "head": run("rev-parse", "HEAD"),
        "origin_main": run("rev-parse", "origin/main"),
        "core_autocrlf": run("config", "--get", "core.autocrlf"),
        "tracked_modified": run("status", "--short", "--untracked-files=no"),
        "untracked": run("status", "--short", "--untracked-files=normal"),
    }


def main() -> int:
    manifest_path = ROOT / MANIFEST_REL
    if manifest_path.exists():
        print(f"refusing: {MANIFEST_REL} already exists", file=sys.stderr)
        return 2

    input_paths = list(REQUIRED_INPUTS) + list(ENV_INPUTS) + list(SRC_FILES)
    input_lines = []
    for relative in sorted(input_paths):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing input: {relative}")
        input_lines.append(f"{sha256_file(path)}  {relative}")
    inputs_path = ROOT / INPUTS_REL
    inputs_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8", newline="\n")
    inputs_hash = sha256_file(inputs_path)

    env_lock = json.loads((ROOT / f"{PKT_REL}/run/ENVIRONMENT_LOCK.json").read_text("utf-8"))
    files = [file_record(path, role) for path, role in packet_files()]
    files.extend(file_record(ROOT / relative, "library_input") for relative in SRC_FILES)
    files.sort(key=lambda item: item["path"])

    manifest = {
        "identity": IDENTITY,
        "evidence_role": "EXPLORATORY_NON_CONFIRMATORY",
        "status": "FROZEN_WAITING_FOR_REVIEW_NOT_EXECUTED",
        "generated_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ),
        "root": str(ROOT),
        "packet": PKT_REL,
        "hash_policy": (
            "SHA-256 of raw file bytes; the driver uses hashlib in binary mode. "
            "Text files are UTF-8; LF is the packet convention; no byte-order "
            "mark. The manifest excludes itself and runtime artifacts; the "
            "reviewer hashes the manifest separately."
        ),
        "git": git_info(),
        "environment": {
            "python_version": env_lock["python_version"],
            "platform": env_lock["platform"],
            "prefix": env_lock["prefix"],
            "prefix_is_project_local": bool(env_lock["prefix_is_project_local"]),
            "packages": {
                name: entry["version"]
                for name, entry in env_lock["packages"].items()
            },
            "wheel_sha256": {
                name: entry["wheel_sha256"]
                for name, entry in env_lock["packages"].items()
            },
            "intended_blas_env": env_lock["intended_blas_env"],
            "smoke_test_pass": bool(env_lock["smoke_test"]["pass"]),
        },
        "inputs_checksum": {
            "path": INPUTS_REL,
            "sha256": inputs_hash,
            "entries": len(input_lines),
            "required_inputs": list(REQUIRED_INPUTS),
            "binds": (
                "the packet run modules, the config, the environment lock and "
                "the 12 imported src/phase_graph_cme files"
            ),
        },
        "files": files,
        "counts": {
            "packet_files": len([item for item in files if not item["path"].startswith("src/")]),
            "library_inputs": len(SRC_FILES),
            "total": len(files),
        },
        "output": {
            "path": f"{PKT_REL}/run/output/E10_shape_output.json",
            "exists": (ROOT / f"{PKT_REL}/run/output/E10_shape_output.json").exists(),
            "marker_exists": (
                ROOT / f"{PKT_REL}/run/output/E10_shape_output.json.started"
            ).exists(),
            "note": "not executed; the one execution is gated by the review and the run instruction",
        },
        "notes": [
            "The packet is frozen with this manifest. The manifest does not hash itself.",
            "The v01-v07 folders are audit trails and are not part of this freeze.",
            "The .venv, the wheel cache and the output directory are not committed and are not in this manifest.",
            "The v02 SEDIAG output is unreviewed (T4A.5) and is not evidence here.",
            "No commit and no push; APPROVE_GITHUB_PUSH has not been supplied.",
        ],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "manifest": MANIFEST_REL,
                "manifest_sha256": sha256_file(manifest_path),
                "inputs_checksum": INPUTS_REL,
                "inputs_checksum_sha256": inputs_hash,
                "entries": len(files),
                "inputs": len(input_lines),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
