"""Replay the frozen v08 packet in an isolated reconstruction of its original layout.

Round 3 version. Changes from the first version:

* the cleanup deletes only the paths that this run created, and refuses to start if an unrelated
  ``src/`` tree already exists;
* the fresh output is kept under ``replay_output/`` instead of being discarded;
* the comparator covers every scientific field (records, summaries, ordering, collapse, labels,
  controls) with an explicit tolerance, and skips only timing and path metadata.

    .venv\\Scripts\\python.exe -B reproducibility\\v08_run\\replay.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
PACKET = HERE.parent
WORK = ROOT / "docs" / "claude_review" / "R2C01_DIRA_SHAPE_v08"
SRC = ROOT / "src"
LIB = SRC / "phase_graph_cme"
STORED = PACKET / "run" / "output" / "E10_shape_output.json"
FRESH_COPY = PACKET / "replay_output" / "E10_shape_output.json"
REPORT = PACKET / "REPLAY_RESULT_ROUND3.json"
GATE = "DIRA_SHAPE_V08_GATE_PASSED"
TOL = 1e-12


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree() -> list[Path]:
    if WORK.exists() or SRC.exists():
        raise SystemExit("refusing to start: docs/ or src/ already exists")
    created: list[Path] = []
    (WORK / "run" / "output").mkdir(parents=True)
    (WORK / "run" / "temp").mkdir(parents=True)
    created.extend([SRC, WORK])
    for item in (PACKET / "run").iterdir():
        if item.name in {"output", "temp", "__pycache__"}:
            continue
        target = WORK / "run" / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.copy2(PACKET / "ARTIFACT_MANIFEST.json", WORK / "ARTIFACT_MANIFEST.json")
    shutil.copytree(PACKET / "vendor" / "phase_graph_cme", LIB)
    return created


def compare(a: object, b: object, path: str = "") -> tuple[list[str], float]:
    """Recursive comparison with an explicit tolerance; returns mismatch paths and max float diff."""

    mismatches: list[str] = []
    max_diff = 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                mismatches.append(f"{path}/{key}: missing on one side")
                continue
            sub, diff = compare(a[key], b[key], f"{path}/{key}")
            mismatches.extend(sub)
            max_diff = max(max_diff, diff)
        return mismatches, max_diff
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"], max_diff
        for index, (x, y) in enumerate(zip(a, b)):
            sub, diff = compare(x, y, f"{path}[{index}]")
            mismatches.extend(sub)
            max_diff = max(max_diff, diff)
        return mismatches, max_diff
    if isinstance(a, bool) or isinstance(b, bool):
        return ([] if a == b else [f"{path}: {a} vs {b}"]), max_diff
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = abs(float(a) - float(b))
        rel = diff / max(abs(float(a)), abs(float(b)), 1e-300)
        return ([] if rel <= TOL else [f"{path}: {a} vs {b}"]), max(rel, max_diff)
    return ([] if a == b else [f"{path}: {a!r} vs {b!r}"]), max_diff


def scientific_view(payload: dict) -> dict:
    """Drop timing, path and environment metadata from an output record."""

    view = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "started_utc",
            "finished_utc",
            "timing_seconds",
            "execution",
            "environment",
            "inputs",
            "library_path",
            "output",
            "marker",
            "note",
        }
    }
    return view


def main() -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    created = copy_tree()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TEMP": str((WORK / "run" / "temp").resolve()),
            "TMP": str((WORK / "run" / "temp").resolve()),
        }
    )
    command = [
        sys.executable,
        "-B",
        str(WORK / "run" / "E10_shape_run.py"),
        "--gate-confirmation",
        GATE,
        "--output",
        "docs/claude_review/R2C01_DIRA_SHAPE_v08/run/output/E10_shape_output.json",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    fresh_path = WORK / "run" / "output" / "E10_shape_output.json"
    result = {
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
        "created_paths": [str(p) for p in created],
    }
    if fresh_path.exists():
        FRESH_COPY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fresh_path, FRESH_COPY)
        stored = json.loads(STORED.read_text(encoding="utf-8"))
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
        mismatches, max_diff = compare(scientific_view(stored), scientific_view(fresh))
        result["comparison"] = {
            "tolerance": TOL,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:50],
            "max_relative_difference": max_diff,
            "sections_compared": sorted(scientific_view(stored)),
            "records": len(fresh.get("records", [])),
            "reproduced": len(mismatches) == 0,
        }
        result["fresh_output"] = {
            "path": str(FRESH_COPY.relative_to(ROOT)),
            "sha256": sha256(FRESH_COPY),
        }
    else:
        result["comparison"] = None
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for path in reversed(created):
        shutil.rmtree(path, ignore_errors=True)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
