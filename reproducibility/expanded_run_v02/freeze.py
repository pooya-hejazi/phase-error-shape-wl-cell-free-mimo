"""Freeze the expanded system packet: environment lock, checksums and manifest.

Run once, before the single execution:

    .venv\\Scripts\\python.exe -B reproducibility\\expanded_run\\freeze.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PKT = Path(__file__).resolve().parent
ROOT = PKT.parents[1]
REL = "reproducibility/expanded_run_v02"
IDENTITY = "R2C01_DIRA_SHAPE_EXP_v02"

REQUIRED = [
    "run/E10_shape_run.py",
    "run/E10_shape_config.json",
    "run/shape_core.py",
    "run/shape_model.py",
    "run/shape_estimators.py",
    "run/shape_stats.py",
    "run/shape_rows.py",
    "run/shape_ordering.py",
    "run/shape_run_support.py",
    "run/ENVIRONMENT_LOCK.json",
    "run/requirements.lock",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock_path = PKT / "run/ENVIRONMENT_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["identity"] = IDENTITY
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")

    lines = []
    files = []
    for rel in REQUIRED:
        path = PKT / rel
        digest = sha256(path)
        lines.append(f"{digest}  {REL}/{rel}")
        files.append(
            {
                "path": f"{REL}/{rel}",
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    checksum_path = PKT / "run/E10_shape_inputs.sha256"
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_digest = sha256(checksum_path)
    files.append(
        {
            "path": f"{REL}/run/E10_shape_inputs.sha256",
            "bytes": checksum_path.stat().st_size,
            "sha256": checksum_digest,
        }
    )
    for rel in ("SCOPE.md", "AUTHORIZATIONS.md"):
        path = PKT / rel
        files.append(
            {"path": f"{REL}/{rel}", "bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    manifest = {
        "identity": IDENTITY,
        "evidence_role": "EXPANDED_SYSTEM_CAMPAIGN",
        "status": "FROZEN_WAITING_FOR_EXECUTION",
        "packet": REL,
        "inputs_checksum": {
            "path": f"{REL}/run/E10_shape_inputs.sha256",
            "sha256": checksum_digest,
            "entries": len(lines),
        },
        "files": files,
    }
    (PKT / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (PKT / "run/temp").mkdir(parents=True, exist_ok=True)
    (PKT / "output").mkdir(parents=True, exist_ok=True)
    print("entries", len(lines), "checksum_sha", checksum_digest[:16])


if __name__ == "__main__":
    main()
