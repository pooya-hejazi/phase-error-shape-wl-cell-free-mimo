"""Write the vendor manifest for the restored v08 dependency set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
VENDOR = PACKET / "vendor" / "phase_graph_cme"
OUT = PACKET / "VENDOR_MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = []
    for path in sorted(VENDOR.rglob("*.py")):
        rel = path.relative_to(PACKET).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "purpose": "exact original dependency set of the v08 packet, restored from the archived tree",
        "source": "_archive/2026-10-26_public_cleanup/src/phase_graph_cme",
        "files": files,
        "count": len(files),
        "note": "the files are copied byte for byte; no newer library is substituted",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("vendored", len(files), "files")


if __name__ == "__main__":
    main()
