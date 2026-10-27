"""Analyze the R2C01_DIRA_SHAPE_REP_v01 fresh-draw replication.

Reads the replication output and prints the primary ordering, collapse, controls and the
per-topology / jackknife robustness. It does not run the diagnostic and writes nothing.

Run from the repository root:
    $env:PYTHONDONTWRITEBYTECODE="1"
    .venv\\Scripts\\python.exe -B docs\\claude_review\\R2C01_DIRA_SHAPE_REP_v01\\analyze_replication.py
    Remove-Item Env:\\PYTHONDONTWRITEBYTECODE
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "docs/claude_review/R2C01_DIRA_SHAPE_REP_v01/run/output/E10_shape_output.json"
V08 = ROOT / "docs/claude_review/R2C01_DIRA_SHAPE_v08/run/output/E10_shape_output.json"


def ratios(records: list[dict], cell: str) -> dict[int, float]:
    return {
        r["topology"]: r["mse_B2"] / r["mse_B1"]
        for r in records
        if r["antenna_case"] == "N4_PRIMARY" and r["cell"] == cell
    }


def effect(b: dict[int, float], g: dict[int, float]) -> float:
    mb = statistics.fmean(b.values())
    mg = statistics.fmean(g.values())
    return (mb - 1.0) / (mg - 1.0)


def report(data: dict, tag: str) -> None:
    records = data["records"]
    u = ratios(records, f"U_K20_{tag}")
    g = ratios(records, f"G_K20_{tag}")
    b = ratios(records, f"B_K20_{tag}")
    keys = sorted(u)
    n = len(keys)
    print(f"\n{tag}: n={n}")
    print("full_order", sum(1 for k in keys if b[k] > u[k] > g[k]), "/", n)
    print("B>U", sum(1 for k in keys if b[k] > u[k]), "/", n)
    print("U>G", sum(1 for k in keys if u[k] > g[k]), "/", n)
    print("B>G", sum(1 for k in keys if b[k] > g[k]), "/", n)
    for name, a, c in (("B-U", b, u), ("U-G", u, g), ("B-G", b, g)):
        d = [a[k] - c[k] for k in keys]
        print(
            f"{name} min={min(d):.10f} median={statistics.median(d):.10f} "
            f"max={max(d):.10f} positive={sum(1 for x in d if x > 0)}/{n}"
        )
    print(f"effect={effect(b, g):.10f}")


def main() -> int:
    data = json.loads(REP.read_text("utf-8"))
    print("identity", data.get("identity"))
    print("status", data.get("status"), data.get("status_detail"))
    print("integrity_failures", len(data.get("integrity_failures", [])))
    ordering = data.get("ordering", [{}])[0]
    collapse = data.get("collapse", [{}])[0]
    print("ordering_label", ordering.get("label"), "combined", ordering.get("combined_label"))
    print("effect_size", ordering.get("effect_size"), "interval", ordering.get("effect_size_interval"))
    print("collapse_label", collapse.get("label"), "effect", collapse.get("effect_size"),
          "interval", collapse.get("effect_size_interval"))
    print("controls",
          f"witness={len(data.get('witness_checks', []))}",
          f"equality={len(data.get('equality_controls', []))}",
          f"moment={len(data.get('empirical_moment_checks', []))}",
          f"b2={data.get('b2_equality_checks', {}).get('evaluated_pairs', 'n/a')}")

    report(data, "M090")
    report(data, "M099")

    if V08.is_file():
        old = json.loads(V08.read_text("utf-8"))
        print("\nv08_reference_effect", old["ordering"][0]["effect_size"])
        print("v08_reference_collapse", old["collapse"][0]["effect_size"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
