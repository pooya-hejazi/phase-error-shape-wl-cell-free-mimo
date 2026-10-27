"""Analysis of the expanded system campaign.

Reads the frozen output and reports, for each block, the per-topology MSE ratios,
the ordering counts, the effect size with a bootstrap interval, the calibration
capture, and the control status.

    python -B reproducibility/expanded_run_v02/analyze_expanded.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
OUT = Path(os.environ.get("E10_OUTPUT", HERE.parent / "output" / "E10_shape_output.json"))
CASE = "N4_PRIMARY"
RECEIVER = "RZF_L1"


def ratios(records: list[dict], cell: str) -> dict[int, float]:
    return {
        int(r["topology"]): float(r["mse_B2"]) / float(r["mse_B1"])
        for r in records
        if r["antenna_case"] == CASE and r["cell"] == cell
    }


def effect(b: dict, u: dict, g: dict) -> float:
    keys = sorted(set(b) & set(u) & set(g))
    return (statistics.fmean(b[k] for k in keys) - 1.0) / (
        statistics.fmean(g[k] for k in keys) - 1.0
    )


def bootstrap_effect(b, u, g, draws: int = 10000, seed: int = 20261027) -> list[float]:
    keys = sorted(set(b) & set(u) & set(g))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.choice(keys, size=len(keys), replace=True)
        values.append(
            (statistics.fmean(b[k] for k in pick) - 1.0)
            / (statistics.fmean(g[k] for k in pick) - 1.0)
        )
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


SHAPE_CODE = {"binary": "B", "uniform": "U", "wrapped_gaussian": "G"}


def resolve(records: list[dict], prefix: str, m1: str) -> dict[str, str]:
    present = {r["cell"] for r in records}
    resolved = {}
    for shape, code in SHAPE_CODE.items():
        candidates = [f"{code}_{prefix}_{m1}"]
        if code == "G":
            candidates.append(f"W_{prefix}_{m1}")
        for candidate in candidates:
            if candidate in present:
                resolved[shape] = candidate
                break
        else:
            raise KeyError(f"{shape} cell for {prefix}_{m1} is absent")
    return resolved


def block(records: list[dict], prefix: str, m1: str) -> dict:
    cells = resolve(records, prefix, m1)
    r = {SHAPE_CODE[s]: ratios(records, cid) for s, cid in cells.items()}
    keys = sorted(set(r["B"]) & set(r["U"]) & set(r["G"]))
    if not keys:
        raise KeyError("block not present")
    full = sum(1 for k in keys if r["B"][k] > r["U"][k] > r["G"][k])
    return {
        "cells": cells,
        "n": len(keys),
        "mean_B": statistics.fmean(r["B"][k] for k in keys),
        "mean_U": statistics.fmean(r["U"][k] for k in keys),
        "mean_G": statistics.fmean(r["G"][k] for k in keys),
        "effect": effect(r["B"], r["U"], r["G"]),
        "interval": bootstrap_effect(r["B"], r["U"], r["G"]),
        "full_order": full,
        "B_gt_U": sum(1 for k in keys if r["B"][k] > r["U"][k]),
        "U_gt_G": sum(1 for k in keys if r["U"][k] > r["G"][k]),
    }


def calibration_block(records: list[dict], level: int) -> dict:
    out = {}
    for shape, code in SHAPE_CODE.items():
        oracle_ids = resolve(records, "K20", "M090")
        present = {r["cell"] for r in records}
        cal_candidates = [f"{code}_K20_M090_C{level}"]
        if code == "G":
            cal_candidates.append(f"W_K20_M090_C{level}")
        cal_id = next(c for c in cal_candidates if c in present)
        oracle = ratios(records, oracle_ids[shape])
        cal = ratios(records, cal_id)
        keys = sorted(set(oracle) & set(cal))
        if not keys:
            raise KeyError(f"calibration block {shape}_C{level} not present")
        capture = [(cal[k] - 1.0) / (oracle[k] - 1.0) for k in keys]
        out[code] = {
            "n": len(keys),
            "mean_ratio": statistics.fmean(cal[k] for k in keys),
            "capture_mean": statistics.fmean(capture),
            "capture_p05": float(np.quantile(capture, 0.05)),
            "capture_p50": float(np.quantile(capture, 0.50)),
            "capture_p95": float(np.quantile(capture, 0.95)),
            "capture_negative": sum(1 for c in capture if c < 0.0),
        }
    present = {r["cell"] for r in records}
    g_code = "G" if f"G_K20_M090_C{level}" in present else "W"
    b = ratios(records, f"B_K20_M090_C{level}")
    u = ratios(records, f"U_K20_M090_C{level}")
    g = ratios(records, f"{g_code}_K20_M090_C{level}")
    keys = sorted(set(b) & set(u) & set(g))
    out["ordering"] = {
        "n": len(keys),
        "full_order": sum(1 for k in keys if b[k] > u[k] > g[k]),
        "effect": effect(b, u, g),
        "interval": bootstrap_effect(b, u, g),
    }
    return out


def main() -> None:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    records = data["records"]
    print("identity", data.get("identity"))
    print("status", data.get("status"), data.get("status_detail"))
    print("summaries", len(data.get("summaries", [])))
    timing = data.get("timing_seconds") or {}
    print("elapsed", timing.get("total_seconds") if isinstance(timing, dict) else timing)

    report = {"identity": data.get("identity"), "status": data.get("status")}
    for prefix, m1 in (
        ("K20", "M090"),
        ("K20", "M099"),
        ("K100", "M090"),
        ("K100", "M099"),
        ("K5", "M090"),
        ("K5", "M099"),
        ("K20", "M070"),
    ):
        try:
            row = block(records, prefix, m1)
        except KeyError:
            continue
        report[f"{prefix}_{m1}"] = row
        print(
            f"{prefix}_{m1}: n={row['n']} "
            f"U={row['mean_U']:.4f} G={row['mean_G']:.4f} B={row['mean_B']:.4f} "
            f"effect={row['effect']:.4f} [{row['interval'][0]:.4f},{row['interval'][1]:.4f}] "
            f"order={row['full_order']}/{row['n']}"
        )

    for level in (1024, 4096):
        try:
            row = calibration_block(records, level)
        except KeyError:
            continue
        report[f"calibration_{level}"] = row
        for shape in ("B", "U", "G"):
            s = row[shape]
            print(
                f"CAL{level} {shape}: ratio={s['mean_ratio']:.4f} "
                f"capture={s['capture_mean']:.3f} "
                f"[{s['capture_p05']:.3f},{s['capture_p95']:.3f}] "
                f"neg={s['capture_negative']}/{s['n']}"
            )
        o = row["ordering"]
        print(
            f"CAL{level} order={o['full_order']}/{o['n']} "
            f"effect={o['effect']:.4f} [{o['interval'][0]:.4f},{o['interval'][1]:.4f}]"
        )

    controls = {
        "witness": f"{sum(1 for w in data.get('witness_checks', []) if w['pass'])}"
        f"/{len(data.get('witness_checks', []))}",
        "equality": f"{sum(1 for w in data.get('equality_controls', []) if w.get('pass'))}"
        f"/{len(data.get('equality_controls', []))}",
        "moment": f"{sum(1 for w in data.get('empirical_moment_checks', []) if w.get('pass'))}"
        f"/{len(data.get('empirical_moment_checks', []))}",
        "b2": data.get("b2_equality_checks", {}).get("all_pass"),
    }
    report["controls"] = controls
    print("controls", controls)
    Path(os.environ.get("E10_ANALYSIS", HERE.parent / "output" / "expanded_analysis.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", os.environ.get("E10_ANALYSIS", HERE.parent / "output" / "expanded_analysis.json"))


if __name__ == "__main__":
    main()
