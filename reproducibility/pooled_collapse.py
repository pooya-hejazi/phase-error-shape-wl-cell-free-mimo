"""Pool the three independent draws of the primary and collapse cells.

The three runs share the design (kappa = 20, m1 = 0.90 and 0.99, three laws) and use independent
seeds, so their per-topology ratios can be pooled. The pooled interval is a post-hoc analysis, not a
pre-registered test; the paper labels it that way.

    .venv\\Scripts\\python.exe -B reproducibility\\pooled_collapse.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "v08": ROOT / "reproducibility/v08_run/run/output/E10_shape_output.json",
    "rep": ROOT / "reproducibility/system_run/output/E10_shape_output.json",
    "exp": ROOT / "reproducibility/expanded_run_v02/output/E10_shape_output.json",
}
CELLS = {
    "M090": {"B": "B_K20_M090", "U": "U_K20_M090", "G": "G_K20_M090"},
    "M099": {"B": "B_K20_M099", "U": "U_K20_M099", "G": "G_K20_M099"},
}


def load(path: Path) -> dict[str, dict[int, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[int, float]] = {}
    for record in data["records"]:
        if record["antenna_case"] != "N4_PRIMARY":
            continue
        out.setdefault(record["cell"], {})[int(record["topology"])] = (
            float(record["mse_B2"]) / float(record["mse_B1"])
        )
    return out


def effect(b: list[float], g: list[float]) -> float:
    return (statistics.fmean(b) - 1.0) / (statistics.fmean(g) - 1.0)


def main() -> None:
    data = {name: load(path) for name, path in RUNS.items()}
    summary = {}
    rng = np.random.default_rng(20261030)
    for key, cells in CELLS.items():
        per_run = {}
        pooled_b: list[float] = []
        pooled_u: list[float] = []
        pooled_g: list[float] = []
        for name, table in data.items():
            b = table[cells["B"]]
            u = table[cells["U"]]
            g = table[cells["G"]]
            keys = sorted(set(b) & set(u) & set(g))
            per_run[name] = {
                "n": len(keys),
                "effect": effect([b[k] for k in keys], [g[k] for k in keys]),
                "full_order": sum(1 for k in keys if b[k] > u[k] > g[k]),
            }
            pooled_b += [b[k] for k in keys]
            pooled_u += [u[k] for k in keys]
            pooled_g += [g[k] for k in keys]

        point = effect(pooled_b, pooled_g)
        draws = []
        sizes = [per_run[name]["n"] for name in RUNS]
        for _ in range(10000):
            bb, gg = [], []
            start = 0
            for size in sizes:
                pick = rng.integers(0, size, size)
                bb += [pooled_b[start + i] for i in pick]
                gg += [pooled_g[start + i] for i in pick]
                start += size
            draws.append(effect(bb, gg))
        summary[key] = {
            "per_run": per_run,
            "pooled_n": len(pooled_b),
            "pooled_effect": point,
            "pooled_interval": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "pooled_full_order": int(
                np.sum(
                    (np.array(pooled_b) > np.array(pooled_u))
                    & (np.array(pooled_u) > np.array(pooled_g))
                )
            ),
        }
        print(key, json.dumps(summary[key], indent=1))

    out = ROOT / "reproducibility/pooled_collapse.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
