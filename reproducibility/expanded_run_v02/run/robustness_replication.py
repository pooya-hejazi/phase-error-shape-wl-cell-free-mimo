"""Post-hoc power and jackknife arithmetic for the replication.

Reads the R2C01_DIRA_SHAPE_REP_v01 output. Prints the paired-difference SE, the 80%-power MDD,
the effect-size interval, the collapse margin and the leave-one-out / summary sensitivity.
Writes nothing.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/claude_review/R2C01_DIRA_SHAPE_REP_v01/run/output/E10_shape_output.json"
Z_ALPHA = float(norm.ppf(1.0 - 0.05 / 3.0 / 2.0))
Z_95 = float(norm.ppf(0.975))
Z_POWER = float(norm.ppf(0.80))


def ratios(records, cell):
    return {
        r["topology"]: r["mse_B2"] / r["mse_B1"]
        for r in records
        if r["antenna_case"] == "N4_PRIMARY" and r["cell"] == cell
    }


def effect(b, g):
    return (statistics.fmean(b.values()) - 1.0) / (statistics.fmean(g.values()) - 1.0)


def main():
    data = json.loads(OUT.read_text("utf-8"))
    u = ratios(data["records"], "U_K20_M090")
    g = ratios(data["records"], "G_K20_M090")
    b = ratios(data["records"], "B_K20_M090")
    keys = sorted(u)
    print("n", len(keys))
    for name, a, c in (("B-U", b, u), ("U-G", u, g), ("B-G", b, g)):
        d = [a[k] - c[k] for k in keys]
        mean = statistics.fmean(d)
        se = statistics.stdev(d) / math.sqrt(len(d))
        mdd = (Z_ALPHA + Z_POWER) * se
        power = norm.cdf(mean / se - Z_ALPHA) + norm.cdf(-mean / se - Z_ALPHA)
        print(f"{name} mean={mean:.10f} se={se:.10f} t={mean/se:.3f} power={power:.6f} mdd={mdd:.10f}")

    o = data["ordering"][0]
    c = data["collapse"][0]
    lo, hi = o["effect_size_interval"]
    se_eff = (hi - lo) / (2 * Z_ALPHA)
    print("effect", o["effect_size"], "interval", o["effect_size_interval"], "se", se_eff)
    print("power_vs_1.5", norm.cdf((o["effect_size"] - 1.5) / se_eff - Z_ALPHA))
    print("mdd_above_1", (Z_ALPHA + Z_POWER) * se_eff)
    clo, chi = c["effect_size_interval"]
    se_c = (chi - clo) / (2 * Z_95)
    print("collapse", c["effect_size"], "interval", c["effect_size_interval"], "se95", se_c)
    print("collapse_margin_to_1.1", (1.1 - c["effect_size"]) / se_c, "se")

    effects = []
    for drop in keys:
        keep = [k for k in keys if k != drop]
        effects.append(
            (statistics.fmean(b[k] for k in keep) - 1.0)
            / (statistics.fmean(g[k] for k in keep) - 1.0)
        )
    print("jackknife_effect", min(effects), max(effects))
    for name, fn in (
        ("median", statistics.median),
        ("geometric", lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs))),
    ):
        mb = fn([b[k] for k in keys])
        mg = fn([g[k] for k in keys])
        print(name, (mb - 1.0) / (mg - 1.0))


if __name__ == "__main__":
    main()
