"""Generate revised main figures from verified analytical formulas and stored outputs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
REPRO = ROOT / "reproducibility"
sys.path.insert(0, str(ANALYSIS))
from phase_laws import law_rows, lens_bound  # noqa: E402

OUT = ROOT / "generated/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "binary": "#1b4f72",
    "uniform": "#b9770e",
    "wrapped_gaussian": "#1e8449",
    "von_mises": "#7d3c98",
}
LAWS = ("binary", "uniform", "wrapped_gaussian", "von_mises")
LABELS = {"binary": "B", "uniform": "U", "wrapped_gaussian": "G", "von_mises": "vM"}
MARKERS = {"binary": "o", "uniform": "s", "wrapped_gaussian": "^", "von_mises": "D"}

KAPPA = 20.0
L2 = KAPPA / (KAPPA + 1.0)
SG2 = 1.0 / (KAPPA + 1.0)
BETA = 1.0


def risk(m1: float, c: float, s: float) -> tuple[float, float]:
    v = L2 * (1.0 - m1 * m1) + SG2
    p = L2 * c
    cy = BETA * BETA * v + s
    b2 = v * s / cy
    e = BETA**2 * p * p * s / (v * (cy * cy - BETA**4 * p * p))
    b1 = b2 * (1.0 - e)
    return b1, b2


def _box(ax, x, y, w, h, text, fc="#eef3f7", ec="#31556f", fs=6.4):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#18344a", linespacing=1.15)


def _arrow(ax, p, q, color="#31556f", ls="-", lw=0.8):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw,
                                 color=color, linestyle=ls, shrinkA=1, shrinkB=1))


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    ax.text(2, 57, "Propagation and pilot acquisition", fontsize=7.2, weight="bold",
            color="#18344a", ha="left")
    ax.text(51, 57, "CPU: calibration and channel estimation", fontsize=7.2,
            weight="bold", color="#18344a", ha="left")
    for i, y in enumerate((43, 31, 19)):
        _box(ax, 3, y, 8, 5, f"UE {i+1}" if i < 2 else "UE $K$", fc="#f8fafb")
        _box(ax, 17, y, 10, 8, "AP/RF\nchain $l$\n($N$ antennas)", fc="#e8eff4")
        _arrow(ax, (11.2, y + 2.5), (16.8, y + 4))
    _box(ax, 3, 5, 8, 5, "Cal. UE", fc="#fbf4e9", ec="#b7772b")
    _box(ax, 17, 3, 10, 9, "reference link\nknown $\\ell_{\\rm ref}$\nno reuse", fc="#fbf4e9", ec="#b7772b")
    _arrow(ax, (11.2, 7.5), (16.8, 7.5), color="#b7772b")


    _box(ax, 33.5, 3, 12, 46, "", fc="#ffffff", ec="#8ba0ae")
    ax.text(39.5, 46.5, "pilot groups", fontsize=6.3, color="#31556f", ha="center")
    _box(ax, 34.5, 31, 10, 10, "despread $\\mathbf{y}_l$\n$\\mathbf{C}_y,\\mathbf{P}_y$\nper group", fc="#eef3f7", fs=5.6)
    _box(ax, 34.5, 12, 10, 12, "moment\nestimator\n$\\hat m_1,\\hat c$", fc="#fbf4e9", ec="#b7772b", fs=5.6)
    _arrow(ax, (27.2, 7.5), (34.3, 17), color="#b7772b", ls=(0, (3, 2)))

    _box(ax, 50, 35, 20, 8, "B2 complex-affine\nB1 widely-linear", fc="#eef3f7")
    _box(ax, 50, 22, 20, 8, "clip first moment\nthen signed feasible set", fc="#fbf4e9", ec="#b7772b")
    _box(ax, 74, 29, 21, 10, "shared residual moments\nfor links on the same RF chain\n(stationarity assumed)", fc="#eef3f7", fs=5.4)
    _box(ax, 50, 7, 20, 9, "channel MSE\non disjoint evaluation blocks", fc="#eef3f7")
    _box(ax, 74, 7, 21, 9, "uplink RZF/UatF\nseparate receiver study", fc="#f3f6f8", fs=5.7)
    _arrow(ax, (44.8, 36), (49.8, 39))
    _arrow(ax, (44.8, 18), (49.8, 26), color="#b7772b")
    _arrow(ax, (70.2, 26), (73.8, 32), color="#b7772b")
    _arrow(ax, (73.8, 35), (70.2, 38), color="#b7772b")
    ax.plot([50, 47.6, 47.6, 50], [39, 39, 11.5, 11.5], color="#31556f", lw=0.75)
    _arrow(ax, (47.6, 11.5), (49.8, 11.5))
    _arrow(ax, (70.2, 11.5), (73.8, 11.5))
    ax.plot([52, 68], [48.5, 48.5], color="#b7772b", lw=0.7, ls=(0, (2, 2)))
    ax.text(60, 49.2, "calibration window: $M$ reference observations", fontsize=5.6,
            color="#b7772b", ha="center")
    ax.text(60, 2.2, "evaluation blocks are distinct from the calibration window", fontsize=5.6,
            color="#66757f", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_shape_panels() -> None:
    m1 = np.linspace(0.20, 0.999, 1000)
    rows = [law_rows(float(x)) for x in m1]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    ax = axes[0]
    lens = np.array([lens_bound(float(x)) for x in m1])
    ax.fill_between(m1, -lens, lens, color="#eef3f7", zorder=0, label="feasible lens")
    ax.plot(m1, lens, color="#7b8794", lw=0.8)
    ax.plot(m1, -lens, color="#7b8794", lw=0.8)
    for law in COLORS:
        ax.plot(m1, [r[law]["c"] for r in rows], color=COLORS[law],
                ls="-" if law != "von_mises" else ":", lw=1.0, label=LABELS[law])
    ax.axhline(0, color="#333333", lw=0.5)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(r"first circular moment $m_1$")
    ax.set_ylabel(r"signed pseudo-covariance $c$")
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    ax.set_title("(a) feasible set and selected loci", fontsize=8)

    ax = axes[1]
    idx = np.where(m1 >= 0.90)[0]
    for law in COLORS:
        ax.plot(m1[idx], [abs(rows[i][law]["c"]) for i in idx],
                color=COLORS[law], marker=MARKERS[law], markevery=40, ms=3,
                lw=1.0, label=LABELS[law])
    ax.set_xlim(0.90, 1.0)
    ax.set_xlabel(r"$m_1$ (high-tracking regime)")
    ax.set_ylabel(r"$|c|$")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(b) high-tracking close-up", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig2_shape_panels.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_shape_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_scalar_benefit() -> None:
    m1 = 0.90
    rows = law_rows(m1)
    snr = np.logspace(-5, 1, 160)
    v = L2 * (1.0 - m1 * m1) + SG2
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    ax = axes[0]
    for law in COLORS:
        c = rows[law]["c"]
        b1, b2 = zip(*(risk(m1, c, BETA**2 * v / x) for x in snr))
        ax.loglog(snr, b1, color=COLORS[law], label=LABELS[law])
        ax.loglog(snr, b2, color=COLORS[law], ls=":", lw=0.9)
    ax.set_xlabel(r"pilot observation SNR $\beta^2V/s$")
    ax.set_ylabel("channel MSE, normalized")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)

    ax = axes[1]
    grid = np.linspace(0.50, 0.995, 250)
    ref_s = 0.01
    for law in COLORS:
        frac = []
        for x in grid:
            c = law_rows(float(x))[law]["c"]
            b1, b2 = risk(float(x), c, ref_s)
            frac.append(1.0 - b1 / b2)
        ax.plot(grid, frac, color=COLORS[law], label=LABELS[law], lw=1.1)
    ax.set_xlabel(r"$m_1$")
    ax.set_ylabel(r"fractional MSE reduction $1-\mathrm{MSE}_{WL}/\mathrm{MSE}_{CF}$")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(0, 1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig3_scalar_benefit.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_scalar_benefit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_mismatch() -> None:
    mats = {
        0.90: np.array([[1.0000, 0.9202, 0.8836, 0.8911],
                        [0.8432, 1.0000, 0.2352, 0.9956],
                        [0.9356, 0.7849, 1.0000, 0.7508],
                        [0.7536, 0.9949, -0.0201, 1.0000]]),
        0.70: np.array([[1.0000, 0.6881, -8.0101, 0.5009],
                        [-0.6885, 1.0000, -68.3459, 0.8838],
                        [0.4417, 0.2134, 1.0000, 0.1457],
                        [-5.4361, 0.7248, -180.2522, 1.0000]]),
    }
    labels = ["U", "G", "B", "vM"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95))
    for idx, (ax, (m1, mat)) in enumerate(zip(axes, mats.items())):
        im = ax.imshow(mat, cmap="RdBu_r", norm=SymLogNorm(linthresh=0.1, vmin=-200, vmax=1))
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("assumed law")
        if idx == 0:
            ax.set_ylabel("true law")
        ax.set_title(rf"$m_1={m1:.2f}$", fontsize=8)
        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val) > 0.8 else "black")
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02,
                        label=r"retained fraction $\rho$ (symlog)")
    fig.savefig(OUT / "fig4_mismatch.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_mismatch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_calibration() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    ax = axes[0]
    for law in LAWS:
        curve = data["laws"][law]["curve"]
        m = np.array([r["frames"] for r in curve])
        y = np.array([r["capture_mean"] for r in curve])
        lo = np.array([r["capture_ci95"][0] for r in curve])
        hi = np.array([r["capture_ci95"][1] for r in curve])
        ax.errorbar(m, y, yerr=[y - lo, hi - y], color=COLORS[law], marker=MARKERS[law],
                    ms=3, lw=1.0, capsize=2, label=LABELS[law])
    ax.axhline(0.99, color="#888888", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("calibration frames $M$")
    ax.set_ylabel("mean captured oracle improvement")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax.set_title("(a) mean and 95% interval", fontsize=8)

    ax = axes[1]
    for law in LAWS:
        curve = data["laws"][law]["curve"]
        m = np.array([r["frames"] for r in curve])
        p10 = np.array([r["capture_p10"] for r in curve])
        p90 = np.array([r["capture_p90"] for r in curve])
        mean = np.array([r["capture_mean"] for r in curve])
        ax.fill_between(m, p10, p90, color=COLORS[law], alpha=0.12)
        ax.plot(m, mean, color=COLORS[law], marker=MARKERS[law], ms=3, lw=0.9,
                label=LABELS[law])
    ax.axhline(0.0, color="#888888", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("calibration frames $M$")
    ax.set_ylabel("individual-window capture\n(10th--90th percentile)")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.set_title("(b) individual-window range", fontsize=8)

    ax = axes[2]
    xs, ys, labels = [], [], []
    for law in LAWS:
        s = data["summary"][law]
        xs.append(s["frames_for_99pct_measured"])
        ys.append(s["frames_for_99pct_model"])
        labels.append(LABELS[law])
    ax.scatter(xs, ys, c=[COLORS[l] for l in LAWS], s=22)
    lim = [1500, 5000]
    ax.plot(lim, lim, color="#555555", lw=0.8, ls="--")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"measured $M^\star$")
    ax.set_ylabel(r"model $M^\star$")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(c) crossing prediction", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig5_calibration_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_calibration_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_calibrated_network() -> None:
    data = load(REPRO / "calibrated_network/output/calibrated_network_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
    ax = axes[0]
    width = 0.19
    for i, est in enumerate(("oracle_B1", "cal_B1", "oracle_B2", "cal_B2")):
        vals = []
        for law in LAWS:
            rs = [r for r in data["records"] if r["law"] == law]
            vals.append(np.mean([r["estimators"][est]["mse"] / r["estimators"][est]["energy"] for r in rs]))
        ax.bar(np.arange(4) + (i - 1.5) * width, vals, width=width,
               color=["#1b4f72", "#5b8db8", "#b7772b", "#d5a15b"][i],
               label=est.replace("_", " "))
    ax.set_xticks(range(4), ["B", "U", "G", "vM"])
    ax.set_yscale("log")
    ax.set_ylabel("channel NMSE")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title("(a) channel estimation", fontsize=8)

    ax = axes[1]
    for i, est in enumerate(("oracle_B1", "cal_B1", "oracle_B2", "cal_B2")):
        vals = []
        for law in LAWS:
            rs = [r for r in data["records"] if r["law"] == law]
            vals.append(np.mean([r["estimators"][est]["se_mean"] for r in rs]))
        ax.bar(np.arange(4) + (i - 1.5) * width, vals, width=width,
               color=["#1b4f72", "#5b8db8", "#b7772b", "#d5a15b"][i])
    ax.set_xticks(range(4), ["B", "U", "G", "vM"])
    ax.set_ylabel("UatF SE (bit/s/Hz/user)")
    ax.set_title("(b) spectral efficiency", fontsize=8)

    ax = axes[2]
    law = "uniform"
    for est, color, label in (("cal_B1", COLORS["uniform"], "calibrated WL"),
                              ("cal_B2", "#7b8794", "complex affine")):
        vals = []
        for r in data["records"]:
            if r["law"] == law:
                vals.extend(r["estimators"][est]["se_per_user"])
        ax.plot(np.sort(vals), np.linspace(0, 1, len(vals)), color=color, label=label)
    ax.set_xlabel("per-user SE")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(c) uniform law", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig6_calibrated_network.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig6_calibrated_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_operational() -> None:
    data = load(REPRO / "calibrated_network/output/operational_transfer_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
    ax = axes[0]
    for law in LAWS:
        rows = [r for r in data["operational"] if r["law"] == law and r["calibration_frames"] == 1024]
        ax.plot([r["horizon_blocks"] for r in rows], [r["cal_B1_net_se_mean"] for r in rows],
                color=COLORS[law], marker=MARKERS[law], ms=3, label=LABELS[law])
    ax.set_xscale("log")
    ax.set_xlabel("evaluation horizon (blocks)")
    ax.set_ylabel("net SE after overhead")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.legend(fontsize=5.5, frameon=False)
    ax.set_title("(a) $M=1024$", fontsize=8)

    ax = axes[1]
    ind = [r for r in data["transfer"] if not r["shared_lo"]]
    sha = [r for r in data["transfer"] if r["shared_lo"]]
    for i, law in enumerate(LAWS):
        ri = next(r for r in ind if r["law"] == law)
        rs = next(r for r in sha if r["law"] == law)
        ax.errorbar(i, ri["se_gain_B1_minus_B2_mean"],
                    yerr=[[ri["se_gain_B1_minus_B2_mean"] - ri["se_gain_interval"][0]],
                          [ri["se_gain_interval"][1] - ri["se_gain_B1_minus_B2_mean"]]],
                    fmt="o", color=COLORS[law], ms=4, capsize=2)
        ax.errorbar(i + 0.18, rs["se_gain_B1_minus_B2_mean"],
                    yerr=[[rs["se_gain_B1_minus_B2_mean"] - rs["se_gain_interval"][0]],
                          [rs["se_gain_interval"][1] - rs["se_gain_B1_minus_B2_mean"]]],
                    fmt="s", color=COLORS[law], ms=4, capsize=2, alpha=0.6)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(4), ["B", "U", "G", "vM"])
    ax.set_ylabel("SE gain B1-B2")
    ax.set_title("(b) independent / common per AP", fontsize=7)

    ax = axes[2]
    for i, row in enumerate(data["mismatch"]):
        val, ci = row["se_gain_B1_minus_oracle_B2_mean"], row["se_gain_interval"]
        ax.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt="o",
                    color="#b7772b", ms=4, capsize=2)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(data["mismatch"])),
                  [f"{r['reference_law'][0]}->{r['data_law'][0]}" for r in data["mismatch"]],
                  fontsize=6)
    ax.set_ylabel("SE gain vs oracle B2")
    ax.set_title("(c) reference/data mismatch", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig7_operational.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig7_operational.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_supplementary_convergence() -> None:
    data = load(REPRO / "calibrated_network/output/calibrated_network_corrected_20261029.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for ax, metric in zip(axes, ("se_mean", "mse")):
        for law in LAWS:
            rows = [r for r in data["records"] if r["law"] == law]
            curves = []
            for est in ("cal_B1",):
                per_r = [x["estimators"][est]["per_realization"] for x in rows]
                means = []
                ns = sorted(int(k) for k in rows[0]["estimators"][est]["uatf_prefix_per_user"])
                for n in ns:
                    vals = []
                    for row, rr in zip(rows, per_r):
                        if metric == "se_mean":
                            vals.append(np.mean(row["estimators"][est]["uatf_prefix_per_user"][str(n)]))
                        else:
                            vals.append(np.mean([v["mse"] / v["energy"] for v in rr[:n]]))
                    means.append(np.mean(vals))
                ax.plot(ns, means, color=COLORS[law],
                        marker=MARKERS[law], ms=3, label=LABELS[law])
        ax.set_xlabel("inner realizations per topology")
        ax.set_ylabel("UatF from cumulative moments" if metric == "se_mean" else "cumulative NMSE")
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figS1_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS1_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_supplementary_sensitivity() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    std = load(ROOT / "reproducibility" / "standard_channel" / "results.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ax = axes[0]
    expanded = load(ROOT / "reproducibility" / "expanded_run_v02" / "output" / "expanded_analysis.json")
    ks = [5, 20, 100]
    eff = [expanded[f"K{k}_M090"]["effect"] for k in ks]
    ci = [expanded[f"K{k}_M090"]["interval"] for k in ks]
    ax.errorbar(range(3), eff, yerr=[[e - c[0] for e, c in zip(eff, ci)],
                                      [c[1] - e for e, c in zip(eff, ci)]],
                fmt="o-", color="#1b4f72", ms=4, capsize=3)
    ax.set_xticks(range(3), ["5", "20", "100"])
    ax.set_xlabel(r"Rician factor $\kappa$")
    ax.set_ylabel("effect statistic")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(a) Rician sensitivity", fontsize=8)

    ax = axes[1]
    variants = list(std["variants"])
    eff = [v["effect"] for v in variants]
    order = [v["full_order"] for v in variants]
    ax.bar(range(2), eff, color=["#1b4f72", "#b9770e"], width=0.55)
    for i, (v, o) in enumerate(zip(eff, order)):
        ax.text(i, v + 0.03, f"{v:.4f}; {o}/64", ha="center", fontsize=6)
    ax.set_xticks(range(2), ["all LoS", "8.4% LoS"])
    ax.set_ylabel("effect statistic")
    ax.set_ylim(0, 2.25)
    ax.set_title("(b) UMa LoS sensitivity", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figS2_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS2_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_supplementary_calibration() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    ax = axes[0]
    for law in LAWS:
        row = next(r for r in data["laws"][law]["curve"] if r["frames"] == 1024)
        draws = row["diagnostic_draws"]
        x = np.array(draws["m1_hat"]) - 0.9
        y = np.array(draws["c_hat"]) - data["laws"][law]["c"]
        ax.scatter(x, y, s=4, alpha=0.3, color=COLORS[law], label=LABELS[law])
    ax.set_xlabel(r"$\hat m_1-m_1$")
    ax.set_ylabel(r"$\hat c-c$")
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(fontsize=5.5, frameon=False, markerscale=3)
    ax.set_title("(a) joint errors, $M=1024$", fontsize=8)

    ax = axes[1]
    for law in LAWS:
        curve = data["laws"][law]["curve"]
        ax.plot([r["frames"] for r in curve], [r["projection_fraction"] for r in curve],
                color=COLORS[law], marker=MARKERS[law], ms=3, label=LABELS[law])
    ax.set_xscale("log")
    ax.set_xlabel("calibration frames $M$")
    ax.set_ylabel("projection frequency")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.set_title("(b) projection rate", fontsize=8)

    ax = axes[2]
    for law in LAWS:
        curve = data["laws"][law]["curve"]
        ax.plot([r["frames"] for r in curve], [r["fraction_worse_than_b2"] for r in curve],
                color=COLORS[law], marker=MARKERS[law], ms=3, label=LABELS[law])
    ax.set_xscale("log")
    ax.set_xlabel("calibration frames $M$")
    ax.set_ylabel(r"$P(\mathrm{WL\ worse\ than\ B2})$")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.set_title("(c) failure probability", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figS3_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS3_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_moment_controls() -> None:
    from calibration import mse
    m1, m2 = 0.90, 0.6355778670923117
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ax = axes[0]
    var_x = (1 + m2) / 2 - m1**2
    a, b = 0.1, var_x / 0.1
    x_a = np.array([m1 - a, m1 + b])
    w_a = np.array([b, a]) / (a + b)
    x_b = np.array([m1 - var_x / (1 - m1), 1.0])
    w_b = np.array([(1 - m1) / (1 - x_b[0]), (m1 - x_b[0]) / (1 - x_b[0])])
    cs = []
    for xs, ws, label, marker, color in ((x_a, w_a, "law A", "o", COLORS["binary"]),
                                        (x_b, w_b, "law B", "s", COLORS["uniform"])):
        assert np.all(np.abs(xs) <= 1) and np.all(ws >= 0)
        assert abs(ws.sum() - 1) < 1e-12
        assert abs(ws @ xs - m1) < 1e-12
        assert abs(ws @ (2 * xs**2 - 1) - m2) < 1e-12
        cs.append(float(ws @ (2 * xs**2 - 1) - m1**2))
        ax.plot(xs, ws, marker, color=color, label=label)
    ax.set_xlabel(r"$X=\cos\varepsilon$")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(a) distinct laws with matched moments", fontsize=8)
    ax = axes[1]
    b1 = [mse(m1, c) for c in cs]
    b2 = [mse(m1, c, m1, 0.0) for c in cs]
    assert abs(b1[0] - b1[1]) < 1e-12 and abs(b2[0] - b2[1]) < 1e-12
    (ROOT / "generated/figures/moment_control.json").write_text(json.dumps({
        "m1": m1, "m2": m2, "law_A_X": x_a.tolist(), "law_A_probability": w_a.tolist(),
        "law_B_X": x_b.tolist(), "law_B_probability": w_b.tolist(), "B1_risk": b1,
        "B2_risk": b2, "phase_lift": "each X is lifted to +/- arccos(X) with equal conditional probabilities"}, indent=2))
    xpos = np.arange(2)
    ax.bar(xpos - 0.16, b1, width=0.3, color=COLORS["binary"], label="WL")
    ax.bar(xpos + 0.16, b2, width=0.3, color="#7b8794", label="complex affine")
    ax.set_xticks(xpos, ["law A", "law B"])
    ax.set_ylabel("affine channel MSE")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(b) identical affine risk", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figS4_moment_controls.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS4_moment_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def figure_network() -> None:
    d1 = load(REPRO / "v08_run/run/output/E10_shape_output.json")
    d2 = load(REPRO / "system_run/output/E10_shape_output.json")
    de = load(REPRO / "expanded_run_v02/output/expanded_analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    ax = axes[0]
    x = np.array([0, 1, 3, 4])
    vals = [d1["ordering"][0]["effect_size"], d2["ordering"][0]["effect_size"],
            d1["collapse"][0]["effect_size"], d2["collapse"][0]["effect_size"]]
    ints = [d1["ordering"][0]["effect_size_interval"], d2["ordering"][0]["effect_size_interval"],
            d1["collapse"][0]["effect_size_interval"], d2["collapse"][0]["effect_size_interval"]]
    yerr = np.array([[v - i[0] for v, i in zip(vals, ints)], [i[1] - v for v, i in zip(vals, ints)]])
    ax.errorbar(x, vals, yerr=yerr, fmt="o", ms=3, color="#1b4f72", capsize=2)
    ax.set_xticks(x, ["D1\n.90", "D2\n.90", "D1\n.99", "D2\n.99"])
    ax.axhline(1.0, color="#888888", lw=0.6)
    ax.set_ylabel("ratio of excess channel-MSE ratios")
    ax.set_title("(a) independent draws", fontsize=8)

    ax = axes[1]
    recs = d2["records"]
    for code, law in (("B", "binary"), ("U", "uniform"), ("G", "wrapped_gaussian")):
        tab = {int(r["topology"]): r["mse_B2"] / r["mse_B1"] for r in recs
               if r["antenna_case"] == "N4_PRIMARY" and r["cell"] == f"{code}_K20_M090"}
        ys = np.sort(np.array(list(tab.values())))
        ax.plot(ys, np.linspace(0, 1, ys.size), color=COLORS[law], label=LABELS[law])
    ax.set_xlabel(r"$r_t$ at $m_1=0.90$")
    ax.set_ylabel("empirical CDF")
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    ks = [5, 20, 100]
    eff = [de["K5_M090"]["effect"], de["K20_M090"]["effect"], de["K100_M090"]["effect"]]
    ci = [de["K5_M090"]["interval"], de["K20_M090"]["interval"], de["K100_M090"]["interval"]]
    yerr = np.array([[v - i[0] for v, i in zip(eff, ci)], [i[1] - v for v, i in zip(eff, ci)]])
    ax.errorbar(range(3), eff, yerr=yerr, fmt="o-", color="#1b4f72", ms=3, capsize=2)
    ax.set_xticks(range(3), ["5", "20", "100"])
    ax.set_xlabel(r"Rician factor $\kappa$")
    ax.set_ylabel("effect statistic")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(c) expanded draw", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig6_network_evidence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig6_network_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_architecture()
    figure_shape_panels()
    figure_scalar_benefit()
    figure_calibration()
    figure_mismatch()
    figure_network()
    figure_calibrated_network()
    figure_operational()
    figure_supplementary_convergence()
    figure_supplementary_sensitivity()
    figure_supplementary_calibration()
    figure_moment_controls()
    print("wrote revised figure PDFs")
