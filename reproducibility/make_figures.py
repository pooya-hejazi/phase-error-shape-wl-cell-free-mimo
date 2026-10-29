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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter, PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
REPRO = ROOT / "reproducibility"
sys.path.insert(0, str(ANALYSIS))
from phase_laws import law_rows, lens_bound  # noqa: E402

OUT = ROOT / "generated/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "binary": "#0072B2",
    "uniform": "#D55E00",
    "wrapped_gaussian": "#009E73",
    "von_mises": "#9C4A8B",
}
LAWS = ("binary", "uniform", "wrapped_gaussian", "von_mises")
LABELS = {"binary": "B", "uniform": "U", "wrapped_gaussian": "G", "von_mises": "vM"}
MARKERS = {"binary": "o", "uniform": "s", "wrapped_gaussian": "^", "von_mises": "D"}


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": .65,
    "lines.linewidth": 1.45, "lines.markersize": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.axisbelow": True, "axes.labelpad": 4,
})
STYLES = {"binary": "-", "uniform": "--", "wrapped_gaussian": "-.", "von_mises": ":"}


def save_figure(fig, stem):
    """Export vector artwork and a 600 dpi preview with consistent print typography."""
    for ax in fig.axes:
        if ax.get_label() != "<colorbar>" and ax.axison:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out",length=3,width=.65,labelsize=7.5)
            ax.title.set_fontsize(8.5)
            ax.title.set_fontweight("bold")
            if ax.get_legend():
                for text in ax.get_legend().get_texts():text.set_fontsize(7)
        # Legends copy line styles at creation; update their handles with the curves.
        for line in [*ax.lines, *(ax.get_legend().get_lines() if ax.get_legend() else [])]:
            for law,color in COLORS.items():
                if line.get_color()==color and line.get_linestyle()=="-":
                    line.set_linestyle(STYLES[law])
                    line.set_linewidth(max(line.get_linewidth(),1.35))
    fig.set_layout_engine("constrained")
    fig.savefig(OUT/(stem+".pdf"),bbox_inches="tight",pad_inches=.025,
                metadata={"Creator":"Matplotlib; reproducible scientific figure", "CreationDate":None})
    fig.savefig(OUT/(stem+".png"),dpi=600,bbox_inches="tight",pad_inches=.025)


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
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    ax.text(1,57,"(a) Evaluation: reused pilots, channel estimates and receiver metrics",
            fontsize=8.5,weight="bold",color="#18344a")
    _box(ax,1,37,18,12,"$K$ users\nreused pilots",fs=8)
    _box(ax,26,37,22,12,"AP $l$: $N$ antennas\nunit-norm despreading",fs=7.5)
    _box(ax,55,37,20,12,"B1: $m_1$ and $c$\nB2: $m_1$ only",fs=8)
    _box(ax,82,37,17,12,"Channel MSE\nRZF / UatF SE",fs=7.5)
    for left,right in ((19,26),(48,55),(75,82)):
        _arrow(ax,(left+.4,43),(right-.4,43),lw=1.15)
    ax.text(1,26,"(b) Calibration: a separate window of $M$ observations per AP",
            fontsize=8.5,weight="bold",color="#18344a")
    _box(ax,1,8,27,12,"Isolated reference link\nknown complex coefficient",fc="#FFF3E8",ec="#D55E00",fs=7.5)
    _box(ax,36,8,27,12,"Estimate first moment\nand pseudo-covariance",fc="#FFF3E8",ec="#D55E00",fs=7.5)
    _box(ax,71,8,28,12,"Clip first moment\nthen signed feasible set",fc="#FFF3E8",ec="#D55E00",fs=7.5)
    _arrow(ax,(28.4,14),(35.6,14),color="#D55E00",lw=1.15)
    _arrow(ax,(63.4,14),(70.6,14),color="#D55E00",lw=1.15)
    ax.plot([85,85,65],[20.5,31,31],color="#D55E00",lw=1.1)
    _arrow(ax,(65,31),(65,36.5),color="#D55E00",lw=1.1)
    ax.text(77,32,"stationary moment sharing",fontsize=7,color="#A14500",ha="center")
    ax.text(50,1.5,"Calibration and evaluation observations are disjoint.",fontsize=7.5,ha="center",color="#425466")
    save_figure(fig,"fig1_architecture")
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
    save_figure(fig,"fig2_shape_panels")
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
    ax.loglog(snr,b2,color="#4B5563",ls=":",lw=1.4,label="B2 (all laws)")
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
    ax.set_ylabel("WL MSE reduction (%)")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(0, .085)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    save_figure(fig,"fig3_scalar_benefit")
    plt.close(fig)


def figure_mismatch() -> None:
    # Recompute retained gains from the same normalized scalar risk as the paper.
    mats={}
    laws=("uniform","wrapped_gaussian","binary","von_mises")
    for m in (.90,.70):
        rows=law_rows(m);v=L2*(1-m*m)+SG2;cy=v+.01
        ps=[L2*rows[l]["c"] for l in laws]
        matrix=[]
        for p in ps:
            b2=v*.01/cy
            oracle=b2*(1-p*p*.01/(v*(cy*cy-p*p)))
            values=[]
            for ph in ps:
                a=(v*cy-ph*ph)/(cy*cy-ph*ph);b=ph*.01/(cy*cy-ph*ph)
                assumed_risk=v+(a*a+b*b)*cy-2*v*a-2*b*p+2*a*b*p
                values.append((b2-assumed_risk)/(b2-oracle))
            matrix.append(values)
        mats[m]=np.asarray(matrix)
        assert np.allclose(np.diag(mats[m]),1,atol=1e-10)
    assert abs(mats[.9][3,2]+.0201)<5e-5
    assert abs(mats[.7][1,2]+68.3459)<5e-5
    labels = ["U", "G", "B", "vM"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95))
    for idx, (ax, (m1, mat)) in enumerate(zip(axes, mats.items())):
        im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-200,vcenter=0,vmax=1))
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("assumed law")
        if idx == 0:
            ax.set_ylabel("true law")
        ax.set_title(rf"$m_1={m1:.2f}$", fontsize=8)
        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if val>.65 or val < -130 else "#17212B")
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02,
                        label=r"retained gain $\rho$ (zero-centered)")
    cbar.set_ticks([-200,-100,0,.5,1])
    save_figure(fig,"fig4_mismatch")
    plt.close(fig)


def figure_calibration() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
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
    ax.set_ylabel("mean oracle-gain capture")
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
    ax.set_ylabel("individual-window capture")
    ax.grid(alpha=0.2, which="both", lw=0.35)
    ax.set_title("(b) 10th-90th percentiles", fontsize=8)

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
    save_figure(fig,"fig5_calibration_diagnostics")
    plt.close(fig)


def figure_calibrated_network() -> None:
    data=load(REPRO/"calibrated_network/output/calibrated_network_corrected_20261029.json")
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.9))
    ests=("oracle_B1","cal_B1","oracle_B2","cal_B2")
    colors=("#0072B2","#0072B2","#D55E00","#D55E00")
    for j,ax in enumerate(axes[:2]):
        for i,est in enumerate(ests):
            vals=[]
            for law in LAWS:
                rows=[r["estimators"][est] for r in data["records"] if r["law"]==law]
                vals.append(np.mean([r["mse"]/r["energy"] if j==0 else r["se_mean"] for r in rows]))
            ax.plot(np.arange(4)+(i-1.5)*.15,vals,linestyle="none",marker="o" if i<2 else "s",
                    markersize=4.5,markeredgewidth=.9,color=colors[i],
                    markerfacecolor="white" if i%2==0 else colors[i],label=est.replace("_"," "))
        ax.set_xticks(range(4),[LABELS[l] for l in LAWS]);ax.set_xlabel("phase law")
        ax.grid(axis="y",alpha=.22,lw=.5)
    axes[0].set_ylabel("channel NMSE")
    axes[0].ticklabel_format(axis="y",style="sci",scilimits=(0,0),useMathText=True)
    axes[0].set_title("(a) Channel estimation")
    axes[1].set_ylabel("SE (bit/s/Hz/user)");axes[1].set_title("(b) Uplink UatF")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=4,frameon=False,
               columnspacing=1,handletextpad=.3,fontsize=7)
    ax=axes[2]
    for est,color,label,style in (("cal_B1","#0072B2","calibrated B1","-"),("cal_B2","#D55E00","calibrated B2","--")):
        vals=[v for r in data["records"] if r["law"]=="uniform" for v in r["estimators"][est]["se_per_user"]]
        ax.step(np.sort(vals),np.arange(1,len(vals)+1)/len(vals),where="post",color=color,label=label,ls=style)
    ax.set_xlabel("SE (bit/s/Hz/user)");ax.set_ylabel("empirical CDF")
    ax.set_ylim(0,1);ax.grid(alpha=.22,lw=.5);ax.legend(loc="upper left",frameon=False)
    ax.set_title("(c) Uniform law")
    save_figure(fig,"fig6_calibrated_network")
    plt.close(fig)


def figure_operational() -> None:
    data = load(REPRO / "calibrated_network/output/operational_transfer_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    ax = axes[0]
    for law in LAWS:
        rows = [r for r in data["operational"] if r["law"] == law and r["calibration_frames"] == 1024]
        ax.plot([r["horizon_blocks"] for r in rows], [r["cal_B1_net_se_mean"] for r in rows],
                color=COLORS[law], marker=MARKERS[law], ms=3, label=LABELS[law])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([32,128,512]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("evaluation horizon (blocks)")
    ax.set_ylabel("net SE (bit/s/Hz/user)")
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
    ax.set_ylabel("SE gain (bit/s/Hz/user)")
    ax.legend(handles=[Line2D([],[],marker="o",ls="none",color="#425466",label="independent"),
                       Line2D([],[],marker="s",ls="none",color="#425466",alpha=.6,label="common per AP")],
              loc="upper right",frameon=False,handletextpad=.3)
    ax.set_title("(b) Residual sharing", fontsize=8)

    ax = axes[2]
    for i, row in enumerate(data["mismatch"]):
        val, ci = row["se_gain_B1_minus_oracle_B2_mean"], row["se_gain_interval"]
        ax.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt="o",
                    color="#b7772b", ms=4, capsize=2)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(data["mismatch"])),
                  [f"{LABELS[r['reference_law']]} → {LABELS[r['data_law']]}" for r in data["mismatch"]],
                  fontsize=6)
    ax.set_ylabel("SE gain (bit/s/Hz/user)")
    ax.set_title("(c) Reference-law mismatch", fontsize=8)
    save_figure(fig,"fig7_operational")
    plt.close(fig)


def figure_supplementary_convergence() -> None:
    data = load(REPRO / "calibrated_network/output/calibrated_network_corrected_20261029.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
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
        ax.set_ylabel("UatF SE (bit/s/Hz/user)" if metric == "se_mean" else "cumulative NMSE")
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].legend(fontsize=5.5, frameon=False)
    save_figure(fig,"figS1_convergence")
    plt.close(fig)


def figure_supplementary_sensitivity() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    std = load(ROOT / "reproducibility" / "standard_channel" / "results.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    ax = axes[0]
    expanded = load(ROOT / "reproducibility" / "expanded_run_v02" / "output" / "expanded_analysis.json")
    ks = [5, 20, 100]
    eff = [expanded[f"K{k}_M090"]["effect"] for k in ks]
    ci = [expanded[f"K{k}_M090"]["interval"] for k in ks]
    ax.errorbar(range(3), eff, yerr=[[e - c[0] for e, c in zip(eff, ci)],
                                      [c[1] - e for e, c in zip(eff, ci)]],
                fmt="o-", color="#0072B2", ms=4, capsize=3)
    ax.set_xticks(range(3), ["5", "20", "100"])
    ax.set_xlabel(r"Rician factor $\kappa$")
    ax.set_ylabel("effect statistic")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(a) Rician sensitivity", fontsize=8)

    ax = axes[1]
    variants = list(std["variants"])
    eff = [v["effect"] for v in variants]
    order = [v["full_order"] for v in variants]
    ax.bar(range(2), eff, color=["#0072B2", "#D55E00"], width=0.55)
    for i, (v, o) in enumerate(zip(eff, order)):
        ax.text(i, v + 0.03, f"{v:.4f}; {o}/64", ha="center", fontsize=6)
    ax.set_xticks(range(2), ["all LoS", "8.4% LoS"])
    ax.set_ylabel("effect statistic")
    ax.set_ylim(0, 2.25)
    ax.set_title("(b) UMa LoS sensitivity", fontsize=8)
    save_figure(fig,"figS2_sensitivity")
    plt.close(fig)


def figure_supplementary_calibration() -> None:
    data = load(ANALYSIS / "results/cp2_calibration_law_corrected_20261029.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
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
    save_figure(fig,"figS3_calibration")
    plt.close(fig)


def figure_moment_controls() -> None:
    from calibration import mse
    m1, m2 = 0.90, 0.6355778670923117
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
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
    ax.set_title("(a) Matched moments", fontsize=8)
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
    ax.set_ylim(0,max(b2)*1.25)
    ax.legend(fontsize=7, frameon=False,loc="upper center",ncol=2)
    ax.set_title("(b) identical affine risk", fontsize=8)
    save_figure(fig,"figS4_moment_controls")
    plt.close(fig)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def figure_network() -> None:
    d1 = load(REPRO / "v08_run/run/output/E10_shape_output.json")
    d2 = load(REPRO / "system_run/output/E10_shape_output.json")
    de = load(REPRO / "expanded_run_v02/output/expanded_analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    ax = axes[0]
    x = np.array([0, 1, 3, 4])
    vals = [d1["ordering"][0]["effect_size"], d2["ordering"][0]["effect_size"],
            d1["collapse"][0]["effect_size"], d2["collapse"][0]["effect_size"]]
    ints = [d1["ordering"][0]["effect_size_interval"], d2["ordering"][0]["effect_size_interval"],
            d1["collapse"][0]["effect_size_interval"], d2["collapse"][0]["effect_size_interval"]]
    yerr = np.array([[v - i[0] for v, i in zip(vals, ints)], [i[1] - v for v, i in zip(vals, ints)]])
    ax.errorbar(x, vals, yerr=yerr, fmt="o", ms=3, color="#0072B2", capsize=2)
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
    ax.errorbar(range(3), eff, yerr=yerr, fmt="o-", color="#0072B2", ms=3, capsize=2)
    ax.set_xticks(range(3), ["5", "20", "100"])
    ax.set_xlabel(r"Rician factor $\kappa$")
    ax.set_ylabel("effect statistic")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("(c) expanded draw", fontsize=8)
    save_figure(fig,"fig6_network_evidence")
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
