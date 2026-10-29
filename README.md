# Phase-Error Shape in Widely-Linear Channel Estimation for Cell-Free Massive MIMO

Companion code and numerical results by **Pooya Hejazi and Masoud Sabaei**, Amirkabir University of Technology.

**Paper-matched version: `submission-2026-10-29-v2`.** Use the tag cited in the manuscript, rather than a future version of `main`. This repository distributes reproduction code, configurations, synthetic numerical outputs and execution instructions. Manuscript sources, bibliographies, article PDFs and review reports are maintained separately.

At a fixed first circular phase moment, the complex-affine estimator B2 is unchanged while the widely-linear estimator B1 depends on the complementary moment `c = m2 - m1²`. The study separates analytic scalar ordering, finite-window moment acquisition, and synthetic network consequences. B1 is optimal within the real-affine class, not generally the unrestricted Bayesian MMSE estimator.

## Start here

```bash
git clone https://github.com/pooya-hejazi/phase-error-shape-wl-cell-free-mimo.git
cd phase-error-shape-wl-cell-free-mimo
git checkout submission-2026-10-29-v2
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
python -B reproducibility/verify_release.py
python -B reproducibility/checks/check_corrections.py
python -B reproducibility/calibrated_network/summarize.py
python -B reproducibility/make_figures.py
```

Python **3.12** is required for the frozen replay; the recorded version is 3.12.10. CPU execution uses NumPy/SciPy, Matplotlib and PyYAML. There are no external datasets, pretrained models or paid solvers. The quick path inspects/reaggregates archived results; it is **not** a full Monte Carlo rerun.

- [Exact reproduction commands and expected outputs](REPRODUCE.md)
- [Paper-to-code/result mapping](RESULTS_MAP.md)
- [Recorded environment](ENVIRONMENT.json)
- [Release file checksums](RELEASE_MANIFEST.json)

Generated diagnostic JSON and plots appear in `generated/`, which is excluded from publication. Plots can be opened locally after generation. All archived scientific outputs are readable JSON.

## Principal results

| Quantity | Binary | Uniform | Wrapped Gaussian | von Mises |
|---|---:|---:|---:|---:|
| Scalar oracle gain captured at 1024 frames | 0.983891 | 0.965604 | 0.960891 | 0.957540 |
| Interpolated 99% crossing (frames) | 2216 | 4141 | 3898 | 3931 |
| Calibrated uplink SE gain (bit/s/Hz/user) | 0.247729 | 0.176781 | 0.117127 | 0.109196 |
| Oracle MR downlink SE gain, 128 frames | 0.009757 | 0.006520 | 0.003667 | not evaluated |

Calibration crossings are Monte Carlo estimates, not guaranteed stationarity windows. The uplink SE intervals and topology-level quantities are reconstructed by `reproducibility/calibrated_network/summarize.py`. Three independent historical channel-MSE draws yield excess-ratio effects 2.1196, 2.2079 and 2.0814 at first moment 0.90 and Rician factor 20; these are neither percentage MSE reductions nor SE gains.

## Layout

```text
analysis/                         scalar laws, calibration, archived scalar outputs
reproducibility/calibrated_network/  corrected uplink, transfer and summaries
reproducibility/downlink/          historical 16-frame and final 128-frame results
reproducibility/standard_channel/  UMa-inspired sensitivity
reproducibility/v08_run/           frozen primary campaign and vendored library
reproducibility/system_run/        independent replication
reproducibility/expanded_run_v02/   Rician-factor and moment-error sensitivity
reproducibility/checks/            independent numerical checks
reproducibility/replay_frozen.py   portable replay of any of the three campaigns
reproducibility/make_figures.py    local regeneration of plots
```

## Version and scope

The `corrected_20261029.json` outputs supersede earlier calibration, calibrated-network and transfer files. The corrected UatF rate averages effective-channel moments over frames **before** SINR and logarithm; reference calibration is independent per AP. Historical files remain for provenance and must not be substituted for the mapped final results. The 128-frame downlink extension supersedes the earlier 16-frame conclusion.

All experiments are synthetic. Baselines share observations and receiver rules; oracle versus calibrated information is labeled. The network order is empirical, the von Mises order is numerical, common-per-AP residuals deliberately test covariance mismatch, and UMa-inspired sensitivity is not a full standardized channel model. The work does not establish hardware tracking performance or certify every tail condition of the calibration theorem.

Questions and reproducibility discrepancies can be raised through repository Issues. Please include the version, environment, command and comparison report. Do not upload confidential manuscripts or credentials.

The v2 tag improves print typography, color/line differentiation, legends and diagram flow. It computes the mismatch matrix directly from the scalar risk and uses the standard empirical CDF steps. Archived simulation data and reported numerical results are unchanged from v1. Generated vector PDFs and 600 dpi PNGs stay local.
