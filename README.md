# Phase-Error Shape in Widely-Linear Channel Estimation for Cell-Free Massive MIMO

Companion code and numerical data for the manuscript:

> **Phase-Error Shape in Widely-Linear Channel Estimation for Cell-Free Massive MIMO**  
> Pooya Hejazi and Masoud Sabaei  
> Amirkabir University of Technology

**Publication status:** the manuscript is under review and has not been published. The manuscript source and PDF are intentionally not included in this repository; this repository contains the code, configuration, and numerical outputs needed to inspect and reproduce the reported results.

---

## Overview

Cell-free massive MIMO systems rely on line-of-sight phase tracking. The residual phase error is commonly summarized by its first circular moment,

$$
m_1 = \mathbb{E}[\cos\varepsilon].
$$

This paper shows that $m_1$ alone is insufficient for widely-linear (WL) channel estimation. The WL estimator also depends on the second circular moment through the pseudo-covariance

$$
c = m_2 - m_1^2,
\qquad
m_2 = \mathbb{E}[\cos 2\varepsilon].
$$

For a fixed $m_1$, the pseudo-covariance is a free coordinate in the interval

$$
-(1-m_1^2) \le c \le 1-m_1^2.
$$

The complex-affine estimator **B2** uses $m_1$ only, whereas the widely-linear estimator **B1** uses both $m_1$ and $c$. The repository provides the scalar phase-law analysis, the calibration study, the cell-free system experiments, and the post-hoc analyses reported in the manuscript.

---

## Key Results

- For $0<m_1<1$, the Binary, Uniform, and Wrapped-Gaussian phase laws have a strict scalar ordering in $|c|$, and therefore in the WL gain.
- The von Mises comparison is numerical over a 20,001-point grid; it is not presented as an analytic theorem.
- For the declared phase laws, the calibration experiment gives:
  - captured gain at $M=1024$: approximately **0.984 / 0.966 / 0.961** for B/U/G;
  - interpolated 99% crossings: approximately **2,216 / 4,141 / 3,898** frames;
  - these values are Monte Carlo estimates, not operational guarantees.
- In three independent cell-free uplink draws with pilot reuse and RZF, the shape-effect estimates are **2.1196**, **2.2079**, and **2.0814**, with the full order in all 64 topologies.
- The pooled collapse estimate is **1.0590** with a 95% bootstrap interval of **[1.0348, 1.0861]**.
- The downlink study is limited to maximum-ratio precoding; a full downlink ordering is not claimed.

---

## Repository Layout

```text
analysis/
  phase_laws.py                  # phase laws, feasible lens, vM numerical check
  calibration.py                 # baseline calibration, MSE, and risk utilities
  calibration_law.py             # Monte Carlo calibration on the actual phase laws
  requirements.txt
  results/
    cp1_phase_laws.json          # phase-law ordering and feasible-set results
    cp2_calibration_law.json     # capture curves, covariance, bias, M*

reproducibility/
  v08_run/                       # frozen v08 packet, replay, and vendored code
  system_run/                    # frozen replication packet and analyses
  expanded_run_v02/              # expanded campaign: kappa and c-sensitivity
  standard_channel/              # 3GPP UMa standard-channel study
  downlink/                      # downlink MR-precoding study
  pooled_collapse.py             # post-hoc pooled analysis
  pooled_collapse.json
```

Each frozen packet retains its configuration, environment lock, checksums, and manifest for provenance. Scientific outputs are stored as JSON so that reviewers can inspect them without rerunning the original experiments.

---

## Requirements

The analyses were produced with:

```text
Python 3.12
numpy  2.5.2
scipy  1.18.1
PyYAML 6.0.3
```

The scalar scripts use the packages listed in `analysis/requirements.txt`. The frozen packets include their own environment locks.

All commands below must be run **from the repository root**.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r analysis\requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
```

---

## Quick Review Path

These commands regenerate and verify the main scalar results in about one minute, depending on hardware:

```bash
python -B analysis/phase_laws.py
python -B analysis/calibration_law.py
python -B reproducibility/pooled_collapse.py
```

Expected outcomes:

- `analysis/phase_laws.py` writes `analysis/results/cp1_phase_laws.json` and prints the phase-law ordering checks.
- `analysis/calibration_law.py` writes `analysis/results/cp2_calibration_law.json` and prints capture, bias, projection, and $M^\star$ values for each law.
- `reproducibility/pooled_collapse.py` reads the three stored system outputs and writes `reproducibility/pooled_collapse.json`.

The calibration script uses deterministic SHA-256-derived seeds. Re-running it should produce byte-identical JSON.

---

## Detailed Reproduction Workflow

### Step 1: Scalar phase-law and calibration analyses

```bash
python -B analysis/phase_laws.py
python -B analysis/calibration_law.py
```

| Command | Reads | Writes | Terminal output |
|---|---|---|---|
| `analysis/phase_laws.py` | Closed-form phase-law definitions | `analysis/results/cp1_phase_laws.json` | Ordering, sinc inequality, and von Mises grid checks |
| `analysis/calibration_law.py` | Actual phase-law generators and proper noise | `analysis/results/cp2_calibration_law.json` | Capture, standard error, bias, projection fraction, and $M^\star$ |

`calibration_law.py` performs the 2,000-replicate Monte Carlo calibration. It is the heaviest scalar computation but normally completes in tens of seconds.

### Step 2: Analyze the frozen system outputs

```bash
python -B reproducibility/expanded_run_v02/analyze_expanded.py
python -B reproducibility/system_run/analyze_replication.py
python -B reproducibility/system_run/robustness_replication.py
```

| Command | Reads | Writes | Terminal output |
|---|---|---|---|
| `expanded_run_v02/analyze_expanded.py` | `expanded_run_v02/output/E10_shape_output.json` | `expanded_run_v02/output/expanded_analysis.json` | Block effects and controls |
| `system_run/analyze_replication.py` | `system_run/output/E10_shape_output.json` | No file | Ordering, collapse, controls, and paired contrasts |
| `system_run/robustness_replication.py` | `system_run/output/E10_shape_output.json` | No file | Standard errors, power, MDD, jackknife, and distance to the collapse threshold |

To retain the terminal-only reports:

```powershell
python -B reproducibility/system_run/analyze_replication.py | Tee-Object -FilePath replication_report.txt
python -B reproducibility/system_run/robustness_replication.py | Tee-Object -FilePath robustness_report.txt
```

On Linux or macOS, use `tee` instead of `Tee-Object`.

### Step 3: Downlink and standard-channel studies

```bash
python -B reproducibility/downlink/run_downlink.py
python -B reproducibility/standard_channel/run_standard.py
```

| Command | Writes | Main contents |
|---|---|---|
| `downlink/run_downlink.py` | `reproducibility/downlink/results.json` | Deterministic self-test, mean SE, paired intervals, and positive-topology counts |
| `standard_channel/run_standard.py` | `reproducibility/standard_channel/results.json` | Both UMa variants, ordering counts, and validation checks |

The downlink run normally takes about one minute. The standard-channel run takes several minutes; both depend on hardware.

### Step 4: Pooled analysis

```bash
python -B reproducibility/pooled_collapse.py
```

This script reads:

```text
reproducibility/v08_run/run/output/E10_shape_output.json
reproducibility/system_run/output/E10_shape_output.json
reproducibility/expanded_run_v02/output/E10_shape_output.json
```

and writes `reproducibility/pooled_collapse.json`.

Reference values:

```text
M090: pooled_effect = 2.1372452505
      interval      = [2.0789193293, 2.2074038563]
M099: pooled_effect = 1.0589764846
      interval      = [1.0347764372, 1.0861185945]
```

### Step 5: Inspect a JSON result

To print a formatted JSON file:

```bash
python -B -c "import json; p='analysis/results/cp2_calibration_law.json'; print(json.dumps(json.load(open(p, encoding='utf-8')), indent=2, ensure_ascii=False))"
```

To print only the calibration summary:

```bash
python -B -c "import json; d=json.load(open('analysis/results/cp2_calibration_law.json', encoding='utf-8')); print(json.dumps(d['summary'], indent=2))"
```

If `jq` is installed:

```bash
jq '.summary' analysis/results/cp2_calibration_law.json
jq '.M099.pooled_effect, .M099.pooled_interval' reproducibility/pooled_collapse.json
```

---

## Files Overwritten by the Analysis Scripts

The reviewer workflow overwrites only these derived outputs:

- `analysis/results/cp1_phase_laws.json`
- `analysis/results/cp2_calibration_law.json`
- `reproducibility/expanded_run_v02/output/expanded_analysis.json`
- `reproducibility/downlink/results.json`
- `reproducibility/standard_channel/results.json`
- `reproducibility/pooled_collapse.json`

The files named `E10_shape_output.json` are frozen scientific outputs. The analysis commands above do not overwrite them.

Do **not** run the frozen experiment drivers as part of a normal review:

```text
reproducibility/v08_run/run/E10_shape_run.py
reproducibility/system_run/E10_shape_run.py
reproducibility/expanded_run_v02/run/E10_shape_run.py
```

They are the original experiment drivers, not analysis scripts.

---

## Full v08 Replay

The complete v08 replay is available but is not required for the quick review path:

```bash
python -B reproducibility/v08_run/replay.py
```

On the original hardware, the replay took approximately 70 minutes. It reconstructs the historical directory layout, refuses to start if an unrelated `src/` directory exists, and removes only the paths it created. The comparison covers 20 scientific sections and 1,600 records at a relative tolerance of `1e-12`.

Replay outputs:

| Path | Contents |
|---|---|
| `reproducibility/v08_run/replay_output/E10_shape_output.json` | Fresh replay output retained for comparison |
| `reproducibility/v08_run/REPLAY_RESULT_ROUND3.json` | Section list, record count, SHA-256, and mismatch report |

---

## Reference Data

| File | Contents |
|---|---|
| `analysis/results/cp1_phase_laws.json` | Analytic and numerical phase-law ordering results |
| `analysis/results/cp2_calibration_law.json` | Capture curves, joint covariance, bias, projection, and $M^\star$ |
| `reproducibility/v08_run/run/output/E10_shape_output.json` | Frozen primary v08 output |
| `reproducibility/system_run/output/E10_shape_output.json` | Frozen replication output |
| `reproducibility/expanded_run_v02/output/E10_shape_output.json` | Frozen expanded and sensitivity output |
| `reproducibility/pooled_collapse.json` | Pooled analysis of the three draws |
| `reproducibility/downlink/results.json` | Downlink results and deterministic self-test |
| `reproducibility/standard_channel/results.json` | 3GPP UMa standard-channel results |

---

## Scope and Limitations

The following qualifications are part of the manuscript's claims:

- The B/U/G ordering is analytic for $0<m_1<1$. The von Mises ordering is numerical over a finite grid.
- The cell-free results are from a synthetic channel model, not a hardware testbed or multi-site measurement.
- The expanded calibration cells are $c$-only sensitivity checks. They do not validate a closed-loop tracker.
- The scalar calibration result combines a conditional theorem with Monte Carlo; the estimated window is not a universal operational guarantee.
- The binary law is a boundary case, and a substantial fraction of its calibration windows are projected onto the feasible boundary.
- The downlink study uses maximum-ratio precoding only. The paired intervals do not resolve the full downlink ordering.
- The standard-channel study is frequency-flat and does not model delay dispersion.

---

## Contact

Questions about the code or data can be directed to the authors:

- Pooya Hejazi — `pooya.hejazi@aut.ac.ir`
- Masoud Sabaei — `sabaei@aut.ac.ir`
