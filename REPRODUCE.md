# Reviewer reproduction guide

Run commands from the repository root after installing `requirements.txt` in Python 3.12. The archived environment used Python 3.12.10. Exact versions are recorded in `ENVIRONMENT.json`. No manuscript, bibliography, or article PDF is needed to execute the code. All data are synthetic; no external dataset or paid solver is required.

## 1. Inspect and check (seconds to a few minutes)

```bash
python -B reproducibility/verify_release.py
python -B reproducibility/checks/check_corrections.py
python -B reproducibility/checks/check_theory.py
python -B reproducibility/checks/check_calibration_moments.py
python -B reproducibility/calibrated_network/summarize.py
python -B reproducibility/make_figures.py
```

The checks cover UatF aggregation before SINR/log, independent AP calibration, batched/direct estimator agreement, feasible moments, complex-vector analytic/Monte Carlo MSE, minimax endpoint, actual-law moments and calibration covariance, and mutation-sensitive comparison of the retained v08 replay. Diagnostics go to `generated/checks/`; numerical summaries to `generated/network_summary.json`; locally generated plots to `generated/figures/`. These paths are ignored by Git. Figures are generated from the archived corrected JSON, not from an arbitrary new run.

## 2. Regenerate corrected Monte Carlo outputs

Use new output filenames: existing corrected results are protected. On PowerShell:

```powershell
$env:CALIBRATION_OUTPUT='calibration_reproduced.json'
python -B analysis/calibration_law.py
$env:CAL_OUTPUT='network_reproduced.json'
python -B reproducibility/calibrated_network/run_calibrated_network.py
$env:TRANSFER_OUTPUT='transfer_reproduced.json'
python -B reproducibility/calibrated_network/run_operational_transfer.py
$env:DOWNLINK_OUTPUT='downlink_reproduced.json'
python -B reproducibility/downlink/run_convergence.py
```

On Linux/macOS, the equivalent is:

```bash
CALIBRATION_OUTPUT=calibration_reproduced.json python -B analysis/calibration_law.py
CAL_OUTPUT=network_reproduced.json python -B reproducibility/calibrated_network/run_calibrated_network.py
TRANSFER_OUTPUT=transfer_reproduced.json python -B reproducibility/calibrated_network/run_operational_transfer.py
DOWNLINK_OUTPUT=downlink_reproduced.json python -B reproducibility/downlink/run_convergence.py
```

The scalar study uses 2,000 windows per law/sample-size cell. Calibrated uplink uses 32 topologies × 64 frames × four laws × four estimators. Transfer uses eight topologies × 64 frames; downlink uses 32 topologies × 128 frames. Scalar runs usually take minutes; network campaigns can take tens of minutes or longer depending on CPU/BLAS. No GPU is required. Seed integers are identifiers, not dates. Leave `CAL_TOPOLOGIES` and `CAL_REALIZATIONS` unset to reproduce the paper.

Compare every scientific JSON field, rather than only the printed averages:

```bash
python -B reproducibility/compare_results.py analysis/results/cp2_calibration_law_corrected_20261029.json analysis/results/calibration_reproduced.json
python -B reproducibility/compare_results.py reproducibility/calibrated_network/output/calibrated_network_corrected_20261029.json reproducibility/calibrated_network/output/network_reproduced.json
python -B reproducibility/compare_results.py reproducibility/calibrated_network/output/operational_transfer_corrected_20261029.json reproducibility/calibrated_network/output/transfer_reproduced.json
python -B reproducibility/compare_results.py reproducibility/downlink/downlink_convergence.json reproducibility/downlink/downlink_reproduced.json
```

Exit code zero means no differences above `rtol=1e-10, atol=1e-12`. Nonzero means investigate; floating-point and random-generator versions can matter. The release manifest verifies archived bytes separately. Do not replace reference files merely to make a check pass.

## 3. Replay all three frozen channel-MSE campaigns

```bash
python -B reproducibility/replay_frozen.py v08
python -B reproducibility/replay_frozen.py replication
python -B reproducibility/replay_frozen.py expanded
```

Each command reconstructs the original layout under a **new** `generated/replay_*` directory, preserves the scientific driver/configuration, verifies the frozen hashes, and runs the original seeds. It restores a text file's newline convention only if the resulting bytes match its frozen SHA-256. It replaces the historical author-machine/path lock with Python 3.12, pinned-package, and single-thread BLAS checks; this portability adaptation is recorded in `PORTABLE_ENVIRONMENT.json`. Numerical algorithms and experimental cells are unchanged.

Fresh output and `COMPARISON.json` remain in that directory. Comparison uses the existing scientific-field comparator at relative tolerance `1e-12`, ignoring only documented timing/path/environment metadata. A mismatch produces a nonzero exit status. Never run the frozen drivers directly from their archived locations.

For a preflight without simulation, add `--prepare-only --run-name preflight_v08` (or another new name). This still checks hashes, configuration and dependencies. The directory is intentionally retained; choose another `--run-name` for an actual run. The original v08 replay took about 70 minutes. Replication and expanded campaigns may also be long. The release preflight validates all three; it does not represent a fresh full execution of all three campaigns.

## 4. Remaining analyses

```bash
python -B analysis/phase_laws.py
python -B reproducibility/expanded_run_v02/analyze_expanded.py
python -B reproducibility/system_run/analyze_replication.py
python -B reproducibility/system_run/robustness_replication.py
python -B reproducibility/pooled_collapse.py
python -B reproducibility/standard_channel/run_standard.py
```

These historical analysis commands regenerate their documented derived JSON in place, except the two replication summaries that print to the terminal. `standard_channel/run_standard.py` also regenerates its archived `results.json`. Use a separate clone for these reruns and compare against the release before replacing anything. The 16-frame `downlink/run_downlink.py` is historical; the paper uses `run_convergence.py` and its 128-frame results.

## Interpretation

The B/U/G ordering is a scalar theorem, not a universal network SE ordering. The von Mises comparison is a finite-grid numerical result. Uplink comparisons fix receiver architecture and share calibration information. Shared-per-AP residuals are a deliberately mismatched-receiver sensitivity, not a matched shared-LO theorem. The standard-channel run is UMa-inspired, not a full 3GPP CDL implementation. Topology intervals do not eliminate finite-frame expectation bias. No hardware validation is claimed.
