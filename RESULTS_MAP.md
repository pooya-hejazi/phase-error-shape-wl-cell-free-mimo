# Mapping the manuscript to executable evidence

Version: `submission-2026-10-29-v2`. Figure/table names below identify the audited manuscript, regardless of later numbering changes. The local manuscript and its bibliography are deliberately not included here.

| Paper result | Final numerical evidence | Regeneration / inspection |
|---|---|---|
| Feasible moment interval; B/U/G order; finite-grid von Mises comparison | `analysis/results/cp1_phase_laws.json` | `analysis/phase_laws.py`; `make_figures.py` shape panels |
| Scalar MSE / mismatch / scalar benefit | closed-form `risk` in `reproducibility/make_figures.py`; scalar utilities in `analysis/calibration.py` | scalar-benefit and mismatch plots; no random data required |
| Finite-window capture table and calibration figures | `analysis/results/cp2_calibration_law_corrected_20261029.json` | `analysis/calibration_law.py` |
| Calibrated channel NMSE / uplink UatF / per-user CDF | `reproducibility/calibrated_network/output/calibrated_network_corrected_20261029.json` | `run_calibrated_network.py`; `summarize.py`; calibrated-network plot |
| Uplink prefix convergence, 8/16/32/64 | `uatf_prefix_per_user` in the same corrected network JSON | `summarize.py`; convergence plot |
| Three-draw channel-MSE table, first draw | `reproducibility/v08_run/run/output/E10_shape_output.json` | `replay_frozen.py v08` |
| Same table, independent second draw | `reproducibility/system_run/output/E10_shape_output.json` | `replay_frozen.py replication`; `system_run/analyze_replication.py` |
| Same table, expanded draw / Rician factor / c-only errors | `reproducibility/expanded_run_v02/output/E10_shape_output.json` and `expanded_analysis.json` | `replay_frozen.py expanded`; `expanded_run_v02/analyze_expanded.py` |
| Pooled collapse, 192 topologies | `reproducibility/pooled_collapse.json` | `reproducibility/pooled_collapse.py` |
| Reference overhead, common-per-AP residuals and law-transfer | `reproducibility/calibrated_network/output/operational_transfer_corrected_20261029.json` | `run_operational_transfer.py`; operational plot |
| UMa-inspired channel variants | `reproducibility/standard_channel/results.json` | `reproducibility/standard_channel/run_standard.py` |
| Downlink paragraph and supplement prefix table | `reproducibility/downlink/downlink_convergence.json`, `summary/128` for final results | `reproducibility/downlink/run_convergence.py` |
| Valid distinct laws with identical first two moments | generated `moment_control.json` | `make_figures.py`, `figure_moment_controls()` asserts probability/support/moment conditions |
| Vector equations / actual-law calibration moments | diagnostics under `generated/checks/` | `checks/check_theory.py`, `checks/check_calibration_moments.py` |

Paths shortened in the last column are relative to their corresponding study folder. Run all commands from the root as specified in REPRODUCE.md. The plotting script also emits one historical channel-MSE diagnostic not used as a main figure; it is not an additional claimed result.

## Exact interpretation of the statistics

- A topology's channel error sums squared errors over its frames, antennas and users. Its NMSE divides by the corresponding channel energy. The calibrated-network summary averages these topology-specific NMSEs.
- Frozen channel-MSE ratio is `r = MSE_B2/MSE_B1`. The reported shape effect is `(mean(r_B)-1)/(mean(r_G)-1)`. A value near 2 does **not** mean halving MSE.
- Uplink UatF takes effective-channel first/second moments over inner frames, then forms SINR and log, then averages per-user rates across topologies. Confidence intervals resample paired topologies, not the dependent user observations as if independent.
- Both calibrated B1 and B2 receive the same estimated first moment and reference cost. Only B1 uses complementary covariance. Oracle curves use true population moments, not instantaneous phases.
- Extra reference cost is `4M/(4M+200H)` under ideal parallel isolated acquisition. Net SE multiplies the already-prelogged UatF rate by `200H/(4M+200H)`.
- Historical primary/replication contrasts use the recorded simultaneous adjustment. Other stated 95% topology intervals are nominal; finite-frame bias is not included.
- The 99% scalar crossing interpolates capture linearly in `log(M)`; it is not a measured deterministic minimum.

## Files retained only for provenance

`cp2_calibration_law.json`, unversioned calibrated-network/operational-transfer JSON, `calibrated_convergence.json`, `calibrated_network_summary.json`, and `downlink/results.json` are historical. Only the explicitly mapped corrected outputs and 128-frame extension determine final calibrated results. Do not combine old SE summaries with the new manuscript.
