"""One summary row per (antenna case, cell, receiver) for R2C01_DIRA_SHAPE_v08."""

from __future__ import annotations

import numpy as np

from shape_stats import (
    bootstrap_interval,
    excludes_zero,
    mse_ratio_values,
    se_difference_values,
    se_even_odd_difference_values,
    se_user_differences,
    t_interval,
)


def mse_row(
    case: dict,
    cell: dict,
    cid: str,
    receiver: dict,
    spec,
    records: dict,
    complete: list[int],
    status: str,
    stats: dict,
    rng: np.random.Generator,
) -> dict:
    """Build one row: primary MSE ratio plus secondary SE diagnostics."""

    role = str(cell["role"])
    row = {
        "antenna_case": case["id"],
        "cell": cid,
        "kappa": float(cell["kappa"]),
        "shape": cell["shape"],
        "role": role,
        "m1_design": float(spec.m1),
        "m2_design": float(spec.m2),
        "c_design": float(spec.c),
        "receiver": receiver["id"],
        "receiver_role": receiver["role"],
        "topologies_complete": len(complete),
        "mse_B1_sum": None,
        "mse_B2_sum": None,
        "channel_energy_sum": None,
        "nmse_B1": None,
        "nmse_B2": None,
        "mse_ratio_of_sums_B2_over_B1": None,
        "mse_ratio_per_topology_mean": None,
        "mse_ratio_per_topology_sd": None,
        "mse_ratio_excess_mean": None,
        "mse_ratio_note": None,
        "per_topology_mse_ratios": [],
        "se_difference_per_topology": [],
        "se_difference_mean": None,
        "se_difference_sd": None,
        "se_t_interval": [None, None],
        "se_bootstrap_interval": [None, None],
        "se_bias_shift_interval": [None, None],
        "se_even_odd_gap_mean": None,
        "se_even_odd_gap_sd": None,
        "se_note": None,
        "se_user_p10": None,
        "se_user_p50": None,
        "se_user_p90": None,
        "se_negative_user_fraction": None,
        "bias_sensitive": None,
        "bias_sensitive_note": None,
        "se_label": "INCONCLUSIVE",
        "se_label_alpha": float(stats["alpha"]),
        "se_label_role": "SECONDARY_DESCRIPTIVE",
        "empirical_m1_mean": None,
        "empirical_m2_mean": None,
        "empirical_m1_sd": None,
        "empirical_m2_sd": None,
        "moment_max_abs_z1": None,
        "moment_max_abs_z2": None,
        "moment_pass": None,
        "moment_note": None,
    }
    if not complete:
        if status == "FAIL_INTEGRITY":
            row["se_label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
            row["bias_sensitive_note"] = "FAIL_INTEGRITY"
        elif status == "INCOMPLETE_RESOURCE":
            row["se_label"] = "INCOMPLETE_RESOURCE"
            row["bias_sensitive_note"] = "INCOMPLETE_RESOURCE"
        elif role == "STRUCTURAL_EQUALITY_CONTROL":
            row["se_label"] = "CONTROL_NOT_EVALUATED"
            row["bias_sensitive_note"] = "CONTROL_ROW_NOT_EVALUATED"
        else:
            row["se_label"] = "INSUFFICIENT_TOPOLOGIES"
        return row

    mse_b1 = float(
        sum(float(records[(case["id"], cid, t)]["mse"]["B1"]) for t in complete)
    )
    mse_b2 = float(
        sum(float(records[(case["id"], cid, t)]["mse"]["B2"]) for t in complete)
    )
    energy = float(
        sum(float(records[(case["id"], cid, t)]["channel_energy"]) for t in complete)
    )
    row["mse_B1_sum"] = mse_b1
    row["mse_B2_sum"] = mse_b2
    row["channel_energy_sum"] = energy
    row["nmse_B1"] = mse_b1 / energy if energy > 0.0 else None
    row["nmse_B2"] = mse_b2 / energy if energy > 0.0 else None
    row["mse_ratio_of_sums_B2_over_B1"] = mse_b2 / mse_b1 if mse_b1 > 0.0 else None

    ratios = mse_ratio_values(case, cid, records, complete)
    row["per_topology_mse_ratios"] = ratios.tolist()
    if ratios.size and np.all(np.isfinite(ratios)):
        row["mse_ratio_per_topology_mean"] = float(np.mean(ratios))
        row["mse_ratio_per_topology_sd"] = (
            float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0
        )
        row["mse_ratio_excess_mean"] = float(np.mean(ratios) - 1.0)
    else:
        row["mse_ratio_note"] = "NONFINITE_OR_EMPTY_MSE_RATIO"

    m1_emp = np.asarray(
        [float(records[(case["id"], cid, t)]["m1_emp"]) for t in complete]
    )
    m2_emp = np.asarray(
        [float(records[(case["id"], cid, t)]["m2_emp"]) for t in complete]
    )
    row["empirical_m1_mean"] = float(m1_emp.mean())
    row["empirical_m2_mean"] = float(m2_emp.mean())
    row["empirical_m1_sd"] = float(m1_emp.std(ddof=1)) if m1_emp.size > 1 else 0.0
    row["empirical_m2_sd"] = float(m2_emp.std(ddof=1)) if m2_emp.size > 1 else 0.0
    z1 = [abs(float(records[(case["id"], cid, t)]["moment_z1"])) for t in complete]
    z2 = [abs(float(records[(case["id"], cid, t)]["moment_z2"])) for t in complete]
    row["moment_max_abs_z1"] = float(max(z1))
    row["moment_max_abs_z2"] = float(max(z2))
    row["moment_pass"] = bool(
        all(bool(records[(case["id"], cid, t)]["moment_pass"]) for t in complete)
    )
    notes = [
        str(records[(case["id"], cid, t)].get("moment_note"))
        for t in complete
        if records[(case["id"], cid, t)].get("moment_note")
    ]
    if notes:
        row["moment_note"] = notes[0]

    differences = se_difference_values(case, cid, receiver["id"], records, complete)
    row["se_difference_per_topology"] = differences.tolist()
    if not np.all(np.isfinite(differences)):
        row["se_label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        row["se_note"] = "NONFINITE_SE_DIFFERENCE"
        return row
    se_ci, _, _ = t_interval(differences, float(stats["alpha"]))
    row["se_t_interval"] = se_ci
    row["se_bootstrap_interval"] = bootstrap_interval(
        differences, rng, int(stats["bootstrap_resamples"]), float(stats["alpha"])
    )
    if differences.size:
        row["se_difference_mean"] = float(differences.mean())
        row["se_difference_sd"] = (
            float(differences.std(ddof=1)) if differences.size > 1 else 0.0
        )
    even_odd = se_even_odd_difference_values(
        case, cid, receiver["id"], records, complete
    )
    if even_odd.size and np.all(np.isfinite(even_odd)):
        row["se_even_odd_gap_mean"] = float(even_odd.mean())
        row["se_even_odd_gap_sd"] = (
            float(even_odd.std(ddof=1)) if even_odd.size > 1 else 0.0
        )
    shifted = differences - np.minimum(differences, 0.0)
    shifted_ci, _, _ = t_interval(shifted, float(stats["alpha"]))
    row["se_bias_shift_interval"] = shifted_ci
    if role == "STRUCTURAL_EQUALITY_CONTROL":
        row["bias_sensitive"] = False
        row["bias_sensitive_note"] = "CONTROL_ROW"
    else:
        row["bias_sensitive"] = bool(
            excludes_zero(se_ci) != excludes_zero(shifted_ci)
        )
        row["bias_sensitive_note"] = "SHIFT_HEURISTIC_NOT_SPLIT_HALF"

    users = se_user_differences(case, cid, receiver["id"], records, complete)
    if users is not None and users.size:
        row["se_user_p10"] = float(np.quantile(users, 0.10))
        row["se_user_p50"] = float(np.quantile(users, 0.50))
        row["se_user_p90"] = float(np.quantile(users, 0.90))
        row["se_negative_user_fraction"] = float(np.mean(users < 0.0))

    if status == "FAIL_INTEGRITY":
        row["se_label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
    elif status == "INCOMPLETE_RESOURCE":
        row["se_label"] = "INCOMPLETE_RESOURCE"
    elif role == "STRUCTURAL_EQUALITY_CONTROL":
        row["se_label"] = "CONTROL_EQUALITY_CHECKED"
    elif differences.size < 2:
        row["se_label"] = "INCONCLUSIVE"
    elif se_ci[0] is not None and se_ci[0] > 0.0:
        row["se_label"] = "DETECTED_POSITIVE"
    elif se_ci[1] is not None and se_ci[1] < 0.0:
        row["se_label"] = "DETECTED_NEGATIVE"
    else:
        row["se_label"] = "INCONCLUSIVE"
    return row
