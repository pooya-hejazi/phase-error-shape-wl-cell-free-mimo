"""Primary ordering, collapse and B2-equality checks for R2C01_DIRA_SHAPE_v08.

The label set is defined once here and mirrored in SCOPE.md section 6.  The
primary ordering label depends only on the three predicted signs and the three
Bonferroni intervals.  The effect-size label is separate; the combined label is
reported for convenience and must not be confused with the ordering test.
"""

from __future__ import annotations

import math

import numpy as np

from shape_stats import mse_ratio_values


def _status_label(status: str) -> str:
    if status == "FAIL_INTEGRITY":
        return "NOT_INTERPRETED_FAIL_INTEGRITY"
    if status == "INCOMPLETE_RESOURCE":
        return "INCOMPLETE_RESOURCE"
    if status != "COMPLETE":
        return "NOT_INTERPRETED"
    return "COMPLETE"


def ordering_summary(
    case: dict,
    receiver: dict,
    records: dict,
    complete: list[int],
    stats: dict,
    status: str,
    rng: np.random.Generator,
) -> dict:
    family = list(stats["primary_family"])
    result = {
        "antenna_case": case["id"],
        "receiver": receiver["id"],
        "metric": stats["primary_metric"],
        "alpha": float(stats["alpha"]),
        "adjustment": stats["primary_adjustment"],
        "topologies": len(complete),
        "cells": family,
        "contrasts": [],
        "effect_size": None,
        "effect_size_interval": [None, None],
        "effect_size_threshold": float(stats["effect_size_threshold"]),
        "effect_size_met": None,
        "effect_size_label": "EFFECT_SIZE_UNDEFINED",
        "label": _status_label(status),
        "combined_label": None,
        "note": None,
    }
    if status != "COMPLETE":
        result["note"] = "NOT_INTERPRETED_OUTSIDE_COMPLETE_STATUS"
        result["combined_label"] = result["label"]
        return result
    if len(complete) < int(stats["minimum_complete_topologies_for_an_interval"]):
        result["label"] = "INSUFFICIENT_TOPOLOGIES"
        result["combined_label"] = "INSUFFICIENT_TOPOLOGIES"
        result["note"] = "FEWER_THAN_TWO_COMPLETE_TOPOLOGIES"
        return result

    ratios = {cid: mse_ratio_values(case, cid, records, complete) for cid in family}
    if any(values.size != len(complete) for values in ratios.values()):
        result["label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        result["combined_label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        result["note"] = "MISSING_MSE_RATIO_VALUES"
        return result
    if any(not np.all(np.isfinite(values)) for values in ratios.values()):
        result["label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        result["combined_label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        result["note"] = "NONFINITE_MSE_RATIO"
        return result

    n = len(complete)
    indices = rng.integers(0, n, size=(int(stats["bootstrap_resamples"]), n))
    pairs = list(stats["primary_pairwise"])
    signs = list(stats["primary_predicted_signs"])
    alpha_p = float(stats["alpha"]) / len(pairs)
    all_sign_ok = True
    all_interval_ok = True
    for (a, b), predicted in zip(pairs, signs):
        difference = ratios[a] - ratios[b]
        mean = float(difference.mean())
        boot = difference[indices].mean(axis=1)
        interval = np.quantile(
            boot, [alpha_p / 2.0, 1.0 - alpha_p / 2.0], method="linear"
        ).tolist()
        sign_ok = (mean > 0.0) if predicted == "POSITIVE" else (mean < 0.0)
        interval_ok = (
            (interval[0] > 0.0) if predicted == "POSITIVE" else (interval[1] < 0.0)
        )
        all_sign_ok = all_sign_ok and sign_ok
        all_interval_ok = all_interval_ok and interval_ok
        result["contrasts"].append(
            {
                "a": a,
                "b": b,
                "predicted": predicted,
                "mean_difference": mean,
                "interval": interval,
                "alpha_adjusted": alpha_p,
                "sign_ok": bool(sign_ok),
                "interval_ok": bool(interval_ok),
            }
        )

    uniform_id, gaussian_id, binary_id = family
    boot_gaussian = ratios[gaussian_id][indices].mean(axis=1)
    boot_binary = ratios[binary_id][indices].mean(axis=1)
    point_denominator = float(ratios[gaussian_id].mean()) - 1.0
    point_effect = (
        (float(ratios[binary_id].mean()) - 1.0) / point_denominator
        if point_denominator > 0.0
        else None
    )
    boot_denominator = boot_gaussian - 1.0
    valid = np.isfinite(boot_denominator) & (boot_denominator > 0.0)
    if np.any(valid):
        boot_effect = (boot_binary[valid] - 1.0) / boot_denominator[valid]
        effect_interval = np.quantile(
            boot_effect, [alpha_p / 2.0, 1.0 - alpha_p / 2.0], method="linear"
        ).tolist()
    else:
        effect_interval = [None, None]
    result["effect_size"] = point_effect
    result["effect_size_interval"] = effect_interval
    if point_effect is None:
        result["effect_size_met"] = None
        result["effect_size_label"] = "EFFECT_SIZE_UNDEFINED"
    elif point_effect >= float(stats["effect_size_threshold"]):
        result["effect_size_met"] = True
        result["effect_size_label"] = "EFFECT_SIZE_MET"
    else:
        result["effect_size_met"] = False
        result["effect_size_label"] = "EFFECT_SIZE_NOT_MET"

    if all_sign_ok and all_interval_ok:
        result["label"] = "ORDER_CONFIRMED"
    else:
        result["label"] = "ORDER_NOT_CONFIRMED"

    if result["label"] == "ORDER_CONFIRMED":
        if result["effect_size_label"] == "EFFECT_SIZE_MET":
            result["combined_label"] = "ORDER_CONFIRMED_EFFECT_MET"
        elif result["effect_size_label"] == "EFFECT_SIZE_NOT_MET":
            result["combined_label"] = "ORDER_CONFIRMED_EFFECT_BELOW_THRESHOLD"
        else:
            result["combined_label"] = "ORDER_CONFIRMED_EFFECT_UNDEFINED"
    else:
        result["combined_label"] = result["label"]
    return result


def collapse_summary(
    case: dict,
    receiver: dict,
    records: dict,
    complete: list[int],
    stats: dict,
    status: str,
    rng: np.random.Generator,
) -> dict:
    cells = list(stats["collapse_cells"])
    result = {
        "antenna_case": case["id"],
        "receiver": receiver["id"],
        "cells": cells,
        "topologies": len(complete),
        "effect_size": None,
        "effect_size_interval": [None, None],
        "threshold": float(stats["collapse_threshold"]),
        "label": _status_label(status),
        "note": None,
    }
    if status != "COMPLETE":
        result["note"] = "NOT_INTERPRETED_OUTSIDE_COMPLETE_STATUS"
        return result
    if len(complete) < int(stats["minimum_complete_topologies_for_an_interval"]):
        result["label"] = "INSUFFICIENT_TOPOLOGIES"
        return result
    ratios = {cid: mse_ratio_values(case, cid, records, complete) for cid in cells}
    if any(not np.all(np.isfinite(values)) for values in ratios.values()):
        result["label"] = "NOT_INTERPRETED_FAIL_INTEGRITY"
        result["note"] = "NONFINITE_MSE_RATIO"
        return result
    uniform_id, gaussian_id, binary_id = cells
    denominator = float(ratios[gaussian_id].mean()) - 1.0
    effect = (
        (float(ratios[binary_id].mean()) - 1.0) / denominator
        if denominator > 0.0
        else None
    )
    result["effect_size"] = effect
    n = len(complete)
    indices = rng.integers(0, n, size=(int(stats["bootstrap_resamples"]), n))
    boot_gaussian = ratios[gaussian_id][indices].mean(axis=1) - 1.0
    boot_binary = ratios[binary_id][indices].mean(axis=1) - 1.0
    valid = np.isfinite(boot_gaussian) & (boot_gaussian > 0.0)
    if np.any(valid):
        boot_effect = boot_binary[valid] / boot_gaussian[valid]
        result["effect_size_interval"] = np.quantile(
            boot_effect, [0.025, 0.975], method="linear"
        ).tolist()
    if effect is None:
        result["label"] = "COLLAPSE_EFFECT_UNDEFINED"
    elif effect <= float(stats["collapse_threshold"]):
        result["label"] = "COLLAPSE_CONSISTENT"
    else:
        result["label"] = "COLLAPSE_NOT_CONSISTENT"
    return result


def b2_equality_summary(
    cases: list[dict],
    cells: dict,
    records: dict,
    complete_by_case: dict,
    sigma: float,
) -> dict:
    """Check that B2 (which uses m1 only) is invariant to the shape at fixed m1."""

    groups = []
    evaluated = 0
    failures = 0
    for case in cases:
        complete = complete_by_case[case["id"]]
        buckets: dict[tuple, list[str]] = {}
        for cid in case["cells"]:
            cell = cells[cid]
            if cell["shape"] in {"uniform", "wrapped_gaussian", "binary"}:
                buckets.setdefault(
                    (float(cell["kappa"]), float(cell["m1"])), []
                ).append(cid)
        for (kappa, m1), cids in sorted(buckets.items()):
            by_shape = {cells[cid]["shape"]: cid for cid in cids}
            if set(by_shape) != {"uniform", "wrapped_gaussian", "binary"}:
                continue
            if len(complete) < 2:
                groups.append(
                    {
                        "antenna_case": case["id"],
                        "kappa": kappa,
                        "m1": m1,
                        "cells": by_shape,
                        "topologies": len(complete),
                        "pass": None,
                        "note": "INSUFFICIENT_TOPOLOGIES",
                    }
                )
                continue
            mse = {
                shape: np.asarray(
                    [
                        float(records[(case["id"], cid, t)]["mse"]["B2"])
                        for t in complete
                    ]
                )
                for shape, cid in by_shape.items()
            }
            scale = float(np.mean(np.concatenate(list(mse.values()))))
            floor = 1e-12 * max(scale, 1.0)
            pairs = []
            for a, b in (
                ("uniform", "wrapped_gaussian"),
                ("uniform", "binary"),
                ("wrapped_gaussian", "binary"),
            ):
                difference = mse[a] - mse[b]
                mean = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(difference.size))
                bound = sigma * se + floor
                passed = abs(mean) <= bound
                evaluated += 1
                if not passed:
                    failures += 1
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "mean_difference": mean,
                        "se": se,
                        "tolerance": bound,
                        "z": (mean / se) if se > 0.0 else None,
                        "pass": bool(passed),
                    }
                )
            groups.append(
                {
                    "antenna_case": case["id"],
                    "kappa": kappa,
                    "m1": m1,
                    "cells": by_shape,
                    "topologies": len(complete),
                    "pairs": pairs,
                    "pass": all(pair["pass"] for pair in pairs),
                }
            )
    all_pass = (
        True
        if evaluated > 0 and failures == 0
        else (False if failures > 0 else None)
    )
    return {
        "sigma": float(sigma),
        "evaluated_pairs": evaluated,
        "failures": failures,
        "all_pass": all_pass,
        "groups": groups,
    }
