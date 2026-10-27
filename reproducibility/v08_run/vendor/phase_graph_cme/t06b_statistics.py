"""Registered T06B finite-frame analysis; no physical simulation or gate promotion."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any, Mapping
import numpy as np
from . import baseline_precision_e0_centered as centered

RECEIVERS = ("MR", "RZF")
SPLITS = ("full", "even", "odd")
PRIMARY = "CORE_K5_D90"
SCHEMA = "r2c01-t06b-statistics/1"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _check_contract(cfg: Mapping[str, Any]) -> None:
    expected = {"q_diagonal_authoritative": True, "residual_correction": True,
        "divisor": "S_NOT_S_MINUS_1", "clipping": False, "primary_reducer": "NEUMAIER",
        "check_reducers": ["FSUM", "NUMPY"], "primitive_atol": centered.PRIMITIVE_ATOL,
        "primitive_rtol": centered.PRIMITIVE_RTOL,
        "pairwise_se_oracle_bound_max": centered.ABSOLUTE_SE_ORACLE_CEILING,
        "witness_rounding_se_bound_max": centered.ABSOLUTE_SE_ORACLE_CEILING,
        "log_series_terms": centered.ATANH_SERIES_TERMS,
        "log_interval_width_max": float(centered.MAX_EXACT_INTERVAL_WIDTH)}
    if any(cfg["numerical_contract"].get(k) != v for k, v in expected.items()):
        raise ValueError("T06B numerical contract differs from frozen certifier")


def _compact(value: Mapping[str, Any]) -> dict[str, Any]:
    oracle, errors = value["oracle"], value["errors"]
    return {"user_index": value["user_index"], "sample_count": value["sample_count"],
        "status": value["status"], "failure_reasons": value["failure_reasons"],
        "full_diagnostic_sha256": _digest(value), "witness_sha256": oracle["shared_witness_sha256"],
        "gates": value["gates"], "oracle_bounds": {
            "se_pair": errors["se_pair"], "se_oracle_by_path": errors["se_oracle_by_path"],
            "centered_witness_rounding": errors["centered_witness_rounding"],
            "log2_interval": oracle["log2_one_plus_sinr_interval"],
            "se_interval": oracle["spectral_efficiency_interval"],
            "rounded_log2_interval": oracle["rounded_witness_log2_one_plus_sinr_interval"],
            "rounded_se_interval": oracle["rounded_witness_spectral_efficiency_interval"]}}


def analyze_cell(c: np.ndarray, q: np.ndarray, r: np.ndarray,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Certify all method/MR,RZF/full,even,odd caches without deleting failures.

    c/r shape (method,2,frame,user); q adds a final user axis. Stored qkk is
    authoritative, never replaced with abs(c)**2. This function never draws RNG.
    """
    _check_contract(cfg)
    c, q, r = np.asarray(c), np.asarray(q), np.asarray(r)
    methods, users = list(cfg["methods"]), int(cfg["system"]["users"])
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("method identities must be unique and nonempty")
    if c.dtype != np.complex128 or q.dtype != np.float64 or r.dtype != np.float64:
        raise ValueError("c/q/r must retain complex128/float64/float64 dtype")
    if (c.ndim != 4 or c.shape[:2] != (len(methods), 2) or c.shape[-1] != users
            or r.shape != c.shape or q.shape != c.shape + (users,)):
        raise ValueError("c/q/r cache shapes differ from registered axes")
    samples = c.shape[2]
    if samples < 2 or samples % 2:
        raise ValueError("full/even/odd requires a positive even frame count")
    powers = np.full(users, float(cfg["system"]["payload_power"]), dtype=np.float64)
    noise = float(cfg["system"]["payload_noise_variance"])
    tau_p, tau_c = (int(cfg["system"][k]) for k in ("pilot_symbols", "coherence_symbols"))
    rows = []
    for mi, method in enumerate(methods):
        for ri, receiver in enumerate(RECEIVERS):
            base = centered.SufficientContributionCache(c[mi, ri], q[mi, ri], r[mi, ri])
            for split in SPLITS:
                indices = np.arange(samples, dtype=np.int64)
                if split != "full":
                    indices = indices[indices % 2 == (split == "odd")]
                mask = np.zeros(samples, dtype=bool)
                mask[indices] = True
                sub = base.select(mask)
                cache = centered.build_centered_contribution_cache(sub)
                args = (cache, powers, noise, tau_p, tau_c)
                stats = {"neumaier": centered.centered_neumaier_statistics(*args),
                    "fsum": centered.fixed_order_fsum_statistics(*args),
                    "numpy": centered.ordinary_direct_statistics(*args)}
                diagnostics = []
                certificate = centered.certify_uatf_statistics(
                    stats["neumaier"], stats["fsum"], stats["numpy"], contribution_cache=sub,
                    centered_cache=cache, powers=powers, noise_variance=noise, tau_p=tau_p,
                    tau_c=tau_c, context={"condition_index": 0, "condition_id": "T06B_CELL",
                        "receiver_index": mi * 2 + ri, "receiver_id": method + ":" + receiver, "split": split},
                    diagnostic_sink=lambda x: diagnostics.append(_compact(x)), raise_on_failure=False)
                se = np.asarray(stats["neumaier"]["spectral_efficiency"])
                finite = bool(np.all(np.isfinite(se)))
                rows.append({"method": method, "receiver": receiver, "split": split,
                    "frame_indices": indices.tolist(), "sample_count": int(indices.size),
                    "status": certificate["status"], "se_per_user": se.tolist(),
                    "mean_user_se": float(np.mean(se)) if finite else None,
                    "p05_user_se": float(np.quantile(se, .05, method="linear")) if finite else None,
                    "outage_fraction": float(np.mean(se < float(cfg["decision"]["outage_se_threshold"]))) if finite else None,
                    "reducer_se": {key: np.asarray(v["spectral_efficiency"]).tolist() for key, v in stats.items()},
                    "certificate": certificate, "diagnostics": diagnostics})
    return _safe({"schema_version": SCHEMA, "status": "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL_INTEGRITY",
        "row_count": len(rows), "diagnostic_count": sum(len(x["diagnostics"]) for x in rows), "rows": rows})


def _failure(label: str, reasons: list[str]) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "decision": label, "reasons": reasons,
        "endpoints": [], "scientific_gates_promoted": False,
        "boundary": "FINITE_FRAME_EXPLORATORY_NOT_POPULATION_OR_NOVELTY_CERTIFICATION"}


def screen(cells: list[Mapping[str, Any]], cfg: Mapping[str, Any], bootstrap_seed: int) -> dict[str, Any]:
    """Explicit offline paired analysis; caller must authorize reserved seed use."""
    methods, count, dec = list(cfg["methods"]), int(cfg["topologies"]), cfg["decision"]
    conditions = {c["id"]: int(c["frames_per_topology"]) for c in cfg["conditions"]}
    comparators = list(dec["gain_margins_bpshz_per_user"])
    if (len(comparators) != 6 or dec["tail_guard_comparators"] != comparators or dec["endpoint_count"] != 20
            or dec["lower_percentile"] != dec["nominal_family_alpha"] / 40
            or dec["upper_percentile"] != 1 - dec["nominal_family_alpha"] / 40
            or cfg["sampling"]["quantile_method"] != "linear" or cfg["sampling"]["bootstrap_generator"] != "PCG64"):
        raise ValueError("T06B endpoint/multiplicity/bootstrap contract mismatch")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, (int, np.integer)) or not 0 <= int(bootstrap_seed) < 2**63:
        raise ValueError("bootstrap seed must be a nonnegative signed63 integer")
    expected = {(j, cond) for j in range(count) for cond in conditions}
    mapped, reasons = {}, []
    for cell in cells:
        key = (cell.get("topology_index"), cell.get("condition"))
        if key not in expected or key in mapped:
            reasons.append("INVALID_OR_DUPLICATE_CELL_ID")
            continue
        mapped[key] = cell
        if cell.get("analysis", {}).get("status") != "PASS":
            reasons.append(f"NUMERICAL_INTEGRITY:{key}")
        if key[1] == "NO_LOS" and cell.get("negative_control_pass") is not True:
            reasons.append(f"NEGATIVE_CONTROL:{key}")
    if reasons:
        return _failure("FAIL_INTEGRITY", reasons)
    if set(mapped) != expected:
        return _failure("INCOMPLETE_RESOURCE", ["INCOMPLETE_REGISTERED_GRID"])
    indexed = {}
    expected_rows = {(m, recv, split) for m in methods for recv in RECEIVERS for split in SPLITS}
    for (j, cond), cell in mapped.items():
        rows, seen = cell["analysis"].get("rows", []), set()
        if cell["analysis"].get("row_count") != len(expected_rows):
            reasons.append(f"WRONG_ROW_COUNT:{j}:{cond}")
        if cell["analysis"].get("diagnostic_count") != len(expected_rows) * int(cfg["system"]["users"]):
            reasons.append(f"WRONG_DIAGNOSTIC_COUNT:{j}:{cond}")
        for row in rows:
            rid = (row.get("method"), row.get("receiver"), row.get("split"))
            if rid in seen or rid not in expected_rows:
                reasons.append(f"INVALID_OR_DUPLICATE_ROW:{j}:{cond}")
                continue
            seen.add(rid)
            frames = list(range(conditions[cond]))
            if rid[2] != "full":
                frames = frames[0 if rid[2] == "even" else 1::2]
            se = np.asarray(row.get("se_per_user"), dtype=float)
            if (row.get("status") != "PASS" or row.get("frame_indices") != frames or row.get("sample_count") != len(frames)
                    or se.shape != (int(cfg["system"]["users"]),) or not np.all(np.isfinite(se)) or np.any(se < 0)):
                reasons.append(f"INVALID_ROW:{j}:{cond}:{rid}")
                continue
            metrics = {"mean_user_se": float(np.mean(se)), "p05_user_se": float(np.quantile(se, .05, method="linear")),
                "outage_fraction": float(np.mean(se < float(dec["outage_se_threshold"])))}
            if any(not math.isfinite(value) or row.get(k) != value for k, value in metrics.items()):
                reasons.append(f"METRIC_CACHE_MISMATCH:{j}:{cond}:{rid}")
            indexed[(j, cond, *rid)] = row
        if seen != expected_rows:
            reasons.append(f"MISSING_METHOD_RECEIVER_SPLIT:{j}:{cond}")
        costs = cell.get("costs", {})
        if set(costs) != set(methods) or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in costs.values()):
            reasons.append(f"INVALID_STANDALONE_COSTS:{j}:{cond}")
    if reasons:
        return _failure("FAIL_INTEGRITY", reasons)
    # Finite per-frame seconds do not by themselves ensure finite accumulated
    # denominators. Validate sums before creating a single bootstrap draw.
    for method in methods:
        costs = np.asarray([mapped[(j, PRIMARY)]["costs"][method] for j in range(count)])
        if not np.isfinite(np.sum(costs)) or np.sum(costs) <= 0:
            return _failure("FAIL_INTEGRITY", ["NONFINITE_OR_NONPOSITIVE_SUMMED_COST"])
    replicates = int(cfg["sampling"]["bootstrap_replicates"])
    if count < 2 or replicates < 2:
        raise ValueError("at least two topology clusters/bootstrap replicates required")
    indices = np.random.Generator(np.random.PCG64(int(bootstrap_seed))).integers(0, count, size=(replicates, count), dtype=np.int64)
    def values(method, field, split="full"):
        return np.asarray([indexed[(j, PRIMARY, method, "RZF", split)][field] for j in range(count)], dtype=float)
    endpoints = []
    def endpoint(family, comparator, paired, margin, direction, ceiling, numerator=None, denominator=None):
        if paired is not None:
            estimate, resampled = float(np.mean(paired)), np.mean(paired[indices], axis=1)
            payload = {"topology_values": paired.tolist()}
        else:
            estimate = float(np.sum(numerator) / np.sum(denominator))
            resampled = np.sum(numerator[indices], axis=1) / np.sum(denominator[indices], axis=1)
            payload = {"numerator_costs": numerator.tolist(), "denominator_costs": denominator.tolist()}
        if not math.isfinite(estimate) or not np.all(np.isfinite(resampled)):
            reasons.append("NONFINITE_BOOTSTRAP_STATISTIC:" + family + ":" + comparator)
            return
        lo, hi = map(float, np.quantile(resampled, [dec["lower_percentile"], dec["upper_percentile"]], method="linear"))
        halfwidth = (hi - lo) / 2
        endpoints.append({"id": f"{family}:{comparator}", "family": family, "comparator": comparator, **payload,
            "estimate": estimate, "lower": lo, "upper": hi, "halfwidth": halfwidth, "ceiling": float(ceiling),
            "margin": float(margin), "direction": direction, "precision_pass": bool(halfwidth <= ceiling),
            "margin_pass": bool(lo >= margin if direction == "lower" else hi <= margin),
            "margin_rejected": bool(hi < margin if direction == "lower" else lo > margin)})
    precision = dec["precision_halfwidth"]
    for comparator, margin in dec["gain_margins_bpshz_per_user"].items():
        endpoint("gain", comparator, values("TRANSFER", "mean_user_se") - values(comparator, "mean_user_se"), margin, "lower", precision["gain"])
    endpoint("v3_difference", "V3", values("TRANSFER", "mean_user_se") - values("V3", "mean_user_se"), dec["v3_noninferiority_margin"], "lower", precision["v3_difference"])
    for family, field, margin, direction in (("p05", "p05_user_se", dec["p05_difference_min"], "lower"), ("outage", "outage_fraction", dec["outage_increase_max"], "upper")):
        for comparator in comparators:
            endpoint(family, comparator, values("TRANSFER", field) - values(comparator, field), margin, direction, precision[family])
    endpoint("cost_ratio", "V3", None, dec["cost_ratio_v3_max"], "upper", precision["cost_ratio"],
        np.asarray([mapped[(j, PRIMARY)]["costs"]["TRANSFER"] for j in range(count)]),
        np.asarray([mapped[(j, PRIMARY)]["costs"]["V3"] for j in range(count)]))
    if reasons:
        return _failure("FAIL_INTEGRITY", reasons)
    sensitivity = []
    for method in methods:
        gaps = np.abs(values(method, "mean_user_se", "even") - values(method, "mean_user_se", "odd"))
        sensitivity.append({"kind": "per_topology_method_half_gap", "method": method, "values": gaps.tolist(),
            "limit": dec["per_topology_method_mean_se_half_gap_max"], "pass": bool(np.all(gaps <= dec["per_topology_method_mean_se_half_gap_max"]))})
    for comparator in comparators + ["V3"]:
        full = float(np.mean(values("TRANSFER", "mean_user_se") - values(comparator, "mean_user_se")))
        for split in ("even", "odd"):
            split_value = float(np.mean(values("TRANSFER", "mean_user_se", split) - values(comparator, "mean_user_se", split)))
            drift = abs(split_value - full)
            sensitivity.append({"kind": "pooled_split_contrast_drift", "comparator": comparator, "split": split,
                "full": full, "split_value": split_value, "drift": drift, "limit": dec["pooled_split_contrast_drift_max"],
                "pass": bool(drift <= dec["pooled_split_contrast_drift_max"])})
    precision_pass = all(x["precision_pass"] for x in endpoints) and all(x["pass"] for x in sensitivity)
    if not precision_pass:
        label = "INCONCLUSIVE_PRECISION"
    elif all(x["margin_pass"] for x in endpoints):
        label = dec["positive_label"]
    elif any(x["margin_rejected"] for x in endpoints):
        label = "NOT_SUPPORTED_IN_THIS_SCREEN"
    else:
        label = "INCONCLUSIVE_MARGIN_OVERLAP"
    return {"schema_version": SCHEMA, "decision": label, "reasons": [], "endpoints": endpoints,
        "endpoint_count": len(endpoints), "precision_pass": precision_pass, "split_sensitivity": sensitivity,
        "bootstrap": {"generator": "PCG64", "seed": str(int(bootstrap_seed)), "shape": [replicates, count],
            "indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes(order="C")).hexdigest(),
            "hash_encoding": "SHA256_LITTLE_ENDIAN_INT64_C_ORDER", "lower_percentile": dec["lower_percentile"],
            "upper_percentile": dec["upper_percentile"], "quantile_method": "linear",
            "nominal_family_alpha": dec["nominal_family_alpha"], "exact_finite_sample_familywise_coverage": False},
        "primary_condition": PRIMARY, "primary_receiver": "RZF", "negative_control_pass": True,
        "scientific_gates_promoted": False, "boundary": "FINITE_FRAME_EXPLORATORY_NOT_POPULATION_OR_NOVELTY_CERTIFICATION"}
