"""Packet-local statistic helpers for R2C01_DIRA_SHAPE_v08."""

from __future__ import annotations

import math

import numpy as np


def complete_topologies(case: dict, records: dict) -> list[int]:
    """Return topologies for which every cell of the case has a record."""

    return [
        t
        for t in range(int(case["topologies"]))
        if all((case["id"], cid, t) in records for cid in case["cells"])
    ]


def t_interval(values: np.ndarray, alpha: float) -> tuple[list, float | None, float | None]:
    """Student-t interval for the mean of ``values``."""

    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return [None, None], None, None
    mean = float(x.mean())
    if n < 2:
        return [None, None], mean, None
    se = float(x.std(ddof=1) / math.sqrt(n))
    if not np.isfinite(se):
        return [None, None], mean, se
    if se == 0.0:
        return [mean, mean], mean, 0.0
    from scipy import stats

    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return [mean - critical * se, mean + critical * se], mean, se


def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator, resamples: int, alpha: float
) -> list:
    """Percentile bootstrap interval for the mean of ``values``."""

    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return [None, None]
    if n == 1:
        return [float(x[0]), float(x[0])]
    indices = rng.integers(0, n, size=(int(resamples), n))
    boot = x[indices].mean(axis=1)
    return np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear").tolist()


def excludes_zero(interval: list) -> bool:
    low, high = interval
    return low is not None and high is not None and (low > 0.0 or high < 0.0)


def mse_ratio_values(case: dict, cid: str, records: dict, complete: list[int]) -> np.ndarray:
    values = []
    for t in complete:
        rec = records[(case["id"], cid, t)]
        denominator = float(rec["mse"]["B1"])
        values.append(
            float(rec["mse"]["B2"]) / denominator if denominator > 0.0 else float("nan")
        )
    return np.asarray(values, dtype=np.float64)


def se_difference_values(
    case: dict, cid: str, receiver_id: str, records: dict, complete: list[int]
) -> np.ndarray:
    values = []
    for t in complete:
        rec = records[(case["id"], cid, t)]
        b1 = np.asarray(rec["se"][receiver_id]["B1"]["full"], dtype=np.float64)
        b2 = np.asarray(rec["se"][receiver_id]["B2"]["full"], dtype=np.float64)
        values.append(float(np.mean(b1 - b2)))
    return np.asarray(values, dtype=np.float64)


def se_even_odd_difference_values(
    case: dict, cid: str, receiver_id: str, records: dict, complete: list[int]
) -> np.ndarray:
    """Descriptive even/odd split-half difference of the SE contrast.

    This is reported next to the shift heuristic. It is not a gate and it does
    not replace the pre-registered SE label.
    """

    values = []
    for t in complete:
        rec = records[(case["id"], cid, t)]
        b1 = rec["se"][receiver_id]["B1"]
        b2 = rec["se"][receiver_id]["B2"]
        even_gap = float(b1["even_mean"]) - float(b2["even_mean"])
        odd_gap = float(b1["odd_mean"]) - float(b2["odd_mean"])
        values.append(even_gap - odd_gap)
    return np.asarray(values, dtype=np.float64)


def se_user_differences(
    case: dict, cid: str, receiver_id: str, records: dict, complete: list[int]
) -> np.ndarray | None:
    blocks = []
    for t in complete:
        rec = records[(case["id"], cid, t)]
        b1 = np.asarray(rec["se"][receiver_id]["B1"]["full"], dtype=np.float64)
        b2 = np.asarray(rec["se"][receiver_id]["B2"]["full"], dtype=np.float64)
        blocks.append(b1 - b2)
    if not blocks:
        return None
    return np.concatenate(blocks)
