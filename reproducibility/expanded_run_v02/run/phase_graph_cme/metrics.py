"""Use-and-then-forget uplink SE and paired summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UatFResult:
    sinr: FloatArray
    spectral_efficiency: FloatArray
    coherent_gain: NDArray[np.complex128]
    denominator: FloatArray


def uatf_uplink(
    channel_samples: object,
    combiner_samples: object,
    payload_powers: object,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
) -> UatFResult:
    """Estimate the standard UatF bound from paired channel/combiner samples.

    Both sample arrays have shape ``(realizations, antennas, users)``.
    """

    H = np.asarray(channel_samples, dtype=np.complex128)
    V = np.asarray(combiner_samples, dtype=np.complex128)
    powers = np.asarray(payload_powers, dtype=np.float64)
    if H.ndim != 3 or V.shape != H.shape:
        raise ValueError("channel_samples and combiner_samples must share shape (S,M,K)")
    sample_count, _, user_count = H.shape
    if sample_count < 1 or powers.shape != (user_count,):
        raise ValueError("payload_powers must have one value per user")
    if np.any(powers < 0.0) or noise_variance <= 0.0:
        raise ValueError("powers must be nonnegative and noise variance positive")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(V)):
        raise ValueError("channel and combiner samples must be finite")
    if not np.all(np.isfinite(powers)) or not np.isfinite(noise_variance):
        raise ValueError("powers and noise variance must be finite")
    if not 0 <= tau_p < tau_c:
        raise ValueError("require 0 <= tau_p < tau_c")

    inner = np.einsum("smk,smi->ski", V.conj(), H, optimize=True)
    coherent = np.mean(np.diagonal(inner, axis1=1, axis2=2), axis=0)
    second = np.mean(np.abs(inner) ** 2, axis=0)
    norm = np.mean(np.sum(np.abs(V) ** 2, axis=1), axis=0)
    numerator = powers * np.abs(coherent) ** 2
    denominator = second @ powers - numerator + float(noise_variance) * norm
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("nonfinite UatF numerator or denominator")
    if np.any(denominator <= 0.0):
        raise FloatingPointError("nonpositive UatF denominator")
    sinr = numerator / denominator
    prelog = 1.0 - float(tau_p) / float(tau_c)
    se = prelog * np.log2(1.0 + sinr)
    if not np.all(np.isfinite(sinr)) or not np.all(np.isfinite(se)):
        raise FloatingPointError("nonfinite UatF SINR or spectral efficiency")
    return UatFResult(
        sinr=sinr.astype(np.float64),
        spectral_efficiency=se.astype(np.float64),
        coherent_gain=coherent,
        denominator=denominator.astype(np.float64),
    )


def relative_change_percent(proposed: object, baseline: object) -> FloatArray:
    p = np.asarray(proposed, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if p.shape != b.shape or np.any(b == 0.0):
        raise ValueError("paired arrays must share shape and baseline must be nonzero")
    return (100.0 * (p - b) / b).astype(np.float64)
