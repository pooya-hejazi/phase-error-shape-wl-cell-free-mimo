"""Convergence diagnostics that remain explicitly empirical."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChainDiagnostics:
    maximum_split_rhat: float
    split_rhat_real: FloatArray
    split_rhat_imag: FloatArray
    maximum_chain_mean_disagreement: float
    all_finite: bool


def _univariate_rhat(values: FloatArray) -> float:
    chains, draws = values.shape
    if chains < 2 or draws < 2:
        return float("nan")
    chain_means = np.mean(values, axis=1)
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(draws * np.var(chain_means, ddof=1))
    if within <= np.finfo(float).eps:
        return 1.0 if between <= np.finfo(float).eps else float("inf")
    variance = ((draws - 1.0) / draws) * within + between / draws
    return float(np.sqrt(max(variance / within, 0.0)))


def split_rhat(samples: object) -> tuple[FloatArray, FloatArray]:
    """Return split-Rhat separately for real and imaginary phasor components."""

    array = np.asarray(samples, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError("samples must have shape (chains, draws, variables)")
    chains, draws, variables = array.shape
    half = draws // 2
    if chains < 2 or half < 2:
        nan = np.full(variables, np.nan, dtype=np.float64)
        return nan.copy(), nan
    trimmed = array[:, : 2 * half, :]
    split = np.concatenate((trimmed[:, :half, :], trimmed[:, half:, :]), axis=0)
    real = np.array([_univariate_rhat(split[:, :, i].real) for i in range(variables)])
    imag = np.array([_univariate_rhat(split[:, :, i].imag) for i in range(variables)])
    return real, imag


def summarize_chains(samples: object) -> ChainDiagnostics:
    array = np.asarray(samples, dtype=np.complex128)
    real, imag = split_rhat(array)
    means = np.mean(array, axis=1)
    pooled = np.mean(means, axis=0)
    disagreement = float(np.max(np.abs(means - pooled), initial=0.0))
    finite = bool(np.all(np.isfinite(array)) and np.all(np.isfinite(real)) and np.all(np.isfinite(imag)))
    maximum = float(np.max(np.concatenate((real, imag))))
    return ChainDiagnostics(
        maximum_split_rhat=maximum,
        split_rhat_real=real,
        split_rhat_imag=imag,
        maximum_chain_mean_disagreement=disagreement,
        all_finite=finite,
    )
