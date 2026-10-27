"""MR and positively regularized centralized receive combiners."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def _channel_matrix(estimates: object) -> ComplexArray:
    matrix = np.asarray(estimates, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("estimates must have shape (antennas, users)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("estimates must be finite")
    return matrix


def mr_combiner(estimates: object) -> ComplexArray:
    return _channel_matrix(estimates).copy()


def rzf_combiner(estimates: object, regularization: float) -> ComplexArray:
    H = _channel_matrix(estimates)
    lam = float(regularization)
    if not np.isfinite(lam) or not lam > 0.0:
        raise ValueError("regularization must be finite and strictly positive")
    gram = H.conj().T @ H + lam * np.eye(H.shape[1], dtype=np.complex128)
    return H @ np.linalg.solve(gram, np.eye(H.shape[1], dtype=np.complex128))
