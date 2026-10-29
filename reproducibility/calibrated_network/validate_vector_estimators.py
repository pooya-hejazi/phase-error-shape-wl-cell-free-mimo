"""Targeted analytical/vector-estimator validation (no frozen packet rerun)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "expanded_run_v02/run"))
from shape_model import ShapePilotGroupModel
from shape_estimators import clmmse_shape, widely_linear_mmse_shape

M1 = 0.90
C = -0.174422132908


def model_for(ell: np.ndarray, cov: np.ndarray, beta: np.ndarray, noise: np.ndarray, m1, m2):
    return ShapePilotGroupModel(
        los=ell, covariances=cov, beta=beta, noise_covariance=noise,
        m1=np.full(ell.shape[1], m1), m2=np.full(ell.shape[1], m2),
    )


def main() -> None:
    rng = np.random.default_rng(20261115)
    n, p = 2, 2
    ell = np.array([[1.0 + 0.2j, 0.4 - 0.3j], [0.1j, 0.7 + 0.2j]])
    cov = np.array([np.eye(n) * 0.05, np.eye(n) * 0.08])
    beta = np.array([1.2, 0.9])
    noise = np.eye(n) * 0.01
    model = model_for(ell, cov, beta, noise, M1, M1 * M1 + C)
    cov_y = noise + sum(
        beta[i] ** 2 * (cov[i] + (1.0 - M1 * M1) * np.outer(ell[:, i], ell[:, i].conj()))
        for i in range(p)
    )
    pseudo_y = sum(beta[i] ** 2 * C * np.outer(ell[:, i], ell[:, i]) for i in range(p))
    augmented = np.block([[cov_y, pseudo_y], [pseudo_y.conj(), cov_y.conj()]])
    eig = float(np.min(np.linalg.eigvalsh(augmented)))

    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b1 = widely_linear_mmse_shape(model, y, 0)
    b2 = clmmse_shape(model, y, 0)
    zero = model_for(ell, cov, beta, noise, M1, M1 * M1)
    b1_zero = widely_linear_mmse_shape(zero, y, 0)
    scalar_gap = float(np.linalg.norm(b1_zero - b2))

    phase = np.exp(1j * 0.37)
    rotated = model_for(phase * ell, cov, beta, noise, M1, M1 * M1 + C)
    rotated_estimate = widely_linear_mmse_shape(rotated, phase * y, 0)
    equiv_error = float(np.linalg.norm(rotated_estimate - phase * b1))

    mse_b1, mse_b2 = [], []
    for _ in range(4000):
        y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        mse_b1.append(float(np.linalg.norm(widely_linear_mmse_shape(model, y, 0)) ** 2))
        mse_b2.append(float(np.linalg.norm(clmmse_shape(model, y, 0)) ** 2))

    payload = {
        "seed": 20261115,
        "augmented_min_eigenvalue": eig,
        "augmented_psd": bool(eig > 0),
        "c0_reduction_gap": scalar_gap,
        "phase_equivariance_error": equiv_error,
        "vector_shape": list(ell.shape),
        "covariance_hermitian_error": float(np.max(np.abs(cov_y - cov_y.conj().T))),
        "pseudo_symmetry_error": float(np.max(np.abs(pseudo_y - pseudo_y.T))),
        "mc_relative_std_B1": float(np.std(mse_b1) / np.mean(mse_b1)),
        "mc_relative_std_B2": float(np.std(mse_b2) / np.mean(mse_b2)),
    }
    out = Path(__file__).resolve().parent / "output" / "vector_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
