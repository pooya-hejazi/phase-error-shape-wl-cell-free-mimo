"""Baseline-only identities and Ain-style centralized receiver utilities.

This module deliberately does not import the proposed inference controller.
It supports algebraic source reproduction and a clearly labelled qualitative
centralized-receiver check; it is not a proposed-method experiment runner.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from numpy.typing import NDArray

from .model import PilotGroupModel

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class BaselineConfigurationError(ValueError):
    """Raised when the baseline-only reproduction contract is violated."""


def source_rho(delta: object) -> FloatArray:
    """Evaluate Ain et al. (3), with the zero-width value defined by continuity."""

    widths = np.asarray(delta, dtype=np.float64)
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0) or np.any(widths > np.pi):
        raise ValueError("phase half-widths must be finite and lie in [0, pi]")
    result = np.ones_like(widths)
    nonzero = widths != 0.0
    result[nonzero] = np.sin(widths[nonzero]) / widths[nonzero]
    return result


def _conditional_covariances(model: PilotGroupModel) -> ComplexArray:
    rho = source_rho(model.delta)
    covariances = model.covariances.copy()
    for index in range(model.group_size):
        covariances[index] += (1.0 - rho[index] ** 2) * np.outer(
            model.los[:, index], model.los[:, index].conj()
        )
    return covariances


def ain_eq5_pilot_coordinates(
    model: PilotGroupModel,
    pilot_observation: object,
    target: int,
    *,
    tau_p: int,
) -> ComplexArray:
    """Evaluate Ain et al. (5) in the paper's unnormalized pilot coordinates.

    ``model.beta`` must equal ``sqrt(eta * tau_p)`` and be real nonnegative.
    ``model.noise_covariance`` is the paper's ``sigma^2 I`` term after the
    common factor ``tau_p`` has been removed from the pilot covariance.
    """

    if not isinstance(tau_p, int) or isinstance(tau_p, bool) or tau_p < 1:
        raise ValueError("tau_p must be a positive integer")
    target_index = int(target)
    if not 0 <= target_index < model.group_size:
        raise IndexError("target index is outside the pilot group")
    if np.max(np.abs(model.beta.imag)) > 1e-14 or np.any(model.beta.real < 0.0):
        raise ValueError("Ain source coordinates require real nonnegative pilot coefficients")
    y_pilot = np.asarray(pilot_observation, dtype=np.complex128)
    if y_pilot.shape != (model.antenna_count,) or not np.all(np.isfinite(y_pilot)):
        raise ValueError("pilot_observation must be a finite vector of shape (N,)")

    eta = np.square(model.beta.real) / float(tau_p)
    sqrt_eta = np.sqrt(eta)
    rho = source_rho(model.delta)
    means = model.los * rho[np.newaxis, :]
    sigma = _conditional_covariances(model)
    psi_inverse_argument = model.noise_covariance + np.einsum(
        "p,pij->ij", eta * float(tau_p), sigma, optimize=True
    )
    mean_pilot = means @ (sqrt_eta * float(tau_p))
    innovation = np.linalg.solve(psi_inverse_argument, y_pilot - mean_pilot)
    return means[:, target_index] + sqrt_eta[target_index] * sigma[target_index] @ innovation


def ain_ls_pilot_coordinates(
    pilot_observation: object, *, eta: float, tau_p: int
) -> ComplexArray:
    """Evaluate the collision-blind LS normalization in paper coordinates."""

    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    if not isinstance(tau_p, int) or isinstance(tau_p, bool) or tau_p < 1:
        raise ValueError("tau_p must be a positive integer")
    y_pilot = np.asarray(pilot_observation, dtype=np.complex128)
    if y_pilot.ndim != 1 or not np.all(np.isfinite(y_pilot)):
        raise ValueError("pilot_observation must be a finite vector")
    return y_pilot / (math.sqrt(eta) * float(tau_p))


def ain_error_covariance(model: PilotGroupModel, target: int) -> ComplexArray:
    """Return Ain et al. (6) after the normalized-statistic substitution."""

    target_index = int(target)
    if not 0 <= target_index < model.group_size:
        raise IndexError("target index is outside the pilot group")
    sigma = _conditional_covariances(model)
    psi_inverse_argument = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, sigma, optimize=True
    )
    value = sigma[target_index] - abs(model.beta[target_index]) ** 2 * sigma[
        target_index
    ] @ np.linalg.solve(psi_inverse_argument, sigma[target_index])
    return (0.5 * (value + value.conj().T)).astype(np.complex128)


def widely_linear_statistics(model: PilotGroupModel, target: int) -> dict[str, Any]:
    """Compute independent second-order diagnostics for the affine classes."""

    target_index = int(target)
    if not 0 <= target_index < model.group_size:
        raise IndexError("target index is outside the pilot group")
    rho = source_rho(model.delta).astype(np.complex128)
    nu = np.sinc(2.0 * model.delta / np.pi).astype(np.complex128)
    covariances = _conditional_covariances(model)
    pseudo_covariances = np.empty_like(covariances)
    for index in range(model.group_size):
        pseudo_covariances[index] = (nu[index] - rho[index] ** 2) * np.outer(
            model.los[:, index], model.los[:, index]
        )
    covariance_y = model.noise_covariance + np.einsum(
        "p,pij->ij", np.abs(model.beta) ** 2, covariances, optimize=True
    )
    pseudo_y = np.einsum(
        "p,pij->ij", model.beta**2, pseudo_covariances, optimize=True
    )
    augmented = np.block(
        [[covariance_y, pseudo_y], [pseudo_y.conj(), covariance_y.conj()]]
    )
    cross_complex = model.beta[target_index].conj() * covariances[target_index]
    cross_augmented = np.concatenate(
        (
            cross_complex,
            model.beta[target_index] * pseudo_covariances[target_index],
        ),
        axis=1,
    )
    complex_error = covariances[target_index] - cross_complex @ np.linalg.solve(
        covariance_y, cross_complex.conj().T
    )
    widely_linear_error = covariances[target_index] - cross_augmented @ np.linalg.solve(
        augmented, cross_augmented.conj().T
    )
    gain = np.linalg.solve(augmented.T, cross_augmented.T).T
    return {
        "pseudo_covariance_y_fro": float(np.linalg.norm(pseudo_y, ord="fro")),
        "conjugate_gain_fro": float(
            np.linalg.norm(gain[:, model.antenna_count :], ord="fro")
        ),
        "complex_affine_mse": float(np.trace(complex_error).real),
        "widely_linear_mse": float(np.trace(widely_linear_error).real),
        "mse_improvement": float(
            np.trace(complex_error).real - np.trace(widely_linear_error).real
        ),
    }


def centralized_virtual_mmse(
    estimates: object,
    error_covariances: object,
    payload_powers: object,
    noise_variance: float,
) -> ComplexArray:
    """Evaluate Ain et al. (11) for full cooperation, ``D_k = I``.

    ``estimates`` has shape ``(L*N, K)`` and ``error_covariances`` has shape
    ``(L, K, N, N)``. The returned columns are the centralized combiners.
    """

    channel_estimates = np.asarray(estimates, dtype=np.complex128)
    errors = np.asarray(error_covariances, dtype=np.complex128)
    powers = np.asarray(payload_powers, dtype=np.float64)
    if channel_estimates.ndim != 2 or errors.ndim != 4:
        raise ValueError("invalid estimate or error-covariance dimensions")
    access_points, users, antennas, antennas_again = errors.shape
    if antennas != antennas_again or channel_estimates.shape != (
        access_points * antennas,
        users,
    ):
        raise ValueError("estimate and error-covariance shapes are inconsistent")
    if powers.shape != (users,) or np.any(powers < 0.0) or not np.all(np.isfinite(powers)):
        raise ValueError("payload_powers must be finite, nonnegative, and length K")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if not np.all(np.isfinite(channel_estimates)) or not np.all(np.isfinite(errors)):
        raise ValueError("estimates and error covariances must be finite")

    dimension = access_points * antennas
    z = np.zeros((dimension, dimension), dtype=np.complex128)
    for access_point in range(access_points):
        block = np.einsum(
            "k,kij->ij", powers, errors[access_point], optimize=True
        )
        start = access_point * antennas
        z[start : start + antennas, start : start + antennas] = block
    system = (
        (channel_estimates * powers[np.newaxis, :]) @ channel_estimates.conj().T
        + z
        + float(noise_variance) * np.eye(dimension, dtype=np.complex128)
    )
    rhs = channel_estimates * np.sqrt(powers)[np.newaxis, :]
    combiners = np.linalg.solve(system, rhs)
    if not np.all(np.isfinite(combiners)):
        raise FloatingPointError("centralized virtual-MMSE combiner is nonfinite")
    return combiners.astype(np.complex128)


def oer_instantaneous(
    channels: object,
    combiners: object,
    payload_powers: object,
    noise_variance: float,
    *,
    prelog: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the instantaneous SINR inside Ain et al. (17), using log2."""

    channel_matrix = np.asarray(channels, dtype=np.complex128)
    combiner_matrix = np.asarray(combiners, dtype=np.complex128)
    powers = np.asarray(payload_powers, dtype=np.float64)
    if channel_matrix.ndim != 2 or combiner_matrix.shape != channel_matrix.shape:
        raise ValueError("channels and combiners must share shape (M, K)")
    users = channel_matrix.shape[1]
    if powers.shape != (users,) or np.any(powers < 0.0):
        raise ValueError("payload_powers must be nonnegative and length K")
    if not np.all(np.isfinite(powers)) or not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("powers and noise variance must be finite and valid")
    if not math.isfinite(prelog) or not 0.0 < prelog <= 1.0:
        raise ValueError("prelog must lie in (0, 1]")
    effective = combiner_matrix.conj().T @ channel_matrix
    effective *= np.sqrt(powers)[np.newaxis, :]
    desired = np.abs(np.diag(effective)) ** 2
    interference = np.sum(np.abs(effective) ** 2, axis=1) - desired
    noise = float(noise_variance) * np.sum(np.abs(combiner_matrix) ** 2, axis=0)
    denominator = interference + noise
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("OER denominator must be finite and positive")
    sinr = desired / denominator
    spectral_efficiency = float(prelog) * np.log2(1.0 + sinr)
    return sinr.astype(np.float64), spectral_efficiency.astype(np.float64)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineConfigurationError(f"{label} must be a mapping")
    return value


def load_baseline_reproduction_config(path: str | Path) -> dict[str, Any]:
    """Load the strict, controller-prohibiting baseline reproduction schema."""

    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BaselineConfigurationError(f"cannot read baseline config: {error}") from error
    root = dict(_require_mapping(document, "root"))
    required = {
        "schema_version",
        "experiment_id",
        "stage",
        "lock_status",
        "evidence_label",
        "controller_execution_allowed",
        "estimator_ids",
        "prohibited_estimator_ids",
        "rho_grid_degrees",
        "fixture_seed",
        "tolerances",
        "trend",
        "output_directory",
        "preregistration_path",
    }
    if set(root) != required:
        raise BaselineConfigurationError(
            f"baseline config fields must equal {sorted(required)}; got {sorted(root)}"
        )
    if root["schema_version"] != "baseline_reproduction/1.0":
        raise BaselineConfigurationError("unsupported schema_version")
    if root["stage"] != "BASELINE_REPRODUCTION":
        raise BaselineConfigurationError("stage must be BASELINE_REPRODUCTION")
    if root["lock_status"] != "LOCKED_BEFORE_RESULTS":
        raise BaselineConfigurationError("config must be locked before results")
    if root["controller_execution_allowed"] is not False:
        raise BaselineConfigurationError("the proposed controller must be disabled")
    estimator_ids = tuple(str(value) for value in root["estimator_ids"])
    required_estimators = {
        "bounded_phase_clmmse",
        "widely_linear_lmmse",
        "phase_unaware_lmmse",
        "collision_blind_ls",
        "perfect_phase_oracle",
    }
    if set(estimator_ids) != required_estimators or len(estimator_ids) != len(
        required_estimators
    ):
        raise BaselineConfigurationError("estimator_ids must be exactly the five baseline families")
    prohibited = {str(value) for value in root["prohibited_estimator_ids"]}
    if "phase_graph_controller" not in prohibited or "phase_graph_controller" in estimator_ids:
        raise BaselineConfigurationError("phase_graph_controller must be explicitly prohibited")
    grid = np.asarray(root["rho_grid_degrees"], dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(grid < 0.0) or np.any(grid > 180.0):
        raise BaselineConfigurationError("rho_grid_degrees must be a nontrivial grid in [0,180]")
    tolerances = dict(_require_mapping(root["tolerances"], "tolerances"))
    if set(tolerances) != {"absolute", "relative", "strict_improvement"}:
        raise BaselineConfigurationError("unexpected tolerance fields")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in tolerances.values()):
        raise BaselineConfigurationError("all tolerances must be finite and positive")

    trend = dict(_require_mapping(root["trend"], "trend"))
    trend_required = {
        "enabled",
        "evidence_label",
        "topology_seeds",
        "realizations_per_topology",
        "access_points",
        "antennas_per_ap",
        "users",
        "pilot_symbols",
        "coherence_symbols",
        "area_side",
        "minimum_distance",
        "pathloss_exponent",
        "spatial_correlation",
        "pilot_power",
        "payload_power",
        "pilot_noise_variance",
        "payload_noise_variance",
        "pilot_assignment",
        "kappa_grid",
        "delta_grid_degrees",
        "cooperation",
        "receiver_equation",
        "metric_equation",
        "log_base",
        "prelog",
    }
    if set(trend) != trend_required:
        raise BaselineConfigurationError("trend fields do not match the frozen schema")
    if trend["enabled"] is not True:
        raise BaselineConfigurationError("the registered qualitative trend check must be enabled")
    if trend["evidence_label"] != "QUALITATIVE_MODEL_CHECK":
        raise BaselineConfigurationError("trend must be labelled QUALITATIVE_MODEL_CHECK")
    if trend["cooperation"] != "FULL" or trend["receiver_equation"] != "AIN_EQ_11":
        raise BaselineConfigurationError("trend must use full-cooperation Ain Eq. (11)")
    if trend["metric_equation"] != "AIN_EQ_17_OER" or trend["log_base"] != 2:
        raise BaselineConfigurationError("trend metric must lock log2 Ain Eq. (17)")
    if float(trend["prelog"]) != 1.0:
        raise BaselineConfigurationError("registered literal Eq. (17) check uses prelog 1")
    if not trend["topology_seeds"] or len(set(trend["topology_seeds"])) != len(
        trend["topology_seeds"]
    ):
        raise BaselineConfigurationError("trend topology seeds must be unique and nonempty")
    if int(trend["realizations_per_topology"]) < 1:
        raise BaselineConfigurationError("trend realizations_per_topology must be positive")
    for path_field in ("output_directory", "preregistration_path"):
        value = Path(str(root[path_field]))
        if value.is_absolute() or ".." in value.parts:
            raise BaselineConfigurationError(f"{path_field} must be project-relative and safe")
    return root
