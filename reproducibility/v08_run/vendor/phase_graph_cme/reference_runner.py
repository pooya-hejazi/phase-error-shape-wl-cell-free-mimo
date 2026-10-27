"""Memory-bounded numerical references and repeat-resolution checks.

The tensor routine in this module is exact only for the selected finite
Gauss--Legendre product rule. Neither its status nor any helper below claims
analytic exactness for the continuous phase posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .diagnostics import summarize_chains
from .inference import MomentEstimate, phase_quadrature
from .model import PhaseGraph

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
ZeroDenominatorPolicy = Literal["zero_if_exact_else_inf", "raise"]


SMALL_REFERENCE_ORDER_PAIRS: dict[int, tuple[int, int]] = {
    2: (64, 96),
    3: (32, 48),
    4: (24, 32),
}


@dataclass(frozen=True)
class MomentDiscrepancy:
    """First- and second-moment discrepancies under fixed definitions."""

    maximum_absolute_z: float
    normalized_squared_z: float
    maximum_absolute_Z: float
    normalized_frobenius_Z: float


@dataclass(frozen=True)
class HighBudgetRepeatThresholds:
    """Frozen numerical thresholds for a two-run reference resolution."""

    maximum_split_rhat: float
    maximum_chain_mean_disagreement_z: float
    maximum_chain_mean_disagreement_Z: float
    independent_repeat_maximum_abs_z: float
    independent_repeat_maximum_abs_Z: float

    def __post_init__(self) -> None:
        for name in (
            "maximum_split_rhat",
            "maximum_chain_mean_disagreement_z",
            "maximum_chain_mean_disagreement_Z",
            "independent_repeat_maximum_abs_z",
            "independent_repeat_maximum_abs_Z",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite nonnegative real number")

@dataclass(frozen=True)
class HighBudgetRunDiagnostics:
    """Diagnostics independently reconstructed from one raw chain array."""

    maximum_split_rhat: float
    maximum_chain_mean_disagreement_z: float
    maximum_chain_mean_disagreement_Z: float
    all_finite: bool


@dataclass(frozen=True)
class HighBudgetRepeatResolution:
    """Resolution result; equality with every registered threshold passes."""

    resolved: bool
    status: str
    primary: HighBudgetRunDiagnostics
    repeat: HighBudgetRunDiagnostics
    independent_repeat_maximum_abs_z: float
    independent_repeat_maximum_abs_Z: float
    primary_split_rhat_pass: bool
    repeat_split_rhat_pass: bool
    primary_z_chain_disagreement_pass: bool
    repeat_z_chain_disagreement_pass: bool
    primary_Z_chain_disagreement_pass: bool
    repeat_Z_chain_disagreement_pass: bool
    independent_repeat_z_pass: bool
    independent_repeat_Z_pass: bool


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _validated_graph_arrays(
    graph: PhaseGraph,
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    if not isinstance(graph, PhaseGraph):
        raise TypeError("graph must be a PhaseGraph")
    x = np.asarray(graph.x, dtype=np.complex128)
    A = np.asarray(graph.A, dtype=np.complex128)
    c = np.asarray(graph.c, dtype=np.complex128)
    G = np.asarray(graph.G, dtype=np.complex128)
    delta = np.asarray(graph.delta, dtype=np.float64)
    cholesky = np.asarray(graph.cholesky, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("graph.A must have nonempty shape (N, P)")
    n, p = A.shape
    if x.shape != (n,):
        raise ValueError("graph.x must have shape (N,)")
    if c.shape != (p,) or G.shape != (p, p) or delta.shape != (p,):
        raise ValueError("graph c, G, and delta dimensions must agree with graph.A")
    if cholesky.shape != (n, n):
        raise ValueError("graph.cholesky must have shape (N, N)")
    arrays = (x, A, c, G, delta, cholesky)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all PhaseGraph arrays must be finite")
    if np.any(delta < 0.0) or np.any(delta > np.pi):
        raise ValueError("each graph.delta entry must lie in [0, pi]")
    return x, A, delta


def streaming_tensor_moments(
    graph: PhaseGraph,
    order: int,
    maximum_states: int = 1_100_000,
    chunk_size: int = 8_192,
) -> MomentEstimate:
    """Evaluate a finite product Gauss--Legendre rule in bounded memory.

    States use the same lexicographic order as :func:`tensor_quadrature`.
    Chunk contributions are merged with an online log-sum-exp rescaling, so
    live storage is ``O(chunk_size * (N + P) + P**2)`` rather than
    ``O(order**P)``. The result is exact for the enumerated finite numerical
    rule, subject to floating-point rounding; it is not an exact continuous
    posterior integral.
    """

    quadrature_order = _positive_integer(order, label="order")
    state_cap = _positive_integer(maximum_states, label="maximum_states")
    requested_chunk = _positive_integer(chunk_size, label="chunk_size")
    x, A, delta = _validated_graph_arrays(graph)
    rules = [phase_quadrature(float(width), quadrature_order) for width in delta]
    radices = tuple(len(nodes) for nodes, _ in rules)
    state_count = math.prod(radices)
    if state_count > state_cap:
        raise ValueError(
            f"tensor rule has {state_count} states, above the declared cap"
        )
    if state_count > np.iinfo(np.int64).max:
        raise ValueError("tensor state count exceeds the supported int64 index range")

    p = A.shape[1]
    log_prior = tuple(np.log(weights) for _, weights in rules)
    phasor_nodes = tuple(np.exp(1j * nodes) for nodes, _ in rules)
    running_max = -math.inf
    denominator = 0.0
    first_numerator = np.zeros(p, dtype=np.complex128)
    second_numerator = np.zeros((p, p), dtype=np.complex128)
    chunk_count = 0

    for start in range(0, state_count, requested_chunk):
        stop = min(start + requested_chunk, state_count)
        linear = np.arange(start, stop, dtype=np.int64)
        digits: list[NDArray[np.int64]] = [
            np.empty(linear.size, dtype=np.int64) for _ in range(p)
        ]
        quotient = linear
        for axis in range(p - 1, -1, -1):
            digits[axis] = quotient % radices[axis]
            quotient = quotient // radices[axis]

        u = np.empty((linear.size, p), dtype=np.complex128)
        prior = np.zeros(linear.size, dtype=np.float64)
        for axis in range(p):
            index = digits[axis]
            u[:, axis] = phasor_nodes[axis][index]
            prior += log_prior[axis][index]
        with np.errstate(over="ignore", invalid="ignore"):
            residual = x[np.newaxis, :] - u @ A.T
            log_weight = prior - np.sum(np.abs(residual) ** 2, axis=1)
        if not np.all(np.isfinite(log_weight)):
            raise FloatingPointError("tensor log weights became nonfinite")

        chunk_max = float(np.max(log_weight))
        target_max = max(running_max, chunk_max)
        old_scale = (
            0.0 if running_max == -math.inf else math.exp(running_max - target_max)
        )
        denominator *= old_scale
        first_numerator *= old_scale
        second_numerator *= old_scale
        scaled_weight = np.exp(log_weight - target_max)
        denominator += float(np.sum(scaled_weight))
        first_numerator += scaled_weight @ u
        second_numerator += np.einsum(
            "s,si,sj->ij", scaled_weight, u, u.conj(), optimize=True
        )
        running_max = target_max
        chunk_count += 1

    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("tensor normalization is nonfinite or nonpositive")
    z = first_numerator / denominator
    Z = second_numerator / denominator
    Z = 0.5 * (Z + Z.conj().T)
    np.fill_diagonal(Z, 1.0)
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(Z)):
        raise FloatingPointError("tensor moments became nonfinite")
    return MomentEstimate(
        z=z.astype(np.complex128),
        Z=Z.astype(np.complex128),
        method="streaming_tensor_gauss_legendre",
        status="DISCRETE_QUADRATURE_EXACT",
        sample_count=int(state_count),
        metadata={
            "order": quadrature_order,
            "state_count": int(state_count),
            "chunk_size": requested_chunk,
            "chunk_count": chunk_count,
            "scientific_boundary": (
                "exact_for_selected_finite_product_rule_not_continuous_posterior"
            ),
        },
    )


def small_reference_orders(group_size: int) -> tuple[int, int]:
    """Return the preregistered coarse/fine order pair for P in {2, 3, 4}."""

    size = _positive_integer(group_size, label="group_size")
    try:
        return SMALL_REFERENCE_ORDER_PAIRS[size]
    except KeyError as error:
        raise ValueError(
            "small-reference orders are defined only for P=2, P=3, or P=4"
        ) from error


def _matching_finite_arrays(
    estimate: object, reference: object, *, label: str
) -> tuple[ComplexArray, ComplexArray]:
    left = np.asarray(estimate, dtype=np.complex128)
    right = np.asarray(reference, dtype=np.complex128)
    if left.shape != right.shape:
        raise ValueError(f"{label} arrays must have identical shapes")
    if left.size < 1:
        raise ValueError(f"{label} arrays must be nonempty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"{label} arrays must be finite")
    return left, right


def _zero_denominator_result(
    numerator: float, *, policy: ZeroDenominatorPolicy, label: str
) -> float:
    if policy == "raise":
        raise ZeroDivisionError(f"{label} reference norm is zero")
    if policy != "zero_if_exact_else_inf":
        raise ValueError("unsupported zero-denominator policy")
    return 0.0 if numerator == 0.0 else math.inf


def maximum_absolute_error(estimate: object, reference: object) -> float:
    """Return ``max(abs(estimate-reference))`` for finite matching arrays."""

    left, right = _matching_finite_arrays(estimate, reference, label="moment")
    return float(np.max(np.abs(left - right)))


def normalized_squared_error(
    estimate: object,
    reference: object,
    *,
    zero_denominator: ZeroDenominatorPolicy = "zero_if_exact_else_inf",
) -> float:
    """Return squared Euclidean error divided by squared reference norm.

    By default, a zero reference norm gives zero only for an exactly zero
    numerator and positive infinity otherwise. ``zero_denominator='raise'``
    instead rejects every zero reference norm.
    """

    left, right = _matching_finite_arrays(estimate, reference, label="vector")
    difference = left - right
    numerator = float(np.vdot(difference.ravel(), difference.ravel()).real)
    denominator = float(np.vdot(right.ravel(), right.ravel()).real)
    if denominator == 0.0:
        return _zero_denominator_result(
            numerator, policy=zero_denominator, label="normalized-squared-error"
        )
    if zero_denominator not in ("zero_if_exact_else_inf", "raise"):
        raise ValueError("unsupported zero-denominator policy")
    return numerator / denominator


def normalized_frobenius_error(
    estimate: object,
    reference: object,
    *,
    zero_denominator: ZeroDenominatorPolicy = "zero_if_exact_else_inf",
) -> float:
    """Return Frobenius error divided by the reference Frobenius norm.

    The default zero-denominator rule is zero for exact equality and positive
    infinity for a nonzero error. The ``'raise'`` policy raises instead.
    """

    left, right = _matching_finite_arrays(estimate, reference, label="matrix")
    if left.ndim != 2:
        raise ValueError("normalized Frobenius error requires two-dimensional arrays")
    numerator = float(np.linalg.norm(left - right, ord="fro"))
    denominator = float(np.linalg.norm(right, ord="fro"))
    if denominator == 0.0:
        return _zero_denominator_result(
            numerator, policy=zero_denominator, label="normalized-Frobenius-error"
        )
    if zero_denominator not in ("zero_if_exact_else_inf", "raise"):
        raise ValueError("unsupported zero-denominator policy")
    return numerator / denominator


def compare_moments(
    estimate_z: object,
    estimate_Z: object,
    reference_z: object,
    reference_Z: object,
    *,
    zero_denominator: ZeroDenominatorPolicy = "zero_if_exact_else_inf",
) -> MomentDiscrepancy:
    """Compute the registered first/full-second-moment error quartet."""

    z, z_reference = _matching_finite_arrays(
        estimate_z, reference_z, label="first-moment"
    )
    Z, Z_reference = _matching_finite_arrays(
        estimate_Z, reference_Z, label="second-moment"
    )
    if z.ndim != 1 or Z.shape != (z.size, z.size):
        raise ValueError("z must have shape (P,) and Z must have shape (P, P)")
    return MomentDiscrepancy(
        maximum_absolute_z=maximum_absolute_error(z, z_reference),
        normalized_squared_z=normalized_squared_error(
            z, z_reference, zero_denominator=zero_denominator
        ),
        maximum_absolute_Z=maximum_absolute_error(Z, Z_reference),
        normalized_frobenius_Z=normalized_frobenius_error(
            Z, Z_reference, zero_denominator=zero_denominator
        ),
    )


def _raw_chain_diagnostics(
    samples: object, *, label: str
) -> tuple[ComplexArray, ComplexArray, HighBudgetRunDiagnostics]:
    array = np.asarray(samples, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError(f"{label} samples must have shape (chains, draws, variables)")
    chains, draws, variables = array.shape
    if chains < 2 or draws < 4 or variables < 1:
        raise ValueError(
            f"{label} samples require at least 2 chains, 4 draws, and 1 variable"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} samples must be finite")
    if float(np.max(np.abs(np.abs(array) - 1.0))) > 1e-12:
        raise ValueError(f"{label} samples must be unit-modulus phasors")

    chain_z = np.mean(array, axis=1)
    pooled_z = np.mean(chain_z, axis=0)
    z_disagreement = float(np.max(np.abs(chain_z - pooled_z)))
    chain_Z = np.einsum(
        "cdi,cdj->cij", array, array.conj(), optimize=True
    ) / float(draws)
    pooled_Z = np.mean(chain_Z, axis=0)
    pooled_Z = 0.5 * (pooled_Z + pooled_Z.conj().T)
    np.fill_diagonal(pooled_Z, 1.0)
    Z_disagreement = float(np.max(np.abs(chain_Z - pooled_Z[np.newaxis, :, :])))
    diagnostic = summarize_chains(array)
    finite = bool(
        diagnostic.all_finite
        and math.isfinite(z_disagreement)
        and math.isfinite(Z_disagreement)
        and np.all(np.isfinite(pooled_z))
        and np.all(np.isfinite(pooled_Z))
    )
    return (
        pooled_z.astype(np.complex128),
        pooled_Z.astype(np.complex128),
        HighBudgetRunDiagnostics(
            maximum_split_rhat=float(diagnostic.maximum_split_rhat),
            maximum_chain_mean_disagreement_z=z_disagreement,
            maximum_chain_mean_disagreement_Z=Z_disagreement,
            all_finite=finite,
        ),
    )


def resolve_high_budget_repeat(
    primary_samples: object,
    repeat_samples: object,
    thresholds: HighBudgetRepeatThresholds,
) -> HighBudgetRepeatResolution:
    """Resolve two high-budget runs only when every independent check passes.

    Split-Rhat, between-chain first moments, and between-chain full second
    moments are reconstructed separately from each raw draw array. The two
    pooled runs are then compared in both ``z`` and full ``Z``. All checks
    use inclusive upper thresholds, and an unresolved pair is never labeled a
    numerical reference.
    """

    if not isinstance(thresholds, HighBudgetRepeatThresholds):
        raise TypeError("thresholds must be HighBudgetRepeatThresholds")
    primary_array = np.asarray(primary_samples, dtype=np.complex128)
    repeat_array = np.asarray(repeat_samples, dtype=np.complex128)
    if primary_array.shape != repeat_array.shape:
        raise ValueError(
            "primary and repeat raw-chain arrays must have identical shapes"
        )
    primary_z, primary_Z, primary = _raw_chain_diagnostics(
        primary_array, label="primary"
    )
    repeat_z, repeat_Z, repeat = _raw_chain_diagnostics(
        repeat_array, label="repeat"
    )
    repeat_z_error = maximum_absolute_error(primary_z, repeat_z)
    repeat_Z_error = maximum_absolute_error(primary_Z, repeat_Z)

    checks = {
        "primary_split_rhat_pass": bool(
            primary.all_finite
            and primary.maximum_split_rhat <= thresholds.maximum_split_rhat
        ),
        "repeat_split_rhat_pass": bool(
            repeat.all_finite
            and repeat.maximum_split_rhat <= thresholds.maximum_split_rhat
        ),
        "primary_z_chain_disagreement_pass": bool(
            primary.all_finite
            and primary.maximum_chain_mean_disagreement_z
            <= thresholds.maximum_chain_mean_disagreement_z
        ),
        "repeat_z_chain_disagreement_pass": bool(
            repeat.all_finite
            and repeat.maximum_chain_mean_disagreement_z
            <= thresholds.maximum_chain_mean_disagreement_z
        ),
        "primary_Z_chain_disagreement_pass": bool(
            primary.all_finite
            and primary.maximum_chain_mean_disagreement_Z
            <= thresholds.maximum_chain_mean_disagreement_Z
        ),
        "repeat_Z_chain_disagreement_pass": bool(
            repeat.all_finite
            and repeat.maximum_chain_mean_disagreement_Z
            <= thresholds.maximum_chain_mean_disagreement_Z
        ),
        "independent_repeat_z_pass": bool(
            repeat_z_error <= thresholds.independent_repeat_maximum_abs_z
        ),
        "independent_repeat_Z_pass": bool(
            repeat_Z_error <= thresholds.independent_repeat_maximum_abs_Z
        ),
    }
    resolved = all(checks.values())
    return HighBudgetRepeatResolution(
        resolved=resolved,
        status="RESOLVED_NUMERICAL_REFERENCE" if resolved else "UNRESOLVED_NOT_TRUTH",
        primary=primary,
        repeat=repeat,
        independent_repeat_maximum_abs_z=repeat_z_error,
        independent_repeat_maximum_abs_Z=repeat_Z_error,
        **checks,
    )
