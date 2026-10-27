"""Deterministic numerical repair for the consumed E0-01 UatF check.

This module has a distinct engineering identity. It reuses the frozen E0-01
physical model, B1 estimator, receiver definitions, cases, and frame registry.
Only the numerical equivalence calculation changes: each combiner is formed
once, per-frame sufficient contributions are bounded and cached, and three
reduction paths are certified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .baseline_precision_e0 import (
    EXPERIMENT_ID as PARENT_EXPERIMENT_ID,
    BaselinePrecisionConfig,
    BaselinePrecisionProducts,
    BaselinePrecisionProtocolError,
    Registries,
    _statistics_json,
    assemble_frame_estimates,
    batched_b1,
    build_registries,
    generate_channel_frame,
    generate_topology,
    pilot_group,
    receiver_combiner,
    stack_models,
    widely_linear_mmse,
)


EXPERIMENT_ID = "R2C01_V3_BASELINE_ONLY_PRECISION_E0_02"
SCIENTIFIC_BOUNDARY = "NON_SCIENTIFIC_BASELINE_ONLY_ENGINEERING"
PRIMITIVE_RTOL = 1e-12
PRIMITIVE_ATOL = 1e-12
IEEE_BINARY64_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps / 2.0)
FORWARD_ERROR_SAFETY_FACTOR = 16.0
ABSOLUTE_SE_ENVELOPE_CEILING = 1e-9

# Frozen E0-01 dimensions and explicit cache byte caps.
MAX_SAMPLES_PER_CONDITION = 64
MAX_ANTENNAS = 128
MAX_USERS = 16
MAX_RECEIVERS = 6
MAX_SUFFICIENT_CACHE_BYTES = 2 * 1024 * 1024

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SufficientContributionCache:
    """Once-computed per-frame contributions needed by every UatF reduction."""

    coherent: ComplexArray
    second: FloatArray
    norm: FloatArray

    @property
    def sample_count(self) -> int:
        return int(self.coherent.shape[0])

    @property
    def user_count(self) -> int:
        return int(self.coherent.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.coherent.nbytes + self.second.nbytes + self.norm.nbytes)

    def select(self, mask: object) -> SufficientContributionCache:
        selected = np.asarray(mask)
        if selected.dtype != np.bool_ or selected.shape != (self.sample_count,) or not np.any(selected):
            raise ValueError("contribution selection must be a nonempty Boolean sample mask")
        result = SufficientContributionCache(
            np.asarray(self.coherent[selected], dtype=np.complex128),
            np.asarray(self.second[selected], dtype=np.float64),
            np.asarray(self.norm[selected], dtype=np.float64),
        )
        _validate_contribution_cache(result)
        return result


def _validate_contribution_cache(cache: SufficientContributionCache) -> None:
    coherent = np.asarray(cache.coherent)
    second = np.asarray(cache.second)
    norm = np.asarray(cache.norm)
    if coherent.dtype != np.dtype(np.complex128):
        raise ValueError("coherent contributions must have dtype complex128")
    if second.dtype != np.dtype(np.float64) or norm.dtype != np.dtype(np.float64):
        raise ValueError("power and norm contributions must have dtype float64")
    if coherent.ndim != 2:
        raise ValueError("coherent contributions must have shape (S,K)")
    samples, users = coherent.shape
    if not 1 <= samples <= MAX_SAMPLES_PER_CONDITION or not 1 <= users <= MAX_USERS:
        raise ValueError("contribution cache exceeds registered sample/user bounds")
    if second.shape != (samples, users, users) or norm.shape != (samples, users):
        raise ValueError("sufficient-contribution shapes differ")
    if cache.nbytes > MAX_SUFFICIENT_CACHE_BYTES:
        raise ValueError("sufficient-contribution cache exceeds its byte cap")
    if not np.all(np.isfinite(coherent)) or not np.all(np.isfinite(second)) or not np.all(np.isfinite(norm)):
        raise ValueError("sufficient contributions must be finite")
    if np.any(second < 0.0) or np.any(norm < 0.0):
        raise ValueError("power and norm contributions must be nonnegative")


def frame_sufficient_contributions(
    channel: object,
    combiner: object,
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """Compute one frame's UatF contributions exactly once."""

    h = np.asarray(channel, dtype=np.complex128)
    v = np.asarray(combiner, dtype=np.complex128)
    if h.ndim != 2 or h.shape != v.shape:
        raise ValueError("one frame must contain matched (M,K) channel/combiner arrays")
    antennas, users = h.shape
    if not 1 <= antennas <= MAX_ANTENNAS or not 1 <= users <= MAX_USERS:
        raise ValueError("one frame exceeds registered antenna/user bounds")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        raise ValueError("channel and combiner values must be finite")
    inner = v.conj().T @ h
    coherent = np.asarray(np.diag(inner), dtype=np.complex128)
    second = np.asarray(np.abs(inner) ** 2, dtype=np.float64)
    norm = np.asarray(np.sum(np.abs(v) ** 2, axis=0), dtype=np.float64)
    if not np.all(np.isfinite(coherent)) or not np.all(np.isfinite(second)) or not np.all(np.isfinite(norm)):
        raise FloatingPointError("nonfinite per-frame sufficient contribution")
    return coherent, second, norm


def sufficient_contribution_cache(
    channel_samples: object,
    combiner_samples: object,
) -> SufficientContributionCache:
    """Build a bounded contribution cache in registered frame order."""

    channels = np.asarray(channel_samples, dtype=np.complex128)
    combiners = np.asarray(combiner_samples, dtype=np.complex128)
    if channels.ndim != 3 or combiners.shape != channels.shape:
        raise ValueError("sample arrays must share shape (S,M,K)")
    samples, antennas, users = channels.shape
    if not 1 <= samples <= MAX_SAMPLES_PER_CONDITION:
        raise ValueError("sample count exceeds the registered per-condition bound")
    if not 1 <= antennas <= MAX_ANTENNAS or not 1 <= users <= MAX_USERS:
        raise ValueError("sample dimensions exceed registered bounds")
    coherent = np.empty((samples, users), dtype=np.complex128)
    second = np.empty((samples, users, users), dtype=np.float64)
    norm = np.empty((samples, users), dtype=np.float64)
    for sample in range(samples):
        coherent[sample], second[sample], norm[sample] = frame_sufficient_contributions(
            channels[sample], combiners[sample]
        )
    result = SufficientContributionCache(coherent, second, norm)
    _validate_contribution_cache(result)
    return result


class OnlineNeumaierReducer:
    """Elementwise, fixed-arrival-order Neumaier summation for real arrays."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._sum = np.zeros(shape, dtype=np.float64)
        self._correction = np.zeros(shape, dtype=np.float64)
        self.count = 0

    def update(self, value: object) -> None:
        sample = np.asarray(value, dtype=np.float64)
        if sample.shape != self._sum.shape or not np.all(np.isfinite(sample)):
            raise ValueError("Neumaier sample has the wrong shape or is nonfinite")
        trial = self._sum + sample
        correction = np.where(
            np.abs(self._sum) >= np.abs(sample),
            (self._sum - trial) + sample,
            (sample - trial) + self._sum,
        )
        self._correction += correction
        self._sum = trial
        self.count += 1

    def mean(self) -> FloatArray:
        if self.count < 1:
            raise BaselinePrecisionProtocolError("cannot finalize an empty Neumaier reducer")
        return np.asarray((self._sum + self._correction) / float(self.count), dtype=np.float64)


class NeumaierUatFAccumulator:
    """Online reducer of cached sufficient contributions, including complex gain."""

    def __init__(self, users: int) -> None:
        if not 1 <= users <= MAX_USERS:
            raise ValueError("user count exceeds registered bound")
        self.users = users
        self._coherent_real = OnlineNeumaierReducer((users,))
        self._coherent_imag = OnlineNeumaierReducer((users,))
        self._second = OnlineNeumaierReducer((users, users))
        self._norm = OnlineNeumaierReducer((users,))

    @property
    def count(self) -> int:
        return self._second.count

    def update_contributions(self, coherent: object, second: object, norm: object) -> None:
        gain = np.asarray(coherent, dtype=np.complex128)
        power = np.asarray(second, dtype=np.float64)
        combiner_norm = np.asarray(norm, dtype=np.float64)
        if gain.shape != (self.users,) or power.shape != (self.users, self.users) or combiner_norm.shape != (self.users,):
            raise ValueError("online contribution shapes differ")
        if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(power)) or not np.all(np.isfinite(combiner_norm)):
            raise ValueError("online contributions must be finite")
        self._coherent_real.update(gain.real)
        self._coherent_imag.update(gain.imag)
        self._second.update(power)
        self._norm.update(combiner_norm)

    def finalize(
        self,
        powers: object,
        noise_variance: float,
        tau_p: int,
        tau_c: int,
    ) -> dict[str, Any]:
        counts = {
            self._coherent_real.count,
            self._coherent_imag.count,
            self._second.count,
            self._norm.count,
        }
        if len(counts) != 1 or self.count < 1:
            raise BaselinePrecisionProtocolError("Neumaier sufficient-statistic counts differ")
        coherent = self._coherent_real.mean().astype(np.complex128)
        coherent += 1j * self._coherent_imag.mean()
        return statistics_from_primitives(
            coherent,
            self._second.mean(),
            self._norm.mean(),
            powers,
            noise_variance,
            tau_p,
            tau_c,
            count=self.count,
        )


def _fixed_order_fsum_axis0(values: object) -> FloatArray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
        raise ValueError("fsum input must have a nonempty finite sample axis")
    flat = samples.reshape(samples.shape[0], -1)
    totals = np.fromiter(
        (
            math.fsum(float(flat[row, column]) for row in range(flat.shape[0]))
            for column in range(flat.shape[1])
        ),
        dtype=np.float64,
        count=flat.shape[1],
    )
    return np.asarray(totals.reshape(samples.shape[1:]) / float(samples.shape[0]), dtype=np.float64)


def fixed_order_fsum_statistics(
    cache: SufficientContributionCache,
    powers: object,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
) -> dict[str, Any]:
    """Independently reduce cached contributions with scalar ``math.fsum``."""

    _validate_contribution_cache(cache)
    coherent = _fixed_order_fsum_axis0(cache.coherent.real).astype(np.complex128)
    coherent += 1j * _fixed_order_fsum_axis0(cache.coherent.imag)
    return statistics_from_primitives(
        coherent,
        _fixed_order_fsum_axis0(cache.second),
        _fixed_order_fsum_axis0(cache.norm),
        powers,
        noise_variance,
        tau_p,
        tau_c,
        count=cache.sample_count,
    )


def statistics_from_primitives(
    coherent: object,
    second: object,
    norm: object,
    powers: object,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
    *,
    count: int,
) -> dict[str, Any]:
    """Derive UatF fields without pre-empting diagnostic certification."""

    gain = np.asarray(coherent, dtype=np.complex128)
    effective_power = np.asarray(second, dtype=np.float64)
    combiner_norm = np.asarray(norm, dtype=np.float64)
    payload_powers = np.asarray(powers, dtype=np.float64)
    if gain.ndim != 1:
        raise ValueError("coherent gain must be one-dimensional")
    users = gain.size
    if effective_power.shape != (users, users) or combiner_norm.shape != (users,) or payload_powers.shape != (users,):
        raise ValueError("primitive UatF dimensions differ")
    if count < 1 or not np.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("count and noise variance must be positive")
    if not 0 <= tau_p < tau_c:
        raise ValueError("require 0 <= tau_p < tau_c")
    if np.any(payload_powers < 0.0):
        raise ValueError("payload powers must be nonnegative")
    if not all(np.all(np.isfinite(value)) for value in (gain, effective_power, combiner_norm, payload_powers)):
        raise ValueError("primitive UatF inputs must be finite")
    numerator = np.asarray(payload_powers * np.abs(gain) ** 2, dtype=np.float64)
    denominator = np.asarray(
        effective_power @ payload_powers - numerator + float(noise_variance) * combiner_norm,
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sinr = np.asarray(numerator / denominator, dtype=np.float64)
        spectral_efficiency = np.asarray(
            (1.0 - float(tau_p) / float(tau_c)) * np.log2(1.0 + sinr),
            dtype=np.float64,
        )
    return {
        "count": int(count),
        "coherent_gain": gain,
        "second_effective_power": effective_power,
        "combiner_norm": combiner_norm,
        "numerator": numerator,
        "denominator": denominator,
        "sinr": sinr,
        "spectral_efficiency": spectral_efficiency,
    }


def ordinary_direct_statistics(
    cache: SufficientContributionCache,
    powers: object,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
) -> dict[str, Any]:
    """Ordinary ``np.mean`` reduction of the exact shared contribution cache."""

    _validate_contribution_cache(cache)
    return statistics_from_primitives(
        np.mean(cache.coherent, axis=0, dtype=np.complex128),
        np.mean(cache.second, axis=0, dtype=np.float64),
        np.mean(cache.norm, axis=0, dtype=np.float64),
        powers,
        noise_variance,
        tau_p,
        tau_c,
        count=cache.sample_count,
    )


def _gamma(operation_count: int) -> float:
    operations = max(1, int(operation_count))
    product = operations * IEEE_BINARY64_UNIT_ROUNDOFF
    if product >= 1.0:
        return math.inf
    return product / (1.0 - product)


def _outward(value: float) -> float:
    if math.isnan(value) or value < 0.0:
        return math.inf
    if math.isinf(value):
        return value
    return float(np.nextafter(np.float64(value), np.float64(math.inf)))


def _float(value: object) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _complex_scalar(value: object) -> dict[str, float]:
    number = complex(np.asarray(value, dtype=np.complex128))
    return {"real": float(number.real), "imag": float(number.imag)}


def _diagnostic_json_safe(value: object) -> object:
    """Preserve nonfinite failure values as explicit JSON-safe tokens."""

    if isinstance(value, Mapping):
        return {str(key): _diagnostic_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if number == math.inf:
            return "Infinity"
        if number == -math.inf:
            return "-Infinity"
        return number
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported diagnostic value {type(value)!r}")


def _path_values(statistics: Mapping[str, Any], user: int) -> dict[str, Any]:
    return {
        "coherent_gain": _complex_scalar(np.asarray(statistics["coherent_gain"])[user]),
        "second_effective_power_row": np.asarray(
            statistics["second_effective_power"], dtype=np.float64
        )[user].tolist(),
        "combiner_norm": _float(np.asarray(statistics["combiner_norm"])[user]),
        "numerator": _float(np.asarray(statistics["numerator"])[user]),
        "denominator": _float(np.asarray(statistics["denominator"])[user]),
        "sinr": _float(np.asarray(statistics["sinr"])[user]),
        "spectral_efficiency": _float(np.asarray(statistics["spectral_efficiency"])[user]),
    }


def _pair_key(first: str, second: str) -> str:
    return f"{first}__vs__{second}"


def _primitive_pair_errors(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    user: int,
) -> dict[str, float]:
    first_gain = complex(np.asarray(first["coherent_gain"])[user])
    second_gain = complex(np.asarray(second["coherent_gain"])[user])
    return {
        "coherent_gain_real": abs(first_gain.real - second_gain.real),
        "coherent_gain_imag": abs(first_gain.imag - second_gain.imag),
        "second_effective_power": float(
            np.max(
                np.abs(
                    np.asarray(first["second_effective_power"])[user]
                    - np.asarray(second["second_effective_power"])[user]
                )
            )
        ),
        "combiner_norm": float(
            abs(np.asarray(first["combiner_norm"])[user] - np.asarray(second["combiner_norm"])[user])
        ),
        "numerator": float(
            abs(np.asarray(first["numerator"])[user] - np.asarray(second["numerator"])[user])
        ),
        "denominator": float(
            abs(np.asarray(first["denominator"])[user] - np.asarray(second["denominator"])[user])
        ),
    }


def _primitive_pair_passes(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    user: int,
) -> dict[str, bool]:
    first_gain = complex(np.asarray(first["coherent_gain"])[user])
    second_gain = complex(np.asarray(second["coherent_gain"])[user])

    def scalar_close(left: float, right: float) -> bool:
        tolerance = PRIMITIVE_ATOL + PRIMITIVE_RTOL * max(abs(left), abs(right))
        return bool(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance)

    def array_close(left: object, right: object) -> bool:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        tolerance = PRIMITIVE_ATOL + PRIMITIVE_RTOL * np.maximum(
            np.abs(left_array), np.abs(right_array)
        )
        return bool(
            np.all(np.isfinite(left_array))
            and np.all(np.isfinite(right_array))
            and np.all(np.abs(left_array - right_array) <= tolerance)
        )

    return {
        "coherent_gain_real": scalar_close(first_gain.real, second_gain.real),
        "coherent_gain_imag": scalar_close(first_gain.imag, second_gain.imag),
        "second_effective_power": array_close(
            np.asarray(first["second_effective_power"])[user],
            np.asarray(second["second_effective_power"])[user],
        ),
        "combiner_norm": scalar_close(
            float(np.asarray(first["combiner_norm"])[user]),
            float(np.asarray(second["combiner_norm"])[user]),
        ),
        "numerator": scalar_close(
            float(np.asarray(first["numerator"])[user]),
            float(np.asarray(second["numerator"])[user]),
        ),
        "denominator": scalar_close(
            float(np.asarray(first["denominator"])[user]),
            float(np.asarray(second["denominator"])[user]),
        ),
    }


def _absolute_means(
    cache: SufficientContributionCache,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        _fixed_order_fsum_axis0(np.abs(cache.coherent)),
        _fixed_order_fsum_axis0(np.abs(cache.second)),
        _fixed_order_fsum_axis0(np.abs(cache.norm)),
    )


def _decimal_from_float(value: float) -> Decimal:
    if not math.isfinite(value):
        raise DecimalException("nonfinite binary64 value")
    return Decimal.from_float(value)


def _formula_identity_entry(
    observed: float,
    expected: Decimal,
    absolute_bound: float,
) -> dict[str, Any]:
    """Compare one binary64 field with an 80-digit formula evaluation."""

    observed_decimal = _decimal_from_float(observed)
    expected_is_finite = expected.is_finite()
    absolute_error = (
        abs(observed_decimal - expected) if expected_is_finite else Decimal("Infinity")
    )
    raw_bound = float(absolute_bound)
    bound_is_valid = math.isfinite(raw_bound) and raw_bound >= 0.0
    outward_bound = _outward(raw_bound) if bound_is_valid else raw_bound
    identity_passes = (
        expected_is_finite
        and absolute_error.is_finite()
        and bound_is_valid
        and math.isfinite(outward_bound)
        and absolute_error <= Decimal.from_float(outward_bound)
    )
    return {
        "observed": observed,
        "high_precision_expected": str(expected),
        "absolute_error": float(absolute_error),
        "outward_absolute_bound": outward_bound,
        "pass": bool(identity_passes),
    }


def _roundoff_bound(scale: float, *, operation_count: int = 1) -> float:
    factor = IEEE_BINARY64_UNIT_ROUNDOFF if operation_count == 1 else _gamma(operation_count)
    return _outward(
        FORWARD_ERROR_SAFETY_FACTOR * factor * max(1.0, abs(float(scale)))
    )


def _high_precision_formula_identity(
    statistics: Mapping[str, Any],
    powers: FloatArray,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
    user: int,
) -> dict[str, Any]:
    """Evaluate sequential and fully recomposed formulas at Decimal precision 80."""

    try:
        gain = complex(np.asarray(statistics["coherent_gain"])[user])
        second_row = np.asarray(statistics["second_effective_power"], dtype=np.float64)[user]
        norm = float(np.asarray(statistics["combiner_norm"])[user])
        observed = {
            "numerator": float(np.asarray(statistics["numerator"])[user]),
            "denominator": float(np.asarray(statistics["denominator"])[user]),
            "sinr": float(np.asarray(statistics["sinr"])[user]),
            "spectral_efficiency": float(
                np.asarray(statistics["spectral_efficiency"])[user]
            ),
        }
        with localcontext() as context:
            context.prec = 80
            real = _decimal_from_float(gain.real)
            imag = _decimal_from_float(gain.imag)
            power = _decimal_from_float(float(powers[user]))
            numerator = power * (real * real + imag * imag)
            received_terms = [
                _decimal_from_float(float(second_row[index]))
                * _decimal_from_float(float(powers[index]))
                for index in range(powers.size)
            ]
            received = sum(received_terms, Decimal(0))
            noise_term = _decimal_from_float(noise_variance) * _decimal_from_float(norm)
            observed_numerator = _decimal_from_float(observed["numerator"])
            observed_denominator = _decimal_from_float(observed["denominator"])
            observed_sinr = _decimal_from_float(observed["sinr"])
            local_denominator = received - observed_numerator + noise_term
            local_sinr = observed_numerator / observed_denominator
            prelog = Decimal(tau_c - tau_p) / Decimal(tau_c)
            local_spectral_efficiency = prelog * (
                (Decimal(1) + observed_sinr).ln() / Decimal(2).ln()
            )
            recomposed_denominator = received - numerator + noise_term
            recomposed_sinr = numerator / recomposed_denominator
            recomposed_spectral_efficiency = prelog * (
                (Decimal(1) + recomposed_sinr).ln() / Decimal(2).ln()
            )

            numerator_scale = max(
                Decimal(1), abs(numerator), abs(observed_numerator)
            )
            denominator_scale = max(
                Decimal(1),
                sum((abs(term) for term in received_terms), Decimal(0))
                + abs(observed_numerator)
                + abs(noise_term)
                + abs(observed_denominator),
            )
            numerator_bound = _roundoff_bound(
                float(numerator_scale), operation_count=4
            )
            denominator_bound = _roundoff_bound(
                float(denominator_scale), operation_count=powers.size + 3
            )
            local = {
                "numerator": _formula_identity_entry(
                    observed["numerator"], numerator, numerator_bound
                ),
                "denominator": _formula_identity_entry(
                    observed["denominator"], local_denominator, denominator_bound
                ),
                "sinr": _formula_identity_entry(
                    observed["sinr"],
                    local_sinr,
                    _roundoff_bound(
                        max(abs(observed["sinr"]), abs(float(local_sinr)))
                    ),
                ),
                "spectral_efficiency": _formula_identity_entry(
                    observed["spectral_efficiency"],
                    local_spectral_efficiency,
                    _roundoff_bound(
                        max(
                            abs(observed["spectral_efficiency"]),
                            abs(float(local_spectral_efficiency)),
                        )
                    ),
                ),
            }
            recomposed_denominator_float = float(recomposed_denominator)
            recomposed_sinr_float = float(recomposed_sinr)
            denominator_margin = min(
                observed["denominator"], recomposed_denominator_float
            ) - denominator_bound
            if denominator_margin > 0.0 and recomposed_sinr_float >= 0.0:
                recomposed_sinr_bound = _outward(
                    numerator_bound / denominator_margin
                    + (abs(float(numerator)) + numerator_bound)
                    * denominator_bound
                    / denominator_margin**2
                    + _roundoff_bound(
                        max(abs(observed["sinr"]), abs(recomposed_sinr_float))
                    )
                )
                recomposed_gamma_minus = max(
                    0.0,
                    min(observed["sinr"], recomposed_sinr_float)
                    - recomposed_sinr_bound,
                )
                recomposed_se_bound = _outward(
                    (float(prelog) * recomposed_sinr_bound)
                    / (math.log(2.0) * (1.0 + recomposed_gamma_minus))
                    + _roundoff_bound(
                        max(
                            abs(observed["spectral_efficiency"]),
                            abs(float(recomposed_spectral_efficiency)),
                        )
                    )
                )
            else:
                recomposed_sinr_bound = math.inf
                recomposed_se_bound = math.inf
            recomposed = {
                "numerator": _formula_identity_entry(
                    observed["numerator"], numerator, numerator_bound
                ),
                "denominator": _formula_identity_entry(
                    observed["denominator"],
                    recomposed_denominator,
                    denominator_bound,
                ),
                "sinr": _formula_identity_entry(
                    observed["sinr"], recomposed_sinr, recomposed_sinr_bound
                ),
                "spectral_efficiency": _formula_identity_entry(
                    observed["spectral_efficiency"],
                    recomposed_spectral_efficiency,
                    recomposed_se_bound,
                ),
            }
            return {
                "precision_digits": 80,
                "local": local,
                "recomposed": recomposed,
                "all_pass": all(
                    entry["pass"]
                    for mode in (local, recomposed)
                    for entry in mode.values()
                ),
            }
    except (ArithmeticError, DecimalException, OverflowError, ValueError):
        failed = {
            field: {
                "observed": float(np.asarray(statistics[field])[user]),
                "high_precision_expected": None,
                "absolute_error": math.inf,
                "outward_absolute_bound": math.inf,
                "pass": False,
            }
            for field in ("numerator", "denominator", "sinr", "spectral_efficiency")
        }
        return {
            "precision_digits": 80,
            "local": dict(failed),
            "recomposed": dict(failed),
            "all_pass": False,
        }


def certify_uatf_statistics(
    online_neumaier: Mapping[str, Any],
    fixed_order_fsum: Mapping[str, Any],
    ordinary_direct: Mapping[str, Any],
    *,
    contribution_cache: SufficientContributionCache,
    powers: object,
    noise_variance: float,
    tau_p: int,
    tau_c: int,
    context: Mapping[str, Any],
    diagnostic_sink: Callable[[Mapping[str, Any]], None],
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Certify primitive equality and propagated SINR/SE forward error.

    Every user diagnostic is emitted before a possible aggregate exception.
    The quotient bound uses numerator/denominator forward uncertainty and
    positive denominator lower margins. The SE bound applies the exact log2
    Lipschitz factor on the observed nonnegative interval.
    """

    _validate_contribution_cache(contribution_cache)
    if not callable(diagnostic_sink):
        raise TypeError("diagnostic_sink must be callable")
    required_context = {
        "condition_index",
        "condition_id",
        "receiver_index",
        "receiver_id",
        "split",
    }
    if set(context) != required_context:
        raise ValueError("diagnostic context keys differ from exact registered coordinate")
    paths = {
        "online_neumaier": online_neumaier,
        "fixed_order_fsum": fixed_order_fsum,
        "ordinary_direct": ordinary_direct,
    }
    users = contribution_cache.user_count
    expected_count = contribution_cache.sample_count
    required_fields = {
        "count",
        "coherent_gain",
        "second_effective_power",
        "combiner_norm",
        "numerator",
        "denominator",
        "sinr",
        "spectral_efficiency",
    }
    expected_shapes = {
        "coherent_gain": (users,),
        "second_effective_power": (users, users),
        "combiner_norm": (users,),
        "numerator": (users,),
        "denominator": (users,),
        "sinr": (users,),
        "spectral_efficiency": (users,),
    }
    expected_dtypes = {
        "coherent_gain": np.dtype(np.complex128),
        "second_effective_power": np.dtype(np.float64),
        "combiner_norm": np.dtype(np.float64),
        "numerator": np.dtype(np.float64),
        "denominator": np.dtype(np.float64),
        "sinr": np.dtype(np.float64),
        "spectral_efficiency": np.dtype(np.float64),
    }
    for path_name, statistics in paths.items():
        if set(statistics) != required_fields:
            raise ValueError(f"{path_name} statistic keys differ")
        if int(statistics["count"]) != expected_count:
            raise ValueError(f"{path_name} sample count differs from contribution cache")
        for field, shape in expected_shapes.items():
            array = np.asarray(statistics[field])
            if array.shape != shape:
                raise ValueError(f"{path_name}.{field} has wrong shape")
            if array.dtype != expected_dtypes[field]:
                raise ValueError(f"{path_name}.{field} has wrong dtype")

    payload_powers = np.asarray(powers, dtype=np.float64)
    if payload_powers.shape != (users,) or np.any(payload_powers < 0.0) or not np.all(np.isfinite(payload_powers)):
        raise ValueError("payload powers differ from certified user dimension")
    if not np.isfinite(noise_variance) or noise_variance <= 0.0 or not 0 <= tau_p < tau_c:
        raise ValueError("noise variance or coherence interval is invalid")

    pair_names = (
        ("NEUMAIER_VS_MATH_FSUM", "online_neumaier", "fixed_order_fsum"),
        ("NEUMAIER_VS_ORDINARY_ARRAY", "online_neumaier", "ordinary_direct"),
        ("MATH_FSUM_VS_ORDINARY_ARRAY", "fixed_order_fsum", "ordinary_direct"),
    )
    coherent_abs_mean, second_abs_mean, norm_abs_mean = _absolute_means(contribution_cache)
    reduction_gamma = _gamma(expected_count)
    dot_gamma = _gamma(users + 3)
    primitive_scale = FORWARD_ERROR_SAFETY_FACTOR * reduction_gamma
    coherent_uncertainty = np.nextafter(
        np.asarray(primitive_scale * coherent_abs_mean, dtype=np.float64), np.inf
    )
    second_uncertainty = np.nextafter(
        np.asarray(primitive_scale * second_abs_mean, dtype=np.float64), np.inf
    )
    norm_uncertainty = np.nextafter(
        np.asarray(primitive_scale * norm_abs_mean, dtype=np.float64), np.inf
    )
    second_dot_uncertainty = second_uncertainty @ np.abs(payload_powers)
    noise_uncertainty = abs(float(noise_variance)) * norm_uncertainty
    path_numerator_uncertainty: dict[str, FloatArray] = {}
    path_denominator_uncertainty: dict[str, FloatArray] = {}
    path_denominator_scale: dict[str, FloatArray] = {}
    for path_name, statistics in paths.items():
        gain = np.asarray(statistics["coherent_gain"], dtype=np.complex128)
        second = np.asarray(statistics["second_effective_power"], dtype=np.float64)
        norm = np.asarray(statistics["combiner_norm"], dtype=np.float64)
        numerator = np.asarray(statistics["numerator"], dtype=np.float64)
        numerator_uncertainty = (
            payload_powers
            * (2.0 * np.abs(gain) * coherent_uncertainty + coherent_uncertainty**2)
            + FORWARD_ERROR_SAFETY_FACTOR
            * _gamma(4)
            * np.maximum(1.0, np.abs(numerator))
        )
        denominator_scale = (
            np.abs(second) @ np.abs(payload_powers)
            + np.abs(numerator)
            + abs(float(noise_variance)) * np.abs(norm)
        )
        denominator_uncertainty = (
            second_dot_uncertainty
            + numerator_uncertainty
            + noise_uncertainty
            + FORWARD_ERROR_SAFETY_FACTOR
            * dot_gamma
            * np.maximum(1.0, denominator_scale)
        )
        path_numerator_uncertainty[path_name] = np.nextafter(
            np.asarray(numerator_uncertainty, dtype=np.float64), np.inf
        )
        path_denominator_uncertainty[path_name] = np.nextafter(
            np.asarray(denominator_uncertainty, dtype=np.float64), np.inf
        )
        path_denominator_scale[path_name] = np.asarray(
            denominator_scale, dtype=np.float64
        )

    failures: list[tuple[int, list[str]]] = []
    maximum_errors = {
        "coherent_gain": 0.0,
        "second_effective_power": 0.0,
        "combiner_norm": 0.0,
        "numerator": 0.0,
        "denominator": 0.0,
        "sinr": 0.0,
        "spectral_efficiency": 0.0,
    }
    maximum_sinr_envelope = 0.0
    maximum_se_envelope = 0.0
    maximum_cancellation_metric = 0.0
    prelog = 1.0 - float(tau_p) / float(tau_c)

    for user in range(users):
        reasons: list[str] = []
        primitive_errors: dict[str, dict[str, float]] = {}
        primitive_passes: dict[str, dict[str, bool]] = {}
        for pair, first_name, second_name in pair_names:
            errors = _primitive_pair_errors(paths[first_name], paths[second_name], user)
            passes = _primitive_pair_passes(paths[first_name], paths[second_name], user)
            primitive_errors[pair] = errors
            primitive_passes[pair] = passes
            for field, error in errors.items():
                aggregate_field = "coherent_gain" if field.startswith("coherent_gain_") else field
                maximum_errors[aggregate_field] = max(
                    maximum_errors[aggregate_field], error
                )
            for field, passed in passes.items():
                if not passed:
                    reasons.append(f"PRIMITIVE_{field.upper()}_{pair}_MISMATCH")

        formula_identity = {
            path_name: _high_precision_formula_identity(
                statistics,
                payload_powers,
                float(noise_variance),
                tau_p,
                tau_c,
                user,
            )
            for path_name, statistics in paths.items()
        }
        for path_name, identity in formula_identity.items():
            for mode_name in ("local", "recomposed"):
                for field, result in identity[mode_name].items():
                    if not result["pass"]:
                        reasons.append(
                            f"FORMULA_IDENTITY_{field.upper()}_{mode_name.upper()}_"
                            f"{path_name.upper()}_MISMATCH"
                        )

        path_denominators = {
            name: float(np.asarray(value["denominator"])[user])
            for name, value in paths.items()
        }
        denominator_positive_by_path = {
            name: math.isfinite(value) and value > 0.0
            for name, value in path_denominators.items()
        }
        for path_name, passed in denominator_positive_by_path.items():
            if not passed:
                reasons.append(
                    f"NONPOSITIVE_OR_NONFINITE_DENOMINATOR_{path_name.upper()}"
                )
        cancellation_by_path = {
            name: (
                float(path_denominator_scale[name][user] / abs(path_denominators[name]))
                if denominator_positive_by_path[name]
                else math.inf
            )
            for name in paths
        }
        cancellation_metric = max(cancellation_by_path.values())
        maximum_cancellation_metric = max(maximum_cancellation_metric, cancellation_metric)

        derived_errors: dict[str, dict[str, float]] = {}
        derived_bounds: dict[str, dict[str, Any]] = {}
        for pair, first_name, second_name in pair_names:
            first = paths[first_name]
            second = paths[second_name]
            first_numerator = float(np.asarray(first["numerator"])[user])
            second_numerator = float(np.asarray(second["numerator"])[user])
            first_denominator = float(np.asarray(first["denominator"])[user])
            second_denominator = float(np.asarray(second["denominator"])[user])
            first_sinr = float(np.asarray(first["sinr"])[user])
            second_sinr = float(np.asarray(second["sinr"])[user])
            first_se = float(np.asarray(first["spectral_efficiency"])[user])
            second_se = float(np.asarray(second["spectral_efficiency"])[user])
            sinr_error = abs(first_sinr - second_sinr)
            se_error = abs(first_se - second_se)
            maximum_errors["sinr"] = max(maximum_errors["sinr"], sinr_error)
            maximum_errors["spectral_efficiency"] = max(
                maximum_errors["spectral_efficiency"], se_error
            )

            first_numerator_uncertainty = float(
                path_numerator_uncertainty[first_name][user]
            )
            second_numerator_uncertainty = float(
                path_numerator_uncertainty[second_name][user]
            )
            first_denominator_uncertainty = float(
                path_denominator_uncertainty[first_name][user]
            )
            second_denominator_uncertainty = float(
                path_denominator_uncertainty[second_name][user]
            )
            delta_numerator = abs(first_numerator - second_numerator)
            delta_denominator = abs(first_denominator - second_denominator)
            numerator_error_budget = _outward(
                delta_numerator
                + first_numerator_uncertainty
                + second_numerator_uncertainty
            )
            denominator_error_budget = _outward(
                delta_denominator
                + first_denominator_uncertainty
                + second_denominator_uncertainty
            )
            lower_first = first_denominator - first_denominator_uncertainty
            lower_second = second_denominator - second_denominator_uncertainty
            denominator_lower_margin = min(lower_first, lower_second)
            numerator_upper = max(
                abs(first_numerator) + first_numerator_uncertainty,
                abs(second_numerator) + second_numerator_uncertainty,
            )
            finite_domain = all(
                math.isfinite(value)
                for value in (
                    first_numerator,
                    second_numerator,
                    first_sinr,
                    second_sinr,
                    first_se,
                    second_se,
                    numerator_error_budget,
                    denominator_error_budget,
                    denominator_lower_margin,
                    numerator_upper,
                )
            )
            denominator_margin_pass = bool(
                finite_domain and denominator_lower_margin > 0.0
            )
            sinr_domain_pass = bool(
                finite_domain and first_sinr >= 0.0 and second_sinr >= 0.0
            )
            if denominator_margin_pass and sinr_domain_pass:
                sinr_envelope = _outward(
                    numerator_error_budget / denominator_lower_margin
                    + numerator_upper
                    * denominator_error_budget
                    / denominator_lower_margin**2
                    + _roundoff_bound(max(abs(first_sinr), abs(second_sinr)))
                )
                gamma_minus = max(
                    0.0, min(first_sinr, second_sinr) - sinr_envelope
                )
                lipschitz = prelog / (
                    math.log(2.0) * (1.0 + gamma_minus)
                )
                se_envelope = _outward(
                    lipschitz * sinr_envelope
                    + _roundoff_bound(max(abs(first_se), abs(second_se)))
                )
            else:
                sinr_envelope = math.inf
                gamma_minus = 0.0
                se_envelope = math.inf
                reasons.append(f"DERIVED_DOMAIN_{pair}_INVALID")

            maximum_sinr_envelope = max(maximum_sinr_envelope, sinr_envelope)
            maximum_se_envelope = max(maximum_se_envelope, se_envelope)
            sinr_envelope_pass = bool(
                math.isfinite(sinr_error)
                and math.isfinite(sinr_envelope)
                and sinr_error <= sinr_envelope
            )
            se_envelope_pass = bool(
                math.isfinite(se_error)
                and math.isfinite(se_envelope)
                and se_error <= se_envelope
            )
            se_ceiling_pass = bool(
                math.isfinite(se_envelope)
                and se_envelope <= ABSOLUTE_SE_ENVELOPE_CEILING
            )
            if not denominator_margin_pass:
                reasons.append(
                    f"NONPOSITIVE_DENOMINATOR_UNCERTAINTY_MARGIN_{pair}"
                )
            if not sinr_envelope_pass:
                reasons.append(f"SINR_FORWARD_ERROR_{pair}_EXCEEDED")
            if not se_envelope_pass:
                reasons.append(f"SE_FORWARD_ERROR_{pair}_EXCEEDED")
            if not se_ceiling_pass:
                reasons.append(f"SE_ENVELOPE_CEILING_{pair}_EXCEEDED")
            derived_errors[pair] = {
                "numerator_absolute_difference": delta_numerator,
                "denominator_absolute_difference": delta_denominator,
                "sinr_absolute_error": sinr_error,
                "spectral_efficiency_absolute_error": se_error,
            }
            derived_bounds[pair] = {
                "first_path": first_name,
                "second_path": second_name,
                "first_numerator_uncertainty": first_numerator_uncertainty,
                "second_numerator_uncertainty": second_numerator_uncertainty,
                "first_denominator_uncertainty": first_denominator_uncertainty,
                "second_denominator_uncertainty": second_denominator_uncertainty,
                "E_N": numerator_error_budget,
                "E_D": denominator_error_budget,
                "first_denominator_lower_margin": lower_first,
                "second_denominator_lower_margin": lower_second,
                "d_minus": denominator_lower_margin,
                "n_plus": numerator_upper,
                "sinr_forward_error_envelope": sinr_envelope,
                "gamma_minus": gamma_minus,
                "spectral_efficiency_forward_error_envelope": se_envelope,
                "gates": {
                    "both_stored_denominators_positive": bool(
                        denominator_positive_by_path[first_name]
                        and denominator_positive_by_path[second_name]
                    ),
                    "denominator_uncertainty_margin_positive": denominator_margin_pass,
                    "sinr_domain_valid": sinr_domain_pass,
                    "sinr_error_within_envelope": sinr_envelope_pass,
                    "spectral_efficiency_error_within_envelope": se_envelope_pass,
                    "spectral_efficiency_envelope_within_absolute_ceiling": se_ceiling_pass,
                },
            }

        reasons = sorted(set(reasons))
        diagnostic = {
            "schema_version": "r2c01-v3-baseline-precision-e0-02-diagnostic/1.0",
            "experiment_id": EXPERIMENT_ID,
            **dict(context),
            "user_index": user,
            "sample_count": expected_count,
            "values": {name: _path_values(value, user) for name, value in paths.items()},
            "errors": {"primitive": primitive_errors, "derived": derived_errors},
            "formula_identity": formula_identity,
            "bounds": {
                "primitive_rtol": PRIMITIVE_RTOL,
                "primitive_atol": PRIMITIVE_ATOL,
                "ieee_binary64_unit_roundoff": IEEE_BINARY64_UNIT_ROUNDOFF,
                "forward_error_safety_factor": FORWARD_ERROR_SAFETY_FACTOR,
                "reduction_gamma": reduction_gamma,
                "dot_gamma": dot_gamma,
                "contribution_scales": {
                    "coherent_gain_abs_mean": float(coherent_abs_mean[user]),
                    "second_effective_power_abs_mean_row": second_abs_mean[user].tolist(),
                    "combiner_norm_abs_mean": float(norm_abs_mean[user]),
                },
                "primitive_uncertainties": {
                    "coherent_gain": float(coherent_uncertainty[user]),
                    "second_effective_power_row": second_uncertainty[user].tolist(),
                    "combiner_norm": float(norm_uncertainty[user]),
                },
                "path_uncertainties": {
                    name: {
                        "numerator": float(path_numerator_uncertainty[name][user]),
                        "denominator": float(
                            path_denominator_uncertainty[name][user]
                        ),
                        "denominator_absolute_term_scale": float(
                            path_denominator_scale[name][user]
                        ),
                    }
                    for name in paths
                },
                "stored_denominator_positive_by_path": denominator_positive_by_path,
                "derived": derived_bounds,
                "absolute_se_envelope_ceiling": ABSOLUTE_SE_ENVELOPE_CEILING,
            },
            "cancellation_metric": {
                "by_path": cancellation_by_path,
                "maximum": cancellation_metric,
            },
            "primitive_pair_passes": primitive_passes,
            "status": "PASS" if not reasons else "FAIL",
            "failure_reasons": reasons,
        }
        diagnostic_sink(MappingProxyType(_diagnostic_json_safe(diagnostic)))
        if reasons:
            failures.append((user, reasons))

    if failures and raise_on_failure:
        first_user, first_reasons = failures[0]
        raise BaselinePrecisionProtocolError(
            "E0-02 UatF numerical certification failed at "
            f"condition={context['condition_id']}, receiver={context['receiver_id']}, "
            f"split={context['split']}, user={first_user}: {','.join(first_reasons)}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "user_count": users,
        "diagnostic_count": users,
        "failed_user_count": len(failures),
        "failed_user_indices": [user for user, _ in failures],
        "first_failure_reasons": failures[0][1] if failures else [],
        "maximum_absolute_differences": maximum_errors,
        "maximum_sinr_forward_error_envelope": maximum_sinr_envelope,
        "maximum_spectral_efficiency_forward_error_envelope": maximum_se_envelope,
        "maximum_cancellation_metric": maximum_cancellation_metric,
    }


def require_complete_numerical_certification(
    certificates: list[Mapping[str, Any]],
    *,
    expected_rows: int,
    expected_diagnostics: int,
) -> None:
    """Raise only after all prospective rows and diagnostics were completed."""

    diagnostic_count = sum(int(value["diagnostic_count"]) for value in certificates)
    if len(certificates) != expected_rows or diagnostic_count != expected_diagnostics:
        raise BaselinePrecisionProtocolError(
            "E0-02 numerical diagnostic completeness differs: "
            f"rows={len(certificates)}/{expected_rows}, "
            f"diagnostics={diagnostic_count}/{expected_diagnostics}"
        )
    failures = [value for value in certificates if value["status"] != "PASS"]
    if failures:
        failed_users = sum(int(value["failed_user_count"]) for value in failures)
        raise BaselinePrecisionProtocolError(
            "E0-02 aggregate UatF numerical certification failed only after "
            f"all {diagnostic_count} diagnostics; failed_rows={len(failures)}, "
            f"failed_users={failed_users}"
        )


def _validate_parent_contract(
    config: BaselinePrecisionConfig,
    registries: Registries,
) -> None:
    if config.raw.get("experiment_id") != PARENT_EXPERIMENT_ID:
        raise BaselinePrecisionProtocolError("E0-02 requires frozen E0-01 parent configuration")
    if config.frame_count != 192 or config.realizations != 64:
        raise BaselinePrecisionProtocolError("E0-02 requires all 192 registered E0-01 frames")
    if len(config.conditions) != 3 or len(config.receivers) != 6:
        raise BaselinePrecisionProtocolError("E0-02 condition/receiver registry differs from E0-01")
    expected = build_registries(config)
    if (
        registries.frame_seeds.shape != (3, 64)
        or not np.array_equal(registries.frame_seeds, expected.frame_seeds)
        or registries.scalar_witness_keys != expected.scalar_witness_keys
        or tuple(dict(row) for row in registries.seed_rows)
        != tuple(dict(row) for row in expected.seed_rows)
        or tuple(dict(row) for row in registries.frame_rows)
        != tuple(dict(row) for row in expected.frame_rows)
    ):
        raise BaselinePrecisionProtocolError("E0-02 registries are not exact frozen E0-01 replay")


def execute_baseline_precision_e0_repair(
    parent_config: BaselinePrecisionConfig,
    registries: Registries,
    *,
    progress_sink: Callable[[Mapping[str, Any]], None],
    resource_guard: Callable[[str], None],
    diagnostic_sink: Callable[[Mapping[str, Any]], None],
) -> BaselinePrecisionProducts:
    """Replay the exact E0-01 cases with prospective E0-02 numerics."""

    _validate_parent_contract(parent_config, registries)
    if not callable(progress_sink) or not callable(resource_guard) or not callable(diagnostic_sink):
        raise TypeError("all persistence/resource callbacks must be callable")
    channels = np.empty((3, 64, 128, 16), dtype=np.complex128)
    estimates = np.empty_like(channels)
    witness_identity: list[tuple[int, int, int, int]] = []
    witness_users: list[NDArray[np.int64]] = []
    witness_models: list[Any] = []
    witness_observations: list[ComplexArray] = []
    witness_batched: list[ComplexArray] = []
    witness_scalar: list[ComplexArray] = []
    rows: list[dict[str, Any]] = []
    certificates: list[Mapping[str, Any]] = []
    completed = 0
    maximum_primitive_difference = 0.0
    maximum_se_envelope = 0.0
    maximum_sinr_envelope = 0.0
    maximum_cancellation_metric = 0.0

    for condition_index, condition in enumerate(parent_config.conditions):
        scenario = parent_config.scenario(condition)
        topology = generate_topology(scenario, seed=parent_config.topology_seed)
        coherent_cache = np.empty((6, 64, 16), dtype=np.complex128)
        second_cache = np.empty((6, 64, 16, 16), dtype=np.float64)
        norm_cache = np.empty((6, 64, 16), dtype=np.float64)
        sufficient_cache_bytes = (
            coherent_cache.nbytes + second_cache.nbytes + norm_cache.nbytes
        )
        if sufficient_cache_bytes > MAX_SUFFICIENT_CACHE_BYTES:
            raise BaselinePrecisionProtocolError(
                "per-frame sufficient-contribution cache exceeds byte cap"
            )
        online = {
            (receiver_index, split): NeumaierUatFAccumulator(16)
            for receiver_index in range(6)
            for split in ("full", "even", "odd")
        }

        for realization in range(64):
            resource_guard("BEFORE_FRAME")
            frame = generate_channel_frame(
                topology,
                scenario,
                seed=int(registries.frame_seeds[condition_index, realization]),
            )
            models: list[Any] = []
            observations: list[Any] = []
            groups: list[NDArray[np.int64]] = []
            coordinates: list[tuple[int, int]] = []
            for access_point in range(32):
                for pilot in range(4):
                    model, observation, users, _ = pilot_group(
                        topology,
                        frame,
                        scenario,
                        access_point=access_point,
                        pilot=pilot,
                    )
                    if model.group_size != 4 or users.size != 4:
                        raise BaselinePrecisionProtocolError("balanced P=4 group invariant failed")
                    models.append(model)
                    observations.append(observation)
                    groups.append(users)
                    coordinates.append((access_point, pilot))
            batched = batched_b1(stack_models(models, observations))
            for local_index, ((access_point, pilot), users) in enumerate(zip(coordinates, groups)):
                if (condition_index, realization, local_index) in registries.scalar_witness_keys:
                    scalar = np.column_stack(
                        [
                            widely_linear_mmse(
                                models[local_index], observations[local_index], target
                            )
                            for target in range(4)
                        ]
                    )
                    if not np.allclose(batched[local_index], scalar, rtol=5e-12, atol=5e-12):
                        raise BaselinePrecisionProtocolError("batched/scalar B1 equivalence failed")
                    witness_identity.append(
                        (condition_index, realization, access_point, pilot)
                    )
                    witness_users.append(np.asarray(users, dtype=np.int64))
                    witness_models.append(models[local_index])
                    witness_observations.append(
                        np.asarray(observations[local_index], dtype=np.complex128)
                    )
                    witness_batched.append(
                        np.asarray(batched[local_index], dtype=np.complex128)
                    )
                    witness_scalar.append(np.asarray(scalar, dtype=np.complex128))
            estimate_lnk = assemble_frame_estimates(batched, groups, coordinates)
            channel = np.asarray(frame.channels.reshape(128, 16), dtype=np.complex128)
            estimate = np.asarray(estimate_lnk.reshape(128, 16), dtype=np.complex128)
            channels[condition_index, realization] = channel
            estimates[condition_index, realization] = estimate
            split = "even" if realization % 2 == 0 else "odd"
            for receiver_index, receiver in enumerate(parent_config.receivers):
                # Sole combiner construction for this frame/profile.
                combiner = receiver_combiner(estimate, receiver)
                coherent, second, norm = frame_sufficient_contributions(channel, combiner)
                coherent_cache[receiver_index, realization] = coherent
                second_cache[receiver_index, realization] = second
                norm_cache[receiver_index, realization] = norm
                online[(receiver_index, "full")].update_contributions(
                    coherent, second, norm
                )
                online[(receiver_index, split)].update_contributions(coherent, second, norm)
            completed += 1
            progress_sink(
                MappingProxyType(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "parent_experiment_id": PARENT_EXPERIMENT_ID,
                        "frame_index": completed - 1,
                        "condition_index": condition_index,
                        "condition_id": condition.id,
                        "realization_index": realization,
                        "frame_seed": int(
                            registries.frame_seeds[condition_index, realization]
                        ),
                        "completed_frames": completed,
                        "expected_frames": parent_config.frame_count,
                        "status": "COMPLETE",
                    }
                )
            )
            resource_guard("AFTER_FRAME")

        for receiver_index, receiver in enumerate(parent_config.receivers):
            full_cache = SufficientContributionCache(
                coherent_cache[receiver_index],
                second_cache[receiver_index],
                norm_cache[receiver_index],
            )
            _validate_contribution_cache(full_cache)
            for split, mask in (
                ("full", np.ones(64, dtype=bool)),
                ("even", np.arange(64) % 2 == 0),
                ("odd", np.arange(64) % 2 == 1),
            ):
                cache = full_cache.select(mask)
                powers = np.full(16, scenario.payload_power, dtype=np.float64)
                neumaier = online[(receiver_index, split)].finalize(
                    powers,
                    scenario.payload_noise_variance,
                    scenario.pilot_symbols,
                    scenario.coherence_symbols,
                )
                fsum = fixed_order_fsum_statistics(
                    cache,
                    powers,
                    scenario.payload_noise_variance,
                    scenario.pilot_symbols,
                    scenario.coherence_symbols,
                )
                direct = ordinary_direct_statistics(
                    cache,
                    powers,
                    scenario.payload_noise_variance,
                    scenario.pilot_symbols,
                    scenario.coherence_symbols,
                )
                context = {
                    "condition_index": condition_index,
                    "condition_id": condition.id,
                    "receiver_index": receiver_index,
                    "receiver_id": receiver.id,
                    "split": split,
                }
                certificate = certify_uatf_statistics(
                    neumaier,
                    fsum,
                    direct,
                    contribution_cache=cache,
                    powers=powers,
                    noise_variance=scenario.payload_noise_variance,
                    tau_p=scenario.pilot_symbols,
                    tau_c=scenario.coherence_symbols,
                    context=context,
                    diagnostic_sink=diagnostic_sink,
                    raise_on_failure=False,
                )
                certificates.append(certificate)
                maximum_primitive_difference = max(
                    maximum_primitive_difference,
                    max(
                        certificate["maximum_absolute_differences"][key]
                        for key in (
                            "coherent_gain",
                            "second_effective_power",
                            "combiner_norm",
                            "numerator",
                            "denominator",
                        )
                    ),
                )
                maximum_sinr_envelope = max(
                    maximum_sinr_envelope,
                    float(certificate["maximum_sinr_forward_error_envelope"]),
                )
                maximum_se_envelope = max(
                    maximum_se_envelope,
                    float(
                        certificate[
                            "maximum_spectral_efficiency_forward_error_envelope"
                        ]
                    ),
                )
                maximum_cancellation_metric = max(
                    maximum_cancellation_metric,
                    float(certificate["maximum_cancellation_metric"]),
                )
                rows.append(
                    {
                        **context,
                        "online_neumaier": _statistics_json(neumaier),
                        "fixed_order_fsum": _statistics_json(fsum),
                        "array_reconstruction": _statistics_json(direct),
                        "certificate": certificate,
                        "equivalence": certificate["status"],
                    }
                )
        resource_guard("AFTER_CONDITION_CERTIFICATION")

    if completed != 192 or len(witness_identity) != 24:
        raise BaselinePrecisionProtocolError("pipeline completion or witness count is wrong")
    require_complete_numerical_certification(
        certificates,
        expected_rows=54,
        expected_diagnostics=864,
    )
    identity_array = np.asarray(witness_identity, dtype=np.int64)
    order = np.lexsort(
        (
            identity_array[:, 3],
            identity_array[:, 2],
            identity_array[:, 1],
            identity_array[:, 0],
        )
    )
    identity_array = identity_array[order]
    witness_users_array = np.asarray(witness_users, dtype=np.int64)[order]
    ordered_models = [witness_models[index] for index in order]
    witness_observations_array = np.asarray(witness_observations)[order]
    witness_batched_array = np.asarray(witness_batched)[order]
    witness_scalar_array = np.asarray(witness_scalar)[order]
    scalar_errors = np.max(
        np.abs(witness_batched_array - witness_scalar_array), axis=(1, 2)
    )

    arrays = {
        "channels": channels,
        "b1_estimates": estimates,
        "frame_seeds": np.asarray(registries.frame_seeds, dtype=np.int64),
        "witness_identity": identity_array,
        "witness_users": witness_users_array,
        "witness_los": np.stack([model.los for model in ordered_models]),
        "witness_covariances": np.stack(
            [model.covariances for model in ordered_models]
        ),
        "witness_beta": np.stack([model.beta for model in ordered_models]),
        "witness_noise_covariance": np.stack(
            [model.noise_covariance for model in ordered_models]
        ),
        "witness_delta": np.stack([model.delta for model in ordered_models]),
        "witness_observation": witness_observations_array,
        "witness_batched_estimate": witness_batched_array,
        "witness_scalar_estimate": witness_scalar_array,
    }
    scalar_document = {
        "schema_version": "r2c01-v3-baseline-precision-scalar-equivalence/1.0",
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "estimator_id": "B1",
        "selection_rule": "INHERITED_E0_01_LOWEST_EIGHT_SHA256_RANKED_LOCAL_GROUPS_PER_CONDITION",
        "witness_count": 24,
        "targets_per_witness": 4,
        "rtol": 5e-12,
        "atol": 5e-12,
        "maximum_absolute_error": float(np.max(scalar_errors)),
        "per_witness_maximum_absolute_error": scalar_errors.tolist(),
        "equivalence": "PASS",
    }
    uatf_document = {
        "schema_version": "r2c01-v3-baseline-precision-uatf-equivalence/2.0",
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "estimator_id": "B1",
        "receiver_ids": [receiver.id for receiver in parent_config.receivers],
        "splits": ["full", "even", "odd"],
        "primitive_rtol": PRIMITIVE_RTOL,
        "primitive_atol": PRIMITIVE_ATOL,
        "ieee_binary64_unit_roundoff": IEEE_BINARY64_UNIT_ROUNDOFF,
        "forward_error_safety_factor": FORWARD_ERROR_SAFETY_FACTOR,
        "absolute_se_envelope_ceiling": ABSOLUTE_SE_ENVELOPE_CEILING,
        "row_count": len(rows),
        "maximum_absolute_primitive_difference": maximum_primitive_difference,
        "maximum_sinr_forward_error_envelope": maximum_sinr_envelope,
        "maximum_spectral_efficiency_forward_error_envelope": maximum_se_envelope,
        "maximum_cancellation_metric": maximum_cancellation_metric,
        "rows": rows,
        "effect_statistics": "ABSENT_BY_CONTRACT",
        "equivalence": "PASS",
    }
    summary = {
        "schema_version": "r2c01-v3-baseline-precision-e0-repair-summary/1.0",
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "execution_status": "COMPLETE",
        "repair_scope": "DETERMINISTIC_UATF_NUMERICAL_CERTIFICATION_ONLY",
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "topology_count": 1,
        "condition_count": 3,
        "frames_per_condition": 64,
        "frame_count": 192,
        "inherited_frame_seed_count": 192,
        "local_groups_per_frame": 128,
        "batched_group_size": 4,
        "estimator_ids": ["B1"],
        "receiver_ids": [receiver.id for receiver in parent_config.receivers],
        "scalar_witness_count": 24,
        "scalar_equivalence": "PASS",
        "uatf_three_path_numerical_certification": "PASS",
        "proposed_method_import_or_dispatch": "ABSENT_BY_CONTRACT",
        "scientific_effect_metrics": "ABSENT_BY_CONTRACT",
        "lambda_selection": "NOT_PERFORMED",
        "realization_count_selection": "NOT_PERFORMED",
        "final_partition": "SEALED_UNINSPECTED",
        "automatic_advance": "PROHIBITED",
        "independent_validation": "PENDING",
    }
    return BaselinePrecisionProducts(
        MappingProxyType(arrays),
        MappingProxyType(scalar_document),
        MappingProxyType(uatf_document),
        MappingProxyType(summary),
    )
