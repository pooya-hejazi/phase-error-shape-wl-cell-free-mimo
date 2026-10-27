"""Exact-reference centered-UatF certification for the final E0-03 repair.

Cached binary64 values are interpreted as exact rationals. A componentwise
RN-even common center and every centered diagonal sample are rounded once and
shared by all three reductions. Nothing is clamped.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .baseline_precision_e0 import (
    EXPERIMENT_ID as ROOT_REPLAY_SOURCE_ID,
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

EXPERIMENT_ID = "R2C01_V3_BASELINE_ONLY_PRECISION_E0_03"
IMMEDIATE_FAILED_PREDECESSOR_ID = "R2C01_V3_BASELINE_ONLY_PRECISION_E0_02"
PARENT_EXPERIMENT_ID = ROOT_REPLAY_SOURCE_ID
SCIENTIFIC_BOUNDARY = "NON_SCIENTIFIC_BASELINE_ONLY_ENGINEERING"
PRIMITIVE_RTOL = 1e-12
PRIMITIVE_ATOL = 1e-12
ABSOLUTE_SE_ORACLE_CEILING = 1e-9
ABSOLUTE_SE_ENVELOPE_CEILING = ABSOLUTE_SE_ORACLE_CEILING
ATANH_SERIES_TERMS = 64
MAX_EXACT_INTERVAL_WIDTH = Fraction(1, 10**60)
IEEE_BINARY64_UNIT_ROUNDOFF = 2.0**-53
MAX_SAMPLES_PER_CONDITION = 64
MAX_ANTENNAS = 128
MAX_USERS = 16
MAX_RECEIVERS = 6
MAX_SUFFICIENT_CACHE_BYTES = 2 * 1024 * 1024

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SufficientContributionCache:
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

    def select(self, mask: object) -> "SufficientContributionCache":
        selected = np.asarray(mask)
        if (selected.dtype != np.dtype(np.bool_)
                or selected.shape != (self.sample_count,) or not np.any(selected)):
            raise ValueError("selection must be a nonempty Boolean sample mask")
        result = SufficientContributionCache(
            np.asarray(self.coherent[selected], dtype=np.complex128),
            np.asarray(self.second[selected], dtype=np.float64),
            np.asarray(self.norm[selected], dtype=np.float64),
        )
        _validate_contribution_cache(result)
        return result


@dataclass(frozen=True)
class CenteredContributionCache:
    base: SufficientContributionCache
    center: ComplexArray
    centered_diagonal: FloatArray
    exact_centered_diagonal: tuple[tuple[Fraction, ...], ...]

    @property
    def sample_count(self) -> int:
        return self.base.sample_count

    @property
    def user_count(self) -> int:
        return self.base.user_count


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval endpoints are reversed")

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower


@dataclass(frozen=True)
class ExactUserOracle:
    user_index: int
    coherent_real: Fraction
    coherent_imag: Fraction
    centered_diagonal_mean: Fraction
    centered_residual: Fraction
    rounded_centered_diagonal_mean: Fraction
    rounded_centered_residual: Fraction
    second_row: tuple[Fraction, ...]
    combiner_norm: Fraction
    numerator: Fraction
    legacy_denominator: Fraction
    centered_denominator: Fraction
    rounded_centered_denominator: Fraction
    sinr: Fraction | None
    log2_one_plus_sinr: RationalInterval | None
    spectral_efficiency: RationalInterval | None
    rounded_sinr: Fraction | None
    rounded_log2_one_plus_sinr: RationalInterval | None
    rounded_spectral_efficiency: RationalInterval | None
    exact_centered_identity: bool


def _validate_contribution_cache(cache: SufficientContributionCache) -> None:
    coherent, second, norm = map(np.asarray, (cache.coherent, cache.second, cache.norm))
    if coherent.dtype != np.dtype(np.complex128):
        raise ValueError("coherent contributions must have dtype complex128")
    if second.dtype != np.dtype(np.float64) or norm.dtype != np.dtype(np.float64):
        raise ValueError("power and norm contributions must have dtype float64")
    if coherent.ndim != 2:
        raise ValueError("coherent contributions must have shape (S,K)")
    samples, users = coherent.shape
    if not 1 <= samples <= MAX_SAMPLES_PER_CONDITION or not 1 <= users <= MAX_USERS:
        raise ValueError("contribution cache exceeds registered bounds")
    if second.shape != (samples, users, users) or norm.shape != (samples, users):
        raise ValueError("sufficient-contribution shapes differ")
    if cache.nbytes > MAX_SUFFICIENT_CACHE_BYTES:
        raise ValueError("sufficient-contribution cache exceeds byte cap")
    if not all(np.all(np.isfinite(x)) for x in (coherent, second, norm)):
        raise ValueError("sufficient contributions must be finite")
    if np.any(second < 0.0) or np.any(norm < 0.0):
        raise ValueError("power and norm contributions must be nonnegative")


def frame_sufficient_contributions(
    channel: object, combiner: object
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    h = np.asarray(channel, dtype=np.complex128)
    v = np.asarray(combiner, dtype=np.complex128)
    if h.ndim != 2 or v.shape != h.shape:
        raise ValueError("one frame must contain matched (M,K) arrays")
    antennas, users = h.shape
    if not 1 <= antennas <= MAX_ANTENNAS or not 1 <= users <= MAX_USERS:
        raise ValueError("one frame exceeds registered bounds")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        raise ValueError("channel and combiner values must be finite")
    inner = v.conj().T @ h
    coherent = np.asarray(np.diag(inner), dtype=np.complex128)
    second = np.asarray(np.abs(inner) ** 2, dtype=np.float64)
    norm = np.asarray(np.sum(np.abs(v) ** 2, axis=0), dtype=np.float64)
    if not all(np.all(np.isfinite(x)) for x in (coherent, second, norm)):
        raise FloatingPointError("nonfinite per-frame contribution")
    return coherent, second, norm


def sufficient_contribution_cache(
    channel_samples: object, combiner_samples: object
) -> SufficientContributionCache:
    channels = np.asarray(channel_samples, dtype=np.complex128)
    combiners = np.asarray(combiner_samples, dtype=np.complex128)
    if channels.ndim != 3 or combiners.shape != channels.shape:
        raise ValueError("sample arrays must share shape (S,M,K)")
    samples, antennas, users = channels.shape
    if (not 1 <= samples <= MAX_SAMPLES_PER_CONDITION
            or not 1 <= antennas <= MAX_ANTENNAS or not 1 <= users <= MAX_USERS):
        raise ValueError("sample dimensions exceed registered bounds")
    coherent = np.empty((samples, users), np.complex128)
    second = np.empty((samples, users, users), np.float64)
    norm = np.empty((samples, users), np.float64)
    for sample in range(samples):
        coherent[sample], second[sample], norm[sample] = frame_sufficient_contributions(
            channels[sample], combiners[sample]
        )
    result = SufficientContributionCache(coherent, second, norm)
    _validate_contribution_cache(result)
    return result


def _as_fraction(value: object) -> Fraction:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("only finite binary64 values have exact semantics")
    return Fraction.from_float(scalar)


def _mean_fraction(values: object) -> Fraction:
    flat = np.asarray(values).reshape(-1)
    if flat.size < 1:
        raise ValueError("cannot form an exact empty mean")
    return sum((_as_fraction(x) for x in flat), Fraction(0)) / int(flat.size)


def _binary64_lsb(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0] & 1


def _round_fraction_to_binary64(value: Fraction) -> float:
    """Round exact rational to finite binary64, ties to even."""
    if value == 0:
        return 0.0
    initial = float(value)
    if not math.isfinite(initial):
        raise OverflowError("exact value is outside finite binary64")
    candidates = {initial}
    for direction in (-math.inf, math.inf):
        neighbor = math.nextafter(initial, direction)
        if math.isfinite(neighbor):
            candidates.add(neighbor)
    ordered = sorted(candidates)
    exact_candidates = [Fraction.from_float(x) for x in ordered]
    bracketed = value in exact_candidates or any(
        left <= value <= right
        for left, right in zip(exact_candidates, exact_candidates[1:])
    )
    if not bracketed:
        raise ArithmeticError("initial rational conversion did not supply an adjacent bracket")
    return float(min(
        candidates,
        key=lambda x: (abs(Fraction.from_float(x) - value), _binary64_lsb(x), abs(x)),
    ))


def _componentwise_center(coherent: ComplexArray) -> ComplexArray:
    return np.asarray([
        complex(
            _round_fraction_to_binary64(_mean_fraction(coherent[:, user].real)),
            _round_fraction_to_binary64(_mean_fraction(coherent[:, user].imag)),
        )
        for user in range(coherent.shape[1])
    ], dtype=np.complex128)


def build_centered_contribution_cache(
    cache: SufficientContributionCache,
) -> CenteredContributionCache:
    """Compute exact w then perform its sole RN-even binary64 conversion."""
    _validate_contribution_cache(cache)
    center = _componentwise_center(cache.coherent)
    samples, users = cache.coherent.shape
    rounded = np.empty((samples, users), dtype=np.float64)
    exact_rows: list[tuple[Fraction, ...]] = []
    for sample in range(samples):
        row: list[Fraction] = []
        for user in range(users):
            cr = _as_fraction(cache.coherent[sample, user].real)
            ci = _as_fraction(cache.coherent[sample, user].imag)
            ar = _as_fraction(center[user].real)
            ai = _as_fraction(center[user].imag)
            w = (_as_fraction(cache.second[sample, user, user])
                 - 2 * (ar * cr + ai * ci) + ar * ar + ai * ai)
            row.append(w)
            rounded[sample, user] = _round_fraction_to_binary64(w)
        exact_rows.append(tuple(row))
    result = CenteredContributionCache(cache, center, rounded, tuple(exact_rows))
    _validate_centered_contribution_cache(result)
    return result


def _validate_centered_contribution_cache(cache: CenteredContributionCache) -> None:
    _validate_contribution_cache(cache.base)
    samples, users = cache.base.sample_count, cache.base.user_count
    center, w = np.asarray(cache.center), np.asarray(cache.centered_diagonal)
    if center.dtype != np.dtype(np.complex128) or center.shape != (users,):
        raise ValueError("common center has wrong dtype or shape")
    if w.dtype != np.dtype(np.float64) or w.shape != (samples, users):
        raise ValueError("centered diagonal has wrong dtype or shape")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(w)):
        raise ValueError("centered cache must be finite")
    expected_center = _componentwise_center(cache.base.coherent)
    if not np.array_equal(center.view(np.uint64), expected_center.view(np.uint64)):
        raise ValueError("common center differs from exact RN-even mean")
    if len(cache.exact_centered_diagonal) != samples:
        raise ValueError("exact centered cache has wrong shape")
    for sample in range(samples):
        if len(cache.exact_centered_diagonal[sample]) != users:
            raise ValueError("exact centered cache has wrong shape")
        for user in range(users):
            cr = _as_fraction(cache.base.coherent[sample, user].real)
            ci = _as_fraction(cache.base.coherent[sample, user].imag)
            ar = _as_fraction(center[user].real)
            ai = _as_fraction(center[user].imag)
            exact = (_as_fraction(cache.base.second[sample, user, user])
                     - 2 * (ar * cr + ai * ci) + ar * ar + ai * ai)
            if cache.exact_centered_diagonal[sample][user] != exact:
                raise ValueError("exact centered contribution was mutated")
            expected = _round_fraction_to_binary64(exact)
            if struct.pack(">d", float(w[sample, user])) != struct.pack(">d", expected):
                raise ValueError("centered contribution is not shared one-round RN-even")


def _centered_witness_sha256(center: complex, witnesses: object) -> str:
    """Bind one common center and its ordered shared binary64 witnesses."""
    values = np.asarray(witnesses)
    if values.dtype != np.dtype(np.float64) or values.ndim != 1:
        raise ValueError("centered witness hash requires a float64 vector")
    payload = bytearray(struct.pack(">dd", float(center.real), float(center.imag)))
    for value in values:
        payload.extend(struct.pack(">d", float(value)))
    return hashlib.sha256(payload).hexdigest()


class OnlineNeumaierReducer:
    """Elementwise, fixed-order Neumaier reduction for real arrays."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._sum = np.zeros(shape, dtype=np.float64)
        self._correction = np.zeros(shape, dtype=np.float64)
        self.count = 0

    def update(self, value: object) -> None:
        sample = np.asarray(value, dtype=np.float64)
        if sample.shape != self._sum.shape or not np.all(np.isfinite(sample)):
            raise ValueError("Neumaier sample has wrong shape or is nonfinite")
        trial = self._sum + sample
        self._correction += np.where(
            np.abs(self._sum) >= np.abs(sample),
            (self._sum - trial) + sample,
            (sample - trial) + self._sum,
        )
        self._sum = trial
        self.count += 1

    def mean(self) -> FloatArray:
        if self.count < 1:
            raise BaselinePrecisionProtocolError("cannot finalize an empty reducer")
        return np.asarray(
            (self._sum + self._correction) / float(self.count), dtype=np.float64
        )


def _neumaier_mean_axis0(values: object) -> FloatArray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
        raise ValueError("Neumaier input must have a nonempty finite sample axis")
    reducer = OnlineNeumaierReducer(tuple(samples.shape[1:]))
    for sample in samples:
        reducer.update(sample)
    return reducer.mean()


def _fixed_order_fsum_axis0(values: object) -> FloatArray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
        raise ValueError("fsum input must have a nonempty finite sample axis")
    flat = samples.reshape(samples.shape[0], -1)
    totals = np.fromiter(
        (math.fsum(float(flat[r, c]) for r in range(flat.shape[0]))
         for c in range(flat.shape[1])),
        dtype=np.float64,
        count=flat.shape[1],
    )
    return np.asarray(
        totals.reshape(samples.shape[1:]) / float(samples.shape[0]), dtype=np.float64
    )


def statistics_from_centered_primitives(
    coherent: object, second: object, norm: object,
    centered_diagonal_mean: object, center: object, powers: object,
    noise_variance: float, tau_p: int, tau_c: int, *, count: int,
) -> dict[str, Any]:
    """Evaluate centered UatF without nonnegative projection."""
    gain = np.asarray(coherent, dtype=np.complex128)
    power = np.asarray(second, dtype=np.float64)
    combiner_norm = np.asarray(norm, dtype=np.float64)
    centered_mean = np.asarray(centered_diagonal_mean, dtype=np.float64)
    common_center = np.asarray(center, dtype=np.complex128)
    payload_powers = np.asarray(powers, dtype=np.float64)
    if gain.ndim != 1:
        raise ValueError("coherent gain must be one-dimensional")
    users = gain.size
    if (power.shape != (users, users) or combiner_norm.shape != (users,)
            or centered_mean.shape != (users,) or common_center.shape != (users,)
            or payload_powers.shape != (users,)):
        raise ValueError("centered UatF dimensions differ")
    if count < 1 or not math.isfinite(float(noise_variance)) or noise_variance <= 0:
        raise ValueError("count and noise variance must be positive")
    if not 0 <= tau_p < tau_c:
        raise ValueError("require 0 <= tau_p < tau_c")
    if np.any(payload_powers < 0.0):
        raise ValueError("payload powers must be nonnegative")
    if not all(np.all(np.isfinite(x)) for x in (
            gain, power, combiner_norm, centered_mean, common_center, payload_powers)):
        raise ValueError("centered primitive inputs must be finite")
    residual = np.asarray(
        centered_mean - np.abs(gain - common_center) ** 2, dtype=np.float64
    )
    numerator = np.asarray(payload_powers * np.abs(gain) ** 2, dtype=np.float64)
    off_diagonal = np.array(power, copy=True)
    off_diagonal[np.arange(users), np.arange(users)] = 0.0
    denominator = np.asarray(
        payload_powers * residual + off_diagonal @ payload_powers
        + float(noise_variance) * combiner_norm,
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sinr = np.asarray(numerator / denominator, dtype=np.float64)
        se = np.asarray(
            (1.0 - float(tau_p) / float(tau_c)) * np.log2(1.0 + sinr),
            dtype=np.float64,
        )
    return {
        "count": int(count),
        "center": common_center,
        "coherent_gain": gain,
        "second_effective_power": power,
        "centered_diagonal_mean": centered_mean,
        "centered_coherent_residual": residual,
        "combiner_norm": combiner_norm,
        "numerator": numerator,
        "denominator": denominator,
        "sinr": sinr,
        "spectral_efficiency": se,
    }


def _statistics_with_reducer(
    cache: CenteredContributionCache, powers: object, noise_variance: float,
    tau_p: int, tau_c: int, reducer: Callable[[object], FloatArray],
) -> dict[str, Any]:
    _validate_centered_contribution_cache(cache)
    coherent = reducer(cache.base.coherent.real).astype(np.complex128)
    coherent += 1j * reducer(cache.base.coherent.imag)
    return statistics_from_centered_primitives(
        coherent, reducer(cache.base.second), reducer(cache.base.norm),
        reducer(cache.centered_diagonal), cache.center, powers, noise_variance,
        tau_p, tau_c, count=cache.sample_count,
    )


def centered_neumaier_statistics(
    cache: CenteredContributionCache, powers: object, noise_variance: float,
    tau_p: int, tau_c: int,
) -> dict[str, Any]:
    return _statistics_with_reducer(
        cache, powers, noise_variance, tau_p, tau_c, _neumaier_mean_axis0
    )


def fixed_order_fsum_statistics(
    cache: CenteredContributionCache, powers: object, noise_variance: float,
    tau_p: int, tau_c: int,
) -> dict[str, Any]:
    return _statistics_with_reducer(
        cache, powers, noise_variance, tau_p, tau_c, _fixed_order_fsum_axis0
    )


def ordinary_direct_statistics(
    cache: CenteredContributionCache, powers: object, noise_variance: float,
    tau_p: int, tau_c: int,
) -> dict[str, Any]:
    def ordinary(values: object) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        return np.asarray(np.mean(array, axis=0, dtype=np.float64), dtype=np.float64)
    return _statistics_with_reducer(
        cache, powers, noise_variance, tau_p, tau_c, ordinary
    )


def _power_of_two(exponent: int) -> Fraction:
    return (Fraction(1 << exponent, 1) if exponent >= 0
            else Fraction(1, 1 << -exponent))


def _floor_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 domain requires a positive rational")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if _power_of_two(exponent) > value:
        exponent -= 1
    elif _power_of_two(exponent + 1) <= value:
        exponent += 1
    return exponent


def _ln_mantissa_interval(
    value: Fraction, terms: int = ATANH_SERIES_TERMS
) -> RationalInterval:
    """Bound ln(value), 1 <= value <= 2, by an exact atanh series."""
    if not Fraction(1) <= value <= Fraction(2):
        raise ValueError("atanh mantissa must lie in [1,2]")
    if terms < 1:
        raise ValueError("atanh series must contain at least one term")
    z = (value - 1) / (value + 1)
    if z == 0:
        return RationalInterval(Fraction(0), Fraction(0))
    z2, power, partial = z * z, z, Fraction(0)
    for index in range(terms):
        partial += power / (2 * index + 1)
        power *= z2
    lower = 2 * partial
    remainder = (
        2 * z ** (2 * terms + 1)
        / ((2 * terms + 1) * (1 - z2))
    )
    return RationalInterval(lower, lower + remainder)


def log2_interval_atanh(
    value: Fraction, terms: int = ATANH_SERIES_TERMS
) -> RationalInterval:
    """Rigorous exact-rational log2 interval (M=64 by contract)."""
    if value <= 0:
        raise ValueError("log2 domain requires a positive rational")
    exponent = _floor_log2_fraction(value)
    mantissa = value / _power_of_two(exponent)
    ln_y = _ln_mantissa_interval(mantissa, terms)
    ln_2 = _ln_mantissa_interval(Fraction(2), terms)
    if ln_y.lower == 0:
        ratio = RationalInterval(Fraction(0), Fraction(0))
    else:
        ratio = RationalInterval(
            ln_y.lower / ln_2.upper, ln_y.upper / ln_2.lower
        )
    return RationalInterval(exponent + ratio.lower, exponent + ratio.upper)


def exact_uatf_oracle(
    cache: CenteredContributionCache, powers: object, noise_variance: float,
    tau_p: int, tau_c: int,
) -> tuple[ExactUserOracle, ...]:
    """Construct exact cached-binary64 UatF targets for every user."""
    _validate_centered_contribution_cache(cache)
    users = cache.user_count
    payload_powers = np.asarray(powers)
    if payload_powers.dtype != np.dtype(np.float64) or payload_powers.shape != (users,):
        raise ValueError("oracle powers must have registered float64 dtype/shape")
    if np.any(payload_powers < 0.0) or not np.all(np.isfinite(payload_powers)):
        raise ValueError("oracle powers must be finite and nonnegative")
    if not math.isfinite(float(noise_variance)) or noise_variance <= 0:
        raise ValueError("oracle noise must be finite and positive")
    if not 0 <= tau_p < tau_c:
        raise ValueError("oracle requires 0 <= tau_p < tau_c")
    p = tuple(_as_fraction(x) for x in payload_powers)
    noise = _as_fraction(noise_variance)
    prelog = Fraction(tau_c - tau_p, tau_c)
    result: list[ExactUserOracle] = []
    for user in range(users):
        mu_r = _mean_fraction(cache.base.coherent[:, user].real)
        mu_i = _mean_fraction(cache.base.coherent[:, user].imag)
        q_row = tuple(
            _mean_fraction(cache.base.second[:, user, interferer])
            for interferer in range(users)
        )
        norm = _mean_fraction(cache.base.norm[:, user])
        ar = _as_fraction(cache.center[user].real)
        ai = _as_fraction(cache.center[user].imag)
        w_mean = (
            sum((row[user] for row in cache.exact_centered_diagonal), Fraction(0))
            / cache.sample_count
        )
        rounded_w_mean = _mean_fraction(cache.centered_diagonal[:, user])
        residual = w_mean - (mu_r - ar) ** 2 - (mu_i - ai) ** 2
        rounded_residual = (
            rounded_w_mean - (mu_r - ar) ** 2 - (mu_i - ai) ** 2
        )
        mean_power = mu_r * mu_r + mu_i * mu_i
        numerator = p[user] * mean_power
        legacy_denominator = (
            sum((p[i] * q_row[i] for i in range(users)), Fraction(0))
            - numerator + noise * norm
        )
        centered_denominator = (
            p[user] * residual
            + sum((p[i] * q_row[i] for i in range(users) if i != user), Fraction(0))
            + noise * norm
        )
        rounded_centered_denominator = (
            p[user] * rounded_residual
            + sum((p[i] * q_row[i] for i in range(users) if i != user), Fraction(0))
            + noise * norm
        )
        identity = (
            w_mean - (mu_r - ar) ** 2 - (mu_i - ai) ** 2
            == q_row[user] - mean_power
            and legacy_denominator == centered_denominator
        )
        if not identity:
            raise BaselinePrecisionProtocolError("exact centered identity failed")
        sinr: Fraction | None = None
        raw_log: RationalInterval | None = None
        se: RationalInterval | None = None
        rounded_sinr: Fraction | None = None
        rounded_raw_log: RationalInterval | None = None
        rounded_se: RationalInterval | None = None
        if centered_denominator > 0:
            sinr = numerator / centered_denominator
            raw_log = log2_interval_atanh(1 + sinr, ATANH_SERIES_TERMS)
            se = RationalInterval(prelog * raw_log.lower, prelog * raw_log.upper)
        if rounded_centered_denominator > 0:
            rounded_sinr = numerator / rounded_centered_denominator
            rounded_raw_log = log2_interval_atanh(
                1 + rounded_sinr, ATANH_SERIES_TERMS
            )
            rounded_se = RationalInterval(
                prelog * rounded_raw_log.lower,
                prelog * rounded_raw_log.upper,
            )
        result.append(ExactUserOracle(
            user, mu_r, mu_i, w_mean, residual, rounded_w_mean,
            rounded_residual, q_row, norm, numerator, legacy_denominator,
            centered_denominator, rounded_centered_denominator, sinr, raw_log,
            se, rounded_sinr, rounded_raw_log, rounded_se, identity,
        ))
    return tuple(result)


def _fraction_text(value: Fraction) -> str:
    def integer_text(integer: int) -> str:
        # CPython protects decimal conversion of very large integers.  Hex is
        # exact, deterministic, compact, and parsed with int(token, 0).
        try:
            return str(integer)
        except ValueError:
            sign = "-" if integer < 0 else ""
            return f"{sign}0x{abs(integer):x}"
    return f"{integer_text(value.numerator)}/{integer_text(value.denominator)}"


def _outward_float(value: Fraction, direction: float) -> float:
    rounded = float(value)
    exact = Fraction.from_float(rounded)
    if (direction < 0 and exact > value) or (direction > 0 and exact < value):
        rounded = math.nextafter(rounded, direction)
    return rounded


def _upward_float(value: Fraction) -> float:
    if value < 0:
        raise ValueError("upward error bound must be nonnegative")
    rounded = float(value)
    if math.isfinite(rounded) and Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _distance_to_interval(value: float, interval: RationalInterval) -> Fraction | None:
    if not math.isfinite(value):
        return None
    exact = Fraction.from_float(value)
    if exact < interval.lower:
        return interval.lower - exact
    if exact > interval.upper:
        return exact - interval.upper
    return Fraction(0)


def _radius_to_interval(value: float, interval: RationalInterval) -> Fraction | None:
    """Worst endpoint distance, used for the registered pairwise bound."""
    if not math.isfinite(value):
        return None
    exact = Fraction.from_float(value)
    return max(abs(exact - interval.lower), abs(exact - interval.upper))


def _interval_pair_radius(
    first: RationalInterval, second: RationalInterval
) -> Fraction:
    """Worst possible distance between values enclosed by two intervals."""

    return max(
        abs(left - right)
        for left in (first.lower, first.upper)
        for right in (second.lower, second.upper)
    )


def _fraction_sequence_sha256(values: tuple[Fraction, ...]) -> str:
    payload = "".join(
        f"{'-' if value.numerator < 0 else '+'}{abs(value.numerator):x}/"
        f"{value.denominator:x}\n"
        for value in values
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _compact_interval_payload(
    interval: RationalInterval | None,
) -> dict[str, float | str] | None:
    """Persist compact outward evidence while binding exact rational endpoints."""
    if interval is None:
        return None
    return {
        "lower_outward_binary64": _outward_float(interval.lower, -math.inf),
        "upper_outward_binary64": _outward_float(interval.upper, math.inf),
        "width_upper_binary64": _upward_float(interval.width),
        "exact_interval_sha256": _fraction_sequence_sha256(
            (interval.lower, interval.upper)
        ),
        "exact_interval_encoding": (
            "SHA256_ASCII_SIGNED_HEX_NUM_SLASH_HEX_DEN_LF_"
            "LOWER_THEN_UPPER"
        ),
    }


def _close_float_to_fraction(observed: float, expected: Fraction) -> bool:
    if not math.isfinite(observed):
        return False
    actual = Fraction.from_float(observed)
    tolerance = (
        Fraction.from_float(PRIMITIVE_ATOL)
        + Fraction.from_float(PRIMITIVE_RTOL) * max(abs(actual), abs(expected))
    )
    return abs(actual - expected) <= tolerance


_STATISTIC_FIELDS = {
    "count", "center", "coherent_gain", "second_effective_power",
    "centered_diagonal_mean", "centered_coherent_residual", "combiner_norm",
    "numerator", "denominator", "sinr", "spectral_efficiency",
}


def _validate_statistics(
    name: str, statistics: Mapping[str, Any],
    cache: CenteredContributionCache,
) -> None:
    users = cache.user_count
    if set(statistics) != _STATISTIC_FIELDS:
        raise ValueError(f"{name} statistic keys differ")
    if int(statistics["count"]) != cache.sample_count:
        raise ValueError(f"{name} sample count differs")
    shapes = {
        "center": (users,), "coherent_gain": (users,),
        "second_effective_power": (users, users),
        "centered_diagonal_mean": (users,),
        "centered_coherent_residual": (users,), "combiner_norm": (users,),
        "numerator": (users,), "denominator": (users,), "sinr": (users,),
        "spectral_efficiency": (users,),
    }
    for field, shape in shapes.items():
        array = np.asarray(statistics[field])
        dtype = np.dtype(np.complex128) if field in {"center", "coherent_gain"} else np.dtype(np.float64)
        if array.shape != shape:
            raise ValueError(f"{name}.{field} has wrong shape")
        if array.dtype != dtype:
            raise ValueError(f"{name}.{field} has wrong dtype")
    if not np.array_equal(
        np.asarray(statistics["center"]).view(np.uint64),
        cache.center.view(np.uint64),
    ):
        raise ValueError(f"{name}.center differs from common centered cache")


def _scalar_close(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    left_exact = Fraction.from_float(left)
    right_exact = Fraction.from_float(right)
    tolerance = (
        Fraction.from_float(PRIMITIVE_ATOL)
        + Fraction.from_float(PRIMITIVE_RTOL)
        * max(abs(left_exact), abs(right_exact))
    )
    return abs(left_exact - right_exact) <= tolerance


def _absolute_binary64_difference_upper(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.inf
    return _upward_float(
        abs(Fraction.from_float(left) - Fraction.from_float(right))
    )


def _path_values(
    statistics: Mapping[str, Any], user: int,
    powers: FloatArray, noise_variance: float,
) -> dict[str, Any]:
    gain = complex(np.asarray(statistics["coherent_gain"])[user])
    center = complex(np.asarray(statistics["center"])[user])
    second_row = np.asarray(
        statistics["second_effective_power"], dtype=np.float64
    )[user]
    norm = float(np.asarray(statistics["combiner_norm"])[user])
    numerator = float(np.asarray(statistics["numerator"])[user])
    legacy_denominator = float(
        second_row @ powers - numerator + float(noise_variance) * norm
    )
    correction = float(abs(gain - center) ** 2)
    return {
        "coherent_gain": {"real": gain.real, "imag": gain.imag},
        "second_effective_power_row": second_row.tolist(),
        "combiner_norm": norm,
        "center": {"real": center.real, "imag": center.imag},
        "centered_witness_mean": float(
            np.asarray(statistics["centered_diagonal_mean"])[user]
        ),
        "center_correction": correction,
        "centered_self_term": float(
            np.asarray(statistics["centered_coherent_residual"])[user]
        ),
        "numerator": numerator,
        "legacy_denominator": legacy_denominator,
        "denominator": float(np.asarray(statistics["denominator"])[user]),
        "sinr": float(np.asarray(statistics["sinr"])[user]),
        "spectral_efficiency": float(
            np.asarray(statistics["spectral_efficiency"])[user]
        ),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        if math.isnan(scalar):
            return "NaN"
        if math.isinf(scalar):
            return "Infinity" if scalar > 0 else "-Infinity"
        return scalar
    if isinstance(value, Fraction):
        return _fraction_text(value)
    return value


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
    centered_cache: CenteredContributionCache | None = None,
) -> dict[str, Any]:
    """Certify three centered paths against an exact Fraction/log2 oracle."""
    if centered_cache is None:
        centered_cache = build_centered_contribution_cache(contribution_cache)
    _validate_centered_contribution_cache(centered_cache)
    if centered_cache.base is not contribution_cache:
        # Identity is intentional: paths must share the very same prepared cache.
        raise ValueError("centered and sufficient caches do not share one base object")
    if not callable(diagnostic_sink):
        raise TypeError("diagnostic_sink must be callable")
    if set(context) != {
        "condition_index", "condition_id", "receiver_index", "receiver_id", "split"
    }:
        raise ValueError("diagnostic context keys differ")
    paths = {
        "online_neumaier": online_neumaier,
        "fixed_order_fsum": fixed_order_fsum,
        "ordinary_direct": ordinary_direct,
    }
    for name, statistics in paths.items():
        _validate_statistics(name, statistics, centered_cache)
    payload_powers = np.asarray(powers)
    if payload_powers.dtype != np.dtype(np.float64):
        raise ValueError("certified powers must have dtype float64")
    oracle = exact_uatf_oracle(
        centered_cache, payload_powers, noise_variance, tau_p, tau_c
    )
    pair_specs = (
        ("NEUMAIER_VS_MATH_FSUM", "online_neumaier", "fixed_order_fsum"),
        ("NEUMAIER_VS_ORDINARY_ARRAY", "online_neumaier", "ordinary_direct"),
        ("MATH_FSUM_VS_ORDINARY_ARRAY", "fixed_order_fsum", "ordinary_direct"),
    )
    failures: list[tuple[int, list[str]]] = []
    maxima = {
        "coherent_gain": 0.0, "second_effective_power": 0.0,
        "centered_diagonal_mean": 0.0, "centered_coherent_residual": 0.0,
        "centered_correction": 0.0,
        "combiner_norm": 0.0, "numerator": 0.0, "denominator": 0.0,
        "legacy_denominator": 0.0,
        "spectral_efficiency": 0.0,
    }
    maximum_oracle_distance = 0.0
    maximum_oracle_radius = 0.0
    maximum_pair_bound = 0.0
    maximum_log2_interval_width = 0.0
    maximum_se_interval_width = 0.0
    maximum_rounded_log2_interval_width = 0.0
    maximum_rounded_se_interval_width = 0.0
    maximum_cancellation = 0.0
    maximum_sinr_ratio_bound = 0.0
    maximum_witness_denominator_gap = 0.0
    maximum_witness_se_gap = 0.0

    for user, target in enumerate(oracle):
        reasons: list[str] = []
        pair_errors: dict[str, dict[str, float]] = {}
        pair_passes: dict[str, dict[str, bool]] = {}
        for pair, first_name, second_name in pair_specs:
            first, second = paths[first_name], paths[second_name]
            first_gain = complex(np.asarray(first["coherent_gain"])[user])
            second_gain = complex(np.asarray(second["coherent_gain"])[user])
            common_center = complex(centered_cache.center[user])
            numeric = {
                "coherent_gain_real": (first_gain.real, second_gain.real),
                "coherent_gain_imag": (first_gain.imag, second_gain.imag),
                "centered_correction": (
                    abs(first_gain - common_center) ** 2,
                    abs(second_gain - common_center) ** 2,
                ),
            }
            for field in (
                "centered_diagonal_mean", "centered_coherent_residual",
                "combiner_norm", "numerator", "denominator",
            ):
                numeric[field] = (
                    float(np.asarray(first[field])[user]),
                    float(np.asarray(second[field])[user]),
                )
            first_row = np.asarray(first["second_effective_power"])[user]
            second_row = np.asarray(second["second_effective_power"])[user]
            first_legacy = float(
                first_row @ payload_powers
                - float(np.asarray(first["numerator"])[user])
                + float(noise_variance)
                * float(np.asarray(first["combiner_norm"])[user])
            )
            second_legacy = float(
                second_row @ payload_powers
                - float(np.asarray(second["numerator"])[user])
                + float(noise_variance)
                * float(np.asarray(second["combiner_norm"])[user])
            )
            numeric["legacy_denominator"] = (first_legacy, second_legacy)
            errors = {
                field: _absolute_binary64_difference_upper(left, right)
                for field, (left, right) in numeric.items()
            }
            errors["second_effective_power"] = max(
                _absolute_binary64_difference_upper(float(left), float(right))
                for left, right in zip(first_row, second_row, strict=True)
            )
            passes = {
                field: _scalar_close(left, right)
                for field, (left, right) in numeric.items()
            }
            # Explicit symmetric gate; numpy.allclose scales only by its
            # second operand and is therefore not the registered predicate.
            passes["second_effective_power"] = bool(
                np.all(np.isfinite(first_row))
                and np.all(np.isfinite(second_row))
                and all(
                    _scalar_close(float(left), float(right))
                    for left, right in zip(first_row, second_row, strict=True)
                )
            )
            pair_errors[pair], pair_passes[pair] = errors, passes
            for field, error in errors.items():
                aggregate = "coherent_gain" if field.startswith("coherent") else field
                maxima[aggregate] = max(maxima[aggregate], float(error))
                if not passes[field]:
                    reasons.append(f"PRIMITIVE_{field.upper()}_{pair}_MISMATCH")

        exact_targets = {
            "coherent_gain_real": target.coherent_real,
            "coherent_gain_imag": target.coherent_imag,
            # These three executable quantities are induced by the one-rounded
            # shared witness cache.  The unrounded identity target is retained
            # separately below and is never silently conflated with it.
            "centered_diagonal_mean": target.rounded_centered_diagonal_mean,
            "centered_coherent_residual": target.rounded_centered_residual,
            "combiner_norm": target.combiner_norm,
            "numerator": target.numerator,
            "denominator": target.rounded_centered_denominator,
        }
        exact_gates: dict[str, dict[str, bool]] = {}
        exact_errors: dict[str, dict[str, float | str]] = {}
        ratio_oracle: dict[str, dict[str, Any]] = {}
        formula_identity: dict[str, dict[str, Any]] = {}
        se_radii: dict[str, Fraction | None] = {}
        se_distances: dict[str, Fraction | None] = {}
        for path_name, statistics in paths.items():
            gain = complex(np.asarray(statistics["coherent_gain"])[user])
            observed = {
                "coherent_gain_real": gain.real,
                "coherent_gain_imag": gain.imag,
                **{field: float(np.asarray(statistics[field])[user]) for field in (
                    "centered_diagonal_mean", "centered_coherent_residual",
                    "combiner_norm", "numerator", "denominator",
                )},
            }
            stored_witness_mean = observed["centered_diagonal_mean"]
            stored_residual = observed["centered_coherent_residual"]
            stored_norm = observed["combiner_norm"]
            stored_numerator = observed["numerator"]
            stored_denominator = observed["denominator"]
            stored_sinr = float(np.asarray(statistics["sinr"])[user])
            stored_se = float(
                np.asarray(statistics["spectral_efficiency"])[user]
            )
            formula_correction = float(abs(gain - centered_cache.center[user]) ** 2)
            formula_residual = float(stored_witness_mean - formula_correction)
            formula_numerator = float(payload_powers[user] * abs(gain) ** 2)
            formula_second_row = np.asarray(
                statistics["second_effective_power"], dtype=np.float64
            )[user]
            formula_off_diagonal = float(
                sum(
                    formula_second_row[index] * payload_powers[index]
                    for index in range(centered_cache.user_count)
                    if index != user
                )
            )
            formula_centered_denominator = float(
                payload_powers[user] * stored_residual
                + formula_off_diagonal
                + float(noise_variance) * stored_norm
            )
            formula_legacy_denominator = float(
                formula_second_row @ payload_powers
                - stored_numerator
                + float(noise_variance) * stored_norm
            )
            formula_sinr = (
                float(stored_numerator / stored_denominator)
                if math.isfinite(stored_denominator) and stored_denominator != 0.0
                else math.nan
            )
            formula_se = (
                float(
                    (1.0 - float(tau_p) / float(tau_c))
                    * math.log2(1.0 + stored_sinr)
                )
                if math.isfinite(stored_sinr) and stored_sinr > -1.0
                else math.nan
            )
            formula_values = {
                "centered_residual": (stored_residual, formula_residual),
                "numerator": (stored_numerator, formula_numerator),
                "centered_denominator": (
                    stored_denominator, formula_centered_denominator
                ),
                "legacy_vs_centered_denominator": (
                    formula_legacy_denominator, stored_denominator
                ),
                "sinr_division": (stored_sinr, formula_sinr),
                "spectral_efficiency": (stored_se, formula_se),
            }
            formula_path_gates = {
                name: _scalar_close(left, right)
                for name, (left, right) in formula_values.items()
            }
            formula_identity[path_name] = {
                "observed_and_reconstructed": {
                    name: {"observed": left, "reconstructed": right}
                    for name, (left, right) in formula_values.items()
                },
                "gates": formula_path_gates,
                "pass": all(formula_path_gates.values()),
            }
            for name, passed in formula_path_gates.items():
                if not passed:
                    reasons.append(
                        f"FORMULA_{name.upper()}_{path_name.upper()}_MISMATCH"
                    )
            gates: dict[str, bool] = {}
            errors: dict[str, float | str] = {}
            for field, expected in exact_targets.items():
                passed = expected is not None and _close_float_to_fraction(observed[field], expected)
                gates[field] = bool(passed)
                errors[field] = (
                    _upward_float(abs(Fraction.from_float(observed[field]) - expected))
                    if expected is not None and math.isfinite(observed[field]) else "Infinity"
                )
                if not passed:
                    reasons.append(f"EXACT_TARGET_{field.upper()}_{path_name.upper()}_MISMATCH")
            second_row = np.asarray(statistics["second_effective_power"])[user]
            second_pass = all(
                _close_float_to_fraction(float(second_row[i]), target.second_row[i])
                for i in range(centered_cache.user_count)
            )
            gates["second_effective_power"] = second_pass
            if not second_pass:
                reasons.append(f"EXACT_TARGET_SECOND_EFFECTIVE_POWER_{path_name.upper()}_MISMATCH")
            observed_legacy_denominator = float(
                second_row @ payload_powers
                - float(np.asarray(statistics["numerator"])[user])
                + float(noise_variance)
                * float(np.asarray(statistics["combiner_norm"])[user])
            )
            observed_centered_denominator = float(
                np.asarray(statistics["denominator"])[user]
            )
            legacy_centered_pass = _scalar_close(
                observed_legacy_denominator, observed_centered_denominator
            )
            gates["legacy_centered_denominator"] = legacy_centered_pass
            errors["legacy_centered_denominator"] = abs(
                observed_legacy_denominator - observed_centered_denominator
            )
            if not legacy_centered_pass:
                reasons.append(
                    f"LEGACY_CENTERED_DENOMINATOR_{path_name.upper()}_MISMATCH"
                )
            observed_numerator = float(np.asarray(statistics["numerator"])[user])
            observed_denominator = float(np.asarray(statistics["denominator"])[user])
            observed_sinr = float(np.asarray(statistics["sinr"])[user])
            ratio_valid = bool(
                target.rounded_sinr is not None
                and math.isfinite(observed_numerator)
                and math.isfinite(observed_denominator)
                and math.isfinite(observed_sinr)
                and observed_denominator > 0.0
                and target.rounded_centered_denominator > 0
            )
            if ratio_valid:
                numerator_fraction = Fraction.from_float(observed_numerator)
                denominator_fraction = Fraction.from_float(observed_denominator)
                sinr_fraction = Fraction.from_float(observed_sinr)
                exact_path_quotient = numerator_fraction / denominator_fraction
                correctly_rounded_division = _round_fraction_to_binary64(
                    exact_path_quotient
                )
                division_rounding_pass = bool(
                    struct.pack(">d", observed_sinr)
                    == struct.pack(">d", correctly_rounded_division)
                )
                e_numerator = abs(numerator_fraction - target.numerator)
                e_denominator = abs(
                    denominator_fraction - target.rounded_centered_denominator
                )
                division_residual = abs(
                    sinr_fraction - exact_path_quotient
                )
                denominator_lower = min(
                    denominator_fraction, target.rounded_centered_denominator
                )
                numerator_upper = max(
                    abs(numerator_fraction), abs(target.numerator)
                )
                ratio_bound = (
                    division_residual
                    + e_numerator / denominator_lower
                    + numerator_upper * e_denominator
                    / (denominator_lower * denominator_lower)
                )
                ratio_actual = abs(sinr_fraction - target.rounded_sinr)
                ratio_pass = bool(
                    division_rounding_pass and ratio_actual <= ratio_bound
                )
                ratio_oracle[path_name] = {
                    "numerator_error": _upward_float(e_numerator),
                    "denominator_error": _upward_float(e_denominator),
                    "division_residual": _upward_float(division_residual),
                    "correctly_rounded_division": correctly_rounded_division,
                    "division_rounding_pass": division_rounding_pass,
                    "denominator_lower": _outward_float(
                        denominator_lower, -math.inf
                    ),
                    "numerator_upper": _upward_float(numerator_upper),
                    "actual_absolute_sinr_error": _upward_float(ratio_actual),
                    "sinr_error_bound": _upward_float(ratio_bound),
                    "exact_ratio_certificate_sha256": _fraction_sequence_sha256(
                        (
                            e_numerator,
                            e_denominator,
                            division_residual,
                            denominator_lower,
                            numerator_upper,
                            ratio_actual,
                            ratio_bound,
                        )
                    ),
                    "exact_ratio_certificate_encoding": (
                        "SHA256_ASCII_SIGNED_HEX_NUM_SLASH_HEX_DEN_LF_ORDER_"
                        "EN_ED_EDIV_DMIN_NMAX_ACTUAL_BOUND"
                    ),
                    "pass": ratio_pass,
                }
                maximum_sinr_ratio_bound = max(
                    maximum_sinr_ratio_bound, _upward_float(ratio_bound)
                )
            else:
                ratio_pass = False
                ratio_oracle[path_name] = {
                    "numerator_error": "Infinity",
                    "denominator_error": "Infinity",
                    "division_residual": "Infinity",
                    "correctly_rounded_division": "NaN",
                    "division_rounding_pass": False,
                    "denominator_lower": "-Infinity",
                    "numerator_upper": "Infinity",
                    "actual_absolute_sinr_error": "Infinity",
                    "sinr_error_bound": "Infinity",
                    "exact_ratio_certificate_sha256": None,
                    "exact_ratio_certificate_encoding": (
                        "SHA256_ASCII_SIGNED_HEX_NUM_SLASH_HEX_DEN_LF_ORDER_"
                        "EN_ED_EDIV_DMIN_NMAX_ACTUAL_BOUND"
                    ),
                    "pass": False,
                }
            gates["sinr_exact_ratio_bound"] = ratio_pass
            if not ratio_pass:
                reasons.append(
                    f"SINR_EXACT_RATIO_BOUND_{path_name.upper()}_FAILED"
                )
            exact_gates[path_name], exact_errors[path_name] = gates, errors
            se_value = float(np.asarray(statistics["spectral_efficiency"])[user])
            if target.rounded_spectral_efficiency is None:
                se_radii[path_name] = se_distances[path_name] = None
            else:
                se_radii[path_name] = _radius_to_interval(
                    se_value, target.rounded_spectral_efficiency
                )
                se_distances[path_name] = _distance_to_interval(
                    se_value, target.rounded_spectral_efficiency
                )

        se_pair: dict[str, dict[str, Any]] = {}
        for pair, first_name, second_name in pair_specs:
            first_se = float(np.asarray(paths[first_name]["spectral_efficiency"])[user])
            second_se = float(np.asarray(paths[second_name]["spectral_efficiency"])[user])
            actual_fraction = (
                abs(Fraction.from_float(first_se) - Fraction.from_float(second_se))
                if math.isfinite(first_se) and math.isfinite(second_se)
                else None
            )
            actual = (
                _upward_float(actual_fraction)
                if actual_fraction is not None else math.inf
            )
            maxima["spectral_efficiency"] = max(
                maxima["spectral_efficiency"], actual
            )
            radii = (se_radii[first_name], se_radii[second_name])
            if any(radius is None for radius in radii):
                bound_fraction = None
                bound = math.inf
                passes = False
            else:
                bound_fraction = radii[0] + radii[1]  # type: ignore[operator]
                bound = _upward_float(bound_fraction)
                passes = (
                    actual_fraction is not None
                    and actual_fraction <= bound_fraction
                    and bound_fraction <= Fraction.from_float(ABSOLUTE_SE_ORACLE_CEILING)
                )
            if not passes:
                reasons.append(f"SE_EXACT_ORACLE_BOUND_{pair}_FAILED")
            se_pair[pair] = {
                "actual_absolute_difference": actual,
                "exact_oracle_pair_bound": bound,
                "pass": passes,
            }
            maximum_pair_bound = max(maximum_pair_bound, bound)

        witness_denominator_gap = abs(
            target.rounded_centered_denominator - target.centered_denominator
        )
        maximum_witness_denominator_gap = max(
            maximum_witness_denominator_gap,
            _upward_float(witness_denominator_gap),
        )
        if (
            target.spectral_efficiency is not None
            and target.rounded_spectral_efficiency is not None
        ):
            witness_se_gap = _interval_pair_radius(
                target.spectral_efficiency,
                target.rounded_spectral_efficiency,
            )
            witness_rounding_pass = (
                witness_se_gap
                <= Fraction.from_float(ABSOLUTE_SE_ORACLE_CEILING)
            )
            maximum_witness_se_gap = max(
                maximum_witness_se_gap, _upward_float(witness_se_gap)
            )
        else:
            witness_se_gap = None
            witness_rounding_pass = False
        if not witness_rounding_pass:
            reasons.append("CENTERED_WITNESS_ROUNDING_GAP_EXCEEDED")

        if target.centered_denominator <= 0:
            reasons.append("NONPOSITIVE_EXACT_DENOMINATOR")
        if target.rounded_centered_denominator <= 0:
            reasons.append("NONPOSITIVE_ROUNDED_WITNESS_EXACT_DENOMINATOR")
        for path_name, statistics in paths.items():
            denominator = float(np.asarray(statistics["denominator"])[user])
            if not math.isfinite(denominator) or denominator <= 0:
                reasons.append(f"NONPOSITIVE_OR_NONFINITE_DENOMINATOR_{path_name.upper()}")
            distance, radius = se_distances[path_name], se_radii[path_name]
            if distance is None or radius is None:
                reasons.append(f"INVALID_SE_ORACLE_DOMAIN_{path_name.upper()}")
            else:
                maximum_oracle_distance = max(maximum_oracle_distance, _upward_float(distance))
                maximum_oracle_radius = max(maximum_oracle_radius, _upward_float(radius))
        if target.spectral_efficiency is not None:
            assert target.log2_one_plus_sinr is not None
            maximum_log2_interval_width = max(
                maximum_log2_interval_width,
                _upward_float(target.log2_one_plus_sinr.width),
            )
            maximum_se_interval_width = max(
                maximum_se_interval_width,
                _upward_float(target.spectral_efficiency.width),
            )
            log_width_pass = bool(
                target.log2_one_plus_sinr is not None
                and target.log2_one_plus_sinr.width <= MAX_EXACT_INTERVAL_WIDTH
            )
            se_width_pass = (
                target.spectral_efficiency.width <= MAX_EXACT_INTERVAL_WIDTH
            )
            if not log_width_pass:
                reasons.append("LOG2_ORACLE_INTERVAL_WIDTH_EXCEEDED")
            if not se_width_pass:
                reasons.append("SE_ORACLE_INTERVAL_WIDTH_EXCEEDED")
        else:
            log_width_pass = False
            se_width_pass = False
        rounded_log_width_pass = bool(
            target.rounded_log2_one_plus_sinr is not None
            and target.rounded_log2_one_plus_sinr.width
            <= MAX_EXACT_INTERVAL_WIDTH
        )
        rounded_se_width_pass = bool(
            target.rounded_spectral_efficiency is not None
            and target.rounded_spectral_efficiency.width
            <= MAX_EXACT_INTERVAL_WIDTH
        )
        if target.rounded_log2_one_plus_sinr is not None:
            maximum_rounded_log2_interval_width = max(
                maximum_rounded_log2_interval_width,
                _upward_float(target.rounded_log2_one_plus_sinr.width),
            )
        if target.rounded_spectral_efficiency is not None:
            maximum_rounded_se_interval_width = max(
                maximum_rounded_se_interval_width,
                _upward_float(target.rounded_spectral_efficiency.width),
            )
        if not rounded_log_width_pass:
            reasons.append("ROUNDED_WITNESS_LOG2_ORACLE_INTERVAL_WIDTH_EXCEEDED")
        if not rounded_se_width_pass:
            reasons.append("ROUNDED_WITNESS_SE_ORACLE_INTERVAL_WIDTH_EXCEEDED")
        term_scale = float(
            abs(payload_powers[user] * float(target.rounded_centered_residual))
            + sum(abs(payload_powers[i] * float(target.second_row[i]))
                  for i in range(centered_cache.user_count) if i != user)
            + abs(float(noise_variance) * float(target.combiner_norm))
        )
        cancellation = float(
            term_scale / abs(float(target.rounded_centered_denominator))
            if target.rounded_centered_denominator != 0 else math.inf
        )
        maximum_cancellation = max(maximum_cancellation, cancellation)

        interval = target.spectral_efficiency
        exact_center = {
            "real": _fraction_text(_as_fraction(centered_cache.center[user].real)),
            "imag": _fraction_text(_as_fraction(centered_cache.center[user].imag)),
        }
        diagnostic = {
            "schema_version": "r2c01-v3-baseline-precision-e0-03-centered-diagnostic/1.0",
            "experiment_id": EXPERIMENT_ID,
            "root_experiment_id": ROOT_REPLAY_SOURCE_ID,
            "immediate_predecessor_experiment_id": IMMEDIATE_FAILED_PREDECESSOR_ID,
            **dict(context),
            "user_index": user,
            "sample_count": centered_cache.sample_count,
            "values": {
                name: _path_values(
                    value, user, payload_powers, float(noise_variance)
                ) for name, value in paths.items()
            },
            "errors": {
                "primitive_pair": pair_errors,
                "exact_target_by_path": exact_errors,
                "sinr_exact_ratio_by_path": ratio_oracle,
                "se_oracle_by_path": {
                    name: {
                        "distance_to_interval": (
                            _upward_float(se_distances[name])
                            if se_distances[name] is not None else "Infinity"
                        ),
                        "worst_endpoint_radius": (
                            _upward_float(se_radii[name])
                            if se_radii[name] is not None else "Infinity"
                        ),
                    } for name in paths
                },
                "se_pair": se_pair,
                "centered_witness_rounding": {
                    "denominator_gap": _upward_float(witness_denominator_gap),
                    "spectral_efficiency_interval_gap": (
                        _upward_float(witness_se_gap)
                        if witness_se_gap is not None else "Infinity"
                    ),
                    "pass": witness_rounding_pass,
                },
                "formula_identity_by_path": formula_identity,
            },
            "oracle": {
                "center": {
                    "exact_fraction": exact_center,
                    "binary64": {
                        "real": float(centered_cache.center[user].real),
                        "imag": float(centered_cache.center[user].imag),
                    },
                },
                "identity_exact": target.exact_centered_identity,
                "unrounded_centered_witness_mean": _fraction_text(
                    target.centered_diagonal_mean
                ),
                "rounded_centered_witness_mean": _fraction_text(
                    target.rounded_centered_diagonal_mean
                ),
                "unrounded_centered_residual": _fraction_text(
                    target.centered_residual
                ),
                "rounded_centered_residual": _fraction_text(
                    target.rounded_centered_residual
                ),
                "numerator": _fraction_text(target.numerator),
                "legacy_denominator": _fraction_text(target.legacy_denominator),
                "centered_denominator": _fraction_text(target.centered_denominator),
                "rounded_witness_centered_denominator": _fraction_text(
                    target.rounded_centered_denominator
                ),
                "sinr": _fraction_text(target.sinr) if target.sinr is not None else None,
                "rounded_witness_sinr": (
                    _fraction_text(target.rounded_sinr)
                    if target.rounded_sinr is not None else None
                ),
                "spectral_efficiency_interval": _compact_interval_payload(
                    interval
                ),
                "log2_one_plus_sinr_interval": _compact_interval_payload(
                    target.log2_one_plus_sinr
                ),
                "rounded_witness_log2_one_plus_sinr_interval": (
                    _compact_interval_payload(
                        target.rounded_log2_one_plus_sinr
                    )
                ),
                "rounded_witness_spectral_efficiency_interval": (
                    _compact_interval_payload(target.rounded_spectral_efficiency)
                ),
                "maximum_exact_interval_width": _fraction_text(
                    MAX_EXACT_INTERVAL_WIDTH
                ),
                "atanh_terms": ATANH_SERIES_TERMS,
                "shared_witness_sha256": _centered_witness_sha256(
                    centered_cache.center[user],
                    centered_cache.centered_diagonal[:, user],
                ),
                "shared_witness_hash_encoding": (
                    "SHA256_BIG_ENDIAN_BINARY64_CENTER_REAL_IMAG_THEN_"
                    "ORDERED_ROUNDED_WITNESSES"
                ),
            },
            "gates": {
                "primitive_pair": pair_passes,
                "exact_target": exact_gates,
                "exact_centered_identity": target.exact_centered_identity,
                "exact_denominator_positive": target.centered_denominator > 0,
                "rounded_witness_exact_denominator_positive": (
                    target.rounded_centered_denominator > 0
                ),
                "sinr_exact_ratio_by_path": {
                    name: value["pass"] for name, value in ratio_oracle.items()
                },
                "centered_witness_rounding_gap": witness_rounding_pass,
                "formula_identity_by_path": {
                    name: value["pass"]
                    for name, value in formula_identity.items()
                },
                "log2_interval_width": log_width_pass,
                "spectral_efficiency_interval_width": se_width_pass,
                "rounded_witness_log2_interval_width": rounded_log_width_pass,
                "rounded_witness_spectral_efficiency_interval_width": (
                    rounded_se_width_pass
                ),
                "se_pair": {name: value["pass"] for name, value in se_pair.items()},
            },
            "primitive_pair_passes": pair_passes,
            "status": "PASS" if not reasons else "FAIL",
            "failure_reasons": sorted(set(reasons)),
        }
        diagnostic_sink(MappingProxyType(_json_safe(diagnostic)))
        if reasons:
            failures.append((user, sorted(set(reasons))))

    if failures and raise_on_failure:
        user, reasons = failures[0]
        raise BaselinePrecisionProtocolError(
            "E0-03 centered-UatF certification failed at "
            f"condition={context['condition_id']}, receiver={context['receiver_id']}, "
            f"split={context['split']}, user={user}: {','.join(reasons)}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "user_count": centered_cache.user_count,
        "diagnostic_count": centered_cache.user_count,
        "failed_user_count": len(failures),
        "failed_user_indices": [user for user, _ in failures],
        "first_failure_reasons": failures[0][1] if failures else [],
        "maximum_absolute_differences": maxima,
        "maximum_absolute_se_oracle_distance": maximum_oracle_distance,
        "maximum_se_oracle_endpoint_radius": maximum_oracle_radius,
        "maximum_se_exact_oracle_pair_bound": maximum_pair_bound,
        "maximum_log2_interval_width": maximum_log2_interval_width,
        "maximum_spectral_efficiency_interval_width": maximum_se_interval_width,
        "maximum_rounded_witness_log2_interval_width": (
            maximum_rounded_log2_interval_width
        ),
        "maximum_rounded_witness_spectral_efficiency_interval_width": (
            maximum_rounded_se_interval_width
        ),
        "maximum_sinr_exact_ratio_bound": maximum_sinr_ratio_bound,
        "maximum_centered_witness_denominator_gap": (
            maximum_witness_denominator_gap
        ),
        "maximum_centered_witness_se_gap": maximum_witness_se_gap,
        "maximum_cancellation_metric": maximum_cancellation,
    }


def require_complete_numerical_certification(
    certificates: list[Mapping[str, Any]], *,
    expected_rows: int, expected_diagnostics: int,
) -> None:
    diagnostic_count = sum(int(value["diagnostic_count"]) for value in certificates)
    if len(certificates) != expected_rows or diagnostic_count != expected_diagnostics:
        raise BaselinePrecisionProtocolError(
            "E0-03 diagnostic completeness differs: "
            f"rows={len(certificates)}/{expected_rows}, "
            f"diagnostics={diagnostic_count}/{expected_diagnostics}"
        )
    failures = [value for value in certificates if value["status"] != "PASS"]
    if failures:
        failed_users = sum(int(value["failed_user_count"]) for value in failures)
        raise BaselinePrecisionProtocolError(
            "E0-03 aggregate centered-UatF certification failed only after "
            f"all {diagnostic_count} diagnostics; failed_rows={len(failures)}, "
            f"failed_users={failed_users}"
        )


def _validate_root_contract(
    config: BaselinePrecisionConfig, registries: Registries,
) -> None:
    if config.raw.get("experiment_id") != ROOT_REPLAY_SOURCE_ID:
        raise BaselinePrecisionProtocolError("E0-03 requires frozen E0-01 configuration")
    if config.frame_count != 192 or config.realizations != 64:
        raise BaselinePrecisionProtocolError("E0-03 requires all 192 E0-01 frames")
    if len(config.conditions) != 3 or len(config.receivers) != 6:
        raise BaselinePrecisionProtocolError("E0-03 registry dimensions differ")
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
        raise BaselinePrecisionProtocolError("E0-03 is not exact frozen E0-01 replay")


def _centered_statistics_json(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _statistics_json(value)
    center = np.asarray(value["center"], dtype=np.complex128)
    result.update({
        "center": {"real": center.real.tolist(), "imag": center.imag.tolist()},
        "centered_diagonal_mean": np.asarray(
            value["centered_diagonal_mean"], dtype=np.float64
        ).tolist(),
        "centered_coherent_residual": np.asarray(
            value["centered_coherent_residual"], dtype=np.float64
        ).tolist(),
    })
    return result


def execute_baseline_precision_e0_centered(
    parent_config: BaselinePrecisionConfig,
    registries: Registries,
    *,
    progress_sink: Callable[[Mapping[str, Any]], None],
    resource_guard: Callable[[str], None],
    diagnostic_sink: Callable[[Mapping[str, Any]], None],
) -> BaselinePrecisionProducts:
    """Replay exact E0-01 frames with prospective E0-03 centered numerics."""
    _validate_root_contract(parent_config, registries)
    if not all(callable(x) for x in (progress_sink, resource_guard, diagnostic_sink)):
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
    maximum_oracle_distance = 0.0
    maximum_oracle_radius = 0.0
    maximum_pair_bound = 0.0
    maximum_log2_interval_width = 0.0
    maximum_se_interval_width = 0.0
    maximum_rounded_log2_interval_width = 0.0
    maximum_rounded_se_interval_width = 0.0
    maximum_cancellation = 0.0
    maximum_sinr_ratio_bound = 0.0
    maximum_witness_denominator_gap = 0.0
    maximum_witness_se_gap = 0.0

    for condition_index, condition in enumerate(parent_config.conditions):
        scenario = parent_config.scenario(condition)
        topology = generate_topology(scenario, seed=parent_config.topology_seed)
        coherent_cache = np.empty((6, 64, 16), dtype=np.complex128)
        second_cache = np.empty((6, 64, 16, 16), dtype=np.float64)
        norm_cache = np.empty((6, 64, 16), dtype=np.float64)
        if coherent_cache.nbytes + second_cache.nbytes + norm_cache.nbytes > MAX_SUFFICIENT_CACHE_BYTES:
            raise BaselinePrecisionProtocolError("per-condition cache exceeds byte cap")
        for realization in range(64):
            resource_guard("BEFORE_FRAME")
            frame = generate_channel_frame(
                topology, scenario,
                seed=int(registries.frame_seeds[condition_index, realization]),
            )
            models: list[Any] = []
            observations: list[Any] = []
            groups: list[NDArray[np.int64]] = []
            coordinates: list[tuple[int, int]] = []
            for access_point in range(32):
                for pilot in range(4):
                    model, observation, users, _ = pilot_group(
                        topology, frame, scenario,
                        access_point=access_point, pilot=pilot,
                    )
                    if model.group_size != 4 or users.size != 4:
                        raise BaselinePrecisionProtocolError("balanced P=4 invariant failed")
                    models.append(model)
                    observations.append(observation)
                    groups.append(users)
                    coordinates.append((access_point, pilot))
            batched = batched_b1(stack_models(models, observations))
            for local_index, ((access_point, pilot), users) in enumerate(zip(coordinates, groups)):
                if (condition_index, realization, local_index) in registries.scalar_witness_keys:
                    scalar = np.column_stack([
                        widely_linear_mmse(
                            models[local_index], observations[local_index], target
                        ) for target in range(4)
                    ])
                    if not np.allclose(batched[local_index], scalar, rtol=5e-12, atol=5e-12):
                        raise BaselinePrecisionProtocolError("batched/scalar B1 failed")
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
            for receiver_index, receiver in enumerate(parent_config.receivers):
                combiner = receiver_combiner(estimate, receiver)
                coherent, second, norm = frame_sufficient_contributions(channel, combiner)
                coherent_cache[receiver_index, realization] = coherent
                second_cache[receiver_index, realization] = second
                norm_cache[receiver_index, realization] = norm
            completed += 1
            progress_sink(MappingProxyType({
                "experiment_id": EXPERIMENT_ID,
                "root_replay_source_id": ROOT_REPLAY_SOURCE_ID,
                "immediate_failed_predecessor_id": IMMEDIATE_FAILED_PREDECESSOR_ID,
                "frame_index": completed - 1,
                "condition_index": condition_index,
                "condition_id": condition.id,
                "realization_index": realization,
                "frame_seed": int(registries.frame_seeds[condition_index, realization]),
                "completed_frames": completed,
                "expected_frames": parent_config.frame_count,
                "status": "COMPLETE",
            }))
            resource_guard("AFTER_FRAME")

        for receiver_index, receiver in enumerate(parent_config.receivers):
            full = SufficientContributionCache(
                coherent_cache[receiver_index],
                second_cache[receiver_index],
                norm_cache[receiver_index],
            )
            _validate_contribution_cache(full)
            for split, mask in (
                ("full", np.ones(64, dtype=bool)),
                ("even", np.arange(64) % 2 == 0),
                ("odd", np.arange(64) % 2 == 1),
            ):
                base = full.select(mask)
                centered = build_centered_contribution_cache(base)
                powers = np.full(16, scenario.payload_power, dtype=np.float64)
                neumaier = centered_neumaier_statistics(
                    centered, powers, scenario.payload_noise_variance,
                    scenario.pilot_symbols, scenario.coherence_symbols,
                )
                fsum = fixed_order_fsum_statistics(
                    centered, powers, scenario.payload_noise_variance,
                    scenario.pilot_symbols, scenario.coherence_symbols,
                )
                direct = ordinary_direct_statistics(
                    centered, powers, scenario.payload_noise_variance,
                    scenario.pilot_symbols, scenario.coherence_symbols,
                )
                context = {
                    "condition_index": condition_index,
                    "condition_id": condition.id,
                    "receiver_index": receiver_index,
                    "receiver_id": receiver.id,
                    "split": split,
                }
                certificate = certify_uatf_statistics(
                    neumaier, fsum, direct,
                    contribution_cache=base, centered_cache=centered,
                    powers=powers, noise_variance=scenario.payload_noise_variance,
                    tau_p=scenario.pilot_symbols, tau_c=scenario.coherence_symbols,
                    context=context, diagnostic_sink=diagnostic_sink,
                    raise_on_failure=False,
                )
                certificates.append(certificate)
                maximum_primitive_difference = max(
                    maximum_primitive_difference,
                    *(certificate["maximum_absolute_differences"][key] for key in (
                        "coherent_gain", "second_effective_power",
                        "centered_diagonal_mean", "centered_coherent_residual",
                        "centered_correction", "combiner_norm", "numerator",
                        "denominator", "legacy_denominator",
                    )),
                )
                maximum_oracle_distance = max(
                    maximum_oracle_distance,
                    certificate["maximum_absolute_se_oracle_distance"],
                )
                maximum_oracle_radius = max(
                    maximum_oracle_radius,
                    certificate["maximum_se_oracle_endpoint_radius"],
                )
                maximum_pair_bound = max(
                    maximum_pair_bound,
                    certificate["maximum_se_exact_oracle_pair_bound"],
                )
                maximum_log2_interval_width = max(
                    maximum_log2_interval_width,
                    certificate["maximum_log2_interval_width"],
                )
                maximum_se_interval_width = max(
                    maximum_se_interval_width,
                    certificate["maximum_spectral_efficiency_interval_width"],
                )
                maximum_rounded_log2_interval_width = max(
                    maximum_rounded_log2_interval_width,
                    certificate["maximum_rounded_witness_log2_interval_width"],
                )
                maximum_rounded_se_interval_width = max(
                    maximum_rounded_se_interval_width,
                    certificate[
                        "maximum_rounded_witness_spectral_efficiency_interval_width"
                    ],
                )
                maximum_sinr_ratio_bound = max(
                    maximum_sinr_ratio_bound,
                    certificate["maximum_sinr_exact_ratio_bound"],
                )
                maximum_witness_denominator_gap = max(
                    maximum_witness_denominator_gap,
                    certificate["maximum_centered_witness_denominator_gap"],
                )
                maximum_witness_se_gap = max(
                    maximum_witness_se_gap,
                    certificate["maximum_centered_witness_se_gap"],
                )
                maximum_cancellation = max(
                    maximum_cancellation,
                    certificate["maximum_cancellation_metric"],
                )
                rows.append({
                    **context,
                    "online_neumaier": _centered_statistics_json(neumaier),
                    "fixed_order_fsum": _centered_statistics_json(fsum),
                    "array_reconstruction": _centered_statistics_json(direct),
                    "certificate": certificate,
                    "equivalence": certificate["status"],
                })
        resource_guard("AFTER_CONDITION_CERTIFICATION")

    if completed != 192 or len(witness_identity) != 24:
        raise BaselinePrecisionProtocolError("pipeline completion or witness count is wrong")
    require_complete_numerical_certification(
        certificates, expected_rows=54, expected_diagnostics=864
    )
    identity_array = np.asarray(witness_identity, dtype=np.int64)
    order = np.lexsort((
        identity_array[:, 3], identity_array[:, 2],
        identity_array[:, 1], identity_array[:, 0],
    ))
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
        "root_experiment_id": ROOT_REPLAY_SOURCE_ID,
        "immediate_predecessor_experiment_id": IMMEDIATE_FAILED_PREDECESSOR_ID,
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
        "schema_version": "r2c01-v3-baseline-precision-uatf-equivalence/3.0",
        "experiment_id": EXPERIMENT_ID,
        "root_experiment_id": ROOT_REPLAY_SOURCE_ID,
        "immediate_predecessor_experiment_id": IMMEDIATE_FAILED_PREDECESSOR_ID,
        "estimator_id": "B1",
        "receiver_ids": [receiver.id for receiver in parent_config.receivers],
        "splits": ["full", "even", "odd"],
        "primitive_rtol": PRIMITIVE_RTOL,
        "primitive_atol": PRIMITIVE_ATOL,
        "absolute_se_oracle_ceiling": ABSOLUTE_SE_ORACLE_CEILING,
        "atanh_series_terms": ATANH_SERIES_TERMS,
        "row_count": len(rows),
        "diagnostic_count": sum(c["diagnostic_count"] for c in certificates),
        "maximum_absolute_primitive_difference": maximum_primitive_difference,
        "maximum_absolute_se_oracle_distance": maximum_oracle_distance,
        "maximum_se_oracle_endpoint_radius": maximum_oracle_radius,
        "maximum_se_exact_oracle_pair_bound": maximum_pair_bound,
        "maximum_log2_interval_width": maximum_log2_interval_width,
        "maximum_spectral_efficiency_interval_width": maximum_se_interval_width,
        "maximum_rounded_witness_log2_interval_width": (
            maximum_rounded_log2_interval_width
        ),
        "maximum_rounded_witness_spectral_efficiency_interval_width": (
            maximum_rounded_se_interval_width
        ),
        "maximum_sinr_exact_ratio_bound": maximum_sinr_ratio_bound,
        "maximum_centered_witness_denominator_gap": (
            maximum_witness_denominator_gap
        ),
        "maximum_centered_witness_se_gap": maximum_witness_se_gap,
        "maximum_cancellation_metric": maximum_cancellation,
        "rows": rows,
        "effect_statistics": "ABSENT_BY_CONTRACT",
        "uatf_centered_exact_oracle_certification": "PASS",
        "equivalence": "PASS",
    }
    summary = {
        "schema_version": "r2c01-v3-baseline-precision-e0-centered-summary/1.0",
        "experiment_id": EXPERIMENT_ID,
        "root_experiment_id": ROOT_REPLAY_SOURCE_ID,
        "immediate_predecessor_experiment_id": IMMEDIATE_FAILED_PREDECESSOR_ID,
        "execution_status": "COMPLETE",
        "repair_scope": "CENTERED_UATF_EXACT_ORACLE_CERTIFICATION_ONLY",
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
        "uatf_centered_exact_oracle_certification": "PASS",
        "proposed_method_import_or_dispatch": "ABSENT_BY_CONTRACT",
        "scientific_effect_metrics": "ABSENT_BY_CONTRACT",
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
