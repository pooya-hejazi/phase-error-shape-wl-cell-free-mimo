"""Integrity-checked persistence for opt-in retained Gibbs chains.

This module is audit infrastructure. A trace bundle is not scientific evidence
by itself; eligibility is determined only by an independently locked campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .diagnostics import ChainDiagnostics, summarize_chains
from .inference import MomentEstimate

ComplexArray = NDArray[np.complex128]
TRACE_SCHEMA_VERSION = "phase_graph_cme.gibbs_trace.v1"
SCIENTIFIC_BOUNDARY = "AUDIT_TRACE_NOT_SCIENTIFIC_EVIDENCE"
_SAFE_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ARRAY_NAMES = ("retained_phasors", "moment_z", "moment_second")
COMPACT_TRACE_SCHEMA_VERSION = "phase_graph_cme.compact_chain_statistics.v1"
COMPACT_MANIFEST_SCHEMA_VERSION = "phase_graph_cme.compact_trace_manifest.v1"
COMPACT_SPLIT_CONVENTION = (
    "floor(draws/2); first half [0:h); second half [h:2h); "
    "one final odd draw is excluded from split diagnostics but included in full-chain moments"
)
DEFAULT_MAXIMUM_COMPACT_PAYLOAD_BYTES = 2 * 1024 * 1024
ABSOLUTE_MAXIMUM_COMPACT_PAYLOAD_BYTES = 64 * 1024 * 1024
_REQUIRED_SAMPLER_METADATA = {
    "seed",
    "chains",
    "burn_sweeps",
    "retained_draws",
    "thinning_sweeps",
    "maximum_conditional_proposals",
    "retained_samples_returned",
}


@dataclass(frozen=True)
class LoadedGibbsTrace:
    retained_phasors: ComplexArray
    z: ComplexArray
    Z: ComplexArray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedCompactGibbsStatistics:
    document: dict[str, Any]
    z: ComplexArray
    Z: ComplexArray
    diagnostics: ChainDiagnostics
    payload_sha256: str
    manifest_body_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("trace metadata keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported trace metadata type: {type(value)}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _complex_json(value: object) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError("complex compact statistics must be finite")
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _decode_complex(
    value: object, *, expected_shape: tuple[int, ...], label: str
) -> ComplexArray:
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError(f"{label} must have exact real/imag fields")
    try:
        real = np.asarray(value["real"], dtype=np.float64)
        imag = np.asarray(value["imag"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a numeric complex array") from error
    if real.shape != expected_shape or imag.shape != expected_shape:
        raise ValueError(f"{label} has the wrong shape")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError(f"{label} must be finite")
    return (real + 1j * imag).astype(np.complex128)


def _real_vector(value: object, *, variables: int, label: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if array.shape != (variables,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite vector of length {variables}")
    return array


def _statistics_block(values: ComplexArray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("a compact statistics block requires a nonempty draw matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("compact statistics source draws must be finite")
    count = int(values.shape[0])
    phasor_mean = np.mean(values, axis=0)
    second = np.einsum(
        "di,dj->ij", values, values.conj(), optimize=True
    ) / count
    second = 0.5 * (second + second.conj().T)
    np.fill_diagonal(second, 1.0)
    return {
        "count": count,
        "phasor_mean": _complex_json(phasor_mean),
        "second_moment_mean": _complex_json(second),
        "phasor_real_sum": np.sum(values.real, axis=0).tolist(),
        "phasor_imag_sum": np.sum(values.imag, axis=0).tolist(),
        "phasor_real_sum_squares": np.sum(values.real**2, axis=0).tolist(),
        "phasor_imag_sum_squares": np.sum(values.imag**2, axis=0).tolist(),
    }


def _validate_sampler_metadata(
    metadata: object, *, chains: int, draws: int
) -> dict[str, Any]:
    safe = _json_safe(metadata)
    if not isinstance(safe, dict):
        raise ValueError("sampler_metadata must be a JSON-safe mapping")
    missing = sorted(_REQUIRED_SAMPLER_METADATA - set(safe))
    if missing:
        raise ValueError(f"sampler_metadata is missing required fields: {missing}")
    integer_rules = {
        "seed": None,
        "chains": 2,
        "burn_sweeps": 0,
        "retained_draws": 2,
        "thinning_sweeps": 1,
        "maximum_conditional_proposals": 1,
    }
    for name, minimum in integer_rules.items():
        value = safe[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"sampler_metadata.{name} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"sampler_metadata.{name} is below its minimum")
    if safe["chains"] != chains or safe["retained_draws"] != draws:
        raise ValueError("sampler metadata does not match raw chain dimensions")
    if safe["retained_samples_returned"] is not True:
        raise ValueError("compact statistics require an explicit raw-sample opt-in")
    return safe


def build_compact_gibbs_statistics(
    samples: object,
    *,
    task_id: str,
    sampler_metadata: Mapping[str, Any],
    method: str,
    status: str,
    proposal_count: int,
) -> dict[str, Any]:
    """Convert raw phasor chains to strict JSON-safe sufficient statistics.

    The returned object contains no draw-level values. It preserves full-chain
    moments and the aggregates needed to reproduce the package's split-Rhat
    and chain-mean-disagreement diagnostics.
    """

    if _SAFE_TRACE_ID.fullmatch(task_id) is None:
        raise ValueError("task_id must be a safe 1-128 character portable identifier")
    array = np.asarray(samples, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError("samples must have shape (chains, draws, variables)")
    chains, draws, variables = array.shape
    half = draws // 2
    if chains < 2 or half < 2 or variables < 1:
        raise ValueError("compact split diagnostics require two chains and four draws")
    if not np.all(np.isfinite(array)):
        raise ValueError("raw phasor chains must be finite")
    if float(np.max(np.abs(np.abs(array) - 1.0), initial=0.0)) > 1e-12:
        raise ValueError("raw retained samples must be unit-modulus phasors")
    if not isinstance(method, str) or not method or not isinstance(status, str) or not status:
        raise ValueError("method and status must be nonempty strings")
    if (
        not isinstance(proposal_count, int)
        or isinstance(proposal_count, bool)
        or proposal_count < 0
    ):
        raise ValueError("proposal_count must be a nonnegative integer")
    metadata = _validate_sampler_metadata(
        sampler_metadata, chains=chains, draws=draws
    )
    chain_rows: list[dict[str, Any]] = []
    for chain_id in range(chains):
        chain = array[chain_id]
        chain_rows.append(
            {
                "chain_id": chain_id,
                "full": _statistics_block(chain),
                "split_halves": [
                    {
                        "split_half_id": 0,
                        "statistics": _statistics_block(chain[:half]),
                    },
                    {
                        "split_half_id": 1,
                        "statistics": _statistics_block(chain[half : 2 * half]),
                    },
                ],
                "split_ignored_tail_count": draws - 2 * half,
            }
        )
    pooled = _statistics_block(array.reshape((-1, variables)))
    return {
        "schema_version": COMPACT_TRACE_SCHEMA_VERSION,
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "task_id": task_id,
        "method": method,
        "status": status,
        "sample_count": int(chains * draws),
        "proposal_count": proposal_count,
        "sampler_metadata": metadata,
        "dimensions": {
            "chains": chains,
            "draws_per_chain": draws,
            "variables": variables,
            "split_half_draws": half,
            "split_ignored_tail_draws_per_chain": draws - 2 * half,
        },
        "split_convention": COMPACT_SPLIT_CONVENTION,
        "chains": chain_rows,
        "pooled_moments": {
            "count": pooled["count"],
            "phasor_mean": pooled["phasor_mean"],
            "second_moment_mean": pooled["second_moment_mean"],
        },
        "diagnostics_from_raw": _diagnostic_record(array),
    }


def _parse_statistics_block(
    value: object, *, variables: int, label: str
) -> dict[str, Any]:
    required = {
        "count",
        "phasor_mean",
        "second_moment_mean",
        "phasor_real_sum",
        "phasor_imag_sum",
        "phasor_real_sum_squares",
        "phasor_imag_sum_squares",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} fields do not match the compact schema")
    count = value["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"{label}.count must be a positive integer")
    phasor_mean = _decode_complex(
        value["phasor_mean"], expected_shape=(variables,), label=f"{label}.phasor_mean"
    )
    second = _decode_complex(
        value["second_moment_mean"],
        expected_shape=(variables, variables),
        label=f"{label}.second_moment_mean",
    )
    real_sum = _real_vector(
        value["phasor_real_sum"], variables=variables, label=f"{label}.real_sum"
    )
    imag_sum = _real_vector(
        value["phasor_imag_sum"], variables=variables, label=f"{label}.imag_sum"
    )
    real_sumsq = _real_vector(
        value["phasor_real_sum_squares"],
        variables=variables,
        label=f"{label}.real_sumsq",
    )
    imag_sumsq = _real_vector(
        value["phasor_imag_sum_squares"],
        variables=variables,
        label=f"{label}.imag_sumsq",
    )
    if np.any(real_sumsq < 0.0) or np.any(imag_sumsq < 0.0):
        raise ValueError(f"{label} sums of squares must be nonnegative")
    reconstructed_mean = (real_sum + 1j * imag_sum) / count
    if not np.allclose(phasor_mean, reconstructed_mean, rtol=1e-13, atol=1e-13):
        raise ValueError(f"{label} phasor mean disagrees with its sums")
    if not np.allclose(second, second.conj().T, rtol=1e-13, atol=1e-13):
        raise ValueError(f"{label} second moment is not Hermitian")
    if not np.allclose(np.diag(second), 1.0, rtol=1e-13, atol=1e-13):
        raise ValueError(f"{label} second-moment diagonal is not unit phasor power")
    return {
        "count": count,
        "phasor_mean": phasor_mean,
        "second_moment_mean": second,
        "real_sum": real_sum,
        "imag_sum": imag_sum,
        "real_sumsq": real_sumsq,
        "imag_sumsq": imag_sumsq,
    }


def _aggregate_rhat(
    sums: NDArray[np.float64], sums_of_squares: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    means = sums / float(count)
    centered = sums_of_squares - sums**2 / float(count)
    variances = np.maximum(centered / float(count - 1), 0.0)
    within = np.mean(variances, axis=0)
    between = count * np.var(means, axis=0, ddof=1)
    result = np.empty(sums.shape[1], dtype=np.float64)
    flat = within <= np.finfo(float).eps
    result[flat] = np.where(
        between[flat] <= np.finfo(float).eps, 1.0, np.inf
    )
    variance = ((count - 1.0) / count) * within[~flat] + between[~flat] / count
    result[~flat] = np.sqrt(np.maximum(variance / within[~flat], 0.0))
    return result


def _encoded_float_array(value: object, *, variables: int, label: str) -> NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != variables:
        raise ValueError(f"{label} must have length {variables}")
    decoded: list[float] = []
    for item in value:
        if item == "NaN":
            decoded.append(float("nan"))
        elif item == "Infinity":
            decoded.append(float("inf"))
        elif item == "-Infinity":
            decoded.append(float("-inf"))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            decoded.append(float(item))
        else:
            raise ValueError(f"{label} contains an invalid encoded float")
    return np.asarray(decoded, dtype=np.float64)


def _encoded_float(value: object, *, label: str) -> float:
    return _encoded_float_array([value], variables=1, label=label)[0]


def _diagnostics_equal(first: ChainDiagnostics, second: ChainDiagnostics) -> bool:
    return bool(
        first.all_finite == second.all_finite
        and np.allclose(
            first.split_rhat_real,
            second.split_rhat_real,
            rtol=1e-11,
            atol=1e-12,
            equal_nan=True,
        )
        and np.allclose(
            first.split_rhat_imag,
            second.split_rhat_imag,
            rtol=1e-11,
            atol=1e-12,
            equal_nan=True,
        )
        and np.isclose(
            first.maximum_split_rhat,
            second.maximum_split_rhat,
            rtol=1e-11,
            atol=1e-12,
            equal_nan=True,
        )
        and np.isclose(
            first.maximum_chain_mean_disagreement,
            second.maximum_chain_mean_disagreement,
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )
    )


def _parse_compact_document(
    document: object,
) -> tuple[ComplexArray, ComplexArray, ChainDiagnostics]:
    required = {
        "schema_version",
        "scientific_boundary",
        "task_id",
        "method",
        "status",
        "sample_count",
        "proposal_count",
        "sampler_metadata",
        "dimensions",
        "split_convention",
        "chains",
        "pooled_moments",
        "diagnostics_from_raw",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("compact statistics fields do not match the strict schema")
    if document["schema_version"] != COMPACT_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported compact statistics schema version")
    if document["scientific_boundary"] != SCIENTIFIC_BOUNDARY:
        raise ValueError("compact statistics scientific-boundary label is missing")
    task_id = document["task_id"]
    if not isinstance(task_id, str) or _SAFE_TRACE_ID.fullmatch(task_id) is None:
        raise ValueError("compact statistics task_id is invalid")
    if (
        not isinstance(document["method"], str)
        or not document["method"]
        or not isinstance(document["status"], str)
        or not document["status"]
    ):
        raise ValueError("compact method and status must be nonempty strings")
    dimensions = document["dimensions"]
    dimension_fields = {
        "chains",
        "draws_per_chain",
        "variables",
        "split_half_draws",
        "split_ignored_tail_draws_per_chain",
    }
    if not isinstance(dimensions, dict) or set(dimensions) != dimension_fields:
        raise ValueError("compact dimensions do not match the strict schema")
    if any(
        not isinstance(dimensions[name], int) or isinstance(dimensions[name], bool)
        for name in dimension_fields
    ):
        raise ValueError("compact dimensions must be integers")
    chains = dimensions["chains"]
    draws = dimensions["draws_per_chain"]
    variables = dimensions["variables"]
    half = dimensions["split_half_draws"]
    tail = dimensions["split_ignored_tail_draws_per_chain"]
    if chains < 2 or variables < 1 or half < 2 or half != draws // 2 or tail != draws - 2 * half:
        raise ValueError("compact dimensions violate the split convention")
    if document["split_convention"] != COMPACT_SPLIT_CONVENTION:
        raise ValueError("compact split convention is not recognized")
    sample_count = document["sample_count"]
    proposal_count = document["proposal_count"]
    if sample_count != chains * draws:
        raise ValueError("compact sample_count does not match dimensions")
    if (
        not isinstance(proposal_count, int)
        or isinstance(proposal_count, bool)
        or proposal_count < 0
    ):
        raise ValueError("compact proposal_count must be nonnegative")
    _validate_sampler_metadata(document["sampler_metadata"], chains=chains, draws=draws)
    rows = document["chains"]
    if not isinstance(rows, list) or len(rows) != chains:
        raise ValueError("compact chain registry has the wrong length")
    full_blocks: list[dict[str, Any]] = []
    split_blocks: list[dict[str, Any]] = []
    for expected_chain_id, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "chain_id",
            "full",
            "split_halves",
            "split_ignored_tail_count",
        }:
            raise ValueError("compact chain row fields do not match the strict schema")
        if row["chain_id"] != expected_chain_id or row["split_ignored_tail_count"] != tail:
            raise ValueError("compact chain identity or ignored-tail count is invalid")
        full = _parse_statistics_block(
            row["full"], variables=variables, label=f"chain[{expected_chain_id}].full"
        )
        if full["count"] != draws:
            raise ValueError("compact full-chain count disagrees with dimensions")
        full_blocks.append(full)
        halves = row["split_halves"]
        if not isinstance(halves, list) or len(halves) != 2:
            raise ValueError("each compact chain must contain two split halves")
        for expected_half_id, split in enumerate(halves):
            if not isinstance(split, dict) or set(split) != {
                "split_half_id",
                "statistics",
            }:
                raise ValueError("compact split row fields do not match the strict schema")
            if split["split_half_id"] != expected_half_id:
                raise ValueError("compact split-half identity is invalid")
            block = _parse_statistics_block(
                split["statistics"],
                variables=variables,
                label=f"chain[{expected_chain_id}].split[{expected_half_id}]",
            )
            if block["count"] != half:
                raise ValueError("compact split-half count disagrees with dimensions")
            split_blocks.append(block)
    total = float(sum(block["count"] for block in full_blocks))
    z = sum(
        (block["count"] * block["phasor_mean"] for block in full_blocks),
        np.zeros(variables, dtype=np.complex128),
    ) / total
    second = sum(
        (block["count"] * block["second_moment_mean"] for block in full_blocks),
        np.zeros((variables, variables), dtype=np.complex128),
    ) / total
    second = 0.5 * (second + second.conj().T)
    np.fill_diagonal(second, 1.0)
    pooled = document["pooled_moments"]
    if not isinstance(pooled, dict) or set(pooled) != {
        "count",
        "phasor_mean",
        "second_moment_mean",
    }:
        raise ValueError("compact pooled-moment fields do not match the strict schema")
    pooled_z = _decode_complex(
        pooled["phasor_mean"], expected_shape=(variables,), label="pooled.phasor_mean"
    )
    pooled_second = _decode_complex(
        pooled["second_moment_mean"],
        expected_shape=(variables, variables),
        label="pooled.second_moment_mean",
    )
    if pooled["count"] != sample_count or not np.allclose(
        pooled_z, z, rtol=1e-13, atol=1e-13
    ) or not np.allclose(pooled_second, second, rtol=1e-13, atol=1e-13):
        raise ValueError("compact pooled moments do not reconstruct from chain rows")
    split_real_sum = np.asarray([block["real_sum"] for block in split_blocks])
    split_imag_sum = np.asarray([block["imag_sum"] for block in split_blocks])
    split_real_sumsq = np.asarray([block["real_sumsq"] for block in split_blocks])
    split_imag_sumsq = np.asarray([block["imag_sumsq"] for block in split_blocks])
    real_rhat = _aggregate_rhat(split_real_sum, split_real_sumsq, half)
    imag_rhat = _aggregate_rhat(split_imag_sum, split_imag_sumsq, half)
    chain_means = np.asarray([block["phasor_mean"] for block in full_blocks])
    chain_disagreement = float(
        np.max(np.abs(chain_means - np.mean(chain_means, axis=0)), initial=0.0)
    )
    all_finite = bool(
        np.all(np.isfinite(real_rhat))
        and np.all(np.isfinite(imag_rhat))
        and np.isfinite(chain_disagreement)
    )
    reconstructed_diagnostics = ChainDiagnostics(
        maximum_split_rhat=float(np.max(np.concatenate((real_rhat, imag_rhat)))),
        split_rhat_real=real_rhat,
        split_rhat_imag=imag_rhat,
        maximum_chain_mean_disagreement=chain_disagreement,
        all_finite=all_finite,
    )
    stored = document["diagnostics_from_raw"]
    if not isinstance(stored, dict) or set(stored) != {
        "maximum_split_rhat",
        "split_rhat_real",
        "split_rhat_imag",
        "maximum_chain_mean_disagreement",
        "all_finite",
    }:
        raise ValueError("compact diagnostic fields do not match the strict schema")
    stored_diagnostics = ChainDiagnostics(
        maximum_split_rhat=_encoded_float(
            stored["maximum_split_rhat"], label="diagnostics.maximum_split_rhat"
        ),
        split_rhat_real=_encoded_float_array(
            stored["split_rhat_real"], variables=variables, label="diagnostics.real"
        ),
        split_rhat_imag=_encoded_float_array(
            stored["split_rhat_imag"], variables=variables, label="diagnostics.imag"
        ),
        maximum_chain_mean_disagreement=_encoded_float(
            stored["maximum_chain_mean_disagreement"],
            label="diagnostics.chain_disagreement",
        ),
        all_finite=stored["all_finite"],
    )
    if not isinstance(stored_diagnostics.all_finite, bool) or not _diagnostics_equal(
        reconstructed_diagnostics, stored_diagnostics
    ):
        raise ValueError("compact aggregates do not reproduce raw-chain diagnostics")
    return z.astype(np.complex128), second.astype(np.complex128), reconstructed_diagnostics


def reconstruct_compact_moments(
    document: object,
) -> tuple[ComplexArray, ComplexArray]:
    """Strictly validate a compact document and reconstruct pooled ``z``/``Z``."""

    z, second, _ = _parse_compact_document(document)
    return z, second


def reconstruct_compact_diagnostics(document: object) -> ChainDiagnostics:
    """Reconstruct split-Rhat and chain disagreement without raw draws."""

    _, _, diagnostics = _parse_compact_document(document)
    return diagnostics


def write_compact_gibbs_statistics(
    directory: str | Path,
    task_id: str,
    estimate: MomentEstimate,
    *,
    maximum_payload_bytes: int = DEFAULT_MAXIMUM_COMPACT_PAYLOAD_BYTES,
) -> tuple[Path, Path]:
    """Write canonical compact JSON plus a self-checking SHA-256 manifest."""

    if (
        not isinstance(maximum_payload_bytes, int)
        or isinstance(maximum_payload_bytes, bool)
        or not 1 <= maximum_payload_bytes <= ABSOLUTE_MAXIMUM_COMPACT_PAYLOAD_BYTES
    ):
        raise ValueError("maximum_payload_bytes is outside the safe supported range")
    arrays = _validated_arrays(estimate)
    document = build_compact_gibbs_statistics(
        arrays["retained_phasors"],
        task_id=task_id,
        sampler_metadata=estimate.metadata,
        method=estimate.method,
        status=estimate.status,
        proposal_count=int(estimate.proposal_count),
    )
    reconstructed_z, reconstructed_second = reconstruct_compact_moments(document)
    if not np.allclose(reconstructed_z, estimate.z, rtol=1e-13, atol=1e-13) or not np.allclose(
        reconstructed_second, estimate.Z, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("compact statistics do not reproduce the supplied estimate")
    payload_bytes = _canonical_json_bytes(document)
    if len(payload_bytes) > maximum_payload_bytes:
        raise ValueError(
            f"compact payload requires {len(payload_bytes)} bytes, above the declared cap"
        )
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload_path = target / f"{task_id}.compact.json"
    manifest_path = target / f"{task_id}.compact.manifest.json"
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite compact Gibbs statistics")
    manifest_body = {
        "schema_version": COMPACT_MANIFEST_SCHEMA_VERSION,
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "task_id": task_id,
        "payload_file": payload_path.name,
        "payload_bytes": len(payload_bytes),
        "maximum_payload_bytes": maximum_payload_bytes,
        "payload_sha256": _sha256_bytes(payload_bytes),
    }
    manifest = {
        **manifest_body,
        "manifest_body_sha256": _sha256_bytes(_canonical_json_bytes(manifest_body)),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    created_payload = False
    try:
        with payload_path.open("xb") as handle:
            handle.write(payload_bytes)
        created_payload = True
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
    except Exception:
        if created_payload and not manifest_path.exists():
            payload_path.unlink()
        raise
    return payload_path, manifest_path


def load_compact_gibbs_statistics(
    manifest_path: str | Path,
) -> LoadedCompactGibbsStatistics:
    """Validate canonical bytes, both hashes, strict schema, and reconstructions."""

    source = Path(manifest_path)
    try:
        manifest_bytes = source.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compact statistics manifest: {error}") from error
    if manifest_bytes != _canonical_json_bytes(manifest):
        raise ValueError("compact manifest is not in deterministic canonical form")
    required = {
        "schema_version",
        "scientific_boundary",
        "task_id",
        "payload_file",
        "payload_bytes",
        "maximum_payload_bytes",
        "payload_sha256",
        "manifest_body_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("compact manifest fields do not match the strict schema")
    if manifest["schema_version"] != COMPACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported compact manifest schema")
    if manifest["scientific_boundary"] != SCIENTIFIC_BOUNDARY:
        raise ValueError("compact manifest scientific-boundary label is missing")
    task_id = manifest["task_id"]
    if not isinstance(task_id, str) or _SAFE_TRACE_ID.fullmatch(task_id) is None:
        raise ValueError("compact manifest task_id is invalid")
    body = {key: manifest[key] for key in required - {"manifest_body_sha256"}}
    if manifest["manifest_body_sha256"] != _sha256_bytes(_canonical_json_bytes(body)):
        raise ValueError("compact manifest-body SHA-256 mismatch")
    maximum = manifest["maximum_payload_bytes"]
    expected_bytes = manifest["payload_bytes"]
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not 1 <= maximum <= ABSOLUTE_MAXIMUM_COMPACT_PAYLOAD_BYTES
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or not 1 <= expected_bytes <= maximum
    ):
        raise ValueError("compact manifest declares an invalid payload size budget")
    payload_name = manifest["payload_file"]
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError("compact payload_file must be a safe sibling filename")
    payload_path = source.parent / payload_name
    if not payload_path.is_file() or payload_path.stat().st_size != expected_bytes:
        raise ValueError("compact payload is missing or has the wrong byte count")
    payload_bytes = payload_path.read_bytes()
    if manifest["payload_sha256"] != _sha256_bytes(payload_bytes):
        raise ValueError("compact payload SHA-256 mismatch")
    try:
        document = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"compact payload is not valid JSON: {error}") from error
    if payload_bytes != _canonical_json_bytes(document):
        raise ValueError("compact payload is not in deterministic canonical form")
    if not isinstance(document, dict) or document.get("task_id") != task_id:
        raise ValueError("compact manifest and payload task identifiers disagree")
    z, second, diagnostics = _parse_compact_document(document)
    z.setflags(write=False)
    second.setflags(write=False)
    return LoadedCompactGibbsStatistics(
        document=document,
        z=z,
        Z=second,
        diagnostics=diagnostics,
        payload_sha256=str(manifest["payload_sha256"]),
        manifest_body_sha256=str(manifest["manifest_body_sha256"]),
    )


def _reconstruct_moments(samples: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    flat = samples.reshape((-1, samples.shape[2]))
    z = np.mean(flat, axis=0)
    second = np.einsum(
        "si,sj->ij", flat, flat.conj(), optimize=True
    ) / flat.shape[0]
    second = 0.5 * (second + second.conj().T)
    np.fill_diagonal(second, 1.0)
    return z.astype(np.complex128), second.astype(np.complex128)


def _validated_arrays(estimate: MomentEstimate) -> dict[str, ComplexArray]:
    if estimate.retained_samples is None:
        raise ValueError("the estimate has no opt-in retained Gibbs samples")
    samples = np.asarray(estimate.retained_samples, dtype=np.complex128)
    if samples.ndim != 3:
        raise ValueError("retained samples must have shape (chains, draws, variables)")
    chains, draws, variables = samples.shape
    if chains < 2 or draws < 2 or variables < 1:
        raise ValueError("retained sample dimensions are below the Gibbs minimum")
    z = np.asarray(estimate.z, dtype=np.complex128)
    second = np.asarray(estimate.Z, dtype=np.complex128)
    if z.shape != (variables,) or second.shape != (variables, variables):
        raise ValueError("stored moment shapes do not match retained variables")
    if estimate.sample_count != chains * draws:
        raise ValueError("sample_count does not match retained sample dimensions")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(z)) or not np.all(
        np.isfinite(second)
    ):
        raise ValueError("trace samples and moments must be finite")
    if float(np.max(np.abs(np.abs(samples) - 1.0), initial=0.0)) > 1e-12:
        raise ValueError("retained samples are not unit-modulus phasors")
    raw_z, raw_second = _reconstruct_moments(samples)
    if not np.allclose(raw_z, z, rtol=1e-13, atol=1e-13):
        raise ValueError("stored first moment is not reconstructible from samples")
    if not np.allclose(raw_second, second, rtol=1e-13, atol=1e-13):
        raise ValueError("stored second moment is not reconstructible from samples")
    return {
        "retained_phasors": samples,
        "moment_z": z,
        "moment_second": second,
    }


def _diagnostic_record(samples: ComplexArray) -> dict[str, Any]:
    diagnostic = summarize_chains(samples)
    return {
        "maximum_split_rhat": _json_safe(diagnostic.maximum_split_rhat),
        "split_rhat_real": _json_safe(diagnostic.split_rhat_real),
        "split_rhat_imag": _json_safe(diagnostic.split_rhat_imag),
        "maximum_chain_mean_disagreement": _json_safe(
            diagnostic.maximum_chain_mean_disagreement
        ),
        "all_finite": diagnostic.all_finite,
    }


def write_gibbs_trace_bundle(
    directory: str | Path, trace_id: str, estimate: MomentEstimate
) -> tuple[Path, Path]:
    """Write a non-overwriting NPZ payload and JSON integrity manifest."""

    if _SAFE_TRACE_ID.fullmatch(trace_id) is None:
        raise ValueError("trace_id must be a safe 1-128 character portable identifier")
    arrays = _validated_arrays(estimate)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload_path = target / f"{trace_id}.npz"
    manifest_path = target / f"{trace_id}.manifest.json"
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing Gibbs trace bundle")
    metadata = _json_safe(estimate.metadata)
    diagnostics = _diagnostic_record(arrays["retained_phasors"])
    try:
        with payload_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        manifest = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "scientific_boundary": SCIENTIFIC_BOUNDARY,
            "trace_id": trace_id,
            "payload_file": payload_path.name,
            "payload_sha256": _sha256_file(payload_path),
            "arrays": {
                name: {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": _sha256_array(value),
                }
                for name, value in arrays.items()
            },
            "method": estimate.method,
            "status": estimate.status,
            "sample_count": int(estimate.sample_count),
            "proposal_count": int(estimate.proposal_count),
            "sampler_metadata": metadata,
            "diagnostics_recomputed_from_raw": diagnostics,
        }
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        if payload_path.exists() and not manifest_path.exists():
            payload_path.unlink()
        raise
    return payload_path, manifest_path


def load_gibbs_trace_bundle(manifest_path: str | Path) -> LoadedGibbsTrace:
    """Load a trace only after schema, hashes, shapes, and moments validate."""

    source = Path(manifest_path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Gibbs trace manifest: {error}") from error
    required = {
        "schema_version",
        "scientific_boundary",
        "trace_id",
        "payload_file",
        "payload_sha256",
        "arrays",
        "method",
        "status",
        "sample_count",
        "proposal_count",
        "sampler_metadata",
        "diagnostics_recomputed_from_raw",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Gibbs trace manifest fields do not match the strict schema")
    if manifest["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported Gibbs trace schema version")
    if manifest["scientific_boundary"] != SCIENTIFIC_BOUNDARY:
        raise ValueError("Gibbs trace scientific-boundary label is missing")
    payload_name = manifest["payload_file"]
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError("payload_file must be a safe sibling filename")
    payload_path = source.parent / payload_name
    if not payload_path.is_file():
        raise ValueError("Gibbs trace payload is missing")
    if not isinstance(manifest["payload_sha256"], str) or _sha256_file(
        payload_path
    ) != manifest["payload_sha256"].lower():
        raise ValueError("Gibbs trace payload SHA-256 mismatch")
    array_schema = manifest["arrays"]
    if not isinstance(array_schema, dict) or set(array_schema) != set(_ARRAY_NAMES):
        raise ValueError("Gibbs trace array registry does not match the strict schema")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            if set(archive.files) != set(_ARRAY_NAMES):
                raise ValueError("NPZ member names do not match the strict schema")
            arrays = {
                name: np.asarray(archive[name]).copy() for name in _ARRAY_NAMES
            }
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot safely load Gibbs trace payload: {error}") from error
    for name, value in arrays.items():
        specification = array_schema[name]
        if not isinstance(specification, dict) or set(specification) != {
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"invalid array registry entry for {name}")
        if specification["dtype"] != value.dtype.str:
            raise ValueError(f"dtype mismatch for {name}")
        if specification["shape"] != list(value.shape):
            raise ValueError(f"shape mismatch for {name}")
        if specification["sha256"] != _sha256_array(value):
            raise ValueError(f"array SHA-256 mismatch for {name}")
    samples = np.asarray(arrays["retained_phasors"], dtype=np.complex128)
    z = np.asarray(arrays["moment_z"], dtype=np.complex128)
    second = np.asarray(arrays["moment_second"], dtype=np.complex128)
    pseudo = MomentEstimate(
        z=z,
        Z=second,
        method=str(manifest["method"]),
        status=str(manifest["status"]),
        sample_count=int(manifest["sample_count"]),
        proposal_count=int(manifest["proposal_count"]),
        metadata=dict(manifest["sampler_metadata"]),
        retained_samples=samples,
    )
    _validated_arrays(pseudo)
    if _diagnostic_record(samples) != manifest["diagnostics_recomputed_from_raw"]:
        raise ValueError("raw-chain diagnostics do not reproduce the manifest")
    samples.setflags(write=False)
    z.setflags(write=False)
    second.setflags(write=False)
    return LoadedGibbsTrace(
        retained_phasors=samples,
        z=z,
        Z=second,
        manifest=manifest,
    )
