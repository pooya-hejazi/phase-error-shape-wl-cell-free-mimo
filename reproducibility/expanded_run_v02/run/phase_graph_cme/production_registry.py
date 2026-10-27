"""Deterministic seed, frame, task, and reference registries for R2C01.

The production campaign must decide its complete computational registry before
any channel observation or controller diagnostic exists. This module contains
only pure metadata construction and validation; it performs no simulation and
reads no scientific data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence


MASK_63 = (1 << 63) - 1
SMALL_REFERENCE_GROUP_SIZES = (2, 3, 4)
SMALL_REFERENCE_KAPPA_VALUES = (1, 5, 20)
SMALL_REFERENCE_DELTA_DEG_VALUES = (15, 45, 90, 180)
SMALL_REFERENCE_CASE_SEEDS = (950101, 950102, 950103, 950104)


class RegistryError(ValueError):
    """Raised when registered metadata is ambiguous, incomplete, or unsafe."""


def _json_value(value: object, *, path: str) -> object:
    """Validate and normalize an exact built-in JSON value.

    Booleans are deliberately prohibited even though JSON supports them:
    Python's ``bool`` is an ``int`` subclass, and accepting it in a seed tag can
    silently alias an accidental flag with a numeric coordinate. Non-built-in
    scalar types are rejected rather than environment-dependently coerced.
    """

    if type(value) is bool:
        raise RegistryError(f"{path} must not be Boolean")
    if value is None or type(value) in {str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RegistryError(f"{path} must be finite")
        return value
    if type(value) is list:
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RegistryError(f"{path} mapping keys must be strings")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    raise RegistryError(
        f"{path} has non-JSON-safe type {type(value).__name__}; "
        "only exact built-in JSON values are permitted"
    )


def canonical_ascii_json(value: object) -> bytes:
    """Return the unique compact ASCII JSON encoding used by every hash.

    Mapping keys are sorted, non-ASCII code points are escaped, insignificant
    whitespace is omitted, and NaN/infinity are prohibited. A trailing newline
    is intentionally absent because these bytes are a hash payload, not a file.
    """

    normalized = _json_value(value, path="value")
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:  # defensive boundary
        raise RegistryError(f"cannot serialize canonical ASCII JSON: {error}") from error


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value >= 1 << 63:
        raise RegistryError(f"{label} must be a non-Boolean integer in [{minimum},2^63)")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    return _integer(value, label=label, minimum=1)


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value or any(character.isspace() for character in value):
        raise RegistryError(f"{label} must be a nonempty string without whitespace")
    return value


def seed_payload(base_seed: int, *tags: object) -> bytes:
    """Serialize ``[base_seed, *tags]`` under the locked seed contract."""

    base_seed = _integer(base_seed, label="base_seed")
    return canonical_ascii_json([base_seed, *tags])


def derived_seed(base_seed: int, *tags: object) -> int:
    """Derive a stable 63-bit seed from the first eight SHA-256 bytes."""

    digest = hashlib.sha256(seed_payload(base_seed, *tags)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & MASK_63


def source_task_ranking_payload(
    campaign_id: str,
    condition_id: str,
    topology_seed: int,
    realization_index: int,
    ap_index: int,
    pilot_index: int,
) -> bytes:
    """Serialize the exact six-field source-reference ranking identity."""

    campaign_id = _identifier(campaign_id, label="campaign_id")
    condition_id = _identifier(condition_id, label="condition_id")
    topology_seed = _integer(topology_seed, label="topology_seed")
    realization_index = _integer(realization_index, label="realization_index")
    ap_index = _integer(ap_index, label="ap_index")
    pilot_index = _integer(pilot_index, label="pilot_index")
    return canonical_ascii_json(
        [campaign_id, condition_id, topology_seed, realization_index, ap_index, pilot_index]
    )


def source_task_ranking_sha256(
    campaign_id: str,
    condition_id: str,
    topology_seed: int,
    realization_index: int,
    ap_index: int,
    pilot_index: int,
) -> str:
    """Return the lowercase SHA-256 hex rank for a source-scale task."""

    return hashlib.sha256(
        source_task_ranking_payload(
            campaign_id,
            condition_id,
            topology_seed,
            realization_index,
            ap_index,
            pilot_index,
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SmallReferenceCase:
    """One pre-observation case in the 144-case finite-rule qualification."""

    case_index: int
    group_size: int
    kappa: int
    delta_deg: int
    case_seed: int

    @property
    def identity_tuple(self) -> tuple[int, int, int, int]:
        return (self.group_size, self.kappa, self.delta_deg, self.case_seed)


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One registered source-scale topology/condition/realization frame."""

    frame_index: int
    topology_index: int
    condition_index: int
    campaign_id: str
    base_seed: int
    condition_id: str
    topology_seed: int
    realization_index: int
    frame_seed: int

    @property
    def order_tuple(self) -> tuple[int, int, int]:
        return (self.topology_index, self.condition_index, self.realization_index)

    @property
    def identity_tuple(self) -> tuple[str, str, int, int]:
        return (
            self.campaign_id,
            self.condition_id,
            self.topology_seed,
            self.realization_index,
        )


@dataclass(frozen=True, slots=True)
class SourceTask:
    """One registered AP/pilot controller task within a source frame."""

    task_index: int
    frame_index: int
    topology_index: int
    condition_index: int
    campaign_id: str
    base_seed: int
    condition_id: str
    topology_seed: int
    realization_index: int
    ap_index: int
    pilot_index: int
    controller_seed: int
    ranking_sha256: str

    @property
    def order_tuple(self) -> tuple[int, int, int]:
        return (self.frame_index, self.ap_index, self.pilot_index)

    @property
    def identity_tuple(self) -> tuple[str, str, int, int, int, int]:
        return (
            self.campaign_id,
            self.condition_id,
            self.topology_seed,
            self.realization_index,
            self.ap_index,
            self.pilot_index,
        )


@dataclass(frozen=True, slots=True)
class SelectedReferenceTask:
    """A metadata-only source reference selected before observations exist."""

    selection_index: int
    frame_index: int
    reference_rank_within_frame: int
    task: SourceTask

    @property
    def selection_basis(self) -> str:
        return "SHA256_REGISTERED_IDENTITY_PRE_OBSERVATION"


@dataclass(frozen=True, slots=True)
class SeedSetPreflight:
    """Machine-recordable proof of unique, pairwise-disjoint seed sets."""

    set_counts: tuple[tuple[str, int], ...]
    total_seed_count: int
    unique_seed_count: int
    canonical_registry_sha256: str


@dataclass(frozen=True, slots=True)
class FrameRegistryPreflight:
    campaign_id: str
    topology_count: int
    condition_count: int
    realizations_per_topology_condition: int
    frame_count: int
    seed_preflight: SeedSetPreflight


@dataclass(frozen=True, slots=True)
class SourceRegistryPreflight:
    campaign_id: str
    topology_count: int
    condition_count: int
    realizations_per_topology_condition: int
    frame_count: int
    access_point_count: int
    pilot_count: int
    task_count: int
    seed_preflight: SeedSetPreflight


def build_small_reference_registry() -> tuple[SmallReferenceCase, ...]:
    """Build the exact R2C01 3 x 3 x 4 x 4 registry in locked order."""

    registry = tuple(
        SmallReferenceCase(index, group_size, kappa, delta_deg, case_seed)
        for index, (group_size, kappa, delta_deg, case_seed) in enumerate(
            product(
                SMALL_REFERENCE_GROUP_SIZES,
                SMALL_REFERENCE_KAPPA_VALUES,
                SMALL_REFERENCE_DELTA_DEG_VALUES,
                SMALL_REFERENCE_CASE_SEEDS,
            )
        )
    )
    preflight_small_reference_registry(registry)
    return registry


def preflight_small_reference_registry(
    registry: Sequence[SmallReferenceCase],
) -> dict[str, int | str]:
    """Reject missing, duplicated, or reordered small-reference cases."""

    cases = tuple(registry)
    expected_identities = tuple(
        product(
            SMALL_REFERENCE_GROUP_SIZES,
            SMALL_REFERENCE_KAPPA_VALUES,
            SMALL_REFERENCE_DELTA_DEG_VALUES,
            SMALL_REFERENCE_CASE_SEEDS,
        )
    )
    if len(cases) != len(expected_identities):
        raise RegistryError(
            f"small-reference registry must contain {len(expected_identities)} cases"
        )
    for index, (case, expected_identity) in enumerate(zip(cases, expected_identities)):
        if type(case) is not SmallReferenceCase:
            raise RegistryError(f"small-reference row {index} has the wrong type")
        if case.case_index != index:
            raise RegistryError("small-reference case_index sequence is not canonical")
        if case.identity_tuple != expected_identity:
            raise RegistryError("small-reference registry is duplicated, missing, or reordered")
    identities = [case.identity_tuple for case in cases]
    if len(set(identities)) != len(identities):
        raise RegistryError("small-reference registry contains a duplicate identity")
    digest = hashlib.sha256(
        canonical_ascii_json([list(identity) for identity in identities])
    ).hexdigest()
    return {
        "case_count": len(cases),
        "unique_case_count": len(set(identities)),
        "canonical_registry_sha256": digest,
    }


def _materialize_identifiers(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    materialized = tuple(_identifier(value, label=label) for value in values)
    if not materialized:
        raise RegistryError(f"{label} must be nonempty")
    if len(set(materialized)) != len(materialized):
        raise RegistryError(f"{label} must be unique")
    if len({value.casefold() for value in materialized}) != len(materialized):
        raise RegistryError(f"{label} must also be unique case-insensitively")
    return materialized


def _materialize_seeds(values: Iterable[int], *, label: str) -> tuple[int, ...]:
    materialized = tuple(_integer(value, label=label) for value in values)
    if not materialized:
        raise RegistryError(f"{label} must be nonempty")
    if len(set(materialized)) != len(materialized):
        raise RegistryError(f"{label} must be unique")
    return materialized


def build_source_frame_registry(
    *,
    campaign_id: str,
    base_seed: int,
    topology_seeds: Iterable[int],
    condition_ids: Iterable[str],
    realizations_per_topology_condition: int,
) -> tuple[SourceFrame, ...]:
    """Build frames in topology -> condition -> realization order."""

    campaign_id = _identifier(campaign_id, label="campaign_id")
    base_seed = _integer(base_seed, label="base_seed")
    topologies = _materialize_seeds(topology_seeds, label="topology_seed")
    conditions = _materialize_identifiers(condition_ids, label="condition_id")
    realizations = _positive_integer(
        realizations_per_topology_condition,
        label="realizations_per_topology_condition",
    )
    if base_seed in topologies:
        raise RegistryError("base_seed and topology_seeds must be disjoint")

    rows: list[SourceFrame] = []
    for topology_index, topology_seed in enumerate(topologies):
        for condition_index, condition_id in enumerate(conditions):
            for realization_index in range(realizations):
                rows.append(
                    SourceFrame(
                        frame_index=len(rows),
                        topology_index=topology_index,
                        condition_index=condition_index,
                        campaign_id=campaign_id,
                        base_seed=base_seed,
                        condition_id=condition_id,
                        topology_seed=topology_seed,
                        realization_index=realization_index,
                        frame_seed=derived_seed(
                            base_seed,
                            campaign_id,
                            condition_id,
                            topology_seed,
                            realization_index,
                            "frame",
                        ),
                    )
                )
    registry = tuple(rows)
    preflight_source_frame_registry(registry)
    return registry


def build_source_task_registry(
    frames: Sequence[SourceFrame],
    *,
    access_points: int,
    pilot_symbols: int,
) -> tuple[SourceTask, ...]:
    """Build tasks in frame -> AP -> pilot order and preflight all seeds."""

    access_points = _positive_integer(access_points, label="access_points")
    pilot_symbols = _positive_integer(pilot_symbols, label="pilot_symbols")
    canonical_frames = tuple(frames)
    preflight_source_frame_registry(canonical_frames)
    rows: list[SourceTask] = []
    for frame in canonical_frames:
        for ap_index in range(access_points):
            for pilot_index in range(pilot_symbols):
                rows.append(
                    SourceTask(
                        task_index=len(rows),
                        frame_index=frame.frame_index,
                        topology_index=frame.topology_index,
                        condition_index=frame.condition_index,
                        campaign_id=frame.campaign_id,
                        base_seed=frame.base_seed,
                        condition_id=frame.condition_id,
                        topology_seed=frame.topology_seed,
                        realization_index=frame.realization_index,
                        ap_index=ap_index,
                        pilot_index=pilot_index,
                        controller_seed=derived_seed(
                            frame.base_seed,
                            frame.campaign_id,
                            frame.condition_id,
                            frame.topology_seed,
                            frame.realization_index,
                            ap_index,
                            pilot_index,
                            "controller",
                        ),
                        ranking_sha256=source_task_ranking_sha256(
                            frame.campaign_id,
                            frame.condition_id,
                            frame.topology_seed,
                            frame.realization_index,
                            ap_index,
                            pilot_index,
                        ),
                    )
                )
    registry = tuple(rows)
    preflight_source_task_registry(canonical_frames, registry)
    return registry


def preflight_seed_sets(seed_sets: Mapping[str, Iterable[int]]) -> SeedSetPreflight:
    """Require uniqueness within and pairwise disjointness across named sets."""

    if type(seed_sets) is not dict or not seed_sets:
        raise RegistryError("seed_sets must be a nonempty exact dict")
    normalized: dict[str, tuple[int, ...]] = {}
    casefolded: set[str] = set()
    owner: dict[int, str] = {}
    for raw_label, raw_values in seed_sets.items():
        label = _identifier(raw_label, label="seed-set label")
        if label.casefold() in casefolded:
            raise RegistryError("seed-set labels must be unique case-insensitively")
        casefolded.add(label.casefold())
        values = tuple(_integer(value, label=f"{label} seed") for value in raw_values)
        if len(set(values)) != len(values):
            raise RegistryError(f"seed set {label!r} contains a duplicate")
        for value in values:
            previous = owner.get(value)
            if previous is not None:
                raise RegistryError(
                    f"seed {value} overlaps seed sets {previous!r} and {label!r}"
                )
            owner[value] = label
        normalized[label] = values
    document = {label: list(values) for label, values in normalized.items()}
    counts = tuple((label, len(values)) for label, values in normalized.items())
    return SeedSetPreflight(
        set_counts=counts,
        total_seed_count=sum(count for _, count in counts),
        unique_seed_count=len(owner),
        canonical_registry_sha256=hashlib.sha256(
            canonical_ascii_json(document)
        ).hexdigest(),
    )


def preflight_source_frame_registry(
    frames: Sequence[SourceFrame],
    *,
    reserved_seeds: Iterable[int] = (),
) -> FrameRegistryPreflight:
    """Verify a complete rectangular frame registry and its derived seeds."""

    rows = tuple(frames)
    if not rows:
        raise RegistryError("source frame registry must be nonempty")
    if any(type(frame) is not SourceFrame for frame in rows):
        raise RegistryError("source frame registry contains a row of the wrong type")

    campaign_id = _identifier(rows[0].campaign_id, label="campaign_id")
    base_seed = _integer(rows[0].base_seed, label="base_seed")
    topology_by_index: dict[int, int] = {}
    condition_by_index: dict[int, str] = {}
    identities: set[tuple[str, str, int, int]] = set()
    observed_order: list[tuple[int, int, int]] = []

    for expected_index, frame in enumerate(rows):
        if frame.frame_index != expected_index:
            raise RegistryError("source frame_index sequence is not canonical")
        if frame.campaign_id != campaign_id or frame.base_seed != base_seed:
            raise RegistryError("all source frames must share campaign_id and base_seed")
        topology_index = _integer(frame.topology_index, label="topology_index")
        condition_index = _integer(frame.condition_index, label="condition_index")
        topology_seed = _integer(frame.topology_seed, label="topology_seed")
        condition_id = _identifier(frame.condition_id, label="condition_id")
        realization_index = _integer(frame.realization_index, label="realization_index")
        old_topology = topology_by_index.setdefault(topology_index, topology_seed)
        if old_topology != topology_seed:
            raise RegistryError("topology_index maps to multiple topology seeds")
        old_condition = condition_by_index.setdefault(condition_index, condition_id)
        if old_condition != condition_id:
            raise RegistryError("condition_index maps to multiple condition ids")
        if frame.identity_tuple in identities:
            raise RegistryError("source frame registry contains a duplicate identity")
        identities.add(frame.identity_tuple)
        observed_order.append(frame.order_tuple)
        expected_seed = derived_seed(
            base_seed,
            campaign_id,
            condition_id,
            topology_seed,
            realization_index,
            "frame",
        )
        if frame.frame_seed != expected_seed:
            raise RegistryError("source frame has an incorrect derived frame_seed")

    topology_count = len(topology_by_index)
    condition_count = len(condition_by_index)
    if set(topology_by_index) != set(range(topology_count)):
        raise RegistryError("topology_index values must be consecutive from zero")
    if set(condition_by_index) != set(range(condition_count)):
        raise RegistryError("condition_index values must be consecutive from zero")
    realization_count = max(frame.realization_index for frame in rows) + 1
    expected_order = list(
        product(range(topology_count), range(condition_count), range(realization_count))
    )
    if observed_order != expected_order:
        raise RegistryError("source frame registry is missing, duplicated, or reordered")

    topology_seeds = tuple(topology_by_index[index] for index in range(topology_count))
    condition_ids = tuple(condition_by_index[index] for index in range(condition_count))
    if len(set(topology_seeds)) != topology_count:
        raise RegistryError("different topology indices reuse a topology seed")
    if len(set(condition_ids)) != condition_count:
        raise RegistryError("different condition indices reuse a condition id")
    seed_sets: dict[str, Iterable[int]] = {
        "base_seed": (base_seed,),
        "topology_seeds": topology_seeds,
        "frame_seeds": tuple(frame.frame_seed for frame in rows),
    }
    external = tuple(reserved_seeds)
    if external:
        seed_sets["reserved_seeds"] = external
    seed_preflight = preflight_seed_sets(seed_sets)
    return FrameRegistryPreflight(
        campaign_id=campaign_id,
        topology_count=topology_count,
        condition_count=condition_count,
        realizations_per_topology_condition=realization_count,
        frame_count=len(rows),
        seed_preflight=seed_preflight,
    )


def preflight_source_task_registry(
    frames: Sequence[SourceFrame],
    tasks: Sequence[SourceTask],
    *,
    reserved_seeds: Iterable[int] = (),
) -> SourceRegistryPreflight:
    """Verify exact task ordering, rectangularity, hashes, and seed closure."""

    canonical_frames = tuple(frames)
    frame_summary = preflight_source_frame_registry(canonical_frames)
    rows = tuple(tasks)
    if not rows:
        raise RegistryError("source task registry must be nonempty")
    if any(type(task) is not SourceTask for task in rows):
        raise RegistryError("source task registry contains a row of the wrong type")

    identities: set[tuple[str, str, int, int, int, int]] = set()
    observed_order: list[tuple[int, int, int]] = []
    for expected_index, task in enumerate(rows):
        if task.task_index != expected_index:
            raise RegistryError("source task_index sequence is not canonical")
        frame_index = _integer(task.frame_index, label="frame_index")
        if frame_index >= len(canonical_frames):
            raise RegistryError("source task refers to an unknown frame_index")
        frame = canonical_frames[frame_index]
        matching_fields = (
            task.topology_index == frame.topology_index
            and task.condition_index == frame.condition_index
            and task.campaign_id == frame.campaign_id
            and task.base_seed == frame.base_seed
            and task.condition_id == frame.condition_id
            and task.topology_seed == frame.topology_seed
            and task.realization_index == frame.realization_index
        )
        if not matching_fields:
            raise RegistryError("source task metadata does not match its registered frame")
        ap_index = _integer(task.ap_index, label="ap_index")
        pilot_index = _integer(task.pilot_index, label="pilot_index")
        if task.identity_tuple in identities:
            raise RegistryError("source task registry contains a duplicate identity")
        identities.add(task.identity_tuple)
        observed_order.append(task.order_tuple)
        expected_seed = derived_seed(
            frame.base_seed,
            frame.campaign_id,
            frame.condition_id,
            frame.topology_seed,
            frame.realization_index,
            ap_index,
            pilot_index,
            "controller",
        )
        if task.controller_seed != expected_seed:
            raise RegistryError("source task has an incorrect derived controller_seed")
        expected_ranking = source_task_ranking_sha256(*task.identity_tuple)
        if task.ranking_sha256 != expected_ranking:
            raise RegistryError("source task has an incorrect ranking_sha256")

    access_point_count = max(task.ap_index for task in rows) + 1
    pilot_count = max(task.pilot_index for task in rows) + 1
    expected_order = list(
        product(range(len(canonical_frames)), range(access_point_count), range(pilot_count))
    )
    if observed_order != expected_order:
        raise RegistryError("source task registry is missing, duplicated, or reordered")

    frame_stride = (
        frame_summary.condition_count
        * frame_summary.realizations_per_topology_condition
    )
    topology_seeds = tuple(
        canonical_frames[index].topology_seed
        for index in range(0, len(canonical_frames), frame_stride)
    )
    seed_sets: dict[str, Iterable[int]] = {
        "base_seed": (canonical_frames[0].base_seed,),
        "topology_seeds": topology_seeds,
        "frame_seeds": tuple(frame.frame_seed for frame in canonical_frames),
        "controller_seeds": tuple(task.controller_seed for task in rows),
    }
    external = tuple(reserved_seeds)
    if external:
        seed_sets["reserved_seeds"] = external
    seed_preflight = preflight_seed_sets(seed_sets)
    return SourceRegistryPreflight(
        campaign_id=frame_summary.campaign_id,
        topology_count=frame_summary.topology_count,
        condition_count=frame_summary.condition_count,
        realizations_per_topology_condition=(
            frame_summary.realizations_per_topology_condition
        ),
        frame_count=frame_summary.frame_count,
        access_point_count=access_point_count,
        pilot_count=pilot_count,
        task_count=len(rows),
        seed_preflight=seed_preflight,
    )


def preflight_source_registry(
    frames: Sequence[SourceFrame],
    tasks: Sequence[SourceTask],
    *,
    reserved_seeds: Iterable[int] = (),
) -> SourceRegistryPreflight:
    """Alias for the combined source frame/task preflight."""

    return preflight_source_task_registry(
        frames, tasks, reserved_seeds=reserved_seeds
    )


def select_reference_tasks(
    frames: Sequence[SourceFrame],
    tasks: Sequence[SourceTask],
    *,
    references_per_frame: int,
) -> tuple[SelectedReferenceTask, ...]:
    """Select reference tasks from registered identities, before observations.

    Caller order is ignored. The SHA-256 digest is followed by the exact
    six-field identity tuple as a deterministic collision tie-break.
    """

    references_per_frame = _positive_integer(
        references_per_frame, label="references_per_frame"
    )
    canonical_frames = tuple(frames)
    if any(type(task) is not SourceTask for task in tasks):
        raise RegistryError("source task registry contains a row of the wrong type")
    canonical_tasks = tuple(sorted(tasks, key=lambda task: task.task_index))
    summary = preflight_source_task_registry(canonical_frames, canonical_tasks)
    tasks_per_frame = summary.access_point_count * summary.pilot_count
    if references_per_frame > tasks_per_frame:
        raise RegistryError("references_per_frame exceeds tasks per frame")

    selected: list[SelectedReferenceTask] = []
    for frame in canonical_frames:
        start = frame.frame_index * tasks_per_frame
        frame_tasks = canonical_tasks[start : start + tasks_per_frame]
        ranked = sorted(
            frame_tasks,
            key=lambda task: (
                bytes.fromhex(task.ranking_sha256),
                task.identity_tuple,
            ),
        )
        for reference_rank, task in enumerate(ranked[:references_per_frame]):
            selected.append(
                SelectedReferenceTask(
                    selection_index=len(selected),
                    frame_index=frame.frame_index,
                    reference_rank_within_frame=reference_rank,
                    task=task,
                )
            )
    return tuple(selected)


__all__ = (
    "FrameRegistryPreflight",
    "MASK_63",
    "RegistryError",
    "SMALL_REFERENCE_CASE_SEEDS",
    "SMALL_REFERENCE_GROUP_SIZES",
    "SMALL_REFERENCE_KAPPA_VALUES",
    "SMALL_REFERENCE_DELTA_DEG_VALUES",
    "SelectedReferenceTask",
    "SeedSetPreflight",
    "SmallReferenceCase",
    "SourceFrame",
    "SourceRegistryPreflight",
    "SourceTask",
    "build_small_reference_registry",
    "build_source_frame_registry",
    "build_source_task_registry",
    "canonical_ascii_json",
    "derived_seed",
    "preflight_seed_sets",
    "preflight_small_reference_registry",
    "preflight_source_frame_registry",
    "preflight_source_registry",
    "preflight_source_task_registry",
    "seed_payload",
    "select_reference_tasks",
    "source_task_ranking_payload",
    "source_task_ranking_sha256",
)
