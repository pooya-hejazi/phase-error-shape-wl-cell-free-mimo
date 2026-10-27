"""T06B witness-recording copy of frozen development_batch; algorithm unchanged.
Only additions: optional observer argument and observer(fit) after actual MF.
The AST-equivalence test protects every remaining executable statement.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
import numpy as np
from .baselines import clmmse, phase_unaware_lmmse, widely_linear_mmse
from .budget_router import GroupPriority, select_groups, transfer_disagreement, transfer_stack
from .controller_v3 import V3FiniteRunController, V3FatalControllerError
from .mean_field_baseline import mean_field_group_estimate
from .model import PilotGroupModel

ROUTING = ("TRANSFER", "TRANSFER_NO_PRIORITY", "PHASE", "MODEL", "CYCLIC", "RANDOM")
METHODS = ("B0", "B1", "B2", "MF", "V3") + ROUTING


@dataclass(frozen=True)
class GroupTask:
    group_id: str
    model: PilotGroupModel
    observation: np.ndarray
    v3_seed: int


@dataclass
class BatchResult:
    method: str
    estimates: dict[str, np.ndarray]
    rows: list[dict]
    selected_ids: tuple[str, ...]
    elapsed_seconds: float
    component_seconds: dict[str, float] = field(default_factory=dict)


def _check_output(estimate, task):
    value = np.asarray(estimate, dtype=np.complex128)
    if value.shape != task.model.los.shape or not np.isfinite(value).all():
        raise FloatingPointError("invalid estimator output")
    return value.copy()


def estimate_batch(tasks: list[GroupTask], method: str, *, fraction: float = .25,
                   selector_seed: int = 0, frame_index: int = 0,
                   guard: Callable[[], None] | None = None,
                   event: Callable[[dict], None] | None = None,
                   _v3_factory=None, _mf_observer=None) -> BatchResult:
    """Execute independently, with complete outer timing and explicit test injection."""
    started = perf_counter()
    if method not in METHODS:
        raise ValueError("unknown development method")
    if not tasks:
        raise ValueError("empty batch")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if not isinstance(selector_seed, int) or isinstance(selector_seed, bool) or selector_seed < 0:
        raise ValueError("selector_seed must be a nonnegative integer")
    select_groups([], fraction)  # Validate budget for all methods.
    ids = [t.group_id for t in tasks]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("unique nonempty group IDs required")
    ordered = sorted(tasks, key=lambda t: t.group_id)
    for t in ordered:
        t.model.validate()
        if not isinstance(t.v3_seed, int) or isinstance(t.v3_seed, bool) or t.v3_seed < 0:
            raise ValueError("nonnegative integer V3 seed required")
        if np.shape(t.observation) != (t.model.antenna_count,) or not np.isfinite(t.observation).all():
            raise ValueError("invalid observation")

    def checkpoint():
        if guard is not None:
            guard()

    def emit(kind, **values):
        if event is not None:
            event({"kind": kind, "method": method, **values})

    checkpoint()
    values, records, mf = {}, {}, {}
    components = dict(affine=0., mf=0., routing=0., v3=0., v3_construction=0.)
    for t in ordered:
        records[t.group_id] = dict(group_id=t.group_id, method=method,
                                  v3_seed=t.v3_seed, selected=False,
                                  score=None, mf_status=None, mf_fallback=None)
    if method in ("B0", "B1", "B2"):
        estimator = {"B0": phase_unaware_lmmse, "B1": widely_linear_mmse, "B2": clmmse}[method]
        for t in ordered:
            emit("GROUP_START", group_id=t.group_id)
            checkpoint()
            tic = perf_counter()
            values[t.group_id] = _check_output(np.column_stack(
                [estimator(t.model, t.observation, k) for k in range(t.model.group_size)]), t)
            elapsed = perf_counter()-tic
            components["affine"] += elapsed
            records[t.group_id].update(status=method, estimator_seconds=elapsed)
            emit("GROUP_COMPLETE", **records[t.group_id])
    elif method != "V3":
        for t in ordered:
            emit("GROUP_START", group_id=t.group_id)
            checkpoint()
            tic = perf_counter()
            fit = mean_field_group_estimate(t.model, t.observation)
            if _mf_observer is not None:
                _mf_observer(t.group_id, fit)
            elapsed = perf_counter()-tic
            components["mf"] += elapsed
            mf[t.group_id] = fit
            values[t.group_id] = _check_output(fit.estimates, t)
            records[t.group_id].update(
                mf_status=fit.status, mf_fallback=fit.status == "B1_FALLBACK",
                mf_seconds=elapsed, mf_sweeps=fit.total_sweeps,
                mf_coordinate_updates=fit.total_coordinate_updates,
                mf_grid_discrepancy=fit.grid_discrepancy,
                mf_fallback_reason=fit.fallback_reason,
                status=fit.status, estimator_seconds=elapsed)
            emit("MF_COMPLETE", group_id=t.group_id, status=fit.status, seconds=elapsed)

    selected = tuple(t.group_id for t in ordered) if method == "V3" else ()
    if method in ROUTING:
        tic = perf_counter()
        n = len(ordered)
        permutation = np.random.default_rng(selector_seed).permutation(n) if method == "RANDOM" else None
        random_rank = {int(index): rank for rank, index in enumerate(permutation)} if permutation is not None else {}
        priorities = []
        for index, t in enumerate(ordered):
            fit = mf[t.group_id]
            starts = np.stack([f.z for f in fit.all_fits[3:]])
            extra = {}
            if method in ("TRANSFER", "TRANSFER_NO_PRIORITY"):
                d = transfer_disagreement(t.model, starts, fit.fine.z, fit.coarse.z)
                score = d.score
                extra = dict(start_component=d.start_component, grid_component=d.grid_component,
                             channel_energy=d.channel_energy)
            elif method == "PHASE":
                spread = max(float(np.sum(abs(starts[a]-starts[b])**2))
                             for a in range(3) for b in range(a))
                score = (spread+float(np.sum(abs(fit.fine.z-fit.coarse.z)**2)))/t.model.group_size
            elif method == "MODEL":
                B, energy = transfer_stack(t.model)
                score = 0. if energy == 0 else float(np.sum(abs(B)**2)/energy)
            elif method == "CYCLIC":
                score = float(n-((index-frame_index) % n))
            else:
                score = float(n-random_rank[index])
            priority = fit.status == "B1_FALLBACK" and method != "TRANSFER_NO_PRIORITY"
            priorities.append(GroupPriority(t.group_id, score, priority))
            records[t.group_id].update(score=score, priority_fallback=priority, **extra)
        selected = select_groups(priorities, fraction)
        components["routing"] = perf_counter()-tic
        emit("SELECTION_FROZEN", selected_ids=list(selected), budget=len(selected),
             scores={r["group_id"]: r["score"] for r in records.values()})

    if selected:
        checkpoint()
        tic = perf_counter()
        controller = (_v3_factory or V3FiniteRunController)()
        components["v3_construction"] = perf_counter()-tic
        selected_set = set(selected)
        for t in ordered:
            if t.group_id not in selected_set:
                continue
            checkpoint()
            emit("V3_START", group_id=t.group_id, seed=t.v3_seed)
            tic = perf_counter()
            outcome = controller.estimate_all(t.model, t.observation, seed=t.v3_seed,
                                             resource_guard=guard, retain_audit_trace=False)
            elapsed = perf_counter()-tic
            components["v3"] += elapsed
            records[t.group_id].update(selected=True, v3_status=outcome.status,
                v3_mode=outcome.mode, v3_seconds=elapsed, v3_diagnostics=outcome.diagnostics,
                status=outcome.status)
            emit("V3_RETURNED", **records[t.group_id])  # Preserve failure before raising.
            allowed = {
                ("CLMMSE_FALLBACK", "BASELINE_FALLBACK"),
                ("REPLICA_EXCHANGE_FINITE_RUN_APPROXIMATION",
                 "EMPIRICALLY_ACCEPTED_FINITE_RUN_APPROXIMATION_NOT_CERTIFIED")}
            if (outcome.mode, outcome.status) not in allowed:
                raise V3FatalControllerError("inconsistent V3 status and mode")
            if outcome.diagnostics.get("fallback_reason") == "ALLOWED_PRECOMPLETION_SAMPLER_EXCEPTION":
                raise V3FatalControllerError("precompletion sampler exception fails T06A integration")
            if outcome.diagnostics.get("cost_accounting", {}).get("accounting_complete") is not True:
                raise V3FatalControllerError("incomplete V3 resource accounting")
            values[t.group_id] = _check_output(outcome.estimates, t)
            records[t.group_id]["status"] = outcome.status
            records[t.group_id]["estimator_seconds"] = records[t.group_id].get("mf_seconds", 0.)+elapsed
            checkpoint()
    for t in ordered:
        if t.group_id not in values:
            raise RuntimeError("missing group output")
        values[t.group_id] = _check_output(values[t.group_id], t)
    checkpoint()
    return BatchResult(method, values, [records[t.group_id] for t in ordered],
                       selected, perf_counter()-started, components)
