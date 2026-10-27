"""Observable, pre-V3 routing only; disagreement is NOT an error certificate."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

import numpy as np

from .model import PilotGroupModel


@dataclass(frozen=True)
class Disagreement:
    score: float
    start_component: float
    grid_component: float
    channel_energy: float


@dataclass(frozen=True)
class GroupPriority:
    group_id: str
    score: float
    mf_fallback: bool

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a nonempty stable string")
        if not isinstance(self.mf_fallback, bool):
            raise ValueError("mf_fallback must be an explicit Boolean")
        if not isfinite(self.score) or self.score < 0:
            raise ValueError("score must be finite and nonnegative")


def transfer_stack(model: PilotGroupModel) -> tuple[np.ndarray, float]:
    """Return B[k, antenna, phase] and unconditional aggregate channel energy."""
    model.validate()
    if model.group_size < 1 or model.antenna_count < 1:
        raise ValueError("nonempty group and antenna dimensions required")
    solved = np.linalg.solve(model.diffuse_pilot_covariance(), model.weighted_los())
    blocks = []
    for k in range(model.group_size):
        B = -model.beta[k].conjugate() * model.covariances[k] @ solved
        B[:, k] += model.los[:, k]
        blocks.append(B)
    stack = np.stack(blocks)
    energy = float(np.sum(np.abs(model.los) ** 2)
                   + np.trace(model.covariances, axis1=1, axis2=2).real.sum())
    if not np.isfinite(stack).all() or not isfinite(energy) or energy < 0:
        raise FloatingPointError("invalid channel-transfer arithmetic")
    return stack, energy


def _means(value, shape):
    z = np.asarray(value, dtype=np.complex128)
    if z.shape != shape or not np.isfinite(z).all():
        raise ValueError(f"finite phase means with shape {shape} required")
    if np.any(np.abs(z) > 1 + 1e-8):
        raise ValueError("phase-mean modulus exceeds one")
    return z


def transfer_disagreement(model: PilotGroupModel, fine_starts,
                          selected_fine, selected_coarse) -> Disagreement:
    """Use the three fixed fine starts and selected coarse/fine MF means.

    Takes neither true channels nor downstream outcomes. Nonconverged but
    finite capped-start means are allowed; MF fallback priority is separate.
    """
    p = model.group_size
    starts = _means(fine_starts, (3, p))
    fine = _means(selected_fine, (p,))
    coarse = _means(selected_coarse, (p,))
    B, energy = transfer_stack(model)

    def distance(d):
        value = float(np.sum(np.abs(B @ d) ** 2))
        if not isfinite(value):
            raise FloatingPointError("nonfinite transfer disagreement")
        return value

    spread = max(distance(starts[a] - starts[b])
                 for a in range(3) for b in range(a))
    grid = distance(fine - coarse)
    if energy == 0:
        if np.any(B != 0):
            raise FloatingPointError("zero energy with nonzero transfer")
        return Disagreement(0.0, 0.0, 0.0, 0.0)
    start_component, grid_component = spread / energy, grid / energy
    score = start_component + grid_component
    if not isfinite(score):
        raise FloatingPointError("nonfinite normalized score")
    return Disagreement(score, start_component, grid_component, energy)


def select_groups(groups: list[GroupPriority], fraction: float = 0.25) -> tuple[str, ...]:
    """Exact floor call budget; fallback first, then score, then stable ID.

    No sampler call, random generator, file access, or input mutation.
    An empty batch returns no selections.
    """
    if isinstance(fraction, bool) or not isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite in [0, 1]")
    if not all(isinstance(g, GroupPriority) for g in groups):
        raise TypeError("GroupPriority entries required")
    ids = [g.group_id for g in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate stable group ID")
    budget = floor(fraction * len(groups))
    ranked = sorted(groups, key=lambda g: (not g.mf_fallback, -g.score, g.group_id))
    return tuple(g.group_id for g in ranked[:budget])
