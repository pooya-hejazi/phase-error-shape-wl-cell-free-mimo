"""Phase-graph Bayesian channel estimation for residual-LoS-phase CF MIMO.

The package distinguishes exact probabilistic identities from numerical
approximations.  Public APIs return diagnostics/status objects rather than
silently treating an unconverged inference result as exact.
"""

from .model import PhaseGraph, PilotGroupModel, build_phase_graph
from .moments import ChannelMoments, channel_moments
from .controller import (
    ControllerConfig,
    ControllerResult,
    GroupControllerResult,
    InferenceController,
)

__all__ = [
    "ChannelMoments",
    "ControllerConfig",
    "ControllerResult",
    "GroupControllerResult",
    "InferenceController",
    "PhaseGraph",
    "PilotGroupModel",
    "build_phase_graph",
    "channel_moments",
]

__version__ = "0.1.0"
