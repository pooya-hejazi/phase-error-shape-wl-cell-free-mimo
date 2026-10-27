"""Minimal vendored subset of the simulation package.

Only ``model`` and ``scenario`` are needed by the frozen system-run scripts; the
rest of the internal package is not part of the public release.
"""

from .model import PhaseGraph, PilotGroupModel, build_phase_graph
from .scenario import ScenarioConfig, Topology, generate_topology

__all__ = [
    "PhaseGraph",
    "PilotGroupModel",
    "ScenarioConfig",
    "Topology",
    "build_phase_graph",
    "generate_topology",
]
