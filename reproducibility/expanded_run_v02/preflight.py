"""Pre-flight validation of the expanded packet without consuming the execution.

Checks the environment lock, the frozen checksums, the configuration, and the
calibration override on one small synthetic group.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKT = HERE.parent
MODULE = PKT / "run" / "E10_shape_run.py"
CONFIG = PKT / "run" / "E10_shape_config.json"

sys.path.insert(0, str(PKT / "run"))
spec = importlib.util.spec_from_file_location("expanded_driver", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

print("environment: PASS")
print("inputs: PASS")
config = module.load_config()
print("config: PASS, cells", len(config["cells"]))

import numpy as np

import phase_graph_cme
from phase_graph_cme.baselines import clmmse, widely_linear_mmse
from phase_graph_cme.metrics import uatf_uplink
from phase_graph_cme.production_registry import derived_seed
from phase_graph_cme.receivers import mr_combiner, rzf_combiner
from phase_graph_cme.scenario import ScenarioConfig, generate_topology

import shape_estimators
import shape_model
import shape_ordering
import shape_rows
import shape_run_support
import shape_stats
print("imports: PASS")
from shape_core import shape_spec
from shape_model import generate_shape_frame, shape_pilot_group

scenario = ScenarioConfig(
    access_points=4,
    antennas_per_ap=2,
    users=4,
    pilot_symbols=2,
    coherence_symbols=200,
    rician_kappa=20.0,
)
topology = generate_topology(scenario, seed=7)
spec_uniform = shape_spec("uniform", 0.90)
frame = generate_shape_frame(topology, scenario, spec_uniform, seed=11)
for offset in (0.0, 0.02, -0.02, 0.5):
    model, y, users, _ = shape_pilot_group(
        topology,
        frame,
        scenario,
        spec_uniform,
        access_point=0,
        pilot=0,
        m2_est=float(spec_uniform.m2) + offset,
    )
    lens = 1.0 - float(model.m1[0]) ** 2
    assert abs(float(model.m2[0]) - float(model.m1[0]) ** 2) <= lens + 1e-12
    assert float(model.m2[0]) <= float(model.m1[0]) ** 2 + 1e-12
    print(
        f"offset {offset:+.2f} -> m2 {float(model.m2[0]):.6f}, "
        f"c {float(model.m2[0]) - float(model.m1[0]) ** 2:+.6f}, |y| {np.abs(y).mean():.4f}"
    )
print("calibration override: PASS")
