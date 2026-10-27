"""Packet-local control tests for R2C01_DIRA_SHAPE_v08.

These tests do not run the diagnostic. They exercise the moment floor, the
label set, the nonfinite-ratio guard and the config/environment receipts.

Run with the project-local interpreter, without bytecode writes:

    $env:PYTHONDONTWRITEBYTECODE="1"; .venv\\Scripts\\python.exe -B -m unittest discover -s docs\\claude_review\\R2C01_DIRA_SHAPE_v08\\run -p "test_shape_controls.py"
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shape_core import draw_phases, moment_check, shape_spec  # noqa: E402
from shape_ordering import collapse_summary, ordering_summary  # noqa: E402
from shape_rows import mse_row  # noqa: E402
from shape_stats import mse_ratio_values  # noqa: E402


def _stats() -> dict:
    return {
        "primary_family": ["U", "G", "B"],
        "primary_metric": "per_topology_mse_ratio_B2_over_B1",
        "alpha": 0.05,
        "primary_adjustment": "BONFERRONI_3",
        "bootstrap_resamples": 20,
        "effect_size_threshold": 1.5,
        "collapse_cells": ["U", "G", "B"],
        "collapse_threshold": 1.1,
        "minimum_complete_topologies_for_an_interval": 2,
        "primary_pairwise": [["B", "U"], ["U", "G"], ["B", "G"]],
        "primary_predicted_signs": ["POSITIVE", "POSITIVE", "POSITIVE"],
    }


def _case() -> dict:
    return {"id": "N4", "topologies": 2, "cells": ["U", "G", "B"]}


def _records(mse_b1: float = 1.0) -> dict:
    records = {}
    for cid in ("U", "G", "B"):
        for topology in (0, 1):
            records[("N4", cid, topology)] = {
                "mse": {"B1": mse_b1, "B2": 1.1},
                "channel_energy": 1.0,
                "m1_emp": 0.9,
                "m2_emp": 0.62,
                "m1_se": 0.0,
                "m2_se": 0.0,
                "m1_se_effective": 2e-13,
                "m2_se_effective": 2e-13,
                "moment_z1": 0.0,
                "moment_z2": 0.0,
                "moment_atol": 1e-12,
                "moment_pass": True,
                "moment_note": None,
                "max_rel_equal": 0.0,
                "se": {},
            }
    return records


class MomentFloorTest(unittest.TestCase):
    def test_binary_moment_check_passes_at_all_targets(self) -> None:
        for target in (0.99, 0.90, 0.70):
            spec = shape_spec("binary", target)
            # 64 frames x 512 links = 32,768 phases, the real driver's field.
            uniform = np.resize(np.array([0.1, 0.9]), (64, 512))
            phases = draw_phases("binary", spec, uniform)
            result = moment_check(phases, spec, sigma=5.0, atol=1e-12)
            self.assertTrue(result["moment_pass"], (target, result))
            self.assertLess(abs(result["moment_z1"]), 5.0)
            self.assertLess(abs(result["moment_z2"]), 5.0)
            self.assertGreaterEqual(result["m1_se_effective"], 2e-13 - 1e-30)

    def test_binary_moment_check_catches_wrong_spec(self) -> None:
        spec = shape_spec("binary", 0.90)
        uniform = np.resize(np.array([0.1, 0.9]), (64, 512))
        phases = draw_phases("binary", shape_spec("binary", 0.99), uniform)
        result = moment_check(phases, spec, sigma=5.0, atol=1e-12)
        self.assertFalse(result["moment_pass"])
        self.assertGreater(abs(result["moment_z1"]), 5.0)


class LabelTest(unittest.TestCase):
    def test_fail_integrity_labels_are_not_interpreted(self) -> None:
        case = _case()
        stats = _stats()
        ordering = ordering_summary(
            case,
            {"id": "RZF_L1"},
            {},
            [],
            stats,
            "FAIL_INTEGRITY",
            np.random.default_rng(1),
        )
        collapse = collapse_summary(
            case,
            {"id": "RZF_L1"},
            {},
            [],
            stats,
            "FAIL_INTEGRITY",
            np.random.default_rng(2),
        )
        self.assertEqual(ordering["label"], "NOT_INTERPRETED_FAIL_INTEGRITY")
        self.assertEqual(collapse["label"], "NOT_INTERPRETED_FAIL_INTEGRITY")

    def test_nonfinite_ratio_is_integrity_not_a_negative_result(self) -> None:
        case = _case()
        stats = _stats()
        records = _records(mse_b1=0.0)
        ratios = mse_ratio_values(case, "U", records, [0, 1])
        self.assertTrue(np.all(np.isnan(ratios)))
        result = ordering_summary(
            case,
            {"id": "RZF_L1"},
            records,
            [0, 1],
            stats,
            "COMPLETE",
            np.random.default_rng(3),
        )
        self.assertEqual(result["label"], "NOT_INTERPRETED_FAIL_INTEGRITY")
        self.assertEqual(result["note"], "NONFINITE_MSE_RATIO")

    def test_control_row_without_topologies_is_not_evaluated(self) -> None:
        row = mse_row(
            _case(),
            {
                "role": "STRUCTURAL_EQUALITY_CONTROL",
                "kappa": 20.0,
                "shape": "zero_c",
                "m1": 0.9,
            },
            "ZERO",
            {"id": "RZF_L1", "role": "PRIMARY"},
            shape_spec("zero_c", 0.90),
            {},
            [],
            "COMPLETE",
            _stats(),
            np.random.default_rng(4),
        )
        self.assertEqual(row["se_label"], "CONTROL_NOT_EVALUATED")

    def test_control_row_fail_integrity_has_status_precedence(self) -> None:
        for status, expected in (
            ("FAIL_INTEGRITY", "NOT_INTERPRETED_FAIL_INTEGRITY"),
            ("INCOMPLETE_RESOURCE", "INCOMPLETE_RESOURCE"),
        ):
            row = mse_row(
                _case(),
                {
                    "role": "STRUCTURAL_EQUALITY_CONTROL",
                    "kappa": 20.0,
                    "shape": "zero_c",
                    "m1": 0.9,
                },
                "ZERO",
                {"id": "RZF_L1", "role": "PRIMARY"},
                shape_spec("zero_c", 0.90),
                {},
                [],
                status,
                _stats(),
                np.random.default_rng(6),
            )
            self.assertEqual(row["se_label"], expected)

    def test_fail_integrity_row_label(self) -> None:
        row = mse_row(
            _case(),
            {"role": "SHAPE", "kappa": 20.0, "shape": "uniform", "m1": 0.9},
            "U",
            {"id": "RZF_L1", "role": "PRIMARY"},
            shape_spec("uniform", 0.90),
            {},
            [],
            "FAIL_INTEGRITY",
            _stats(),
            np.random.default_rng(5),
        )
        self.assertEqual(row["se_label"], "NOT_INTERPRETED_FAIL_INTEGRITY")


class ConfigAndEnvironmentTest(unittest.TestCase):
    def test_config_declares_descriptive_secondary_role(self) -> None:
        config = json.loads((HERE / "E10_shape_config.json").read_text("utf-8"))
        self.assertEqual(config["identity"], "R2C01_DIRA_SHAPE_v08")
        self.assertEqual(
            config["statistics"]["secondary_descriptive"]["role"],
            "DESCRIPTIVE_ONLY_NOT_A_FORMAL_ORDERING_TEST",
        )
        self.assertEqual(config["controls"]["empirical_moment_atol"], 1e-12)

    def test_environment_lock_is_v08(self) -> None:
        lock = json.loads((HERE / "ENVIRONMENT_LOCK.json").read_text("utf-8"))
        self.assertEqual(lock["identity"], "R2C01_DIRA_SHAPE_v08")
        self.assertTrue(lock["prefix_is_project_local"])
        self.assertTrue(lock["smoke_test"]["pass"])
        self.assertIn("run/temp", lock["temp_policy"])


if __name__ == "__main__":
    unittest.main()
