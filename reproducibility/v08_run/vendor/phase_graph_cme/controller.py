"""Diagnostic/certificate-aware inference with an immutable CLMMSE fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .baselines import clmmse
from .inference import ConditionalSamplingError, MomentEstimate, heat_bath_gibbs, tensor_quadrature
from .model import PilotGroupModel, build_phase_graph
from .moments import channel_moments

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ControllerConfig:
    quadrature_order: int = 24
    quadrature_check_order: int = 32
    quadrature_maximum_states: int = 100_000
    quadrature_moment_tolerance: float = 1e-4
    quadrature_second_moment_tolerance: float = 1e-4
    gibbs_chains: int = 4
    gibbs_burn_sweeps: int = 96
    gibbs_retained_draws: int = 192
    gibbs_thinning_sweeps: int = 1
    maximum_conditional_proposals: int = 100_000
    conditional_batch_size: int = 256
    maximum_split_rhat: float = 1.05
    maximum_chain_mean_disagreement: float = 0.05
    accept_empirical_gibbs: bool = True
    retain_gibbs_samples: bool = False

    def __post_init__(self) -> None:
        positive_counts = (
            self.quadrature_order,
            self.quadrature_check_order,
            self.quadrature_maximum_states,
            self.gibbs_chains,
            self.gibbs_retained_draws,
            self.gibbs_thinning_sweeps,
            self.maximum_conditional_proposals,
            self.conditional_batch_size,
        )
        if any(
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) < 1
            for value in positive_counts
        ):
            raise ValueError("orders, caps, chains, draws, and thinning must be positive integers")
        if (
            not isinstance(self.gibbs_burn_sweeps, Integral)
            or isinstance(self.gibbs_burn_sweeps, (bool, np.bool_))
            or int(self.gibbs_burn_sweeps) < 0
        ):
            raise ValueError("gibbs_burn_sweeps must be a nonnegative integer")
        if not isinstance(self.accept_empirical_gibbs, (bool, np.bool_)):
            raise ValueError("accept_empirical_gibbs must be Boolean")
        if not isinstance(self.retain_gibbs_samples, (bool, np.bool_)):
            raise ValueError("retain_gibbs_samples must be Boolean")
        if self.quadrature_order < 1 or self.quadrature_check_order <= self.quadrature_order:
            raise ValueError("quadrature orders must be positive and check_order > order")
        if self.quadrature_maximum_states < 1:
            raise ValueError("quadrature state cap must be positive")
        if (
            not math.isfinite(self.quadrature_moment_tolerance)
            or not math.isfinite(self.quadrature_second_moment_tolerance)
            or self.quadrature_moment_tolerance < 0.0
            or self.quadrature_second_moment_tolerance < 0.0
        ):
            raise ValueError("quadrature tolerances must be finite and nonnegative")
        if self.gibbs_chains < 2 or self.gibbs_retained_draws < 2:
            raise ValueError("Gibbs requires at least two chains and two retained draws")
        if self.gibbs_burn_sweeps < 0 or self.gibbs_thinning_sweeps < 1:
            raise ValueError("invalid Gibbs sweep budget")
        if self.maximum_conditional_proposals < 1:
            raise ValueError("conditional proposal cap must be positive")
        if (
            not math.isfinite(self.maximum_split_rhat)
            or not math.isfinite(self.maximum_chain_mean_disagreement)
            or self.maximum_split_rhat <= 0.0
            or self.maximum_chain_mean_disagreement < 0.0
        ):
            raise ValueError("diagnostic thresholds must be nonnegative, with positive R-hat")


@dataclass(frozen=True)
class ControllerResult:
    estimate: ComplexArray
    mode: str
    status: str
    phase_moments: MomentEstimate | None
    attempted_phase_moments: MomentEstimate | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupControllerResult:
    """One phase inference reused for every target in a pilot group."""

    estimates: ComplexArray
    mode: str
    status: str
    phase_moments: MomentEstimate | None
    attempted_phase_moments: MomentEstimate | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class InferenceController:
    """Select a calibrated numerical mode or return the affine baseline.

    R-hat and chain disagreement are empirical checks, not proofs.  The mode is
    explicitly labeled so downstream experiments can report fallback frequency.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()

    def _state_count(self, widths: np.ndarray, order: int) -> int:
        """Count tensor states exactly with overflow-safe Python integers."""

        count = 1
        for width in widths:
            if float(width) != 0.0:
                count *= int(order)
        return count

    def _phase_inference(
        self, model: PilotGroupModel, observation: object, *, seed: int
    ) -> tuple[
        str,
        str,
        MomentEstimate | None,
        MomentEstimate | None,
        dict[str, Any],
    ]:
        """Run the immutable phase-inference decision tree once per group."""

        graph = build_phase_graph(model, observation)
        cfg = self.config
        diagnostics: dict[str, Any] = {
            "quadrature_coarse_states": self._state_count(
                graph.delta, cfg.quadrature_order
            ),
            "quadrature_fine_states": self._state_count(
                graph.delta, cfg.quadrature_check_order
            ),
            "retain_gibbs_samples": bool(cfg.retain_gibbs_samples),
        }
        if (
            max(
                diagnostics["quadrature_coarse_states"],
                diagnostics["quadrature_fine_states"],
            )
            <= cfg.quadrature_maximum_states
        ):
            try:
                coarse = tensor_quadrature(
                    graph, cfg.quadrature_order, cfg.quadrature_maximum_states
                )
                fine = tensor_quadrature(
                    graph, cfg.quadrature_check_order, cfg.quadrature_maximum_states
                )
                discrepancy = float(
                    np.max(np.abs(coarse.z - fine.z), initial=0.0)
                )
                second_discrepancy = float(
                    np.max(np.abs(coarse.Z - fine.Z), initial=0.0)
                )
                diagnostics.update(
                    {
                        "order_comparison_max_abs_first_moment": discrepancy,
                        "order_comparison_max_abs_second_moment": second_discrepancy,
                        "coarse_order": cfg.quadrature_order,
                        "fine_order": cfg.quadrature_check_order,
                    }
                )
                if (
                    discrepancy <= cfg.quadrature_moment_tolerance
                    and second_discrepancy
                    <= cfg.quadrature_second_moment_tolerance
                ):
                    return (
                        "CALIBRATED_TENSOR_QUADRATURE",
                        "NUMERICALLY_CALIBRATED_FIRST_AND_SECOND_MOMENTS_NOT_ANALYTICALLY_CERTIFIED",
                        fine,
                        None,
                        diagnostics,
                    )
                diagnostics["quadrature_rejection_reason"] = (
                    "ORDER_DISAGREEMENT_EXCEEDS_FROZEN_TOLERANCE"
                )
            except (FloatingPointError, ValueError, MemoryError) as error:
                diagnostics["quadrature_rejection_reason"] = "QUADRATURE_ERROR"
                diagnostics["quadrature_error"] = str(error)
        else:
            diagnostics["quadrature_rejection_reason"] = "STATE_CAP"

        try:
            gibbs = heat_bath_gibbs(
                graph,
                seed=int(seed),
                chains=cfg.gibbs_chains,
                burn_sweeps=cfg.gibbs_burn_sweeps,
                retained_draws=cfg.gibbs_retained_draws,
                thinning_sweeps=cfg.gibbs_thinning_sweeps,
                maximum_conditional_proposals=cfg.maximum_conditional_proposals,
                conditional_batch_size=cfg.conditional_batch_size,
                return_samples=cfg.retain_gibbs_samples,
            )
        except (ConditionalSamplingError, FloatingPointError, ValueError) as error:
            diagnostics.update(
                {"fallback_reason": "GIBBS_ERROR", "gibbs_error": str(error)}
            )
            audit = getattr(error, "audit", None)
            if audit is not None:
                diagnostics["gibbs_failure_audit"] = dict(audit)
                diagnostics["gibbs_proposal_count"] = int(
                    audit.get("logical_proposal_count", 0)
                )
                diagnostics["gibbs_generated_candidate_value_count"] = int(
                    audit.get("generated_candidate_value_count", 0)
                )
            return (
                "CLMMSE_FALLBACK",
                "BASELINE_FALLBACK",
                None,
                None,
                diagnostics,
            )

        diagnostic = gibbs.diagnostics
        diagnostics.update(
            {
                "gibbs_sample_count": gibbs.sample_count,
                "gibbs_proposal_count": gibbs.proposal_count,
                "gibbs_generated_candidate_value_count": gibbs.metadata.get(
                    "generated_candidate_value_count", gibbs.proposal_count
                ),
                "gibbs_per_chain_logical_proposal_count": gibbs.metadata.get(
                    "per_chain_logical_proposal_count", []
                ),
                "gibbs_per_chain_generated_candidate_value_count": gibbs.metadata.get(
                    "per_chain_generated_candidate_value_count", []
                ),
                "dobrushin_eta": gibbs.metadata["dobrushin_eta"],
                "tv_0p01_sufficient_burn_sweeps": gibbs.metadata[
                    "tv_0p01_sufficient_burn_sweeps"
                ],
            }
        )
        if diagnostic is None or not diagnostic.all_finite:
            diagnostics["fallback_reason"] = "NONFINITE_DIAGNOSTIC"
            return (
                "CLMMSE_FALLBACK",
                "BASELINE_FALLBACK",
                None,
                gibbs,
                diagnostics,
            )
        diagnostics.update(
            {
                "maximum_split_rhat": diagnostic.maximum_split_rhat,
                "maximum_chain_mean_disagreement": (
                    diagnostic.maximum_chain_mean_disagreement
                ),
            }
        )
        accepted = (
            cfg.accept_empirical_gibbs
            and diagnostic.maximum_split_rhat <= cfg.maximum_split_rhat
            and diagnostic.maximum_chain_mean_disagreement
            <= cfg.maximum_chain_mean_disagreement
        )
        if not accepted:
            diagnostics["fallback_reason"] = "CHAIN_DIAGNOSTIC_THRESHOLD"
            return (
                "CLMMSE_FALLBACK",
                "BASELINE_FALLBACK",
                None,
                gibbs,
                diagnostics,
            )
        return (
            "DIAGNOSTIC_GIBBS",
            "EMPIRICALLY_ACCEPTED_NOT_CERTIFIED",
            gibbs,
            gibbs,
            diagnostics,
        )

    def _fallback(
        self,
        model: PilotGroupModel,
        observation: object,
        target: int,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> ControllerResult:
        diagnostics: dict[str, Any] = {"fallback_reason": reason}
        if extra:
            diagnostics.update(extra)
        return ControllerResult(
            estimate=clmmse(model, observation, target),
            mode="CLMMSE_FALLBACK",
            status="BASELINE_FALLBACK",
            phase_moments=None,
            attempted_phase_moments=None,
            diagnostics=diagnostics,
        )

    def estimate(
        self, model: PilotGroupModel, observation: object, target: int, *, seed: int
    ) -> ControllerResult:
        if not 0 <= int(target) < model.group_size:
            raise IndexError("target index is outside the pilot group")
        mode, status, moments, attempted, diagnostics = self._phase_inference(
            model, observation, seed=seed
        )
        if moments is None:
            return ControllerResult(
                estimate=clmmse(model, observation, target),
                mode=mode,
                status=status,
                phase_moments=None,
                attempted_phase_moments=attempted,
                diagnostics=diagnostics,
            )
        channel = channel_moments(
            model, observation, target, moments.z, moments.Z
        )
        return ControllerResult(
            estimate=channel.mean,
            mode=mode,
            status=status,
            phase_moments=moments,
            attempted_phase_moments=attempted,
            diagnostics=diagnostics,
        )

    def estimate_all(
        self, model: PilotGroupModel, observation: object, *, seed: int
    ) -> GroupControllerResult:
        """Estimate every co-pilot channel after a single phase inference."""

        mode, status, moments, attempted, diagnostics = self._phase_inference(
            model, observation, seed=seed
        )
        if moments is None:
            estimates = np.column_stack(
                [
                    clmmse(model, observation, target)
                    for target in range(model.group_size)
                ]
            )
        else:
            estimates = np.column_stack(
                [
                    channel_moments(
                        model, observation, target, moments.z, moments.Z
                    ).mean
                    for target in range(model.group_size)
                ]
            )
        return GroupControllerResult(
            estimates=estimates.astype(np.complex128),
            mode=mode,
            status=status,
            phase_moments=moments,
            attempted_phase_moments=attempted,
            diagnostics=diagnostics,
        )
