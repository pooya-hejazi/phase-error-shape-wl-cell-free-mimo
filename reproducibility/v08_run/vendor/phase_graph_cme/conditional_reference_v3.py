"""Conditional moment estimators; no sampler changes or PSD projection."""

import numpy as np
from phase_graph_cme.redesign_v2 import rao_blackwellized_phase_functions


def coordinate_conditional_moments(graph, samples, order=192):
    u = np.asarray(samples, dtype=np.complex128)
    if u.ndim != 2 or u.shape[1] != graph.group_size:
        raise ValueError("expected samples x phase coordinates")
    m = rao_blackwellized_phase_functions(graph, u, quadrature_order=order)
    z = np.mean(m, axis=0)
    # E[m_i(u_-i) conj(u_j)] = E[u_i conj(u_j)] for i != j.
    directed = m.T @ u.conj() / len(u)
    Z = (directed + directed.conj().T) / 2.0
    np.fill_diagonal(Z, 1.0)
    # Coordinatewise conditioning need not yield a PSD finite-sample matrix.
    # Report this, never project silently or count it as a valid covariance.
    minimum_eigenvalue = float(np.linalg.eigvalsh(Z - np.outer(z, z.conj())).min())
    return (
        z,
        Z,
        {
            "minimum_phase_covariance_eigenvalue": minimum_eigenvalue,
            "projection_applied": False,
            "quadrature_order": order,
        },
    )


def integrate_reference_panel(graph, sample_stack):
    """Return replicate moments and explicit quadrature/PSD eligibility flags."""
    stack = np.asarray(sample_stack)
    if (
        stack.ndim != 3
        or stack.shape[-1] != graph.group_size
        or not np.isfinite(stack).all()
    ):
        raise ValueError("finite replicate x particle x phase samples required")
    if stack.shape[0] < 3 or stack.shape[1] < 1:
        raise ValueError("at least three nonempty independent SMC replicates required")
    moments = []
    order_error = 0.0
    minimum_eigenvalue = float("inf")
    for samples in stack:
        first, second, meta = coordinate_conditional_moments(graph, samples, 192)
        low_first, low_second, _ = coordinate_conditional_moments(graph, samples, 128)
        order_error = max(
            order_error,
            float(np.max(abs(first - low_first))),
            float(np.max(abs(second - low_second))),
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue, meta["minimum_phase_covariance_eigenvalue"]
        )
        moments.append((first, second))
    first = np.mean([item[0] for item in moments], axis=0)
    second = np.mean([item[1] for item in moments], axis=0)
    minimum_eigenvalue = min(
        minimum_eigenvalue,
        float(np.linalg.eigvalsh(second - np.outer(first, first.conj())).min()),
    )
    checks = {
        "conditional_quadrature_agreement": bool(order_error <= 1e-8),
        "conditional_phase_covariance_psd": bool(minimum_eigenvalue >= -1e-12),
    }
    return moments, first, second, checks
