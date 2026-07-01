# -*- coding: utf-8 -*-
"""
Paper-like random periodic RVE generator for continuous fiber reinforced composites.

This script follows the workflow in Chen, Fu and Li (2026):

STEP 1. Establish a packing-optimization model and obtain initial fiber coordinates.
        The paper solves the non-objective constrained model with DCCP:
            (xi-xj)^2 + (yi-yj)^2 >= (ri+rj+dis[i,j])^2
            dis[i,j] = random.uniform(lmin, lmax)
        In this standalone script, two STEP-1 solvers are available:
        (1) periodic penalty/relaxation, which is fast and suitable for large PBC RVEs;
        (2) non-periodic DCCP packing, which is closer to the paper formulation but is
            used without expensive 3x3 periodic-image constraints.
        A hybrid mode can use DCCP first and then apply a short periodic relaxation
        polish so that the final geometry is compatible with PBC.

STEP 2. Re-orientation method.
        Fibers are repeatedly tested for relocation into resin-rich positions.
        A new position is accepted only if it satisfies all inter-fiber distance
        constraints. This reduces fiber clustering and improves randomness.

STEP 3. Periodic process for boundary fibers.
        Boundary-intersecting fibers are copied to opposite sides/corners so
        the exported geometry is periodic.

Outputs:
    1. <prefix>_main.csv
    2. <prefix>_for_abaqus.csv
    3. <prefix>.xlsx
    4. <prefix>_preview.png
    5. <prefix>_randomness_evaluation.xlsx
    6. four statistical figures similar to the paper

Author: Chao Yang
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


@dataclass
class RVEParams:
    # Geometry
    Lx_um: float = 100.0
    Ly_um: float = 100.0
    target_vf: float = 0.70

    # Fiber diameter distribution
    # constant: all fibers have diameter_mean_um
    # normal  : normal distribution, clipped by +/- 3 std
    # uniform : uniform distribution between diameter_min_um and diameter_max_um
    diameter_mode: str = "constant"
    diameter_mean_um: float = 7.0
    diameter_std_um: float = 0.317
    diameter_min_um: float = 5.0
    diameter_max_um: float = 8.0

    # Paper-like random inter-fiber distance constraint:
    # dis[i,j] = random.uniform(lmin_um, lmax_um)
    # Paper examples:
    #   Vf = 60%, 65%: lmin = 0.175 um, lmax = 0.875 um
    #   Vf = 70%     : lmin = 0.175 um, lmax = 0.350 um
    #   Vf = 80%     : lmin = 0.100 um, lmax = 0.150 um
    lmin_um: float = 0.175
    lmax_um: float = 0.350

    # Candidate search
    max_restarts: int = 200
    valid_candidates_to_test: int = 5
    selection_mode: str = "best_valid_randomness"  # first_valid or best_valid_randomness

    # STEP 1 solver method
    #   "relaxation"  : fast periodic minimum-image relaxation. Best for large PBC RVEs.
    #   "dccp"        : non-periodic DCCP packing, followed by optional periodic polish.
    #   "hybrid_dccp" : try DCCP; if DCCP fails, fall back to relaxation result.
    # Important: this DCCP mode does NOT impose strict 3x3 periodic-image constraints,
    # because that version is too expensive for large Vf=70% RVEs.
    step1_method: str = "relaxation"

    # DCCP settings for non-periodic packing
    # boundary_mode="cell"  : centers are constrained by 0<=x<=Lx, 0<=y<=Ly.
    #                         This allows boundary-intersecting fibers after periodic copy.
    # boundary_mode="inside": centers are constrained by r<=x<=Lx-r, r<=y<=Ly-r.
    #                         This is closer to paper Eq.(2), but less useful for PBC.
    dccp_boundary_mode: str = "cell"
    dccp_use_relaxation_warm_start: bool = True
    dccp_regularization: float = 1.0e-6
    dccp_solver: str = "SCS"
    dccp_ccp_times: int = 1
    dccp_max_iter: int = 60
    dccp_tau: float = 0.005
    dccp_mu: float = 1.2
    dccp_tau_max: float = 1.0e8
    dccp_solver_eps: float = 1.0e-4
    dccp_solver_max_iters: int = 30000
    dccp_verbose: bool = False

    # After DCCP, apply a short periodic minimum-image polish.
    # This is the key practical step: DCCP gives a good global packing tendency,
    # while the polish fixes cross-boundary spacing for PBC without using 3x3 DCCP.
    dccp_post_periodic_polish: bool = True
    dccp_post_polish_iter: int = 2500
    dccp_post_polish_move_limit_um: float = 0.05

    # STEP 1: packing optimization / feasibility relaxation
    # These parameters replace the DCCP solver in a standalone SciPy-free manner.
    opt_alpha_start: float = 0.25
    opt_alpha_end: float = 1.0
    opt_alpha_steps: int = 40
    opt_iter_per_alpha: int = 800
    opt_final_iter: int = 6000
    opt_step_scale: float = 0.55
    opt_move_limit_um: float = 0.25
    opt_final_move_limit_um: float = 0.08
    opt_random_shake_um: float = 0.06

    # STEP 2: paper-like re-orientation
    use_reorientation: bool = True
    reorient_passes: int = 3
    reorient_trials_per_fiber: int = 80
    reorient_accept_if_improve_um: float = 0.02
    reorient_random_accept_prob: float = 0.02
    # Candidates are sampled partly globally and partly around resin-rich positions.
    reorient_local_radius_factor: float = 4.0

    # Optional Monte Carlo legal shuffle after reorientation.
    # This is not the main paper step, but is useful for sampling feasible space.
    use_mc_shuffle: bool = False
    mc_steps_per_fiber: int = 300
    mc_move_amp_um: float = 0.05

    # Distance checking
    use_periodic_distance: bool = True
    tol_um: float = 1.0e-7

    # Randomness evaluation settings
    pair_h_min_over_r: float = 2.0
    pair_h_max_over_r: float = 15.0
    pair_dh_over_r: float = 0.35
    pair_smooth_window: int = 1  # 1 = no smoothing; set 5 for a smoother visual curve
    contact_low_over_r: float = 2.0
    # This should usually be consistent with the first pair-distribution bin.
    # If pair_h_min_over_r=2.0 and pair_dh_over_r=0.35, the first bin is 2.00-2.35.
    contact_high_over_r: float = 2.35

    # Reproducibility and output
    seed: Optional[int] = 2026
    output_prefix: str = "fiber_centers_paper_like"


# ============================================================
# Basic functions
# ============================================================

def estimate_fiber_number(p: RVEParams) -> int:
    area = p.Lx_um * p.Ly_um

    if p.diameter_mode == "constant":
        r = p.diameter_mean_um / 2.0
        a = math.pi * r * r
    elif p.diameter_mode == "normal":
        mu = p.diameter_mean_um
        sig = p.diameter_std_um
        a = math.pi * (mu * mu + sig * sig) / 4.0
    elif p.diameter_mode == "uniform":
        d1 = p.diameter_min_um
        d2 = p.diameter_max_um
        mean_d2 = (d1 * d1 + d1 * d2 + d2 * d2) / 3.0
        a = math.pi * mean_d2 / 4.0
    else:
        raise ValueError("diameter_mode must be constant, normal, or uniform.")

    return int(round(p.target_vf * area / a))


def generate_radii(p: RVEParams, n: int, rng: np.random.Generator) -> np.ndarray:
    if p.diameter_mode == "constant":
        d = np.ones(n) * p.diameter_mean_um
    elif p.diameter_mode == "normal":
        d = rng.normal(p.diameter_mean_um, p.diameter_std_um, n)
        d = np.clip(
            d,
            p.diameter_mean_um - 3.0 * p.diameter_std_um,
            p.diameter_mean_um + 3.0 * p.diameter_std_um,
        )
    elif p.diameter_mode == "uniform":
        d = rng.uniform(p.diameter_min_um, p.diameter_max_um, n)
    else:
        raise ValueError("diameter_mode must be constant, normal, or uniform.")

    return d / 2.0


def make_gap_matrix(p: RVEParams, n: int, rng: np.random.Generator) -> np.ndarray:
    gap = rng.uniform(p.lmin_um, p.lmax_um, (n, n))
    gap = np.triu(gap, 1)
    gap = gap + gap.T
    np.fill_diagonal(gap, 0.0)
    return gap


def actual_vf(radii: np.ndarray, p: RVEParams) -> float:
    return float(np.sum(math.pi * radii ** 2) / (p.Lx_um * p.Ly_um))


def min_image_components(dx: np.ndarray, dy: np.ndarray, p: RVEParams) -> Tuple[np.ndarray, np.ndarray]:
    if p.use_periodic_distance:
        dx = dx - p.Lx_um * np.round(dx / p.Lx_um)
        dy = dy - p.Ly_um * np.round(dy / p.Ly_um)
    return dx, dy


def min_image(dx: float, dy: float, p: RVEParams) -> Tuple[float, float]:
    if p.use_periodic_distance:
        dx -= p.Lx_um * round(dx / p.Lx_um)
        dy -= p.Ly_um * round(dy / p.Ly_um)
    return dx, dy


def distance(c1: np.ndarray, c2: np.ndarray, p: RVEParams) -> float:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dx, dy = min_image(dx, dy, p)
    return math.sqrt(dx * dx + dy * dy)


def random_initial_centers(p: RVEParams, n: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.zeros((n, 2), dtype=float)
    centers[:, 0] = rng.uniform(0.0, p.Lx_um, n)
    centers[:, 1] = rng.uniform(0.0, p.Ly_um, n)
    return centers


def pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


# ============================================================
# Metrics and validation
# ============================================================

def compute_metrics(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray, p: RVEParams) -> Dict[str, float]:
    n = len(radii)
    ii, jj = pair_indices(n)

    dx = centers[ii, 0] - centers[jj, 0]
    dy = centers[ii, 1] - centers[jj, 1]
    dx, dy = min_image_components(dx, dy, p)
    d = np.sqrt(dx * dx + dy * dy)

    required_surface = radii[ii] + radii[jj]
    required_constraint = required_surface + gap[ii, jj]

    surface_gap = d - required_surface
    margin = d - required_constraint
    violation = np.maximum(-margin, 0.0)
    overlap = np.maximum(-surface_gap, 0.0)

    return {
        "min_surface_gap": float(np.min(surface_gap)),
        "min_constraint_margin": float(np.min(margin)),
        "max_overlap": float(np.max(overlap)),
        "max_violation": float(np.max(violation)),
        "energy": float(np.sum(violation * violation)),
    }


def get_bad_pairs(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray, p: RVEParams):
    bad = []
    n = len(radii)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(centers[i], centers[j], p)
            required = radii[i] + radii[j] + gap[i, j]
            if d < required - p.tol_um:
                bad.append((i + 1, j + 1, d, required, required - d))
    return bad


# ============================================================
# STEP 1: paper-like packing optimization
# ============================================================

def packing_relax_one_stage(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    alpha: float,
    n_iter: int,
    shake_amp: float,
    move_limit_um: float,
) -> np.ndarray:
    """
    Penalty/relaxation solution of the paper's non-objective constraint model.

    The paper uses DCCP to solve:
        d_ij >= r_i + r_j + dis[i,j]
    Here we impose the same inequality by iteratively separating only the pairs
    that violate it. alpha gradually increases from opt_alpha_start to 1.0.
    """
    centers = centers.copy()
    n = len(radii)
    ii, jj = pair_indices(n)

    for _ in range(int(n_iter)):
        dx = centers[ii, 0] - centers[jj, 0]
        dy = centers[ii, 1] - centers[jj, 1]
        dx, dy = min_image_components(dx, dy, p)
        d = np.sqrt(dx * dx + dy * dy)

        required = alpha * (radii[ii] + radii[jj] + gap[ii, jj])
        violation = required - d
        mask = violation > 0.0

        if not np.any(mask):
            if shake_amp <= 0.0:
                break
            # Continue a little randomness only during early alpha stages.
            centers[:, 0] = (centers[:, 0] + rng.normal(0.0, shake_amp, n)) % p.Lx_um
            centers[:, 1] = (centers[:, 1] + rng.normal(0.0, shake_amp, n)) % p.Ly_um
            continue

        mi = ii[mask]
        mj = jj[mask]
        mdx = dx[mask]
        mdy = dy[mask]
        md = d[mask]
        mv = violation[mask]

        # Direction for pairs with exactly zero distance.
        zero = md < 1.0e-12
        if np.any(zero):
            theta = rng.uniform(0.0, 2.0 * math.pi, np.sum(zero))
            mdx[zero] = np.cos(theta)
            mdy[zero] = np.sin(theta)
            md[zero] = 1.0

        nx = mdx / md
        ny = mdy / md

        # Move each pair half of the violating distance in opposite directions.
        f = 0.5 * mv * p.opt_step_scale
        fx = nx * f
        fy = ny * f

        force = np.zeros_like(centers)
        np.add.at(force[:, 0], mi, fx)
        np.add.at(force[:, 1], mi, fy)
        np.add.at(force[:, 0], mj, -fx)
        np.add.at(force[:, 1], mj, -fy)

        norm = np.sqrt(force[:, 0] ** 2 + force[:, 1] ** 2)
        too_large = norm > move_limit_um
        if np.any(too_large):
            force[too_large, :] *= (move_limit_um / norm[too_large])[:, None]

        centers += force

        if shake_amp > 0.0:
            centers[:, 0] += rng.normal(0.0, shake_amp, n)
            centers[:, 1] += rng.normal(0.0, shake_amp, n)

        centers[:, 0] = centers[:, 0] % p.Lx_um
        centers[:, 1] = centers[:, 1] % p.Ly_um

    return centers


def paper_step1_initial_optimization(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    print("STEP 1: packing-optimization feasibility relaxation starts...")
    alphas = np.linspace(p.opt_alpha_start, p.opt_alpha_end, p.opt_alpha_steps)

    for k, alpha in enumerate(alphas):
        frac = 1.0 - float(k) / float(max(len(alphas) - 1, 1))
        shake = p.opt_random_shake_um * frac
        centers = packing_relax_one_stage(
            centers, radii, gap, p, rng,
            alpha=float(alpha),
            n_iter=p.opt_iter_per_alpha,
            shake_amp=shake,
            move_limit_um=p.opt_move_limit_um,
        )

    # Final polishing at alpha=1.0 without random shaking.
    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=p.opt_final_iter,
        shake_amp=0.0,
        move_limit_um=p.opt_final_move_limit_um,
    )

    return centers



# ============================================================
# Optional STEP 1: non-periodic DCCP packing
# ============================================================

def get_cvxpy_solver(p: RVEParams, cp_module):
    """Map the user solver name to a CVXPY solver constant."""
    name = str(p.dccp_solver).upper()
    if name == "SCS":
        return cp_module.SCS
    if name == "CLARABEL" and hasattr(cp_module, "CLARABEL"):
        return cp_module.CLARABEL
    if name == "ECOS" and hasattr(cp_module, "ECOS"):
        return cp_module.ECOS
    return cp_module.SCS


def prepare_dccp_initial_value(centers: np.ndarray, radii: np.ndarray,
                               p: RVEParams) -> np.ndarray:
    """Prepare an initial value satisfying the selected DCCP boundary box."""
    c0 = centers.copy()
    c0[:, 0] = np.mod(c0[:, 0], p.Lx_um)
    c0[:, 1] = np.mod(c0[:, 1], p.Ly_um)

    if str(p.dccp_boundary_mode).lower() == "inside":
        eps = 1.0e-6
        c0[:, 0] = np.clip(c0[:, 0], radii + eps, p.Lx_um - radii - eps)
        c0[:, 1] = np.clip(c0[:, 1], radii + eps, p.Ly_um - radii - eps)

    return c0


def paper_step1_dccp_nonperiodic_optimization(
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    warm_start_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Non-periodic DCCP version of STEP 1.

    This solves the paper-like packing constraint without 3x3 periodic-image
    constraints:

        ||c_i - c_j||_2 >= r_i + r_j + dis[i,j]

    Why non-periodic DCCP here?
    - The strict periodic DCCP formulation would add 9 image constraints for
      every fiber pair and becomes too expensive for Vf=70%, L=175 um.
    - This function uses DCCP only to obtain a globally coordinated initial
      packing tendency.
    - A short periodic relaxation polish is applied afterwards when
      p.dccp_post_periodic_polish=True, so cross-boundary spacing is corrected
      for the final PBC-compatible RVE.
    """
    try:
        import cvxpy as cp
        import dccp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DCCP mode requires cvxpy and dccp. Install them with:\n"
            "    pip install cvxpy dccp\n"
            "or:\n"
            "    conda install -c conda-forge cvxpy\n"
            "    pip install dccp"
        ) from exc

    import dccp

    n = len(radii)
    print("STEP 1: non-periodic DCCP packing optimization starts...")
    print("  DCCP boundary mode       = %s" % str(p.dccp_boundary_mode))
    print("  number of fibers         = %d" % n)
    print("  pair constraints         = %d" % (n * (n - 1) // 2))
    print("  strict periodic DCCP     = False")

    C = cp.Variable((n, 2), name="fiber_centers")

    if warm_start_centers is None:
        centers0 = random_initial_centers(p, n, rng)
    else:
        centers0 = warm_start_centers.copy()
    centers0 = prepare_dccp_initial_value(centers0, radii, p)
    C.value = centers0

    constraints = []
    boundary_mode = str(p.dccp_boundary_mode).lower()
    if boundary_mode == "inside":
        constraints += [C[:, 0] >= radii, C[:, 0] <= p.Lx_um - radii]
        constraints += [C[:, 1] >= radii, C[:, 1] <= p.Ly_um - radii]
    elif boundary_mode == "cell":
        constraints += [C[:, 0] >= 0.0, C[:, 0] <= p.Lx_um]
        constraints += [C[:, 1] >= 0.0, C[:, 1] <= p.Ly_um]
    else:
        raise ValueError("dccp_boundary_mode must be 'cell' or 'inside'.")

    # Base-cell pairwise constraints only. No periodic image constraints here.
    for i in range(n):
        for j in range(i + 1, n):
            required = float(radii[i] + radii[j] + gap[i, j])
            constraints.append(
                cp.norm(cp.hstack([C[i, 0] - C[j, 0], C[i, 1] - C[j, 1]]), 2) >= required
            )

    objective = cp.Minimize(float(p.dccp_regularization) * cp.sum_squares(C - centers0))
    problem = cp.Problem(objective, constraints)

    if not dccp.is_dccp(problem):
        raise RuntimeError(
            "The constructed non-periodic packing problem is not recognized as DCCP. "
            "Try updating cvxpy/dccp, or use step1_method='relaxation'."
        )

    solver = get_cvxpy_solver(p, cp)
    solver_kwargs = {}
    if str(p.dccp_solver).upper() == "SCS":
        solver_kwargs.update({
            "max_iters": int(p.dccp_solver_max_iters),
            "eps": float(p.dccp_solver_eps),
        })

    try:
        result = problem.solve(
            method="dccp",
            solver=solver,
            ccp_times=int(p.dccp_ccp_times),
            max_iter=int(p.dccp_max_iter),
            tau=float(p.dccp_tau),
            mu=float(p.dccp_mu),
            tau_max=float(p.dccp_tau_max),
            verbose=bool(p.dccp_verbose),
            **solver_kwargs,
        )
    except TypeError:
        # Some cvxpy/dccp versions accept fewer DCCP keywords.
        result = problem.solve(
            method="dccp",
            solver=solver,
            max_iter=int(p.dccp_max_iter),
            tau=float(p.dccp_tau),
            mu=float(p.dccp_mu),
            verbose=bool(p.dccp_verbose),
            **solver_kwargs,
        )

    if C.value is None:
        raise RuntimeError("DCCP failed to return fiber coordinates.")

    centers = np.asarray(C.value, dtype=float)
    centers[:, 0] = np.mod(centers[:, 0], p.Lx_um)
    centers[:, 1] = np.mod(centers[:, 1], p.Ly_um)

    if boundary_mode == "inside":
        centers[:, 0] = np.clip(centers[:, 0], radii, p.Lx_um - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, p.Ly_um - radii)

    print("  DCCP result/status       = %s / %s" % (str(result), str(problem.status)))
    return centers


def periodic_polish_after_dccp(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray,
                               p: RVEParams, rng: np.random.Generator) -> np.ndarray:
    """
    Short minimum-image polishing after non-periodic DCCP.

    This step is intentionally much cheaper than strict periodic DCCP. It only
    applies the existing relaxation with alpha=1.0 under the current
    p.use_periodic_distance setting. When p.use_periodic_distance=True, it fixes
    cross-boundary violations before re-orientation and MC shuffle.
    """
    if not p.dccp_post_periodic_polish:
        return centers

    print("STEP 1b: periodic polish after DCCP starts...")
    centers = packing_relax_one_stage(
        centers,
        radii,
        gap,
        p,
        rng,
        alpha=1.0,
        n_iter=int(p.dccp_post_polish_iter),
        shake_amp=0.0,
        move_limit_um=float(p.dccp_post_polish_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    print(
        "  after polish | min_margin = %.6e | max_violation = %.6e | energy = %.6e"
        % (metrics["min_constraint_margin"], metrics["max_violation"], metrics["energy"])
    )
    return centers


# ============================================================
# STEP 2: paper-like re-orientation method
# ============================================================

def single_fiber_min_margin(k: int, trial_xy: np.ndarray, centers: np.ndarray,
                            radii: np.ndarray, gap: np.ndarray, p: RVEParams) -> float:
    """Minimum constraint margin of one trial position against all other fibers."""
    n = len(radii)
    dx = trial_xy[0] - centers[:, 0]
    dy = trial_xy[1] - centers[:, 1]
    dx, dy = min_image_components(dx, dy, p)
    d = np.sqrt(dx * dx + dy * dy)
    d[k] = np.inf
    required = radii[k] + radii + gap[k, :]
    required[k] = 0.0
    margin = d - required
    return float(np.min(margin))


def is_one_fiber_valid(k: int, centers: np.ndarray, radii: np.ndarray, gap: np.ndarray, p: RVEParams) -> bool:
    margin = single_fiber_min_margin(k, centers[k], centers, radii, gap, p)
    return margin >= -p.tol_um


def estimate_resin_rich_points(centers: np.ndarray, radii: np.ndarray, p: RVEParams,
                               rng: np.random.Generator, n_points: int = 2000,
                               keep: int = 50) -> np.ndarray:
    """
    Sample random points and keep those farthest from existing fiber surfaces.
    These points are used as resin-rich search centers in Step 2.
    """
    pts = np.zeros((n_points, 2), dtype=float)
    pts[:, 0] = rng.uniform(0.0, p.Lx_um, n_points)
    pts[:, 1] = rng.uniform(0.0, p.Ly_um, n_points)

    best_surface_gap = np.empty(n_points, dtype=float)
    for a in range(n_points):
        dx = pts[a, 0] - centers[:, 0]
        dy = pts[a, 1] - centers[:, 1]
        dx, dy = min_image_components(dx, dy, p)
        d = np.sqrt(dx * dx + dy * dy)
        best_surface_gap[a] = np.min(d - radii)

    idx = np.argsort(best_surface_gap)[-min(keep, n_points):]
    return pts[idx]


def paper_step2_reorientation(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Re-orientation method in the spirit of the paper.

    For each fiber, candidate positions are searched in resin-rich regions.
    A candidate is acceptable only if it satisfies all inter-fiber distance
    constraints. It is accepted if it improves the local minimum margin, with a
    small probability of random legal acceptance to avoid over-regularization.
    """
    if not p.use_reorientation:
        return centers

    print("STEP 2: re-orientation method starts...")
    centers = centers.copy()
    n = len(radii)
    local_sigma = p.reorient_local_radius_factor * float(np.mean(radii))

    for pass_id in range(1, p.reorient_passes + 1):
        resin_points = estimate_resin_rich_points(
            centers, radii, p, rng,
            n_points=max(1000, 8 * n),
            keep=max(30, n // 5),
        )

        order = rng.permutation(n)
        moved = 0

        for k in order:
            old_xy = centers[k].copy()
            old_margin = single_fiber_min_margin(k, old_xy, centers, radii, gap, p)
            best_xy = old_xy.copy()
            best_margin = old_margin

            for _ in range(p.reorient_trials_per_fiber):
                if rng.random() < 0.65 and len(resin_points) > 0:
                    base = resin_points[int(rng.integers(0, len(resin_points)))]
                    trial = base + rng.normal(0.0, local_sigma, 2)
                    trial[0] = trial[0] % p.Lx_um
                    trial[1] = trial[1] % p.Ly_um
                else:
                    trial = np.array([
                        rng.uniform(0.0, p.Lx_um),
                        rng.uniform(0.0, p.Ly_um),
                    ])

                margin = single_fiber_min_margin(k, trial, centers, radii, gap, p)
                if margin >= -p.tol_um and margin > best_margin:
                    best_xy = trial.copy()
                    best_margin = margin

            legal_improvement = best_margin >= old_margin + p.reorient_accept_if_improve_um
            legal_random = best_margin >= -p.tol_um and rng.random() < p.reorient_random_accept_prob

            if legal_improvement or legal_random:
                centers[k] = best_xy
                moved += 1

        metrics = compute_metrics(centers, radii, gap, p)
        print(
            "  pass %d/%d | moved = %d | min_margin = %.6e | max_violation = %.6e"
            % (pass_id, p.reorient_passes, moved,
               metrics["min_constraint_margin"], metrics["max_violation"])
        )

        if metrics["max_violation"] > p.tol_um:
            # The move logic should prevent this, but polish if numerical noise appears.
            centers = packing_relax_one_stage(
                centers, radii, gap, p, rng,
                alpha=1.0,
                n_iter=1000,
                shake_amp=0.0,
                move_limit_um=p.opt_final_move_limit_um,
            )

    return centers


# ============================================================
# Optional legal Monte Carlo shuffle
# ============================================================

def monte_carlo_shuffle(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray,
                        p: RVEParams, rng: np.random.Generator) -> np.ndarray:
    if not p.use_mc_shuffle or p.mc_steps_per_fiber <= 0 or p.mc_move_amp_um <= 0.0:
        return centers

    centers = centers.copy()
    n = len(radii)
    n_steps = int(p.mc_steps_per_fiber * n)
    accept = 0

    for _ in range(n_steps):
        k = int(rng.integers(0, n))
        old = centers[k].copy()
        centers[k, 0] = (centers[k, 0] + rng.normal(0.0, p.mc_move_amp_um)) % p.Lx_um
        centers[k, 1] = (centers[k, 1] + rng.normal(0.0, p.mc_move_amp_um)) % p.Ly_um
        if is_one_fiber_valid(k, centers, radii, gap, p):
            accept += 1
        else:
            centers[k] = old

    print("Optional MC shuffle: steps = %d, accept ratio = %.4f" % (n_steps, accept / max(n_steps, 1)))
    return centers


# ============================================================
# STEP 3: periodic copies for Abaqus geometry
# ============================================================

def build_main_dataframe(centers: np.ndarray, radii: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, (c, r) in enumerate(zip(centers, radii), start=1):
        rows.append({
            "id": i,
            "x_um": c[0],
            "y_um": c[1],
            "r_um": r,
            "diameter_um": 2.0 * r,
            "x_mm": c[0] / 1000.0,
            "y_mm": c[1] / 1000.0,
            "r_mm": r / 1000.0,
            "diameter_mm": 2.0 * r / 1000.0,
        })
    return pd.DataFrame(rows)


def build_for_abaqus_dataframe(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> pd.DataFrame:
    rows = []
    new_id = 1

    for i, (c, r) in enumerate(zip(centers, radii), start=1):
        x = c[0]
        y = c[1]

        rows.append({
            "id": new_id,
            "parent_id": i,
            "is_periodic_copy": False,
            "shift_x_um": 0.0,
            "shift_y_um": 0.0,
            "x_um": x,
            "y_um": y,
            "r_um": r,
            "diameter_um": 2.0 * r,
            "x_mm": x / 1000.0,
            "y_mm": y / 1000.0,
            "r_mm": r / 1000.0,
            "diameter_mm": 2.0 * r / 1000.0,
        })
        new_id += 1

        shift_x = [0.0]
        shift_y = [0.0]

        if x - r < 0.0:
            shift_x.append(p.Lx_um)
        if x + r > p.Lx_um:
            shift_x.append(-p.Lx_um)
        if y - r < 0.0:
            shift_y.append(p.Ly_um)
        if y + r > p.Ly_um:
            shift_y.append(-p.Ly_um)

        for sx in shift_x:
            for sy in shift_y:
                if abs(sx) < 1.0e-12 and abs(sy) < 1.0e-12:
                    continue
                xx = x + sx
                yy = y + sy
                rows.append({
                    "id": new_id,
                    "parent_id": i,
                    "is_periodic_copy": True,
                    "shift_x_um": sx,
                    "shift_y_um": sy,
                    "x_um": xx,
                    "y_um": yy,
                    "r_um": r,
                    "diameter_um": 2.0 * r,
                    "x_mm": xx / 1000.0,
                    "y_mm": yy / 1000.0,
                    "r_mm": r / 1000.0,
                    "diameter_mm": 2.0 * r / 1000.0,
                })
                new_id += 1

    return pd.DataFrame(rows)


# ============================================================
# Randomness evaluation, similar to the paper
# ============================================================

def periodic_pairwise_distance_angle(centers: np.ndarray, p: RVEParams) -> Tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    dist = np.zeros((n, n), dtype=float)
    angle = np.zeros((n, n), dtype=float)

    for i in range(n):
        dx = centers[:, 0] - centers[i, 0]
        dy = centers[:, 1] - centers[i, 1]
        dx, dy = min_image_components(dx, dy, p)
        d = np.sqrt(dx * dx + dy * dy)
        theta = np.degrees(np.arctan2(dy, dx))
        theta = np.where(theta < 0.0, theta + 360.0, theta)
        d[i] = np.inf
        theta[i] = np.nan
        dist[i, :] = d
        angle[i, :] = theta

    return dist, angle


def evaluate_nearest_neighbor_distance(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> pd.DataFrame:
    dist, _ = periodic_pairwise_distance_angle(centers, p)
    sorted_dist = np.sort(dist, axis=1)
    r_mean = float(np.mean(radii))
    return pd.DataFrame({
        "fiber_id": np.arange(1, len(centers) + 1),
        "nearest_1_h_over_r": sorted_dist[:, 0] / r_mean,
        "nearest_2_h_over_r": sorted_dist[:, 1] / r_mean,
        "nearest_3_h_over_r": sorted_dist[:, 2] / r_mean,
    })


def evaluate_nearest_neighbor_orientation(centers: np.ndarray, p: RVEParams) -> pd.DataFrame:
    dist, angle = periodic_pairwise_distance_angle(centers, p)
    nearest_id = np.argmin(dist, axis=1)
    nn_angle = np.array([angle[i, nearest_id[i]] for i in range(len(centers))])
    nn_angle_sorted = np.sort(nn_angle)
    cdf = np.arange(1, len(nn_angle_sorted) + 1) / float(len(nn_angle_sorted))
    csr_cdf = nn_angle_sorted / 360.0
    return pd.DataFrame({
        "orientation_degree": nn_angle_sorted,
        "cdf_algorithm": cdf,
        "cdf_CSR": csr_cdf,
        "cdf_difference": cdf - csr_cdf,
    })


def evaluate_ripley_k(centers: np.ndarray, radii: np.ndarray, p: RVEParams,
                      h_min_over_r: float = 2.0, h_max_over_r: float = 15.0,
                      n_h: int = 120) -> pd.DataFrame:
    n = len(centers)
    area = p.Lx_um * p.Ly_um
    r_mean = float(np.mean(radii))
    h_min = h_min_over_r * r_mean
    h_max = min(h_max_over_r * r_mean, 0.5 * min(p.Lx_um, p.Ly_um))
    h_values = np.linspace(h_min, h_max, n_h)

    dist, _ = periodic_pairwise_distance_angle(centers, p)
    K_values = []
    for h in h_values:
        count = np.sum(dist <= h)  # ordered pairs; diagonal is inf
        K_values.append(area * count / float(n * (n - 1)))

    K_values = np.array(K_values)
    K_csr = math.pi * h_values ** 2
    return pd.DataFrame({
        "h_um": h_values,
        "h_over_r": h_values / r_mean,
        "K_algorithm": K_values,
        "K_CSR": K_csr,
        "K_minus_KCSR": K_values - K_csr,
    })


def evaluate_pair_distribution_from_k(df_k: pd.DataFrame) -> pd.DataFrame:
    """Paper definition: g(h) = 1/(2*pi*h) * dK/dh."""
    h = df_k["h_um"].values
    K = df_k["K_algorithm"].values
    dK_dh = np.gradient(K, h)
    g = dK_dh / (2.0 * math.pi * h)
    return pd.DataFrame({
        "h_um": h,
        "h_over_r": df_k["h_over_r"].values,
        "g_algorithm": g,
        "g_CSR": np.ones_like(g),
        "g_minus_1": g - 1.0,
    })


def evaluate_pair_distribution_shell(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> pd.DataFrame:
    """Stable direct annulus-count version; useful for finite-sample plotting."""
    n = len(centers)
    area = p.Lx_um * p.Ly_um
    r_mean = float(np.mean(radii))
    h_min = p.pair_h_min_over_r * r_mean
    h_max = min(p.pair_h_max_over_r * r_mean, 0.5 * min(p.Lx_um, p.Ly_um))
    dh = p.pair_dh_over_r * r_mean

    dist, _ = periodic_pairwise_distance_angle(centers, p)
    pair_dist = dist[np.triu_indices(n, k=1)]
    edges = np.arange(h_min, h_max + 0.5 * dh, dh)
    if len(edges) < 2:
        raise ValueError("Pair distribution range too small. Increase RVE size or reduce h range.")

    count_unordered, _ = np.histogram(pair_dist, bins=edges)
    h_center = 0.5 * (edges[:-1] + edges[1:])
    shell_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    rho = (n - 1) / area
    expected_unordered = 0.5 * n * rho * shell_area
    g_raw = count_unordered / np.maximum(expected_unordered, 1.0e-12)

    if p.pair_smooth_window is not None and p.pair_smooth_window > 1:
        g = pd.Series(g_raw).rolling(
            window=int(p.pair_smooth_window),
            center=True,
            min_periods=1,
        ).mean().values
    else:
        g = g_raw

    return pd.DataFrame({
        "h_um": h_center,
        "h_over_r": h_center / r_mean,
        "count_unordered": count_unordered,
        "g_raw": g_raw,
        "g_algorithm": g,
        "g_CSR": np.ones_like(g),
        "g_minus_1": g - 1.0,
    })


def contact_peak_diagnostics(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Dict[str, float]:
    n = len(centers)
    r_mean = float(np.mean(radii))
    dist, _ = periodic_pairwise_distance_angle(centers, p)
    pair_dist = dist[np.triu_indices(n, k=1)]
    h_over_r = pair_dist / r_mean
    low = p.contact_low_over_r
    high = p.contact_high_over_r
    count = int(np.sum((h_over_r >= low) & (h_over_r < high)))
    total = int(len(h_over_r))
    return {
        "pair_total": total,
        "min_h_over_r": float(np.min(h_over_r)),
        "contact_interval_low": low,
        "contact_interval_high": high,
        "contact_pair_count": count,
        "contact_pair_fraction": count / max(total, 1),
    }


def layout_randomness_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """
    Scalar score used to choose the best candidate among several valid RVEs.

    The score is designed for the user's current purpose:
    1. keep the pair distribution function close to CSR at medium/large h/r;
    2. avoid an excessively high first peak near h/r = 2;
    3. keep the nearest-neighbor orientation reasonably random;
    4. penalize too many near-contact pairs in the first histogram bin.

    Smaller score means a better candidate.
    """
    df_g = evaluate_pair_distribution_shell(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    contact = contact_peak_diagnostics(centers, radii, p)

    g_values = df_g["g_algorithm"].values
    h_values = df_g["h_over_r"].values

    g_rmse = math.sqrt(np.mean((g_values - 1.0) ** 2))
    orientation_ks = np.max(np.abs(df_ori["cdf_difference"].values))
    contact_fraction = contact["contact_pair_fraction"]

    # The first bin is the most sensitive part of the pair distribution curve.
    first_peak = float(g_values[0])

    # Medium/far-field part of g(h). This prevents the candidate selection from
    # focusing only on the first peak.
    far_mask = h_values >= 3.0
    if np.any(far_mask):
        far_rmse = math.sqrt(np.mean((g_values[far_mask] - 1.0) ** 2))
    else:
        far_rmse = g_rmse

    # Empirical weights. The first-peak penalty starts only when the first peak
    # is obviously high. You can adjust 2.2 if your target curve is different.
    score = (
        0.50 * g_rmse
        + 0.80 * far_rmse
        + 0.25 * orientation_ks
        + 0.25 * max(0.0, first_peak - 2.2)
        + 20.0 * contact_fraction
    )

    return score, {
        "score": score,
        "pair_distribution_RMSE_shell": g_rmse,
        "pair_distribution_far_RMSE_shell": far_rmse,
        "orientation_CDF_KS_distance": orientation_ks,
        "contact_pair_fraction": contact_fraction,
        "first_pair_distribution_peak": first_peak,
        "min_h_over_r": contact["min_h_over_r"],
    }


# ============================================================
# Plotting
# ============================================================

def plot_rve(df_for_abaqus: pd.DataFrame, p: RVEParams, save_path: str):
    """Plot the main RVE and its periodic copies for checking boundary periodicity."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((0, 0), p.Lx_um, p.Ly_um, fill=False, linewidth=2.0))

    for _, row in df_for_abaqus.iterrows():
        alpha = 0.35 if bool(row["is_periodic_copy"]) else 0.75
        ax.add_patch(Circle((row["x_um"], row["y_um"]), row["r_um"], fill=True, alpha=alpha, linewidth=0.6))

    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * p.Lx_um, 1.1 * p.Lx_um)
    ax.set_ylim(-0.1 * p.Ly_um, 1.1 * p.Ly_um)
    ax.set_xlabel("x / μm")
    ax.set_ylabel("y / μm")
    ax.set_title("Periodic RVE check, target Vf = %.3f" % p.target_vf)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_rve_main_only(df_for_abaqus: pd.DataFrame, p: RVEParams, save_path: str):
    """
    Plot only the main fibers inside the base RVE.

    This figure is cleaner for papers/reports. The periodic-copy figure above is
    better for checking whether boundary-intersecting fibers have been copied.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((0, 0), p.Lx_um, p.Ly_um, fill=False, linewidth=2.0))

    df_main_only = df_for_abaqus[df_for_abaqus["is_periodic_copy"] == False]
    for _, row in df_main_only.iterrows():
        ax.add_patch(Circle((row["x_um"], row["y_um"]), row["r_um"], fill=True, alpha=0.75, linewidth=0.6))

    ax.set_aspect("equal")
    ax.set_xlim(0.0, p.Lx_um)
    ax.set_ylim(0.0, p.Ly_um)
    ax.set_xlabel("x / μm")
    ax.set_ylabel("y / μm")
    ax.set_title("Main RVE only, target Vf = %.3f" % p.target_vf)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_failed(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray, p: RVEParams):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((0, 0), p.Lx_um, p.Ly_um, fill=False, linewidth=2.0))
    for c, r in zip(centers, radii):
        ax.add_patch(Circle((c[0], c[1]), r, fill=True, alpha=0.6))
    bad = get_bad_pairs(centers, radii, gap, p)
    for item in bad[:30]:
        i = item[0] - 1
        j = item[1] - 1
        ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * p.Lx_um, 1.1 * p.Lx_um)
    ax.set_ylim(-0.1 * p.Ly_um, 1.1 * p.Ly_um)
    ax.set_title("Failed layout")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("failed_layout.png", dpi=300)
    plt.show()


def plot_nearest_neighbor_distance(df_nn: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    bins = 22
    plt.hist(df_nn["nearest_1_h_over_r"].values, bins=bins, density=True, alpha=0.45, label="1st nearest")
    plt.hist(df_nn["nearest_2_h_over_r"].values, bins=bins, density=True, alpha=0.45, label="2nd nearest")
    plt.hist(df_nn["nearest_3_h_over_r"].values, bins=bins, density=True, alpha=0.45, label="3rd nearest")
    plt.xlabel("h / r")
    plt.ylabel("PDF")
    plt.title("Nearest-neighbor distance distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_nearest_neighbor_orientation(df_ori: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df_ori["orientation_degree"].values, df_ori["cdf_algorithm"].values, linewidth=2.0, label="Generated RVE")
    plt.plot(df_ori["orientation_degree"].values, df_ori["cdf_CSR"].values, linestyle="--", linewidth=2.0, label="CSR")
    plt.xlabel("Orientation / degree")
    plt.ylabel("CDF")
    plt.title("Nearest-neighbor orientation CDF")
    plt.xlim(0.0, 360.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_ripley_k(df_k: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df_k["h_over_r"].values, df_k["K_algorithm"].values, linewidth=2.0, label="Generated RVE")
    plt.plot(df_k["h_over_r"].values, df_k["K_CSR"].values, linestyle="--", linewidth=2.0, label="CSR")
    plt.xlabel("h / r")
    plt.ylabel("K(h)")
    plt.title("Ripley's K function")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_pair_distribution(df_g: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df_g["h_over_r"].values, df_g["g_algorithm"].values, linewidth=2.0, label="Generated RVE")
    plt.plot(df_g["h_over_r"].values, df_g["g_CSR"].values, linestyle="--", linewidth=2.0, label="CSR")
    plt.xlabel("h / r")
    plt.ylabel("g(h)")
    plt.title("Pair distribution function")
    plt.xlim(float(np.min(df_g["h_over_r"].values)), float(np.max(df_g["h_over_r"].values)))
    ymax = max(3.0, min(8.0, 1.15 * float(np.nanmax(df_g["g_algorithm"].values))))
    plt.ylim(0.0, ymax)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ============================================================
# Evaluation and main generation
# ============================================================

def evaluate_randomness(centers: np.ndarray, radii: np.ndarray, p: RVEParams, output_prefix: str):
    print("\n" + "=" * 80)
    print("Randomness evaluation starts...")
    print("=" * 80)

    df_nn = evaluate_nearest_neighbor_distance(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    df_k = evaluate_ripley_k(centers, radii, p)

    # The paper defines g(h) from dK/dh; direct-shell version is also saved.
    df_g_from_k = evaluate_pair_distribution_from_k(df_k)
    df_g_shell = evaluate_pair_distribution_shell(centers, radii, p)

    contact_diag = contact_peak_diagnostics(centers, radii, p)
    orientation_ks = np.max(np.abs(df_ori["cdf_difference"].values))
    k_rmse = math.sqrt(np.mean(((df_k["K_algorithm"].values - df_k["K_CSR"].values) / np.maximum(df_k["K_CSR"].values, 1.0e-12)) ** 2))
    g_rmse_shell = math.sqrt(np.mean((df_g_shell["g_algorithm"].values - 1.0) ** 2))
    g_rmse_from_k = math.sqrt(np.mean((df_g_from_k["g_algorithm"].values - 1.0) ** 2))

    summary = pd.DataFrame([{
        "fiber_number": len(centers),
        "mean_radius_um": float(np.mean(radii)),
        "nearest_1_mean_h_over_r": df_nn["nearest_1_h_over_r"].mean(),
        "nearest_1_std_h_over_r": df_nn["nearest_1_h_over_r"].std(),
        "nearest_2_mean_h_over_r": df_nn["nearest_2_h_over_r"].mean(),
        "nearest_3_mean_h_over_r": df_nn["nearest_3_h_over_r"].mean(),
        "orientation_CDF_KS_distance": orientation_ks,
        "Ripley_K_relative_RMSE": k_rmse,
        "pair_distribution_RMSE_from_K_paper_definition": g_rmse_from_k,
        "pair_distribution_RMSE_shell_stable": g_rmse_shell,
        "contact_pair_fraction": contact_diag["contact_pair_fraction"],
        "contact_pair_count": contact_diag["contact_pair_count"],
        "min_h_over_r": contact_diag["min_h_over_r"],
    }])

    plot_nearest_neighbor_distance(df_nn, output_prefix + "_eval_nearest_distance.png")
    plot_nearest_neighbor_orientation(df_ori, output_prefix + "_eval_orientation_cdf.png")
    plot_ripley_k(df_k, output_prefix + "_eval_ripley_k.png")
    plot_pair_distribution(df_g_shell, output_prefix + "_eval_pair_distribution_shell.png")
    plot_pair_distribution(df_g_from_k, output_prefix + "_eval_pair_distribution_from_K.png")

    eval_xlsx = output_prefix + "_randomness_evaluation.xlsx"
    with pd.ExcelWriter(eval_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df_nn.to_excel(writer, sheet_name="nearest_distance", index=False)
        df_ori.to_excel(writer, sheet_name="orientation_cdf", index=False)
        df_k.to_excel(writer, sheet_name="ripley_k", index=False)
        df_g_from_k.to_excel(writer, sheet_name="pair_distribution_from_K", index=False)
        df_g_shell.to_excel(writer, sheet_name="pair_distribution_shell", index=False)

    print("Randomness evaluation finished.")
    print("Evaluation file saved:", eval_xlsx)
    print("orientation CDF KS distance       = %.6e" % orientation_ks)
    print("Ripley K relative RMSE            = %.6e" % k_rmse)
    print("Pair distribution RMSE from K     = %.6e" % g_rmse_from_k)
    print("Pair distribution RMSE shell      = %.6e" % g_rmse_shell)
    print("contact pair fraction             = %.6e" % contact_diag["contact_pair_fraction"])
    print("min h/r                           = %.6e" % contact_diag["min_h_over_r"])
    print("=" * 80)

    return summary, df_nn, df_ori, df_k, df_g_shell


def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate one candidate layout.

    Available STEP 1 methods:
    - relaxation  : fast periodic relaxation only.
    - dccp        : non-periodic DCCP + optional periodic polish.
    - hybrid_dccp : use DCCP if possible; if DCCP fails, use the relaxation warm start.
    """
    method = str(p.step1_method).lower()

    if method == "relaxation":
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ("dccp", "hybrid_dccp"):
        warm = None
        if p.dccp_use_relaxation_warm_start:
            warm = random_initial_centers(p, len(radii), rng)
            warm = paper_step1_initial_optimization(warm, radii, gap, p, rng)

        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == "hybrid_dccp" and warm is not None:
                print("  Warning: DCCP failed in hybrid_dccp mode. Using relaxation warm start instead.")
                print("  DCCP error:", repr(exc))
                centers = warm.copy()
            else:
                raise

        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)

    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    centers = paper_step2_reorientation(centers, radii, gap, p, rng)
    centers = monte_carlo_shuffle(centers, radii, gap, p, rng)
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics

def generate(p: RVEParams):
    if p.seed is None:
        p.seed = int(time.time() * 1000000) % (2 ** 32)

    random.seed(p.seed)
    rng = np.random.default_rng(p.seed)

    n = estimate_fiber_number(p)
    radii = generate_radii(p, n, rng)
    vf = actual_vf(radii, p)

    print("=" * 80)
    print("Paper-like random periodic RVE generation")
    print("Random seed used = %d" % p.seed)
    print("Lx, Ly           = %.6f, %.6f um" % (p.Lx_um, p.Ly_um))
    print("target Vf        = %.6f" % p.target_vf)
    print("actual Vf        = %.6f" % vf)
    print("fiber number     = %d" % n)
    print("diameter mode    = %s" % p.diameter_mode)
    print("lmin, lmax       = %.6f, %.6f um" % (p.lmin_um, p.lmax_um))
    print("periodic metric  = %s" % str(p.use_periodic_distance))
    print("STEP 1 method    = %s" % str(p.step1_method))
    if str(p.step1_method).lower() in ("dccp", "hybrid_dccp"):
        print("DCCP boundary    = %s" % str(p.dccp_boundary_mode))
        print("post DCCP polish = %s" % str(p.dccp_post_periodic_polish))
    print("=" * 80)

    best_centers = None
    best_gap = None
    best_metrics = None
    best_score = None
    best_score_info = None
    best_invalid = None
    valid_found = 0

    for restart in range(1, p.max_restarts + 1):
        print("\nrestart %03d starts..." % restart, flush=True)
        gap = make_gap_matrix(p, n, rng)
        centers, metrics = generate_one_candidate(p, radii, gap, rng)

        print(
            "restart %03d | min_gap = %.6e | min_margin = %.6e | max_overlap = %.6e | max_violation = %.6e | energy = %.6e"
            % (restart, metrics["min_surface_gap"], metrics["min_constraint_margin"],
               metrics["max_overlap"], metrics["max_violation"], metrics["energy"])
        )

        if best_invalid is None or metrics["energy"] < best_invalid[2]["energy"]:
            best_invalid = (centers.copy(), gap.copy(), metrics.copy())

        is_valid = metrics["max_overlap"] <= p.tol_um and metrics["max_violation"] <= p.tol_um
        if not is_valid:
            continue

        valid_found += 1
        print("Valid candidate %d found." % valid_found)

        if p.selection_mode == "first_valid":
            best_centers = centers.copy()
            best_gap = gap.copy()
            best_metrics = metrics.copy()
            best_score = 0.0
            best_score_info = {"score": 0.0}
            print("Accepted first valid layout.")
            break

        candidate_score, candidate_score_info = layout_randomness_score(centers, radii, p)
        print(
            "candidate randomness score = %.6e | g_RMSE(shell) = %.6e | contact_fraction = %.6e | min_h/r = %.6f"
            % (candidate_score_info["score"],
               candidate_score_info["pair_distribution_RMSE_shell"],
               candidate_score_info["contact_pair_fraction"],
               candidate_score_info["min_h_over_r"])
        )

        if best_score is None or candidate_score < best_score:
            best_centers = centers.copy()
            best_gap = gap.copy()
            best_metrics = metrics.copy()
            best_score = candidate_score
            best_score_info = candidate_score_info.copy()
            print("This candidate is currently the best valid layout.")

        if valid_found >= p.valid_candidates_to_test:
            print("\nRequired number of valid candidates reached.")
            break

    if best_centers is None:
        print("\nNo valid candidate found. Reporting best invalid candidate.")
        best_centers, best_gap, best_metrics = best_invalid

    if best_score_info is not None:
        print("\nBest valid candidate score information:")
        for key, value in best_score_info.items():
            if isinstance(value, float):
                print("  %s = %.6e" % (key, value))
            else:
                print("  %s = %s" % (key, value))

    if best_metrics["max_overlap"] > p.tol_um or best_metrics["max_violation"] > p.tol_um:
        print("\nGeneration failed.")
        print("Best metrics:", best_metrics)
        bad = get_bad_pairs(best_centers, radii, best_gap, p)
        print("Bad pair number:", len(bad))
        for item in bad[:20]:
            print("fiber %d - %d | d = %.6f | required = %.6f | violation = %.6e" % item)
        plot_failed(best_centers, radii, best_gap, p)
        raise RuntimeError("生成失败：仍然存在相交或间距不足，不要导入 Abaqus。")

    df_main = build_main_dataframe(best_centers, radii)
    df_for_abaqus = build_for_abaqus_dataframe(best_centers, radii, p)

    summary = pd.DataFrame([{
        "Lx_um": p.Lx_um,
        "Ly_um": p.Ly_um,
        "target_vf": p.target_vf,
        "actual_vf": vf,
        "seed": p.seed,
        "fiber_number": n,
        "diameter_mode": p.diameter_mode,
        "diameter_mean_um": p.diameter_mean_um,
        "diameter_std_um": p.diameter_std_um,
        "diameter_min_um": p.diameter_min_um,
        "diameter_max_um": p.diameter_max_um,
        "lmin_um": p.lmin_um,
        "lmax_um": p.lmax_um,
        "valid_candidates_found": valid_found,
        "selection_mode": p.selection_mode,
        "step1_method": p.step1_method,
        "dccp_boundary_mode": p.dccp_boundary_mode,
        "dccp_post_periodic_polish": p.dccp_post_periodic_polish,
        "best_randomness_score": best_score,
        "use_reorientation": p.use_reorientation,
        "reorient_passes": p.reorient_passes,
        "reorient_trials_per_fiber": p.reorient_trials_per_fiber,
        "use_periodic_distance": p.use_periodic_distance,
        "min_surface_gap_um": best_metrics["min_surface_gap"],
        "min_constraint_margin_um": best_metrics["min_constraint_margin"],
        "max_overlap_um": best_metrics["max_overlap"],
        "max_violation_um": best_metrics["max_violation"],
    }])

    main_csv = p.output_prefix + "_main.csv"
    abaqus_csv = p.output_prefix + "_for_abaqus.csv"
    xlsx_file = p.output_prefix + ".xlsx"
    preview_png = p.output_prefix + "_preview.png"

    df_main.to_csv(main_csv, index=False, encoding="utf-8-sig")
    df_for_abaqus.to_csv(abaqus_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="main_centers", index=False)
        df_for_abaqus.to_excel(writer, sheet_name="for_abaqus", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    plot_rve(df_for_abaqus, p, preview_png)
    plot_rve_main_only(df_for_abaqus, p, p.output_prefix + "_main_only_preview.png")
    evaluate_randomness(best_centers, radii, p, output_prefix=p.output_prefix)

    print("\n" + "=" * 80)
    print("Generation succeeded.")
    print("Saved files:")
    print(main_csv)
    print(abaqus_csv)
    print(xlsx_file)
    print(preview_png)
    print(p.output_prefix + "_main_only_preview.png")
    print(p.output_prefix + "_randomness_evaluation.xlsx")
    print("=" * 80)
    print("actual Vf       = %.8f" % vf)
    print("min gap         = %.8e um" % best_metrics["min_surface_gap"])
    print("min margin      = %.8e um" % best_metrics["min_constraint_margin"])
    print("max overlap     = %.8e um" % best_metrics["max_overlap"])
    print("max violation   = %.8e um" % best_metrics["max_violation"])
    print("=" * 80)


# ============================================================
# Parameter presets
# ============================================================

def make_fast_test_params() -> RVEParams:
    """Practical preset: faster than the full paper statistical size."""
    return RVEParams(
        Lx_um=100.0,
        Ly_um=100.0,
        target_vf=0.70,
        diameter_mode="constant",
        diameter_mean_um=7.0,
        diameter_std_um=0.317,
        lmin_um=0.175,
        lmax_um=0.350,
        max_restarts=80,
        valid_candidates_to_test=3,
        selection_mode="best_valid_randomness",
        step1_method="relaxation",
        dccp_boundary_mode="cell",
        dccp_post_periodic_polish=True,
        opt_alpha_steps=35,
        opt_iter_per_alpha=600,
        opt_final_iter=4000,
        use_reorientation=True,
        reorient_passes=2,
        reorient_trials_per_fiber=60,
        use_mc_shuffle=True,
        mc_steps_per_fiber=150,
        mc_move_amp_um=0.03,
        pair_smooth_window=1,
        contact_low_over_r=2.0,
        contact_high_over_r=2.35,
        seed=2026,
        output_prefix="fiber_centers_paper_like_100um_vf70",
    )


def make_paper_fig10_vf70_params() -> RVEParams:
    """
    Vf=70% preset close to the paper's statistical verification setting.

    The paper uses delta = L/r = 50. With r = 3.5 um, L = 175 um.
    For Vf=70%, use lmin = 0.175 um and lmax = 0.350 um.

    This preset is configured for practical periodic-RVE generation using the
    relaxation method and the minimum-image distance criterion. It is suitable
    for later Abaqus PBC use.
    """
    return RVEParams(
        Lx_um=175.0,
        Ly_um=175.0,
        target_vf=0.70,

        diameter_mode="constant",
        diameter_mean_um=7.0,
        diameter_std_um=0.317,

        lmin_um=0.175,
        lmax_um=0.350,

        max_restarts=160,
        valid_candidates_to_test=2,
        selection_mode="best_valid_randomness",

        step1_method="relaxation",
        dccp_boundary_mode="cell",
        dccp_post_periodic_polish=True,

        opt_alpha_steps=40,
        opt_iter_per_alpha=800,
        opt_final_iter=6000,

        use_reorientation=True,
        reorient_passes=2,
        reorient_trials_per_fiber=60,

        use_mc_shuffle=True,
        mc_steps_per_fiber=200,
        mc_move_amp_um=0.03,

        use_periodic_distance=True,

        pair_smooth_window=1,
        contact_low_over_r=2.0,
        contact_high_over_r=2.35,

        seed=2026,
        output_prefix="fiber_centers_periodic_relaxation_vf70_optimized",
    )


def make_real_diameter_vf60_params() -> RVEParams:
    """
    Preset for the paper's T700/7901 validation idea: measured fiber diameter
    is close to a normal distribution with mean around 6.92 um and std 0.317 um.
    """
    return RVEParams(
        Lx_um=50.0,
        Ly_um=50.0,
        target_vf=0.60,
        diameter_mode="normal",
        diameter_mean_um=6.92,
        diameter_std_um=0.317,
        lmin_um=0.175,
        lmax_um=0.875,
        max_restarts=100,
        valid_candidates_to_test=5,
        selection_mode="best_valid_randomness",
        step1_method="relaxation",
        dccp_boundary_mode="cell",
        dccp_post_periodic_polish=True,
        opt_alpha_steps=35,
        opt_iter_per_alpha=600,
        opt_final_iter=4000,
        use_reorientation=True,
        reorient_passes=3,
        reorient_trials_per_fiber=80,
        use_mc_shuffle=False,
        pair_smooth_window=1,
        seed=2026,
        output_prefix="fiber_centers_real_diameter_vf60",
    )


def build_user_params() -> RVEParams:
    """
    ========================== USER OPERATION CENTER ==========================
    All frequently adjusted parameters are collected here.

    Recommended workflow:
    1. First use PRESET = "fast_100um_vf70" to check whether the code runs.
    2. Then use PRESET = "paper_vf70_periodic" for the formal Vf=70% model.
    3. Change only the values in this function for most daily operations.
    ==========================================================================
    """

    # ----------------------------------------------------------------------
    # 1. Choose a preset
    # ----------------------------------------------------------------------
    # "fast_100um_vf70"     : faster test model, L=100 um, Vf=70%.
    # "paper_vf70_periodic" : formal periodic model, L=175 um, Vf=70%.
    # "real_diameter_vf60"  : normal fiber diameter distribution, Vf=60%.
    PRESET = "paper_vf70_periodic"

    if PRESET == "fast_100um_vf70":
        params = make_fast_test_params()
    elif PRESET == "paper_vf70_periodic":
        params = make_paper_fig10_vf70_params()
    elif PRESET == "real_diameter_vf60":
        params = make_real_diameter_vf60_params()
    else:
        raise ValueError("Unknown PRESET. Check build_user_params().")

    # ----------------------------------------------------------------------
    # 2. Geometry and volume fraction
    # ----------------------------------------------------------------------
    # For the paper Vf=70% statistical setting, use Lx=Ly=175 um, d=7 um.
    # If you change Lx/Ly/target_vf/diameter, the fiber number is recalculated.
    params.Lx_um = params.Lx_um
    params.Ly_um = params.Ly_um
    params.target_vf = params.target_vf

    # ----------------------------------------------------------------------
    # 3. Fiber diameter mode
    # ----------------------------------------------------------------------
    # Options: "constant", "normal", "uniform".
    params.diameter_mode = params.diameter_mode
    params.diameter_mean_um = params.diameter_mean_um
    params.diameter_std_um = params.diameter_std_um
    params.diameter_min_um = params.diameter_min_um
    params.diameter_max_um = params.diameter_max_um

    # ----------------------------------------------------------------------
    # 4. Inter-fiber random spacing
    # ----------------------------------------------------------------------
    # Paper examples:
    #   Vf = 60%, 65%: lmin=0.175 um, lmax=0.875 um
    #   Vf = 70%     : lmin=0.175 um, lmax=0.350 um
    #   Vf = 80%     : lmin=0.100 um, lmax=0.150 um
    params.lmin_um = params.lmin_um
    params.lmax_um = params.lmax_um

    # ----------------------------------------------------------------------
    # 5. STEP 1 solver method
    # ----------------------------------------------------------------------
    # "relaxation"  : fastest. Recommended for many repeated generations.
    # "dccp"        : non-periodic DCCP packing + short periodic polish.
    # "hybrid_dccp" : if DCCP fails, automatically falls back to relaxation.
    #
    # This DCCP is NOT the expensive 3x3 periodic-image DCCP.
    # It is the practical version you asked for: DCCP first, then periodic polish.
    STEP1_METHOD = "hybrid_dccp"
    params.step1_method = STEP1_METHOD

    # DCCP boundary mode:
    # "cell"   : 0<=center<=L. Recommended if you want boundary-intersecting fibers
    #            and periodic copies for Abaqus PBC.
    # "inside" : r<=center<=L-r. Closer to paper Eq.(2), but all fibers stay inside.
    params.dccp_boundary_mode = "cell"

    # DCCP can still be slow for L=175 um and Vf=70%. If it is too slow,
    # change STEP1_METHOD above to "relaxation".
    params.dccp_max_iter = 60
    params.dccp_solver_eps = 1.0e-4
    params.dccp_solver_max_iters = 30000
    params.dccp_verbose = False

    # Important practical trick: after non-periodic DCCP, run a short periodic
    # relaxation polish to remove cross-boundary violations without using strict
    # 3x3 periodic DCCP constraints.
    params.dccp_post_periodic_polish = True
    params.dccp_post_polish_iter = 2500
    params.dccp_post_polish_move_limit_um = 0.05

    # ----------------------------------------------------------------------
    # 6. Periodic geometry switch for Abaqus PBC
    # ----------------------------------------------------------------------
    # True  : use minimum-image distance. Recommended for PBC geometry.
    # False : ordinary non-periodic distance. Usually not recommended for PBC.
    params.use_periodic_distance = True

    # ----------------------------------------------------------------------
    # 7. Candidate search and randomness selection
    # ----------------------------------------------------------------------
    # Larger valid_candidates_to_test gives a better chance of finding a curve
    # with a reasonable pair-distribution first peak, but costs more time.
    params.valid_candidates_to_test = params.valid_candidates_to_test
    params.max_restarts = params.max_restarts
    params.selection_mode = "best_valid_randomness"

    # ----------------------------------------------------------------------
    # 8. Relaxation parameters
    # ----------------------------------------------------------------------
    # Usually do not need to change these. If generation fails, increase
    # opt_iter_per_alpha or opt_final_iter.
    params.opt_alpha_steps = params.opt_alpha_steps
    params.opt_iter_per_alpha = params.opt_iter_per_alpha
    params.opt_final_iter = params.opt_final_iter

    # ----------------------------------------------------------------------
    # 9. Re-orientation parameters
    # ----------------------------------------------------------------------
    # Too strong re-orientation may make the layout overly regular.
    # A practical setting is passes=2, trials=60, then use a small MC shuffle.
    params.use_reorientation = True
    params.reorient_passes = params.reorient_passes
    params.reorient_trials_per_fiber = params.reorient_trials_per_fiber

    # ----------------------------------------------------------------------
    # 10. Optional legal Monte Carlo shuffle
    # ----------------------------------------------------------------------
    # This does not break spacing constraints. It helps reduce artificial
    # near-contact pile-up from the relaxation step.
    params.use_mc_shuffle = params.use_mc_shuffle
    params.mc_steps_per_fiber = params.mc_steps_per_fiber
    params.mc_move_amp_um = params.mc_move_amp_um

    # ----------------------------------------------------------------------
    # 11. Pair distribution settings
    # ----------------------------------------------------------------------
    # First bin = [pair_h_min_over_r, pair_h_min_over_r + pair_dh_over_r].
    # Therefore, if pair_h_min_over_r=2.0 and pair_dh_over_r=0.35, set
    # contact_high_over_r=2.35 for candidate scoring.
    params.pair_h_min_over_r = 2.0
    params.pair_h_max_over_r = 15.0
    params.pair_dh_over_r = 0.35
    params.contact_low_over_r = 2.0
    params.contact_high_over_r = 2.35

    # 1 = raw curve. Use 3 or 5 only for smoother visual display.
    params.pair_smooth_window = 1

    # ----------------------------------------------------------------------
    # 12. Random seed and output name
    # ----------------------------------------------------------------------
    # seed=None gives a new RVE every run. seed=2026 reproduces the same RVE.
    params.seed = 2026
    params.output_prefix = "fiber_centers_hybrid_dccp_periodic_vf70"

    return params


if __name__ == "__main__":
    params = build_user_params()
    generate(params)
