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

    # DCCP constraint reduction / active-set settings.
    # The original all-pair DCCP uses n(n-1)/2 constraints; for n=557 this is
    # 154846 constraints, which is usually the bottleneck. The active-set mode
    # follows the practical idea that only nearby pairs can become active. It
    # solves a reduced DCCP problem, checks all pairs, then adds the violated or
    # near-active pairs and repeats.
    #
    # dccp_pair_strategy:
    #   "active" : use only nearby/nearest pairs and update the set iteratively.
    #   "all"    : use every pair, closest to the direct paper equation but slow.
    dccp_pair_strategy: str = "active"
    dccp_active_outer_iter: int = 3
    dccp_active_margin_um: float = 1.00
    dccp_active_add_margin_um: float = 0.25
    dccp_active_nearest_k: int = 18
    dccp_max_active_constraints: int = 45000
    dccp_constraint_batch_size: int = 6000
    dccp_vectorized_constraints: bool = True
    # If True, DCCP starts from a random/box value instead of the fully-relaxed
    # warm start. This is often more useful for improving pair distribution,
    # because a feasible relaxation warm start plus a regularization objective
    # tends to return almost the same layout.
    dccp_random_start_if_active: bool = True

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



def compute_nonperiodic_pair_margins(centers: np.ndarray, radii: np.ndarray, gap: np.ndarray):
    """Return all base-cell pair margins without minimum-image wrapping."""
    n = len(radii)
    ii, jj = pair_indices(n)
    dx = centers[ii, 0] - centers[jj, 0]
    dy = centers[ii, 1] - centers[jj, 1]
    d = np.sqrt(dx * dx + dy * dy)
    required = radii[ii] + radii[jj] + gap[ii, jj]
    margin = d - required
    return ii, jj, d, required, margin


def build_dccp_active_pairs(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    forced_pairs: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a reduced active pair list for DCCP.

    Using every pair gives O(n^2) constraints. For Vf=70%, L=175 um, n is about
    557 and the all-pair DCCP has 154846 constraints. This function keeps:
      1. pairs whose current margin is smaller than dccp_active_margin_um;
      2. the nearest dccp_active_nearest_k neighbors of every fiber;
      3. pairs forced in by the active-set outer loop.

    The full pair set is still checked after every DCCP solve. Violated or
    near-active pairs are then added in the next outer iteration.
    """
    n = len(radii)
    ii, jj, d, required, margin = compute_nonperiodic_pair_margins(centers, radii, gap)

    active = set()
    near_idx = np.where(margin <= float(p.dccp_active_margin_um))[0]
    for idx in near_idx:
        active.add((int(ii[idx]), int(jj[idx])))

    k = int(p.dccp_active_nearest_k)
    if k > 0:
        # Full distance matrix is cheap compared with CVXPY/DCCP compilation.
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        kk = min(k, n - 1)
        # argpartition is much faster than full sorting for large n.
        neigh = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        for i in range(n):
            for j in neigh[i]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                active.add((a, b))

    if forced_pairs:
        for a, b in forced_pairs:
            if a == b:
                continue
            if a > b:
                a, b = b, a
            active.add((int(a), int(b)))

    if not active:
        # Safety fallback: keep the globally smallest margins.
        order = np.argsort(margin)
        for idx in order[:max(1, min(n, len(order)))]:
            active.add((int(ii[idx]), int(jj[idx])))

    pairs = np.array(sorted(active), dtype=int)

    # Optional cap for very large models. Keep the pairs with the smallest
    # current margins, because they are the most likely to be active.
    max_constraints = int(p.dccp_max_active_constraints)
    if max_constraints > 0 and len(pairs) > max_constraints:
        # Compute margins only for active pairs.
        pi = pairs[:, 0]
        pj = pairs[:, 1]
        dx = centers[pi, 0] - centers[pj, 0]
        dy = centers[pi, 1] - centers[pj, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        req = radii[pi] + radii[pj] + gap[pi, pj]
        mm = dd - req
        keep = np.argsort(mm)[:max_constraints]
        pairs = pairs[keep]

    pair_i = pairs[:, 0]
    pair_j = pairs[:, 1]
    required_active = radii[pair_i] + radii[pair_j] + gap[pair_i, pair_j]
    return pair_i.astype(int), pair_j.astype(int), required_active.astype(float)


def solve_dccp_with_pair_list(
    centers0: np.ndarray,
    radii: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    required: np.ndarray,
    p: RVEParams,
    cp_module,
    dccp_module,
) -> Tuple[np.ndarray, object, str]:
    """Solve one DCCP subproblem using the supplied active pair list."""
    cp = cp_module
    dccp = dccp_module
    n = len(radii)
    C = cp.Variable((n, 2), name="fiber_centers")
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

    batch_size = max(1, int(p.dccp_constraint_batch_size))
    use_vector = bool(p.dccp_vectorized_constraints)

    if use_vector:
        for start in range(0, len(pair_i), batch_size):
            end = min(start + batch_size, len(pair_i))
            bi = pair_i[start:end]
            bj = pair_j[start:end]
            req = required[start:end]
            diff = C[bi, :] - C[bj, :]
            # Vectorized row norms reduce CVXPY expression count drastically.
            constraints.append(cp.norm(diff, 2, axis=1) >= req)
    else:
        for i, j, req in zip(pair_i, pair_j, required):
            constraints.append(
                cp.norm(cp.hstack([C[int(i), 0] - C[int(j), 0], C[int(i), 1] - C[int(j), 1]]), 2) >= float(req)
            )

    objective = cp.Minimize(float(p.dccp_regularization) * cp.sum_squares(C - centers0))
    problem = cp.Problem(objective, constraints)

    if not dccp.is_dccp(problem):
        # Some old cvxpy/dccp combinations do not recognize vectorized DCCP
        # constraints reliably. Fall back to scalar constraints once.
        if use_vector:
            old_flag = p.dccp_vectorized_constraints
            p.dccp_vectorized_constraints = False
            try:
                return solve_dccp_with_pair_list(centers0, radii, pair_i, pair_j, required, p, cp, dccp)
            finally:
                p.dccp_vectorized_constraints = old_flag
        raise RuntimeError("The active DCCP subproblem is not recognized as DCCP.")

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

    return centers, result, str(problem.status)


def paper_step1_dccp_nonperiodic_optimization(
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    warm_start_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Non-periodic DCCP version of STEP 1 with an active-set constraint strategy.

    Direct all-pair DCCP is close to the paper equation but too slow for large
    models: n=557 gives 154846 pair constraints, and n=637 gives 202566. The
    active-set version keeps only local/near-active pairs, solves DCCP, checks
    all pairs, adds the violated/near-active pairs, and repeats.

    This keeps the paper-like DCCP idea but avoids compiling a huge CVXPY model.
    It is still non-periodic; a short periodic polish is applied afterwards for
    PBC-compatible geometry.
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
    all_pair_count = n * (n - 1) // 2
    print("STEP 1: non-periodic DCCP packing optimization starts...")
    print("  DCCP boundary mode       = %s" % str(p.dccp_boundary_mode))
    print("  number of fibers         = %d" % n)
    print("  all pair constraints     = %d" % all_pair_count)
    print("  DCCP pair strategy       = %s" % str(p.dccp_pair_strategy))
    print("  strict periodic DCCP     = False")

    if warm_start_centers is None or (bool(p.dccp_random_start_if_active) and str(p.dccp_pair_strategy).lower() == "active"):
        centers_current = random_initial_centers(p, n, rng)
    else:
        centers_current = warm_start_centers.copy()
    centers_current = prepare_dccp_initial_value(centers_current, radii, p)

    strategy = str(p.dccp_pair_strategy).lower()
    forced_pairs = set()

    if strategy == "all":
        ii, jj = pair_indices(n)
        pair_i = ii.astype(int)
        pair_j = jj.astype(int)
        required = radii[pair_i] + radii[pair_j] + gap[pair_i, pair_j]
        print("  active constraints       = %d (all pairs)" % len(pair_i))
        centers_current, result, status = solve_dccp_with_pair_list(
            centers_current, radii, pair_i, pair_j, required, p, cp, dccp
        )
        print("  DCCP result/status       = %s / %s" % (str(result), str(status)))

    elif strategy == "active":
        for outer in range(1, int(p.dccp_active_outer_iter) + 1):
            pair_i, pair_j, required = build_dccp_active_pairs(
                centers_current, radii, gap, p, forced_pairs=forced_pairs
            )
            print(
                "  active outer %d/%d | active constraints = %d"
                % (outer, int(p.dccp_active_outer_iter), len(pair_i))
            )
            centers_current, result, status = solve_dccp_with_pair_list(
                centers_current, radii, pair_i, pair_j, required, p, cp, dccp
            )
            print("    DCCP result/status     = %s / %s" % (str(result), str(status)))

            ii, jj, d, req_all, margin = compute_nonperiodic_pair_margins(centers_current, radii, gap)
            bad_idx = np.where(margin < -float(p.tol_um))[0]
            near_idx = np.where(margin < float(p.dccp_active_add_margin_um))[0]
            print(
                "    full check | bad pairs = %d | min margin = %.6e"
                % (len(bad_idx), float(np.min(margin)))
            )

            # Add all violated pairs and a limited number of near-active pairs.
            add_idx = np.unique(np.concatenate([bad_idx, near_idx]))
            # Keep the most critical pairs first if too many are near-active.
            if len(add_idx) > int(p.dccp_max_active_constraints):
                order = np.argsort(margin[add_idx])[:int(p.dccp_max_active_constraints)]
                add_idx = add_idx[order]
            for idx in add_idx:
                forced_pairs.add((int(ii[idx]), int(jj[idx])))

            if len(bad_idx) == 0:
                break
    else:
        raise ValueError("dccp_pair_strategy must be 'active' or 'all'.")

    return centers_current

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
            # For active-set DCCP, a fully feasible relaxation warm start plus
            # regularization can make DCCP return almost the same layout. Use a
            # shorter, coarse warm start to reduce severe overlaps without
            # eliminating DCCP's role in the generation stage.
            if str(p.dccp_pair_strategy).lower() == "active":
                warm = packing_relax_one_stage(
                    warm, radii, gap, p, rng,
                    alpha=0.75,
                    n_iter=max(500, int(0.30 * p.opt_final_iter)),
                    shake_amp=0.5 * p.opt_random_shake_um,
                    move_limit_um=p.opt_move_limit_um,
                )
            else:
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
    params.dccp_max_iter = 45
    params.dccp_solver_eps = 2.0e-4
    params.dccp_solver_max_iters = 12000
    params.dccp_verbose = False

    # Active-set DCCP: much faster than all-pair DCCP. If you want the old
    # direct paper equation for a small model, set dccp_pair_strategy="all".
    params.dccp_pair_strategy = "active"
    params.dccp_active_outer_iter = 3
    params.dccp_active_margin_um = 1.00
    params.dccp_active_add_margin_um = 0.25
    params.dccp_active_nearest_k = 18
    params.dccp_max_active_constraints = 45000
    params.dccp_constraint_batch_size = 6000
    params.dccp_vectorized_constraints = True
    params.dccp_random_start_if_active = False

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



# ============================================================
# Overrides: periodic active DCCP + pair-aware local shuffle
# ============================================================

# Save the original user-parameter builder so the override can reuse all presets.
_original_build_user_params = build_user_params


def compute_periodic_pair_margins_and_shifts(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
):
    """
    Return all pair margins under the periodic minimum-image convention, plus
    the integer image shifts used to express each active DCCP constraint.

    For a current pair (i,j), the minimum-image vector is written as
        c_i - c_j - sx*Lx*e_x - sy*Ly*e_y.
    sx and sy are fixed constants inside one DCCP subproblem and are updated
    between active-set outer iterations. This is much cheaper than adding all
    3x3 image constraints, but it still lets DCCP see the currently relevant
    periodic neighbours.
    """
    n = len(radii)
    ii, jj = pair_indices(n)
    raw_dx = centers[ii, 0] - centers[jj, 0]
    raw_dy = centers[ii, 1] - centers[jj, 1]

    if p.use_periodic_distance:
        sx = np.round(raw_dx / p.Lx_um).astype(int)
        sy = np.round(raw_dy / p.Ly_um).astype(int)
        dx = raw_dx - sx * p.Lx_um
        dy = raw_dy - sy * p.Ly_um
    else:
        sx = np.zeros_like(raw_dx, dtype=int)
        sy = np.zeros_like(raw_dy, dtype=int)
        dx = raw_dx
        dy = raw_dy

    d = np.sqrt(dx * dx + dy * dy)
    required = radii[ii] + radii[jj] + gap[ii, jj]
    margin = d - required
    return ii, jj, d, required, margin, sx, sy


def build_dccp_active_pairs_periodic(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    forced_pairs: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Periodic active-pair selector for DCCP.

    It keeps only the pairs that can actually influence the current solution:
    near-margin pairs, k-nearest periodic neighbours, and forced pairs from the
    previous full check. For each pair it also stores the current minimum-image
    shift, so DCCP constraints are periodic-active rather than purely base-cell.
    """
    n = len(radii)
    ii, jj, d, required, margin, sx_all, sy_all = compute_periodic_pair_margins_and_shifts(
        centers, radii, gap, p
    )

    active = set()
    near_idx = np.where(margin <= float(getattr(p, 'dccp_active_margin_um', 1.0)))[0]
    for idx in near_idx:
        active.add((int(ii[idx]), int(jj[idx])))

    k = int(getattr(p, 'dccp_active_nearest_k', 18))
    if k > 0:
        # Periodic distance matrix for k-nearest neighbour selection.
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        if p.use_periodic_distance:
            dx = dx - p.Lx_um * np.round(dx / p.Lx_um)
            dy = dy - p.Ly_um * np.round(dy / p.Ly_um)
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        kk = min(k, n - 1)
        neigh = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        for i in range(n):
            for j in neigh[i]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                active.add((a, b))

    if forced_pairs:
        for item in forced_pairs:
            # Backward-compatible: forced pair can be (i,j) or (i,j,sx,sy).
            a, b = int(item[0]), int(item[1])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            active.add((a, b))

    if not active:
        order = np.argsort(margin)
        for idx in order[:max(1, min(n, len(order)))]:
            active.add((int(ii[idx]), int(jj[idx])))

    pairs = np.array(sorted(active), dtype=int)

    max_constraints = int(getattr(p, 'dccp_max_active_constraints', 45000))
    if max_constraints > 0 and len(pairs) > max_constraints:
        pi = pairs[:, 0]
        pj = pairs[:, 1]
        raw_dx = centers[pi, 0] - centers[pj, 0]
        raw_dy = centers[pi, 1] - centers[pj, 1]
        if p.use_periodic_distance:
            ddx = raw_dx - p.Lx_um * np.round(raw_dx / p.Lx_um)
            ddy = raw_dy - p.Ly_um * np.round(raw_dy / p.Ly_um)
        else:
            ddx, ddy = raw_dx, raw_dy
        dd = np.sqrt(ddx * ddx + ddy * ddy)
        req = radii[pi] + radii[pj] + gap[pi, pj]
        mm = dd - req
        keep = np.argsort(mm)[:max_constraints]
        pairs = pairs[keep]

    pair_i = pairs[:, 0].astype(int)
    pair_j = pairs[:, 1].astype(int)
    raw_dx = centers[pair_i, 0] - centers[pair_j, 0]
    raw_dy = centers[pair_i, 1] - centers[pair_j, 1]
    if p.use_periodic_distance:
        shift_x = np.round(raw_dx / p.Lx_um).astype(int)
        shift_y = np.round(raw_dy / p.Ly_um).astype(int)
    else:
        shift_x = np.zeros_like(raw_dx, dtype=int)
        shift_y = np.zeros_like(raw_dy, dtype=int)
    required_active = radii[pair_i] + radii[pair_j] + gap[pair_i, pair_j]
    return pair_i, pair_j, required_active.astype(float), shift_x.astype(int), shift_y.astype(int)


def solve_dccp_with_periodic_pair_list(
    centers0: np.ndarray,
    radii: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    required: np.ndarray,
    shift_x: np.ndarray,
    shift_y: np.ndarray,
    p: RVEParams,
    cp_module,
    dccp_module,
) -> Tuple[np.ndarray, object, str]:
    """Solve one active DCCP subproblem with fixed periodic image shifts."""
    cp = cp_module
    dccp = dccp_module
    n = len(radii)
    C = cp.Variable((n, 2), name='fiber_centers')
    centers0 = prepare_dccp_initial_value(centers0, radii, p)
    C.value = centers0

    constraints = []
    boundary_mode = str(p.dccp_boundary_mode).lower()
    if boundary_mode == 'inside':
        constraints += [C[:, 0] >= radii, C[:, 0] <= p.Lx_um - radii]
        constraints += [C[:, 1] >= radii, C[:, 1] <= p.Ly_um - radii]
    elif boundary_mode == 'cell':
        constraints += [C[:, 0] >= 0.0, C[:, 0] <= p.Lx_um]
        constraints += [C[:, 1] >= 0.0, C[:, 1] <= p.Ly_um]
    else:
        raise ValueError("dccp_boundary_mode must be 'cell' or 'inside'.")

    batch_size = max(1, int(getattr(p, 'dccp_constraint_batch_size', 6000)))
    use_vector = bool(getattr(p, 'dccp_vectorized_constraints', True))

    if use_vector:
        for start in range(0, len(pair_i), batch_size):
            end = min(start + batch_size, len(pair_i))
            bi = pair_i[start:end]
            bj = pair_j[start:end]
            req = required[start:end]
            sx = shift_x[start:end].astype(float)
            sy = shift_y[start:end].astype(float)
            ex = C[bi, 0] - C[bj, 0] - sx * p.Lx_um
            ey = C[bi, 1] - C[bj, 1] - sy * p.Ly_um
            # Use explicit reshape order to avoid CVXPY future warnings.
            diff = cp.hstack([
                cp.reshape(ex, (end - start, 1), order='C'),
                cp.reshape(ey, (end - start, 1), order='C'),
            ])
            constraints.append(cp.norm(diff, 2, axis=1) >= req)
    else:
        for i, j, req, sx, sy in zip(pair_i, pair_j, required, shift_x, shift_y):
            constraints.append(
                cp.norm(
                    cp.hstack([
                        C[int(i), 0] - C[int(j), 0] - float(sx) * p.Lx_um,
                        C[int(i), 1] - C[int(j), 1] - float(sy) * p.Ly_um,
                    ]),
                    2,
                ) >= float(req)
            )

    # Keep this small. A strong regularization simply preserves the warm start
    # and makes DCCP look ineffective for pair statistics.
    objective = cp.Minimize(float(getattr(p, 'dccp_regularization', 1e-8)) * cp.sum_squares(C - centers0))
    problem = cp.Problem(objective, constraints)

    if not dccp.is_dccp(problem):
        if use_vector:
            old_flag = p.dccp_vectorized_constraints
            p.dccp_vectorized_constraints = False
            try:
                return solve_dccp_with_periodic_pair_list(
                    centers0, radii, pair_i, pair_j, required, shift_x, shift_y, p, cp, dccp
                )
            finally:
                p.dccp_vectorized_constraints = old_flag
        raise RuntimeError('The periodic-active DCCP subproblem is not recognized as DCCP.')

    solver = get_cvxpy_solver(p, cp)
    solver_kwargs = {}
    if str(p.dccp_solver).upper() == 'SCS':
        solver_kwargs.update({
            'max_iters': int(getattr(p, 'dccp_solver_max_iters', 30000)),
            'eps': float(getattr(p, 'dccp_solver_eps', 1e-4)),
        })

    try:
        result = problem.solve(
            method='dccp',
            solver=solver,
            ccp_times=int(getattr(p, 'dccp_ccp_times', 1)),
            max_iter=int(getattr(p, 'dccp_max_iter', 60)),
            tau=float(getattr(p, 'dccp_tau', 0.005)),
            mu=float(getattr(p, 'dccp_mu', 1.2)),
            tau_max=float(getattr(p, 'dccp_tau_max', 1.0e8)),
            verbose=bool(getattr(p, 'dccp_verbose', False)),
            **solver_kwargs,
        )
    except TypeError:
        result = problem.solve(
            method='dccp',
            solver=solver,
            max_iter=int(getattr(p, 'dccp_max_iter', 60)),
            tau=float(getattr(p, 'dccp_tau', 0.005)),
            mu=float(getattr(p, 'dccp_mu', 1.2)),
            verbose=bool(getattr(p, 'dccp_verbose', False)),
            **solver_kwargs,
        )

    if C.value is None:
        raise RuntimeError('DCCP failed to return fiber coordinates.')

    centers = np.asarray(C.value, dtype=float)
    centers[:, 0] = np.mod(centers[:, 0], p.Lx_um)
    centers[:, 1] = np.mod(centers[:, 1], p.Ly_um)
    if boundary_mode == 'inside':
        centers[:, 0] = np.clip(centers[:, 0], radii, p.Lx_um - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, p.Ly_um - radii)

    return centers, result, str(problem.status)


def paper_step1_dccp_nonperiodic_optimization(
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    warm_start_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Periodic-active DCCP STEP 1 override.

    It is still not strict 3x3 DCCP. Instead, it uses a fixed minimum-image
    shift for only the active/near-neighbour pairs in each outer iteration. This
    is the key compromise: DCCP now handles the pairs that caused the later
    periodic polish to change the layout, without exploding to 9*n(n-1)/2
    constraints.
    """
    try:
        import cvxpy as cp
        import dccp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'DCCP mode requires cvxpy and dccp. Install them with:\n'
            '    pip install cvxpy dccp\n'
            'or:\n'
            '    conda install -c conda-forge cvxpy\n'
            '    pip install dccp'
        ) from exc

    import dccp

    n = len(radii)
    all_pair_count = n * (n - 1) // 2
    print('STEP 1: periodic-active DCCP packing optimization starts...')
    print('  DCCP boundary mode       = %s' % str(p.dccp_boundary_mode))
    print('  number of fibers         = %d' % n)
    print('  all pair constraints     = %d' % all_pair_count)
    print('  DCCP pair strategy       = %s' % str(getattr(p, 'dccp_pair_strategy', 'active')))
    print('  periodic-active DCCP     = %s' % str(getattr(p, 'dccp_use_periodic_active_constraints', True)))

    # For pair statistics, a fully-relaxed warm start is too close to the old
    # relaxation result. Use random start by default, or use a very loose warm
    # start only if the user disables dccp_random_start_if_active.
    if warm_start_centers is None or bool(getattr(p, 'dccp_random_start_if_active', True)):
        centers_current = random_initial_centers(p, n, rng)
    else:
        centers_current = warm_start_centers.copy()
    centers_current = prepare_dccp_initial_value(centers_current, radii, p)

    strategy = str(getattr(p, 'dccp_pair_strategy', 'active')).lower()
    forced_pairs = set()

    if strategy == 'all':
        ii, jj = pair_indices(n)
        raw_dx = centers_current[ii, 0] - centers_current[jj, 0]
        raw_dy = centers_current[ii, 1] - centers_current[jj, 1]
        if p.use_periodic_distance:
            sx = np.round(raw_dx / p.Lx_um).astype(int)
            sy = np.round(raw_dy / p.Ly_um).astype(int)
        else:
            sx = np.zeros_like(raw_dx, dtype=int)
            sy = np.zeros_like(raw_dy, dtype=int)
        required = radii[ii] + radii[jj] + gap[ii, jj]
        print('  active constraints       = %d (all base pairs with fixed periodic shifts)' % len(ii))
        centers_current, result, status = solve_dccp_with_periodic_pair_list(
            centers_current, radii, ii.astype(int), jj.astype(int), required, sx, sy, p, cp, dccp
        )
        print('  DCCP result/status       = %s / %s' % (str(result), str(status)))

    elif strategy == 'active':
        for outer in range(1, int(getattr(p, 'dccp_active_outer_iter', 3)) + 1):
            pair_i, pair_j, required, sx, sy = build_dccp_active_pairs_periodic(
                centers_current, radii, gap, p, forced_pairs=forced_pairs
            )
            print(
                '  active outer %d/%d | active constraints = %d'
                % (outer, int(getattr(p, 'dccp_active_outer_iter', 3)), len(pair_i))
            )
            centers_current, result, status = solve_dccp_with_periodic_pair_list(
                centers_current, radii, pair_i, pair_j, required, sx, sy, p, cp, dccp
            )
            print('    DCCP result/status     = %s / %s' % (str(result), str(status)))

            ii, jj, d, req_all, margin, sx_all, sy_all = compute_periodic_pair_margins_and_shifts(
                centers_current, radii, gap, p
            )
            bad_idx = np.where(margin < -float(p.tol_um))[0]
            near_idx = np.where(margin < float(getattr(p, 'dccp_active_add_margin_um', 0.25)))[0]
            print(
                '    periodic full check | bad pairs = %d | min margin = %.6e'
                % (len(bad_idx), float(np.min(margin)))
            )

            add_idx = np.unique(np.concatenate([bad_idx, near_idx]))
            if len(add_idx) > int(getattr(p, 'dccp_max_active_constraints', 45000)):
                order = np.argsort(margin[add_idx])[:int(getattr(p, 'dccp_max_active_constraints', 45000))]
                add_idx = add_idx[order]
            for idx in add_idx:
                forced_pairs.add((int(ii[idx]), int(jj[idx])))

            if len(bad_idx) == 0:
                break
    else:
        raise ValueError("dccp_pair_strategy must be 'active' or 'all'.")

    return centers_current


def pair_aware_local_shuffle(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Legal local shuffle that directly targets the first-shell and medium-range
    peaks in g(h), without recomputing the full pair histogram at every move.

    This is generation-stage optimization, not plot smoothing. It only accepts
    moves that keep all hard constraints valid and reduce a local shell score,
    with a small simulated-annealing probability for exploration.
    """
    if not bool(getattr(p, 'use_pair_aware_shuffle', False)):
        return centers

    centers = centers.copy()
    n = len(radii)
    r_mean = float(np.mean(radii))
    steps = int(float(getattr(p, 'pair_aware_steps_per_fiber', 80)) * n)
    move_amp = float(getattr(p, 'pair_aware_move_amp_um', 0.010))
    t0 = float(getattr(p, 'pair_aware_temperature_start', 0.04))
    t1 = float(getattr(p, 'pair_aware_temperature_end', 0.003))

    # Continuous penalties around the peaks commonly produced by dense, nearly
    # monodisperse hard-core layouts. The first shell is weighted most strongly.
    shell_centers = np.array([2.18, 4.10, 5.70, 7.40], dtype=float)
    shell_widths = np.array([0.22, 0.32, 0.38, 0.45], dtype=float)
    shell_weights = np.array([1.00, 0.45, 0.30, 0.20], dtype=float)

    def local_score(k: int, xy: np.ndarray) -> float:
        dx = xy[0] - centers[:, 0]
        dy = xy[1] - centers[:, 1]
        dx, dy = min_image_components(dx, dy, p)
        d = np.sqrt(dx * dx + dy * dy)
        d[k] = np.inf
        h = d / r_mean
        score = 0.0
        for hc, hw, ww in zip(shell_centers, shell_widths, shell_weights):
            score += ww * float(np.sum(np.exp(-((h - hc) / hw) ** 2)))
        return score

    accept = 0
    improve = 0
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        temp = t0 * (1.0 - frac) + t1 * frac
        k = int(rng.integers(0, n))
        old_xy = centers[k].copy()
        old_score = local_score(k, old_xy)

        trial = old_xy + rng.normal(0.0, move_amp, 2)
        trial[0] = trial[0] % p.Lx_um
        trial[1] = trial[1] % p.Ly_um
        centers[k] = trial

        if not is_one_fiber_valid(k, centers, radii, gap, p):
            centers[k] = old_xy
            continue

        new_score = local_score(k, trial)
        delta = new_score - old_score
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1.0e-12)):
            accept += 1
            if delta <= 0.0:
                improve += 1
        else:
            centers[k] = old_xy

    print(
        'Pair-aware local shuffle: steps = %d, accept ratio = %.4f, improve moves = %d'
        % (steps, accept / max(steps, 1), improve)
    )
    return centers


# Override generate_one_candidate so pair-aware local shuffle is applied after
# DCCP/reorientation and before the final validity check.
def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    method = str(p.step1_method).lower()

    if method == 'relaxation':
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ('dccp', 'hybrid_dccp'):
        warm = None
        if p.dccp_use_relaxation_warm_start and not bool(getattr(p, 'dccp_random_start_if_active', True)):
            warm = random_initial_centers(p, len(radii), rng)
            warm = paper_step1_initial_optimization(warm, radii, gap, p, rng)

        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == 'hybrid_dccp':
                print('  Warning: periodic-active DCCP failed in hybrid_dccp mode. Falling back to relaxation.')
                print('  DCCP error:', repr(exc))
                centers = random_initial_centers(p, len(radii), rng)
                centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)
            else:
                raise

        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)

    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    centers = paper_step2_reorientation(centers, radii, gap, p, rng)
    centers = monte_carlo_shuffle(centers, radii, gap, p, rng)
    centers = pair_aware_local_shuffle(centers, radii, gap, p, rng)

    # Final hard polish to remove tiny violations introduced by stochastic moves.
    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=int(getattr(p, 'final_hard_polish_iter', 1500)),
        shake_amp=0.0,
        move_limit_um=float(p.opt_final_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics


def build_user_params() -> RVEParams:
    params = _original_build_user_params()

    # Recommended debugging setting: first judge DCCP + periodic-active +
    # pair-aware shuffle itself. Avoid strong legal MC, which can wash out DCCP.
    params.step1_method = 'hybrid_dccp'
    params.dccp_pair_strategy = 'active'
    params.dccp_use_periodic_active_constraints = True
    params.dccp_random_start_if_active = True
    params.dccp_regularization = 1.0e-8
    params.dccp_active_outer_iter = 4
    params.dccp_active_nearest_k = 24
    params.dccp_active_margin_um = 1.20
    params.dccp_active_add_margin_um = 0.50
    params.dccp_max_active_constraints = 65000
    params.dccp_constraint_batch_size = 5000
    params.dccp_post_periodic_polish = True
    params.dccp_post_polish_iter = 1200
    params.dccp_post_polish_move_limit_um = 0.035

    # Do not let a long legal random walk erase the DCCP structure while testing.
    params.use_mc_shuffle = False

    # Keep reorientation mild. The pair-aware shuffle below handles shell peaks.
    params.use_reorientation = True
    params.reorient_passes = 1
    params.reorient_trials_per_fiber = 35

    # Directly target the pair curve in the generation stage.
    params.use_pair_aware_shuffle = True
    params.pair_aware_steps_per_fiber = 80
    params.pair_aware_move_amp_um = 0.010
    params.pair_aware_temperature_start = 0.035
    params.pair_aware_temperature_end = 0.003
    params.final_hard_polish_iter = 1500

    # More candidates matter more than forcing one DCCP solve to be perfect.
    params.valid_candidates_to_test = 3
    params.max_restarts = 80
    params.selection_mode = 'best_valid_randomness'
    params.tol_um = max(float(params.tol_um), 1.0e-6)
    params.output_prefix = 'fiber_centers_periodic_active_dccp_pairaware_vf70'
    return params



# ============================================================
# Paper-style statistics + stronger cluster re-orientation override
# ============================================================

def _edge_fraction_in_rectangle(x: float, y: float, h: float, p: RVEParams, angles_cache: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Approximate wk in the paper's Ripley-K edge correction.

    wk is the fraction of the circumference of the circle centered at (x,y)
    with radius h that lies inside the rectangular RVE. The paper uses this
    non-periodic edge correction rather than a periodic minimum-image metric.
    """
    if h <= 1.0e-14:
        return 1.0
    n_ang = int(getattr(p, 'edge_correction_angle_samples', 720))
    if n_ang not in angles_cache:
        theta = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
        angles_cache[n_ang] = (np.cos(theta), np.sin(theta))
    ct, st = angles_cache[n_ang]
    xx = x + h * ct
    yy = y + h * st
    inside = (xx >= 0.0) & (xx <= p.Lx_um) & (yy >= 0.0) & (yy <= p.Ly_um)
    return max(float(np.mean(inside)), 1.0e-6)


def evaluate_ripley_k_paper_edge(
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
    h_min_over_r: Optional[float] = None,
    h_max_over_r: Optional[float] = None,
    n_h: Optional[int] = None,
) -> pd.DataFrame:
    """
    Paper-style Ripley's K function.

    Important difference from the earlier implementation:
    - use ordinary Euclidean distance inside the base RVE;
    - do NOT use periodic minimum-image distance for the statistics;
    - apply the circumference fraction edge correction wk, as written in the paper.

    This is intended to reproduce Fig. 10(c,d) more closely. The generated
    geometry can still be periodic for Abaqus, but the statistical post-processing
    follows the paper's edge-corrected definition.
    """
    n = len(centers)
    area = p.Lx_um * p.Ly_um
    r_mean = float(np.mean(radii))
    if h_min_over_r is None:
        h_min_over_r = float(getattr(p, 'paper_pair_h_min_over_r', 0.05))
    if h_max_over_r is None:
        h_max_over_r = float(getattr(p, 'pair_h_max_over_r', 15.0))
    if n_h is None:
        n_h = int(getattr(p, 'paper_pair_n_h', 160))

    h_min = max(0.0, h_min_over_r * r_mean)
    h_max = min(h_max_over_r * r_mean, math.sqrt(p.Lx_um * p.Lx_um + p.Ly_um * p.Ly_um))
    h_values = np.linspace(h_min, h_max, n_h)

    # Raw, non-periodic pair distances in the base square.
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)

    angles_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    K_values = []
    for h in h_values:
        if h <= 1.0e-14:
            K_values.append(0.0)
            continue
        count_i = np.sum(dist <= h, axis=1).astype(float)
        wk = np.array([
            _edge_fraction_in_rectangle(float(c[0]), float(c[1]), float(h), p, angles_cache)
            for c in centers
        ], dtype=float)
        # Paper formula: K(h)=A/N^2 * sum_k I_k(h)/w_k.
        K_values.append(area * float(np.sum(count_i / wk)) / float(n * n))

    K_values = np.asarray(K_values, dtype=float)
    K_csr = math.pi * h_values ** 2
    return pd.DataFrame({
        'h_um': h_values,
        'h_over_r': h_values / r_mean,
        'K_algorithm': K_values,
        'K_CSR': K_csr,
        'K_minus_KCSR': K_values - K_csr,
    })


def evaluate_pair_distribution_paper_from_k(df_k: pd.DataFrame, p: RVEParams) -> pd.DataFrame:
    """Paper-style pair distribution: g(h)=1/(2*pi*h)*dK/dh, with optional smoothing."""
    h = df_k['h_um'].values.astype(float)
    K = df_k['K_algorithm'].values.astype(float)
    dK_dh = np.gradient(K, h, edge_order=1)
    g = np.zeros_like(h)
    mask = h > 1.0e-12
    g[mask] = dK_dh[mask] / (2.0 * math.pi * h[mask])
    g = np.maximum(g, 0.0)

    win = int(getattr(p, 'paper_pair_smooth_window', 5))
    if win > 1:
        g_plot = pd.Series(g).rolling(window=win, center=True, min_periods=1).mean().values
    else:
        g_plot = g

    return pd.DataFrame({
        'h_um': h,
        'h_over_r': df_k['h_over_r'].values,
        'g_raw': g,
        'g_algorithm': g_plot,
        'g_CSR': np.ones_like(g),
        'g_minus_1': g_plot - 1.0,
    })


def _pair_curve_score_from_df(df_g: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    g_values = df_g['g_algorithm'].values.astype(float)
    h_values = df_g['h_over_r'].values.astype(float)
    # Ignore the physically empty exclusion zone below h/r=2 when scoring.
    valid = h_values >= 2.0
    if not np.any(valid):
        valid = np.ones_like(h_values, dtype=bool)
    first_mask = (h_values >= 2.0) & (h_values <= 2.8)
    mid_mask = (h_values >= 3.0) & (h_values <= 8.0)
    far_mask = h_values >= 8.0
    first_peak = float(np.max(g_values[first_mask])) if np.any(first_mask) else float(np.max(g_values[valid]))
    mid_peak = float(np.max(g_values[mid_mask])) if np.any(mid_mask) else first_peak
    rmse = math.sqrt(float(np.mean((g_values[valid] - 1.0) ** 2)))
    mid_rmse = math.sqrt(float(np.mean((g_values[mid_mask] - 1.0) ** 2))) if np.any(mid_mask) else rmse
    far_rmse = math.sqrt(float(np.mean((g_values[far_mask] - 1.0) ** 2))) if np.any(far_mask) else rmse
    score = 0.55 * rmse + 1.00 * mid_rmse + 0.65 * far_rmse + 0.55 * max(0.0, first_peak - 2.2) + 0.45 * max(0.0, mid_peak - 1.5)
    return score, {
        'score': score,
        'pair_distribution_RMSE_paper': rmse,
        'pair_distribution_mid_RMSE_paper': mid_rmse,
        'pair_distribution_far_RMSE_paper': far_rmse,
        'first_pair_distribution_peak': first_peak,
        'mid_pair_distribution_peak': mid_peak,
    }


def layout_randomness_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """Candidate score aligned with the paper-style pair distribution when enabled."""
    if bool(getattr(p, 'use_paper_style_statistics', True)):
        df_k = evaluate_ripley_k_paper_edge(centers, radii, p)
        df_g = evaluate_pair_distribution_paper_from_k(df_k, p)
        score, info = _pair_curve_score_from_df(df_g)
        df_ori = evaluate_nearest_neighbor_orientation(centers, p)
        orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
        contact = contact_peak_diagnostics(centers, radii, p)
        score += 0.25 * orientation_ks + 15.0 * contact['contact_pair_fraction']
        info.update({
            'orientation_CDF_KS_distance': orientation_ks,
            'contact_pair_fraction': contact['contact_pair_fraction'],
            'min_h_over_r': contact['min_h_over_r'],
            # Backward-compatible key used by generate() print statement.
            'pair_distribution_RMSE_shell': info['pair_distribution_RMSE_paper'],
        })
        return score, info

    # Fallback to the earlier shell-based score.
    df_g = evaluate_pair_distribution_shell(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    contact = contact_peak_diagnostics(centers, radii, p)
    g_values = df_g['g_algorithm'].values
    h_values = df_g['h_over_r'].values
    g_rmse = math.sqrt(np.mean((g_values - 1.0) ** 2))
    orientation_ks = np.max(np.abs(df_ori['cdf_difference'].values))
    contact_fraction = contact['contact_pair_fraction']
    first_peak = float(g_values[0])
    far_mask = h_values >= 3.0
    far_rmse = math.sqrt(np.mean((g_values[far_mask] - 1.0) ** 2)) if np.any(far_mask) else g_rmse
    score = 0.50 * g_rmse + 0.80 * far_rmse + 0.25 * orientation_ks + 0.25 * max(0.0, first_peak - 2.2) + 20.0 * contact_fraction
    return score, {
        'score': score,
        'pair_distribution_RMSE_shell': g_rmse,
        'pair_distribution_far_RMSE_shell': far_rmse,
        'orientation_CDF_KS_distance': orientation_ks,
        'contact_pair_fraction': contact_fraction,
        'first_pair_distribution_peak': first_peak,
        'min_h_over_r': contact['min_h_over_r'],
    }


def evaluate_randomness(centers: np.ndarray, radii: np.ndarray, p: RVEParams, output_prefix: str):
    """Evaluation override: save both stable shell and paper-style edge-corrected g(h)."""
    print('\n' + '=' * 80)
    print('Randomness evaluation starts...')
    print('=' * 80)

    df_nn = evaluate_nearest_neighbor_distance(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    df_k_periodic = evaluate_ripley_k(centers, radii, p)
    df_g_shell = evaluate_pair_distribution_shell(centers, radii, p)

    df_k_paper = evaluate_ripley_k_paper_edge(centers, radii, p)
    df_g_paper = evaluate_pair_distribution_paper_from_k(df_k_paper, p)

    contact_diag = contact_peak_diagnostics(centers, radii, p)
    orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
    k_rmse_paper = math.sqrt(np.mean(((df_k_paper['K_algorithm'].values - df_k_paper['K_CSR'].values) / np.maximum(df_k_paper['K_CSR'].values, 1.0e-12)) ** 2))
    g_rmse_paper = math.sqrt(np.mean((df_g_paper.loc[df_g_paper['h_over_r'] >= 2.0, 'g_algorithm'].values - 1.0) ** 2))
    g_rmse_shell = math.sqrt(np.mean((df_g_shell['g_algorithm'].values - 1.0) ** 2))

    score, score_info = _pair_curve_score_from_df(df_g_paper)
    summary = pd.DataFrame([{
        'fiber_number': len(centers),
        'mean_radius_um': float(np.mean(radii)),
        'nearest_1_mean_h_over_r': df_nn['nearest_1_h_over_r'].mean(),
        'nearest_1_std_h_over_r': df_nn['nearest_1_h_over_r'].std(),
        'nearest_2_mean_h_over_r': df_nn['nearest_2_h_over_r'].mean(),
        'nearest_3_mean_h_over_r': df_nn['nearest_3_h_over_r'].mean(),
        'orientation_CDF_KS_distance': orientation_ks,
        'Ripley_K_relative_RMSE_paper_edge': k_rmse_paper,
        'pair_distribution_RMSE_paper_edge': g_rmse_paper,
        'pair_distribution_RMSE_shell_stable_periodic': g_rmse_shell,
        'first_pair_distribution_peak_paper_edge': score_info['first_pair_distribution_peak'],
        'mid_pair_distribution_peak_paper_edge': score_info['mid_pair_distribution_peak'],
        'contact_pair_fraction': contact_diag['contact_pair_fraction'],
        'contact_pair_count': contact_diag['contact_pair_count'],
        'min_h_over_r': contact_diag['min_h_over_r'],
    }])

    plot_nearest_neighbor_distance(df_nn, output_prefix + '_eval_nearest_distance.png')
    plot_nearest_neighbor_orientation(df_ori, output_prefix + '_eval_orientation_cdf.png')
    plot_ripley_k(df_k_paper, output_prefix + '_eval_ripley_k.png')
    plot_pair_distribution(df_g_paper, output_prefix + '_eval_pair_distribution_from_K.png')
    plot_pair_distribution(df_g_paper, output_prefix + '_eval_pair_distribution_paper_edge.png')
    plot_pair_distribution(df_g_shell, output_prefix + '_eval_pair_distribution_shell_periodic.png')

    eval_xlsx = output_prefix + '_randomness_evaluation.xlsx'
    with pd.ExcelWriter(eval_xlsx, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        df_nn.to_excel(writer, sheet_name='nearest_distance', index=False)
        df_ori.to_excel(writer, sheet_name='orientation_cdf', index=False)
        df_k_paper.to_excel(writer, sheet_name='ripley_k_paper_edge', index=False)
        df_g_paper.to_excel(writer, sheet_name='pair_distribution_paper', index=False)
        df_k_periodic.to_excel(writer, sheet_name='ripley_k_periodic_old', index=False)
        df_g_shell.to_excel(writer, sheet_name='pair_distribution_shell', index=False)

    print('Randomness evaluation finished.')
    print('Evaluation file saved:', eval_xlsx)
    print('orientation CDF KS distance       = %.6e' % orientation_ks)
    print('Ripley K relative RMSE paper-edge = %.6e' % k_rmse_paper)
    print('Pair distribution RMSE paper-edge = %.6e' % g_rmse_paper)
    print('Pair distribution RMSE shell      = %.6e' % g_rmse_shell)
    print('first g(h) peak paper-edge        = %.6e' % score_info['first_pair_distribution_peak'])
    print('contact pair fraction             = %.6e' % contact_diag['contact_pair_fraction'])
    print('min h/r                           = %.6e' % contact_diag['min_h_over_r'])
    print('=' * 80)
    return summary, df_nn, df_ori, df_k_paper, df_g_paper


def _local_shell_score_for_fiber(k: int, xy: np.ndarray, centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> float:
    """Local shell score used for generation-stage pair peak reduction."""
    r_mean = float(np.mean(radii))
    dx = xy[0] - centers[:, 0]
    dy = xy[1] - centers[:, 1]
    dx, dy = min_image_components(dx, dy, p)
    d = np.sqrt(dx * dx + dy * dy)
    d[k] = np.inf
    h = d / r_mean
    # Strongly penalize the first histogram bin [2.0, 2.35], because this is
    # the source of the very high initial peak in the user's curve.
    near_count = float(np.sum((h >= 2.0) & (h < float(getattr(p, 'cluster_first_shell_high', 2.35)))))
    score = float(getattr(p, 'cluster_first_shell_weight', 3.0)) * near_count
    shell_centers = np.array([2.08, 2.35, 4.10, 5.70, 7.40], dtype=float)
    shell_widths = np.array([0.10, 0.20, 0.32, 0.38, 0.45], dtype=float)
    shell_weights = np.array([2.20, 1.20, 0.45, 0.30, 0.20], dtype=float)
    for hc, hw, ww in zip(shell_centers, shell_widths, shell_weights):
        score += ww * float(np.sum(np.exp(-((h - hc) / hw) ** 2)))
    return score


def pair_peak_reorientation(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stronger Step-2 style re-orientation driven by pair-distribution peaks.

    This follows the paper's idea more closely than tiny local moves: identify
    fibers contributing to clustering/first-shell peaks, then try to relocate
    them to resin-rich candidate positions. A move is accepted only if it remains
    feasible and lowers the local shell score.
    """
    if not bool(getattr(p, 'use_pair_peak_reorientation', True)):
        return centers

    centers = centers.copy()
    n = len(radii)
    passes = int(getattr(p, 'pair_peak_reorient_passes', 2))
    trials = int(getattr(p, 'pair_peak_reorient_trials_per_fiber', 120))
    top_fraction = float(getattr(p, 'pair_peak_reorient_top_fraction', 0.35))
    local_sigma = float(getattr(p, 'pair_peak_reorient_local_sigma_factor', 6.0)) * float(np.mean(radii))

    print('STEP 2b: pair-peak resin-rich re-orientation starts...')
    for pass_id in range(1, passes + 1):
        # Rank fibers by first-shell/local shell score.
        scores = np.array([
            _local_shell_score_for_fiber(k, centers[k], centers, radii, p)
            for k in range(n)
        ])
        n_target = max(1, int(top_fraction * n))
        target_ids = np.argsort(scores)[-n_target:]
        target_ids = rng.permutation(target_ids)

        resin_points = estimate_resin_rich_points(
            centers, radii, p, rng,
            n_points=max(2000, 12 * n),
            keep=max(80, n // 4),
        )

        moved = 0
        for k in target_ids:
            k = int(k)
            old_xy = centers[k].copy()
            old_score = _local_shell_score_for_fiber(k, old_xy, centers, radii, p)
            best_xy = old_xy.copy()
            best_score = old_score
            best_margin = single_fiber_min_margin(k, old_xy, centers, radii, gap, p)

            for _ in range(trials):
                if len(resin_points) > 0 and rng.random() < 0.80:
                    base = resin_points[int(rng.integers(0, len(resin_points)))]
                    trial = base + rng.normal(0.0, local_sigma, 2)
                else:
                    # Occasional moderate jump around the current point.
                    trial = old_xy + rng.normal(0.0, 2.5 * float(np.mean(radii)), 2)
                trial[0] = trial[0] % p.Lx_um
                trial[1] = trial[1] % p.Ly_um

                margin = single_fiber_min_margin(k, trial, centers, radii, gap, p)
                if margin < -p.tol_um:
                    continue
                score = _local_shell_score_for_fiber(k, trial, centers, radii, p)
                # Prefer lower shell peak; use margin as tie-breaker.
                if score < best_score - 1.0e-9 or (abs(score - best_score) < 1.0e-9 and margin > best_margin):
                    best_xy = trial.copy()
                    best_score = score
                    best_margin = margin

            if best_score < old_score - float(getattr(p, 'pair_peak_min_improve', 0.20)):
                centers[k] = best_xy
                moved += 1

        metrics = compute_metrics(centers, radii, gap, p)
        print(
            '  pass %d/%d | moved = %d | min_margin = %.6e | max_violation = %.6e'
            % (pass_id, passes, moved, metrics['min_constraint_margin'], metrics['max_violation'])
        )
        if metrics['max_violation'] > p.tol_um:
            centers = packing_relax_one_stage(
                centers, radii, gap, p, rng,
                alpha=1.0,
                n_iter=1000,
                shake_amp=0.0,
                move_limit_um=p.opt_final_move_limit_um,
            )
    return centers


def pair_aware_local_shuffle(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stronger legal local shuffle for fine adjustment after resin-rich reorientation."""
    if not bool(getattr(p, 'use_pair_aware_shuffle', False)):
        return centers
    centers = centers.copy()
    n = len(radii)
    steps = int(float(getattr(p, 'pair_aware_steps_per_fiber', 120)) * n)
    move_amp = float(getattr(p, 'pair_aware_move_amp_um', 0.014))
    t0 = float(getattr(p, 'pair_aware_temperature_start', 0.025))
    t1 = float(getattr(p, 'pair_aware_temperature_end', 0.0015))

    accept = 0
    improve = 0
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        temp = t0 * (1.0 - frac) + t1 * frac
        k = int(rng.integers(0, n))
        old_xy = centers[k].copy()
        old_score = _local_shell_score_for_fiber(k, old_xy, centers, radii, p)

        # Mostly small moves; sometimes a slightly larger move to escape local cages.
        scale = move_amp if rng.random() < 0.90 else 3.0 * move_amp
        trial = old_xy + rng.normal(0.0, scale, 2)
        trial[0] = trial[0] % p.Lx_um
        trial[1] = trial[1] % p.Ly_um
        centers[k] = trial

        if not is_one_fiber_valid(k, centers, radii, gap, p):
            centers[k] = old_xy
            continue
        new_score = _local_shell_score_for_fiber(k, trial, centers, radii, p)
        delta = new_score - old_score
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1.0e-12)):
            accept += 1
            if delta <= 0.0:
                improve += 1
        else:
            centers[k] = old_xy
    print(
        'Pair-aware local shuffle: steps = %d, accept ratio = %.4f, improve moves = %d'
        % (steps, accept / max(steps, 1), improve)
    )
    return centers


def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    method = str(p.step1_method).lower()

    if method == 'relaxation':
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ('dccp', 'hybrid_dccp'):
        warm = None
        if p.dccp_use_relaxation_warm_start and not bool(getattr(p, 'dccp_random_start_if_active', True)):
            warm = random_initial_centers(p, len(radii), rng)
            warm = paper_step1_initial_optimization(warm, radii, gap, p, rng)
        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == 'hybrid_dccp':
                print('  Warning: periodic-active DCCP failed in hybrid_dccp mode. Falling back to relaxation.')
                print('  DCCP error:', repr(exc))
                centers = random_initial_centers(p, len(radii), rng)
                centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)
            else:
                raise
        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)
    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    # This order matters: first make global resin-rich relocations for peak fibers,
    # then a mild paper-like reorientation, then fine pair-aware local moves.
    centers = pair_peak_reorientation(centers, radii, gap, p, rng)
    centers = paper_step2_reorientation(centers, radii, gap, p, rng)
    centers = monte_carlo_shuffle(centers, radii, gap, p, rng)
    centers = pair_aware_local_shuffle(centers, radii, gap, p, rng)

    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=int(getattr(p, 'final_hard_polish_iter', 1500)),
        shake_amp=0.0,
        move_limit_um=float(p.opt_final_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics


# Override parameter builder again: use paper-style statistics and stronger peak reduction.
_previous_build_user_params_pair_paper = build_user_params


def build_user_params() -> RVEParams:
    params = _previous_build_user_params_pair_paper()

    # Match the paper's statistical definition: ordinary distance + edge-corrected K.
    params.use_paper_style_statistics = True
    params.paper_pair_h_min_over_r = 0.05
    params.paper_pair_n_h = 180
    params.paper_pair_smooth_window = 5
    params.edge_correction_angle_samples = 720

    # Stronger Step-2 reorientation aimed at reducing first-shell pair peak.
    params.use_pair_peak_reorientation = True
    params.pair_peak_reorient_passes = 2
    params.pair_peak_reorient_trials_per_fiber = 140
    params.pair_peak_reorient_top_fraction = 0.40
    params.pair_peak_reorient_local_sigma_factor = 7.0
    params.pair_peak_min_improve = 0.10
    params.cluster_first_shell_high = 2.35
    params.cluster_first_shell_weight = 4.0

    # Fine local shuffle after global relocations.
    params.use_pair_aware_shuffle = True
    params.pair_aware_steps_per_fiber = 160
    params.pair_aware_move_amp_um = 0.014
    params.pair_aware_temperature_start = 0.025
    params.pair_aware_temperature_end = 0.0015

    # Do not wash out the DCCP/reorientation result with ordinary random MC.
    params.use_mc_shuffle = False

    # Slightly more candidates because pair distribution is stochastic.
    params.valid_candidates_to_test = 4
    params.max_restarts = 100
    params.output_prefix = 'fiber_centers_periodic_active_dccp_paperstyle_pair_vf70'
    return params


# ============================================================
# V4 override: paper-faithful balance re-orientation
# ============================================================

def _local_density_balance_score(
    k: int,
    xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
) -> float:
    """
    Local density score used by V4.

    The previous V3 focused too strongly on escaping the first shell. That can
    reduce the first g(h) peak but may worsen Ripley's K by creating large-scale
    inhomogeneity. This score therefore penalizes both lower-bound neighbours
    and medium-range local crowding, which is closer to the paper's Step-2 idea:
    relocate fibers from clustered zones into resin-rich zones.
    """
    r_mean = float(np.mean(radii))
    dx = xy[0] - centers[:, 0]
    dy = xy[1] - centers[:, 1]
    # Use non-periodic distance for density-balance scoring, because the paper's
    # statistical K-function is computed in the base RVE with edge correction.
    d_non = np.sqrt(dx * dx + dy * dy)
    d_non[k] = np.inf
    h_non = d_non / r_mean

    # Keep a periodic hard-core contact diagnostic too, because final geometry is
    # still intended for periodic/PBC-compatible use.
    dxp, dyp = min_image_components(dx.copy(), dy.copy(), p)
    d_per = np.sqrt(dxp * dxp + dyp * dyp)
    d_per[k] = np.inf
    h_per = d_per / r_mean

    hard_core = 2.0 + float(p.lmin_um) / r_mean
    first_high = float(getattr(p, 'balance_first_shell_high_over_r', hard_core + 0.22))
    first_count = float(np.sum((h_per >= hard_core - 0.01) & (h_per < first_high)))

    # Medium-range crowding controls Ripley's K. Gaussian kernels are smoother
    # than hard counts and work well for local candidate comparison.
    density_radii = np.asarray(getattr(p, 'balance_density_radii_over_r', [4.0, 6.0, 8.0]), dtype=float)
    density_weights = np.asarray(getattr(p, 'balance_density_weights', [1.0, 0.70, 0.45]), dtype=float)
    score = float(getattr(p, 'balance_first_shell_weight', 1.20)) * first_count
    for rr, ww in zip(density_radii, density_weights):
        score += float(ww) * float(np.sum(np.exp(-((h_non / max(rr, 1.0e-12)) ** 2))))
    return score


def _sample_resin_rich_balance_points(
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    n_points: int,
    keep: int,
) -> np.ndarray:
    """Sample candidate points with low local density and acceptable clearance."""
    pts = np.empty((n_points, 2), dtype=float)

    # If DCCP boundary is set to inside, sample within the same inner box. This
    # matches Eq. (2) in the paper and avoids excessive edge-correction artifacts.
    margin = float(np.mean(radii)) if str(getattr(p, 'dccp_boundary_mode', 'cell')).lower() == 'inside' else 0.0
    pts[:, 0] = rng.uniform(margin, p.Lx_um - margin, n_points)
    pts[:, 1] = rng.uniform(margin, p.Ly_um - margin, n_points)

    r_mean = float(np.mean(radii))
    density_radius = float(getattr(p, 'balance_candidate_density_radius_over_r', 6.0)) * r_mean
    scores = np.empty(n_points, dtype=float)

    for a in range(n_points):
        dx = pts[a, 0] - centers[:, 0]
        dy = pts[a, 1] - centers[:, 1]
        d = np.sqrt(dx * dx + dy * dy)
        clearance = float(np.min(d - radii))
        local_count = float(np.sum(d < density_radius))
        local_kernel = float(np.sum(np.exp(-((d / max(density_radius, 1.0e-12)) ** 2))))
        # Large score means good target: large resin clearance and low density.
        scores[a] = 1.00 * clearance - 0.10 * local_count - 0.35 * local_kernel

    idx = np.argsort(scores)[-min(keep, n_points):]
    return pts[idx]


def paper_balance_reorientation(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Paper-faithful Step-2 re-orientation.

    Instead of directly forcing the first pair-distribution peak down, this step
    follows the paper's logic: identify fibers in locally crowded/clustered
    zones, search random candidate positions in resin-rich/sparse zones, and
    accept only legal relocations that reduce local density imbalance.
    """
    if not bool(getattr(p, 'use_paper_balance_reorientation', True)):
        return centers

    centers = centers.copy()
    n = len(radii)
    passes = int(getattr(p, 'paper_balance_passes', 2))
    trials = int(getattr(p, 'paper_balance_trials_per_fiber', 90))
    top_fraction = float(getattr(p, 'paper_balance_top_fraction', 0.28))
    local_sigma = float(getattr(p, 'paper_balance_local_sigma_factor', 3.5)) * float(np.mean(radii))
    min_improve = float(getattr(p, 'paper_balance_min_improve', 0.08))

    print('STEP 2b: paper-style density-balance re-orientation starts...')
    for pass_id in range(1, passes + 1):
        scores = np.array([
            _local_density_balance_score(k, centers[k], centers, radii, p)
            for k in range(n)
        ])
        n_target = max(1, int(top_fraction * n))
        target_ids = np.argsort(scores)[-n_target:]
        target_ids = rng.permutation(target_ids)

        candidate_pool = _sample_resin_rich_balance_points(
            centers, radii, p, rng,
            n_points=max(3000, 10 * n),
            keep=max(120, n // 3),
        )

        moved = 0
        for k in target_ids:
            k = int(k)
            old_xy = centers[k].copy()
            old_score = _local_density_balance_score(k, old_xy, centers, radii, p)
            best_xy = old_xy.copy()
            best_score = old_score

            for _ in range(trials):
                if len(candidate_pool) > 0 and rng.random() < 0.82:
                    base = candidate_pool[int(rng.integers(0, len(candidate_pool)))]
                    trial = base + rng.normal(0.0, local_sigma, 2)
                else:
                    margin = float(np.mean(radii)) if str(getattr(p, 'dccp_boundary_mode', 'cell')).lower() == 'inside' else 0.0
                    trial = np.array([
                        rng.uniform(margin, p.Lx_um - margin),
                        rng.uniform(margin, p.Ly_um - margin),
                    ])

                trial[0] = trial[0] % p.Lx_um
                trial[1] = trial[1] % p.Ly_um
                if str(getattr(p, 'dccp_boundary_mode', 'cell')).lower() == 'inside':
                    rr = radii[k]
                    trial[0] = np.clip(trial[0], rr, p.Lx_um - rr)
                    trial[1] = np.clip(trial[1], rr, p.Ly_um - rr)

                # Temporarily test the move.
                centers[k] = trial
                if not is_one_fiber_valid(k, centers, radii, gap, p):
                    centers[k] = old_xy
                    continue
                new_score = _local_density_balance_score(k, trial, centers, radii, p)
                centers[k] = old_xy

                if new_score < best_score:
                    best_score = new_score
                    best_xy = trial.copy()

            if best_score < old_score - min_improve:
                centers[k] = best_xy
                moved += 1

        metrics = compute_metrics(centers, radii, gap, p)
        print(
            '  balance pass %d/%d | moved = %d | min_margin = %.6e | max_violation = %.6e'
            % (pass_id, passes, moved, metrics['min_constraint_margin'], metrics['max_violation'])
        )
        if metrics['max_violation'] > p.tol_um:
            centers = packing_relax_one_stage(
                centers, radii, gap, p, rng,
                alpha=1.0,
                n_iter=1000,
                shake_amp=0.0,
                move_limit_um=p.opt_final_move_limit_um,
            )

    return centers


def _paper_k_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """Combined paper-style K and g score; smaller is better."""
    df_k = evaluate_ripley_k_paper_edge(centers, radii, p)
    df_g = evaluate_pair_distribution_paper_from_k(df_k, p)
    h = df_k['h_over_r'].values.astype(float)
    K = df_k['K_algorithm'].values.astype(float)
    Kc = df_k['K_CSR'].values.astype(float)
    g = df_g['g_algorithm'].values.astype(float)
    hg = df_g['h_over_r'].values.astype(float)

    k_mask = (h >= 2.0) & (h <= 14.5)
    k_rmse = math.sqrt(np.mean(((K[k_mask] - Kc[k_mask]) / np.maximum(Kc[k_mask], 1.0e-12)) ** 2)) if np.any(k_mask) else 0.0
    # Penalize systematic large-scale clustering: K>Kcsr at h/r >= 8.
    large_mask = (h >= 8.0) & (h <= 14.5)
    k_large_pos = math.sqrt(np.mean((np.maximum((K[large_mask] - Kc[large_mask]) / np.maximum(Kc[large_mask], 1.0e-12), 0.0)) ** 2)) if np.any(large_mask) else 0.0

    g_mask = hg >= 2.0
    g_rmse = math.sqrt(np.mean((g[g_mask] - 1.0) ** 2)) if np.any(g_mask) else 0.0
    first_mask = (hg >= 2.0) & (hg <= 2.45)
    first_peak = float(np.max(g[first_mask])) if np.any(first_mask) else float(np.max(g))
    mid_mask = (hg >= 3.0) & (hg <= 8.0)
    mid_peak = float(np.max(g[mid_mask])) if np.any(mid_mask) else first_peak

    score = (
        1.80 * k_rmse
        + 2.40 * k_large_pos
        + 0.55 * g_rmse
        + 0.12 * max(0.0, first_peak - 3.0)
        + 0.25 * max(0.0, mid_peak - 1.8)
    )
    return score, {
        'score': score,
        'Ripley_K_relative_RMSE_paper_edge': k_rmse,
        'Ripley_K_large_positive_RMSE': k_large_pos,
        'pair_distribution_RMSE_paper': g_rmse,
        'first_pair_distribution_peak': first_peak,
        'mid_pair_distribution_peak': mid_peak,
        # Backward-compatible key used by generate() print statement.
        'pair_distribution_RMSE_shell': g_rmse,
    }


def layout_randomness_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """V4: candidate selection prioritizes paper-style Ripley's K agreement."""
    score, info = _paper_k_score(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
    contact = contact_peak_diagnostics(centers, radii, p)
    score += 0.20 * orientation_ks + 8.0 * contact['contact_pair_fraction']
    info.update({
        'score': score,
        'orientation_CDF_KS_distance': orientation_ks,
        'contact_pair_fraction': contact['contact_pair_fraction'],
        'min_h_over_r': contact['min_h_over_r'],
    })
    return score, info


def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """V4 generation: DCCP + paper-style density-balance Step 2."""
    method = str(p.step1_method).lower()

    if method == 'relaxation':
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ('dccp', 'hybrid_dccp'):
        warm = None
        if p.dccp_use_relaxation_warm_start and not bool(getattr(p, 'dccp_random_start_if_active', True)):
            warm = random_initial_centers(p, len(radii), rng)
            warm = paper_step1_initial_optimization(warm, radii, gap, p, rng)
        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == 'hybrid_dccp':
                print('  Warning: active DCCP failed in hybrid_dccp mode. Falling back to relaxation.')
                print('  DCCP error:', repr(exc))
                centers = random_initial_centers(p, len(radii), rng)
                centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)
            else:
                raise
        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)
    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    # Paper-like Step 2: global density-balance relocation, then only a very mild
    # original reorientation. Avoid V3's aggressive first-shell escape because it
    # can worsen Ripley's K.
    centers = paper_balance_reorientation(centers, radii, gap, p, rng)
    centers = paper_step2_reorientation(centers, radii, gap, p, rng)

    if bool(getattr(p, 'use_pair_aware_shuffle', False)):
        centers = pair_aware_local_shuffle(centers, radii, gap, p, rng)

    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=int(getattr(p, 'final_hard_polish_iter', 1200)),
        shake_amp=0.0,
        move_limit_um=float(p.opt_final_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics


_previous_build_user_params_v4 = build_user_params


def build_user_params() -> RVEParams:
    params = _previous_build_user_params_v4()

    # Paper comparison mode. Eq. (2) uses centers inside the RVE. This is the
    # most faithful setting for reproducing Fig. 10 statistics. If you mainly
    # need boundary-intersecting fibers for Abaqus PBC geometry, change this back
    # to 'cell' after the statistical comparison is satisfactory.
    params.dccp_boundary_mode = 'inside'

    # Keep periodic distance for final full check/polish. The DCCP active list is
    # still much cheaper than strict 3x3 DCCP.
    params.step1_method = 'hybrid_dccp'
    params.dccp_pair_strategy = 'active'
    params.dccp_random_start_if_active = True
    params.dccp_regularization = 1.0e-8
    params.dccp_active_outer_iter = 4
    params.dccp_active_nearest_k = 20
    params.dccp_active_margin_um = 0.95
    params.dccp_active_add_margin_um = 0.35
    params.dccp_max_active_constraints = 50000
    params.dccp_post_polish_iter = 1400
    params.dccp_post_polish_move_limit_um = 0.035

    # Disable V3's effective gap floor and first-shell escape. They can suppress
    # the first g(h) peak locally but may create large-scale clustering, which is
    # exactly what your new K(h) plot shows.
    params.dccp_use_effective_gap_floor = False
    params.use_first_shell_escape_shuffle = False
    params.use_pair_peak_reorientation = False
    params.use_mc_shuffle = False
    params.use_pair_aware_shuffle = False

    # Paper-style K and g evaluation. Start K at zero like the paper, but do not
    # use an artificial zero mask to hide actual generator behaviour.
    params.use_paper_style_statistics = True
    params.paper_pair_h_min_over_r = 0.0
    params.paper_pair_n_h = 220
    params.paper_pair_smooth_window = 5
    params.edge_correction_angle_samples = 720
    if hasattr(params, 'paper_pair_zero_below_over_r'):
        params.paper_pair_zero_below_over_r = None

    # Paper-like Step-2 density-balance reorientation.
    params.use_paper_balance_reorientation = True
    params.paper_balance_passes = 2
    params.paper_balance_trials_per_fiber = 120
    params.paper_balance_top_fraction = 0.30
    params.paper_balance_local_sigma_factor = 3.0
    params.paper_balance_min_improve = 0.06
    params.balance_first_shell_high_over_r = 2.35
    params.balance_first_shell_weight = 0.80
    params.balance_density_radii_over_r = [4.0, 6.0, 8.0]
    params.balance_density_weights = [1.0, 0.75, 0.50]
    params.balance_candidate_density_radius_over_r = 6.0

    # Mild original Step 2. Large moves focused only on first-shell peaks are not
    # used in V4.
    params.use_reorientation = True
    params.reorient_passes = 1
    params.reorient_trials_per_fiber = 35
    params.reorient_accept_if_improve_um = 0.005
    params.reorient_random_accept_prob = 0.0

    # Candidate selection now weights K(h) strongly, so a few candidates are useful.
    params.valid_candidates_to_test = 4
    params.max_restarts = 100
    params.output_prefix = 'fiber_centers_active_dccp_paper_balance_vf70_v4'
    return params




# ============================================================
# V5: paper-faithful override
# ============================================================
# Design principle of V5:
# 1) Keep the paper-like workflow: DCCP initial packing + re-orientation + periodic copy.
# 2) Do not add aggressive pair-peak or first-shell escape operations, because they
#    can make nearest-neighbor distributions too sharp or worsen Ripley's K.
# 3) Use active periodic DCCP only as a computational acceleration of the paper's
#    all-pair DCCP constraint, followed by full-pair validation.
# 4) Use candidate selection, rather than strong local post-processing, to choose
#    the layout whose K(h), g(h), orientation CDF and nearest-neighbor distributions
#    are most consistent with the paper-style randomness checks.


def _nearest_distribution_shape_metrics(
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
) -> Dict[str, float]:
    """
    Quantify whether nearest-neighbor distributions are too narrow/too peaked.

    This is deliberately used mainly for candidate selection, not as a strong
    relocation target. The paper evaluates nearest-neighbor distance as a
    randomness descriptor; using it only in candidate scoring avoids creating new
    artificial patterns during generation.
    """
    df_nn = evaluate_nearest_neighbor_distance(centers, radii, p)
    out: Dict[str, float] = {}
    for col, tag in [
        ("nearest_1_h_over_r", "nn1"),
        ("nearest_2_h_over_r", "nn2"),
        ("nearest_3_h_over_r", "nn3"),
    ]:
        x = df_nn[col].values.astype(float)
        q05, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
        iqr = q75 - q25
        width90 = q95 - q05
        std = float(np.std(x, ddof=1))

        # Histogram peak with a fixed bin width in h/r units. This makes the
        # peak less sensitive to matplotlib's automatic binning.
        bin_width = float(getattr(p, 'nn_hist_bin_width_over_r', 0.0125))
        xmin = max(1.95, float(np.min(x)) - 0.02)
        xmax = float(np.max(x)) + 0.02
        if xmax <= xmin + bin_width:
            peak_pdf = 0.0
        else:
            bins = np.arange(xmin, xmax + bin_width, bin_width)
            hist, _ = np.histogram(x, bins=bins, density=True)
            peak_pdf = float(np.max(hist)) if len(hist) else 0.0

        out[f'{tag}_mean'] = float(np.mean(x))
        out[f'{tag}_std'] = std
        out[f'{tag}_iqr'] = float(iqr)
        out[f'{tag}_width90'] = float(width90)
        out[f'{tag}_median'] = float(q50)
        out[f'{tag}_peak_pdf'] = peak_pdf

    # Soft penalties. The numbers are not hard physical thresholds; they are used
    # only to avoid selecting candidates with unrealistically sharp nearest-neighbor
    # shells, such as an excessively narrow 3rd-nearest peak.
    min_iqr_1 = float(getattr(p, 'nn_min_iqr_1', 0.022))
    min_iqr_2 = float(getattr(p, 'nn_min_iqr_2', 0.030))
    min_iqr_3 = float(getattr(p, 'nn_min_iqr_3', 0.034))
    peak_limit_1 = float(getattr(p, 'nn_peak_limit_1', 22.0))
    peak_limit_2 = float(getattr(p, 'nn_peak_limit_2', 18.0))
    peak_limit_3 = float(getattr(p, 'nn_peak_limit_3', 18.0))

    narrow_penalty = (
        max(0.0, (min_iqr_1 - out['nn1_iqr']) / max(min_iqr_1, 1.0e-12))
        + max(0.0, (min_iqr_2 - out['nn2_iqr']) / max(min_iqr_2, 1.0e-12))
        + 1.35 * max(0.0, (min_iqr_3 - out['nn3_iqr']) / max(min_iqr_3, 1.0e-12))
    )
    peak_penalty = (
        max(0.0, (out['nn1_peak_pdf'] - peak_limit_1) / max(peak_limit_1, 1.0e-12))
        + 0.75 * max(0.0, (out['nn2_peak_pdf'] - peak_limit_2) / max(peak_limit_2, 1.0e-12))
        + 1.25 * max(0.0, (out['nn3_peak_pdf'] - peak_limit_3) / max(peak_limit_3, 1.0e-12))
    )
    out['nearest_distribution_narrow_penalty'] = float(narrow_penalty)
    out['nearest_distribution_peak_penalty'] = float(peak_penalty)
    out['nearest_distribution_score'] = float(0.65 * narrow_penalty + 0.35 * peak_penalty)
    return out


def _paper_k_g_score_v5(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """Paper-style score from edge-corrected Ripley's K and g(h)."""
    df_k = evaluate_ripley_k_paper_edge(centers, radii, p)
    df_g = evaluate_pair_distribution_paper_from_k(df_k, p)

    h = df_k['h_over_r'].values.astype(float)
    K = df_k['K_algorithm'].values.astype(float)
    Kc = df_k['K_CSR'].values.astype(float)
    g = df_g['g_algorithm'].values.astype(float)
    hg = df_g['h_over_r'].values.astype(float)

    k_mask = (h >= 2.0) & (h <= 14.5)
    k_rel = (K[k_mask] - Kc[k_mask]) / np.maximum(Kc[k_mask], 1.0e-12) if np.any(k_mask) else np.array([0.0])
    k_rmse = math.sqrt(float(np.mean(k_rel * k_rel)))

    large_mask = (h >= 8.0) & (h <= 14.5)
    if np.any(large_mask):
        large_rel = (K[large_mask] - Kc[large_mask]) / np.maximum(Kc[large_mask], 1.0e-12)
        # Penalize both clustering (positive K-Kcsr) and overly regular depletion
        # (negative K-Kcsr), but clustering is slightly more harmful for the user's
        # current results.
        k_large_pos = math.sqrt(float(np.mean(np.maximum(large_rel, 0.0) ** 2)))
        k_large_abs = math.sqrt(float(np.mean(large_rel ** 2)))
    else:
        k_large_pos = 0.0
        k_large_abs = 0.0

    valid_g = hg >= 2.0
    g_rmse = math.sqrt(float(np.mean((g[valid_g] - 1.0) ** 2))) if np.any(valid_g) else 0.0
    first_mask = (hg >= 2.0) & (hg <= 2.65)
    mid_mask = (hg >= 3.0) & (hg <= 8.0)
    far_mask = hg >= 8.0
    first_peak = float(np.max(g[first_mask])) if np.any(first_mask) else float(np.max(g))
    mid_peak = float(np.max(g[mid_mask])) if np.any(mid_mask) else first_peak
    mid_rmse = math.sqrt(float(np.mean((g[mid_mask] - 1.0) ** 2))) if np.any(mid_mask) else g_rmse
    far_rmse = math.sqrt(float(np.mean((g[far_mask] - 1.0) ** 2))) if np.any(far_mask) else g_rmse

    score = (
        1.60 * k_rmse
        + 1.90 * k_large_pos
        + 0.70 * k_large_abs
        + 0.55 * g_rmse
        + 0.55 * mid_rmse
        + 0.20 * far_rmse
        + 0.10 * max(0.0, first_peak - 3.5)
        + 0.20 * max(0.0, mid_peak - 1.8)
    )
    return score, {
        'score': score,
        'Ripley_K_relative_RMSE_paper_edge': k_rmse,
        'Ripley_K_large_positive_RMSE': k_large_pos,
        'Ripley_K_large_abs_RMSE': k_large_abs,
        'pair_distribution_RMSE_paper': g_rmse,
        'pair_distribution_mid_RMSE_paper': mid_rmse,
        'pair_distribution_far_RMSE_paper': far_rmse,
        'first_pair_distribution_peak': first_peak,
        'mid_pair_distribution_peak': mid_peak,
        # Backward-compatible key used by generate() print statement.
        'pair_distribution_RMSE_shell': g_rmse,
    }


def layout_randomness_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """
    V5 candidate selection.

    The score follows the paper's randomness checks rather than directly forcing
    pair peaks during generation. It balances:
      - paper-style Ripley's K agreement;
      - paper-style pair distribution g(h);
      - nearest-neighbor orientation CDF;
      - nearest-neighbor distance distribution width/peak shape.
    """
    score, info = _paper_k_g_score_v5(centers, radii, p)

    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
    contact = contact_peak_diagnostics(centers, radii, p)
    nn_info = _nearest_distribution_shape_metrics(centers, radii, p)

    score += (
        0.22 * orientation_ks
        + 7.0 * contact['contact_pair_fraction']
        + float(getattr(p, 'nearest_distribution_weight', 0.45)) * nn_info['nearest_distribution_score']
    )
    info.update({
        'score': score,
        'orientation_CDF_KS_distance': orientation_ks,
        'contact_pair_fraction': contact['contact_pair_fraction'],
        'min_h_over_r': contact['min_h_over_r'],
        **nn_info,
    })
    return score, info


def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    V5 generation.

    This is intentionally close to the paper workflow:
      STEP 1: active periodic DCCP as a computationally reduced version of the
              paper's all-pair DCCP constraint;
      STEP 1b: full periodic hard-constraint polish/validation;
      STEP 2: original resin-rich re-orientation only, kept mild;
      STEP 3: export periodic copies later in build_for_abaqus_dataframe().

    No aggressive pair-aware shuffle, first-shell escape, effective gap floor, or
    density-balance relocation is used by default in V5. These can improve one
    metric while harming K(h) or nearest-neighbor distributions.
    """
    method = str(p.step1_method).lower()

    if method == 'relaxation':
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ('dccp', 'hybrid_dccp'):
        warm = None
        if p.dccp_use_relaxation_warm_start and not bool(getattr(p, 'dccp_random_start_if_active', True)):
            warm = random_initial_centers(p, len(radii), rng)
            warm = paper_step1_initial_optimization(warm, radii, gap, p, rng)
        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == 'hybrid_dccp':
                print('  Warning: active DCCP failed in hybrid_dccp mode. Falling back to relaxation.')
                print('  DCCP error:', repr(exc))
                centers = random_initial_centers(p, len(radii), rng)
                centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)
            else:
                raise
        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)
    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    # Paper Step 2 only: resin-rich re-orientation. Keep it mild; do not force
    # first-shell escape or pair-targeted moves.
    centers = paper_step2_reorientation(centers, radii, gap, p, rng)

    # Final hard feasibility polish. This should be short; it is not intended to
    # regularize the structure, only to remove tiny numerical violations.
    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=int(getattr(p, 'final_hard_polish_iter', 1200)),
        shake_amp=0.0,
        move_limit_um=float(p.opt_final_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics


def evaluate_randomness(centers: np.ndarray, radii: np.ndarray, p: RVEParams, output_prefix: str):
    """V5 evaluation: save paper-style K/g plus nearest-neighbor shape metrics."""
    print('\n' + '=' * 80)
    print('Randomness evaluation starts...')
    print('=' * 80)

    df_nn = evaluate_nearest_neighbor_distance(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    df_k_periodic = evaluate_ripley_k(centers, radii, p)
    df_g_shell = evaluate_pair_distribution_shell(centers, radii, p)

    df_k_paper = evaluate_ripley_k_paper_edge(centers, radii, p)
    df_g_paper = evaluate_pair_distribution_paper_from_k(df_k_paper, p)

    contact_diag = contact_peak_diagnostics(centers, radii, p)
    orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
    k_score, k_info = _paper_k_g_score_v5(centers, radii, p)
    nn_info = _nearest_distribution_shape_metrics(centers, radii, p)
    g_rmse_shell = math.sqrt(float(np.mean((df_g_shell['g_algorithm'].values - 1.0) ** 2)))

    summary = pd.DataFrame([{
        'fiber_number': len(centers),
        'mean_radius_um': float(np.mean(radii)),
        'nearest_1_mean_h_over_r': df_nn['nearest_1_h_over_r'].mean(),
        'nearest_1_std_h_over_r': df_nn['nearest_1_h_over_r'].std(),
        'nearest_1_iqr_h_over_r': nn_info['nn1_iqr'],
        'nearest_1_peak_pdf': nn_info['nn1_peak_pdf'],
        'nearest_2_mean_h_over_r': df_nn['nearest_2_h_over_r'].mean(),
        'nearest_2_std_h_over_r': df_nn['nearest_2_h_over_r'].std(),
        'nearest_2_iqr_h_over_r': nn_info['nn2_iqr'],
        'nearest_2_peak_pdf': nn_info['nn2_peak_pdf'],
        'nearest_3_mean_h_over_r': df_nn['nearest_3_h_over_r'].mean(),
        'nearest_3_std_h_over_r': df_nn['nearest_3_h_over_r'].std(),
        'nearest_3_iqr_h_over_r': nn_info['nn3_iqr'],
        'nearest_3_peak_pdf': nn_info['nn3_peak_pdf'],
        'nearest_distribution_score': nn_info['nearest_distribution_score'],
        'orientation_CDF_KS_distance': orientation_ks,
        'Ripley_K_relative_RMSE_paper_edge': k_info['Ripley_K_relative_RMSE_paper_edge'],
        'Ripley_K_large_positive_RMSE': k_info['Ripley_K_large_positive_RMSE'],
        'Ripley_K_large_abs_RMSE': k_info['Ripley_K_large_abs_RMSE'],
        'pair_distribution_RMSE_paper_edge': k_info['pair_distribution_RMSE_paper'],
        'pair_distribution_mid_RMSE_paper_edge': k_info['pair_distribution_mid_RMSE_paper'],
        'pair_distribution_far_RMSE_paper_edge': k_info['pair_distribution_far_RMSE_paper'],
        'pair_distribution_RMSE_shell_stable_periodic': g_rmse_shell,
        'first_pair_distribution_peak_paper_edge': k_info['first_pair_distribution_peak'],
        'mid_pair_distribution_peak_paper_edge': k_info['mid_pair_distribution_peak'],
        'contact_pair_fraction': contact_diag['contact_pair_fraction'],
        'contact_pair_count': contact_diag['contact_pair_count'],
        'min_h_over_r': contact_diag['min_h_over_r'],
    }])

    plot_nearest_neighbor_distance(df_nn, output_prefix + '_eval_nearest_distance.png')
    plot_nearest_neighbor_orientation(df_ori, output_prefix + '_eval_orientation_cdf.png')
    plot_ripley_k(df_k_paper, output_prefix + '_eval_ripley_k.png')
    plot_pair_distribution(df_g_paper, output_prefix + '_eval_pair_distribution_from_K.png')
    plot_pair_distribution(df_g_paper, output_prefix + '_eval_pair_distribution_paper_edge.png')
    plot_pair_distribution(df_g_shell, output_prefix + '_eval_pair_distribution_shell_periodic.png')

    eval_xlsx = output_prefix + '_randomness_evaluation.xlsx'
    with pd.ExcelWriter(eval_xlsx, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        df_nn.to_excel(writer, sheet_name='nearest_distance', index=False)
        df_ori.to_excel(writer, sheet_name='orientation_cdf', index=False)
        df_k_paper.to_excel(writer, sheet_name='ripley_k_paper_edge', index=False)
        df_g_paper.to_excel(writer, sheet_name='pair_distribution_paper', index=False)
        df_k_periodic.to_excel(writer, sheet_name='ripley_k_periodic_old', index=False)
        df_g_shell.to_excel(writer, sheet_name='pair_distribution_shell', index=False)

    print('Randomness evaluation finished.')
    print('Evaluation file saved:', eval_xlsx)
    print('orientation CDF KS distance       = %.6e' % orientation_ks)
    print('Ripley K relative RMSE paper-edge = %.6e' % k_info['Ripley_K_relative_RMSE_paper_edge'])
    print('Ripley K large positive RMSE      = %.6e' % k_info['Ripley_K_large_positive_RMSE'])
    print('Pair distribution RMSE paper-edge = %.6e' % k_info['pair_distribution_RMSE_paper'])
    print('Pair distribution RMSE shell      = %.6e' % g_rmse_shell)
    print('first g(h) peak paper-edge        = %.6e' % k_info['first_pair_distribution_peak'])
    print('nearest distribution score        = %.6e' % nn_info['nearest_distribution_score'])
    print('contact pair fraction             = %.6e' % contact_diag['contact_pair_fraction'])
    print('min h/r                           = %.6e' % contact_diag['min_h_over_r'])
    print('=' * 80)
    return summary, df_nn, df_ori, df_k_paper, df_g_paper


_previous_build_user_params_v5_base = build_user_params


def build_user_params() -> RVEParams:
    """
    V5 user operation center.

    The default parameters are intentionally conservative and paper-faithful.
    The generator is not allowed to use aggressive pair-specific post-processing.
    Candidate selection, rather than forced local relocation, is responsible for
    choosing the best statistically random layout.
    """
    params = _previous_build_user_params_v5_base()

    # Choose the formal Vf=70% paper-statistical setting by default.
    params.Lx_um = 175.0
    params.Ly_um = 175.0
    params.target_vf = 0.70
    params.diameter_mode = 'constant'
    params.diameter_mean_um = 7.0
    params.diameter_std_um = 0.317
    params.lmin_um = 0.175
    params.lmax_um = 0.350

    # STEP 1: active periodic DCCP. This keeps the paper DCCP idea while avoiding
    # all 154846 constraints in one CVXPY problem.
    params.step1_method = 'hybrid_dccp'
    params.dccp_pair_strategy = 'active'
    params.dccp_boundary_mode = 'cell'  # PBC-friendly. Use 'inside' for strict Eq.(2) comparison only.
    params.dccp_random_start_if_active = True
    params.dccp_regularization = 1.0e-8
    params.dccp_active_outer_iter = 4
    params.dccp_active_nearest_k = 20
    params.dccp_active_margin_um = 0.95
    params.dccp_active_add_margin_um = 0.35
    params.dccp_max_active_constraints = 50000
    params.dccp_constraint_batch_size = 6000
    params.dccp_vectorized_constraints = True
    params.dccp_post_periodic_polish = True
    params.dccp_post_polish_iter = 1400
    params.dccp_post_polish_move_limit_um = 0.035

    # STEP 2: paper-like resin-rich re-orientation only. Keep it mild.
    params.use_reorientation = True
    params.reorient_passes = 2
    params.reorient_trials_per_fiber = 55
    params.reorient_accept_if_improve_um = 0.012
    params.reorient_random_accept_prob = 0.006
    params.reorient_local_radius_factor = 4.0

    # Disable all non-paper aggressive operations introduced in previous tests.
    params.use_paper_balance_reorientation = False
    params.use_pair_peak_reorientation = False
    params.use_pair_aware_shuffle = False
    params.use_first_shell_escape_shuffle = False
    params.use_mc_shuffle = False
    params.dccp_use_effective_gap_floor = False
    params.final_hard_polish_iter = 1200

    # Paper-style statistics.
    params.use_paper_style_statistics = True
    params.paper_pair_h_min_over_r = 0.0
    params.paper_pair_n_h = 220
    params.paper_pair_smooth_window = 5
    params.edge_correction_angle_samples = 720
    params.pair_h_min_over_r = 2.0
    params.pair_h_max_over_r = 15.0
    params.pair_dh_over_r = 0.35
    params.pair_smooth_window = 1
    params.contact_low_over_r = 2.0
    params.contact_high_over_r = 2.35

    # Nearest-neighbor candidate-scoring settings. These do not force moves; they
    # only help choose a candidate whose nearest-neighbor histograms are not too
    # narrow or over-peaked compared with the paper's statistical intent.
    params.nearest_distribution_weight = 0.45
    params.nn_hist_bin_width_over_r = 0.0125
    params.nn_min_iqr_1 = 0.022
    params.nn_min_iqr_2 = 0.030
    params.nn_min_iqr_3 = 0.034
    params.nn_peak_limit_1 = 22.0
    params.nn_peak_limit_2 = 18.0
    params.nn_peak_limit_3 = 18.0

    # Candidate search. Use more candidates instead of stronger post-processing.
    params.valid_candidates_to_test = 5
    params.max_restarts = 120
    params.selection_mode = 'best_valid_randomness'
    params.tol_um = 1.0e-7
    params.seed = 2026
    params.output_prefix = 'fiber_centers_active_dccp_paper_faithful_v5'
    return params



# ============================================================
# V6: paper-faithful DCCP + density-balanced re-orientation
# ============================================================
# Rationale of V6:
# - V2/V3 directly targeted pair peaks and introduced new artifacts.
# - V5 removed aggressive post-processing, but the results can still show
#   first-neighbor pile-up and K(h)>K_CSR at large h/r, indicating clustering.
# - V6 keeps the paper workflow but makes Step 2 closer to the paper's
#   "cluster to resin-rich region" idea: move a small number of fibers from
#   locally crowded regions to carefully selected resin-rich positions, while
#   rejecting moves that create overly isolated fibers or damage K/NN statistics.
# - The pair distribution is NOT forced by a hard artificial target.


def _periodic_distances_to_point(xy: np.ndarray, centers: np.ndarray, p: RVEParams) -> np.ndarray:
    dx = xy[0] - centers[:, 0]
    dy = xy[1] - centers[:, 1]
    dx, dy = min_image_components(dx, dy, p)
    return np.sqrt(dx * dx + dy * dy)


def _fiber_local_balance_metrics(
    k: int,
    xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
) -> Dict[str, float]:
    """
    Local crowding metrics used by V6 re-orientation.

    This is intentionally not a pair-distribution optimizer. It measures whether
    a fiber sits in a locally crowded region across several physically meaningful
    radii. Moves are accepted only when they reduce crowding without creating an
    isolated fiber. This mirrors the paper's re-orientation idea more closely.
    """
    r_mean = float(np.mean(radii))
    d = _periodic_distances_to_point(xy, centers, p)
    d[k] = np.inf
    h = d / max(r_mean, 1.0e-12)

    near_hi = float(getattr(p, 'v6_near_shell_high_over_r', 2.35))
    mid_hi = float(getattr(p, 'v6_mid_shell_high_over_r', 5.0))
    far_hi = float(getattr(p, 'v6_far_shell_high_over_r', 8.0))

    near_count = float(np.sum((h >= 2.0) & (h < near_hi)))
    mid_count = float(np.sum((h >= near_hi) & (h < mid_hi)))
    far_count = float(np.sum((h >= mid_hi) & (h < far_hi)))

    # Gaussian local density avoids hard discontinuities when comparing trial points.
    gaussian_density = 0.0
    for hc, width, weight in [
        (2.16, 0.20, 1.25),
        (3.2, 0.55, 0.55),
        (5.0, 0.85, 0.28),
        (7.0, 1.10, 0.16),
    ]:
        gaussian_density += weight * float(np.sum(np.exp(-((h - hc) / width) ** 2)))

    sorted_h = np.sort(h)
    nn1 = float(sorted_h[0])
    nn2 = float(sorted_h[1])
    nn3 = float(sorted_h[2])
    nn6 = float(sorted_h[5]) if len(sorted_h) > 5 else nn3

    # Avoid moving a fiber into an excessively empty point. The paper's
    # re-orientation fills resin-rich regions, but not by creating isolated
    # outliers that would make K(h) or nearest-neighbor distributions worse.
    target_nn3_hi = float(getattr(p, 'v6_max_trial_nn3_over_r', 2.55))
    target_nn6_hi = float(getattr(p, 'v6_max_trial_nn6_over_r', 3.20))
    isolation_penalty = (
        1.60 * max(0.0, nn3 - target_nn3_hi)
        + 0.75 * max(0.0, nn6 - target_nn6_hi)
    )

    crowding_score = (
        float(getattr(p, 'v6_near_count_weight', 1.35)) * near_count
        + float(getattr(p, 'v6_mid_count_weight', 0.24)) * mid_count
        + float(getattr(p, 'v6_far_count_weight', 0.07)) * far_count
        + gaussian_density
        + isolation_penalty
    )

    return {
        'score': float(crowding_score),
        'near_count': near_count,
        'mid_count': mid_count,
        'far_count': far_count,
        'gaussian_density': float(gaussian_density),
        'nn1': nn1,
        'nn2': nn2,
        'nn3': nn3,
        'nn6': nn6,
        'isolation_penalty': float(isolation_penalty),
    }


def _sample_resin_rich_points_v6(
    centers: np.ndarray,
    radii: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
    n_points: int,
    keep: int,
) -> np.ndarray:
    """
    Sample resin-rich candidate points with mild density balancing.

    A good candidate should have enough clearance from the nearest fiber, but it
    should not be a very isolated point in a large void. This prevents the K(h)
    curve from increasing at large h/r due to cluster/void separation.
    """
    n = len(radii)
    r_mean = float(np.mean(radii))
    margin = 0.0
    if str(getattr(p, 'dccp_boundary_mode', 'cell')).lower() == 'inside':
        margin = float(np.max(radii)) + 1.0e-6

    pts = np.zeros((int(n_points), 2), dtype=float)
    pts[:, 0] = rng.uniform(margin, p.Lx_um - margin, len(pts))
    pts[:, 1] = rng.uniform(margin, p.Ly_um - margin, len(pts))

    scores = np.empty(len(pts), dtype=float)
    for a, xy in enumerate(pts):
        d = _periodic_distances_to_point(xy, centers, p)
        surface_gap = float(np.min(d - radii))
        h = np.sort(d / max(r_mean, 1.0e-12))
        # Prefer a resin-rich point, but reject extremely isolated void centers.
        nn3 = float(h[2]) if len(h) > 2 else float(h[-1])
        nn6 = float(h[5]) if len(h) > 5 else nn3
        isolation = max(0.0, nn3 - float(getattr(p, 'v6_candidate_nn3_hi_over_r', 2.75)))
        isolation += 0.40 * max(0.0, nn6 - float(getattr(p, 'v6_candidate_nn6_hi_over_r', 3.45)))
        # Small positive surface_gap is good; too large often means a big void.
        scores[a] = surface_gap - float(getattr(p, 'v6_candidate_isolation_weight', 1.20)) * r_mean * isolation

    idx = np.argsort(scores)[-min(int(keep), len(scores)):]
    return pts[idx]


def paper_density_balance_reorientation_v6(
    centers: np.ndarray,
    radii: np.ndarray,
    gap: np.ndarray,
    p: RVEParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    V6 paper-style re-orientation.

    This is the main algorithmic change in V6. It implements a conservative form
    of the paper's Step 2: identify fibers in crowded local environments and try
    relocating them to resin-rich, density-balanced positions. Acceptance is
    based on local crowding reduction and hard feasibility, not on forcibly
    minimizing the pair-distribution curve.
    """
    if not bool(getattr(p, 'use_v6_density_balance_reorientation', True)):
        return centers

    centers = centers.copy()
    n = len(radii)
    passes = int(getattr(p, 'v6_balance_passes', 2))
    top_fraction = float(getattr(p, 'v6_balance_top_fraction', 0.22))
    trials = int(getattr(p, 'v6_balance_trials_per_fiber', 70))
    min_improve = float(getattr(p, 'v6_balance_min_improve', 0.10))
    r_mean = float(np.mean(radii))
    local_sigma = float(getattr(p, 'v6_balance_local_sigma_factor', 2.8)) * r_mean

    print('STEP 2b: V6 paper-style density-balance re-orientation starts...')
    for pass_id in range(1, passes + 1):
        # Rank fibers by local crowding.
        metrics = [
            _fiber_local_balance_metrics(k, centers[k], centers, radii, p)
            for k in range(n)
        ]
        scores = np.array([m['score'] for m in metrics], dtype=float)
        n_target = max(1, int(top_fraction * n))
        target_ids = np.argsort(scores)[-n_target:]
        target_ids = rng.permutation(target_ids)

        resin_points = _sample_resin_rich_points_v6(
            centers, radii, p, rng,
            n_points=max(2500, int(getattr(p, 'v6_resin_points_factor', 14)) * n),
            keep=max(120, n // 3),
        )

        moved = 0
        for k in target_ids:
            k = int(k)
            old_xy = centers[k].copy()
            old_metrics = _fiber_local_balance_metrics(k, old_xy, centers, radii, p)
            old_score = old_metrics['score']
            best_xy = old_xy.copy()
            best_score = old_score
            best_margin = single_fiber_min_margin(k, old_xy, centers, radii, gap, p)

            for _ in range(trials):
                if len(resin_points) > 0 and rng.random() < 0.82:
                    base = resin_points[int(rng.integers(0, len(resin_points)))]
                    trial = base + rng.normal(0.0, local_sigma, 2)
                else:
                    # Occasional moderate local move to avoid over-filling one void.
                    trial = old_xy + rng.normal(0.0, 1.6 * r_mean, 2)

                if str(getattr(p, 'dccp_boundary_mode', 'cell')).lower() == 'inside':
                    eps = 1.0e-6
                    trial[0] = np.clip(trial[0], radii[k] + eps, p.Lx_um - radii[k] - eps)
                    trial[1] = np.clip(trial[1], radii[k] + eps, p.Ly_um - radii[k] - eps)
                else:
                    trial[0] = trial[0] % p.Lx_um
                    trial[1] = trial[1] % p.Ly_um

                margin = single_fiber_min_margin(k, trial, centers, radii, gap, p)
                if margin < -p.tol_um:
                    continue

                trial_metrics = _fiber_local_balance_metrics(k, trial, centers, radii, p)
                trial_score = trial_metrics['score']
                # Do not accept a move that creates a very isolated point.
                if trial_metrics['isolation_penalty'] > float(getattr(p, 'v6_max_isolation_penalty', 0.55)):
                    continue
                if trial_score < best_score - 1.0e-12 or (
                    abs(trial_score - best_score) < 1.0e-12 and margin > best_margin
                ):
                    best_xy = trial.copy()
                    best_score = trial_score
                    best_margin = margin

            if best_score < old_score - min_improve:
                centers[k] = best_xy
                moved += 1

        # Short hard polish only; no pair forcing.
        centers = packing_relax_one_stage(
            centers, radii, gap, p, rng,
            alpha=1.0,
            n_iter=int(getattr(p, 'v6_balance_post_polish_iter', 550)),
            shake_amp=0.0,
            move_limit_um=float(p.opt_final_move_limit_um),
        )
        m = compute_metrics(centers, radii, gap, p)
        print(
            '  pass %d/%d | moved = %d | min_margin = %.6e | max_violation = %.6e'
            % (pass_id, passes, moved, m['min_constraint_margin'], m['max_violation'])
        )
    return centers


def _paper_k_g_score_v6(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """V6 score: prioritizes K agreement and nearest distributions, with g(h) secondary."""
    df_k = evaluate_ripley_k_paper_edge(centers, radii, p)
    df_g = evaluate_pair_distribution_paper_from_k(df_k, p)

    h = df_k['h_over_r'].values.astype(float)
    K = df_k['K_algorithm'].values.astype(float)
    Kc = df_k['K_CSR'].values.astype(float)
    hg = df_g['h_over_r'].values.astype(float)
    g = df_g['g_algorithm'].values.astype(float)

    k_mask = (h >= 2.0) & (h <= 14.5)
    k_rel = (K[k_mask] - Kc[k_mask]) / np.maximum(Kc[k_mask], 1.0e-12) if np.any(k_mask) else np.array([0.0])
    k_rmse = math.sqrt(float(np.mean(k_rel * k_rel)))

    large_mask = (h >= 8.0) & (h <= 14.5)
    if np.any(large_mask):
        large_rel = (K[large_mask] - Kc[large_mask]) / np.maximum(Kc[large_mask], 1.0e-12)
        k_large_pos = math.sqrt(float(np.mean(np.maximum(large_rel, 0.0) ** 2)))
        k_large_abs = math.sqrt(float(np.mean(large_rel ** 2)))
    else:
        k_large_pos = 0.0
        k_large_abs = 0.0

    valid_g = hg >= 2.0
    g_rmse = math.sqrt(float(np.mean((g[valid_g] - 1.0) ** 2))) if np.any(valid_g) else 0.0
    first_mask = (hg >= 2.0) & (hg <= 2.65)
    mid_mask = (hg >= 3.0) & (hg <= 8.0)
    far_mask = hg >= 8.0
    first_peak = float(np.max(g[first_mask])) if np.any(first_mask) else float(np.max(g))
    mid_peak = float(np.max(g[mid_mask])) if np.any(mid_mask) else first_peak
    mid_rmse = math.sqrt(float(np.mean((g[mid_mask] - 1.0) ** 2))) if np.any(mid_mask) else g_rmse
    far_rmse = math.sqrt(float(np.mean((g[far_mask] - 1.0) ** 2))) if np.any(far_mask) else g_rmse

    score = (
        2.10 * k_rmse
        + 2.80 * k_large_pos
        + 0.90 * k_large_abs
        + 0.34 * g_rmse
        + 0.34 * mid_rmse
        + 0.18 * far_rmse
        + 0.10 * max(0.0, first_peak - 3.8)
        + 0.16 * max(0.0, mid_peak - 1.75)
    )
    return score, {
        'score': score,
        'Ripley_K_relative_RMSE_paper_edge': k_rmse,
        'Ripley_K_large_positive_RMSE': k_large_pos,
        'Ripley_K_large_abs_RMSE': k_large_abs,
        'pair_distribution_RMSE_paper': g_rmse,
        'pair_distribution_mid_RMSE_paper': mid_rmse,
        'pair_distribution_far_RMSE_paper': far_rmse,
        'first_pair_distribution_peak': first_peak,
        'mid_pair_distribution_peak': mid_peak,
        'pair_distribution_RMSE_shell': g_rmse,
    }


def layout_randomness_score(centers: np.ndarray, radii: np.ndarray, p: RVEParams) -> Tuple[float, Dict[str, float]]:
    """
    V6 candidate score.

    The current results show that forcing the pair curve alone can damage K(h)
    and nearest-neighbor distributions. V6 therefore selects candidates mainly by
    paper-style K agreement and nearest-neighbor shape, while using g(h) as a
    secondary descriptor.
    """
    score, info = _paper_k_g_score_v6(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    orientation_ks = float(np.max(np.abs(df_ori['cdf_difference'].values)))
    contact = contact_peak_diagnostics(centers, radii, p)
    nn_info = _nearest_distribution_shape_metrics(centers, radii, p)

    score += (
        0.28 * orientation_ks
        + 5.0 * contact['contact_pair_fraction']
        + float(getattr(p, 'nearest_distribution_weight', 0.70)) * nn_info['nearest_distribution_score']
    )
    info.update({
        'score': score,
        'orientation_CDF_KS_distance': orientation_ks,
        'contact_pair_fraction': contact['contact_pair_fraction'],
        'min_h_over_r': contact['min_h_over_r'],
        **nn_info,
    })
    return score, info


def generate_one_candidate(p: RVEParams, radii: np.ndarray, gap: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    V6 generation: DCCP -> hard periodic polish -> original Step-2 -> V6 density
    balance Step-2b -> short hard polish.

    The extra V6 step is not a direct pair-curve optimizer. It is a conservative
    implementation of the paper's cluster-to-resin-rich re-orientation concept.
    """
    method = str(p.step1_method).lower()

    if method == 'relaxation':
        centers = random_initial_centers(p, len(radii), rng)
        centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)

    elif method in ('dccp', 'hybrid_dccp'):
        warm = None
        if bool(getattr(p, 'dccp_use_coarse_warm_start_v6', True)):
            warm = random_initial_centers(p, len(radii), rng)
            # A coarse warm start reduces catastrophic overlaps but avoids the
            # fully relaxed near-contact state that made earlier pair peaks high.
            warm = packing_relax_one_stage(
                warm, radii, gap, p, rng,
                alpha=float(getattr(p, 'v6_warm_alpha', 0.62)),
                n_iter=int(getattr(p, 'v6_warm_iter', 1200)),
                shake_amp=float(getattr(p, 'v6_warm_shake_um', 0.015)),
                move_limit_um=float(getattr(p, 'v6_warm_move_limit_um', 0.16)),
            )
        try:
            centers = paper_step1_dccp_nonperiodic_optimization(
                radii, gap, p, rng, warm_start_centers=warm
            )
        except Exception as exc:
            if method == 'hybrid_dccp':
                print('  Warning: active DCCP failed in hybrid_dccp mode. Falling back to relaxation.')
                print('  DCCP error:', repr(exc))
                centers = random_initial_centers(p, len(radii), rng)
                centers = paper_step1_initial_optimization(centers, radii, gap, p, rng)
            else:
                raise
        centers = periodic_polish_after_dccp(centers, radii, gap, p, rng)
    else:
        raise ValueError("step1_method must be 'relaxation', 'dccp', or 'hybrid_dccp'.")

    # Mild original paper-like resin-rich reorientation.
    centers = paper_step2_reorientation(centers, radii, gap, p, rng)

    # V6 density-balance reorientation: conservative cluster-to-resin-rich moves.
    centers = paper_density_balance_reorientation_v6(centers, radii, gap, p, rng)

    centers = packing_relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=int(getattr(p, 'final_hard_polish_iter', 900)),
        shake_amp=0.0,
        move_limit_um=float(p.opt_final_move_limit_um),
    )
    metrics = compute_metrics(centers, radii, gap, p)
    return centers, metrics


_previous_build_user_params_v6_base = build_user_params


def build_user_params() -> RVEParams:
    """
    V6 user operation center.

    V6 stays closer to the paper than V2/V3: it does not force pair peaks.
    Instead, it improves the Step-2 re-orientation so that clustered fibers are
    moved toward resin-rich but not overly isolated positions. Candidate scoring
    emphasizes the full set of paper randomness descriptors.
    """
    params = _previous_build_user_params_v6_base()

    # Paper Fig.10-like Vf=70 setting.
    params.Lx_um = 175.0
    params.Ly_um = 175.0
    params.target_vf = 0.70
    params.diameter_mode = 'constant'
    params.diameter_mean_um = 7.0
    params.diameter_std_um = 0.317
    params.lmin_um = 0.175
    params.lmax_um = 0.350

    # Active periodic DCCP. Keep constraints manageable, but include enough
    # neighbors to avoid local clusters being missed by the active set.
    params.step1_method = 'hybrid_dccp'
    params.dccp_pair_strategy = 'active'
    params.dccp_boundary_mode = 'cell'
    params.dccp_regularization = 5.0e-9
    params.dccp_active_outer_iter = 4
    params.dccp_active_nearest_k = 26
    params.dccp_active_margin_um = 1.25
    params.dccp_active_add_margin_um = 0.45
    params.dccp_max_active_constraints = 65000
    params.dccp_constraint_batch_size = 6000
    params.dccp_vectorized_constraints = True
    params.dccp_post_periodic_polish = True
    params.dccp_post_polish_iter = 1200
    params.dccp_post_polish_move_limit_um = 0.032
    params.dccp_use_coarse_warm_start_v6 = True
    params.v6_warm_alpha = 0.62
    params.v6_warm_iter = 1200
    params.v6_warm_shake_um = 0.012
    params.v6_warm_move_limit_um = 0.16

    # Original Step-2 reorientation, mild.
    params.use_reorientation = True
    params.reorient_passes = 1
    params.reorient_trials_per_fiber = 45
    params.reorient_accept_if_improve_um = 0.010
    params.reorient_random_accept_prob = 0.004
    params.reorient_local_radius_factor = 4.0

    # V6 paper-style density-balance reorientation.
    params.use_v6_density_balance_reorientation = True
    params.v6_balance_passes = 2
    params.v6_balance_top_fraction = 0.20
    params.v6_balance_trials_per_fiber = 75
    params.v6_balance_min_improve = 0.10
    params.v6_balance_local_sigma_factor = 2.6
    params.v6_balance_post_polish_iter = 500
    params.v6_resin_points_factor = 14
    params.v6_near_shell_high_over_r = 2.35
    params.v6_mid_shell_high_over_r = 5.0
    params.v6_far_shell_high_over_r = 8.0
    params.v6_near_count_weight = 1.25
    params.v6_mid_count_weight = 0.22
    params.v6_far_count_weight = 0.07
    params.v6_max_trial_nn3_over_r = 2.55
    params.v6_max_trial_nn6_over_r = 3.20
    params.v6_candidate_nn3_hi_over_r = 2.75
    params.v6_candidate_nn6_hi_over_r = 3.45
    params.v6_candidate_isolation_weight = 1.20
    params.v6_max_isolation_penalty = 0.55

    # Disable aggressive non-paper operations from previous experimental versions.
    params.use_paper_balance_reorientation = False
    params.use_pair_peak_reorientation = False
    params.use_pair_aware_shuffle = False
    params.use_first_shell_escape_shuffle = False
    params.use_mc_shuffle = False
    params.dccp_use_effective_gap_floor = False
    params.final_hard_polish_iter = 900

    # Paper-style statistics and plotting.
    params.use_paper_style_statistics = True
    params.paper_pair_h_min_over_r = 0.0
    params.paper_pair_n_h = 260
    params.paper_pair_smooth_window = 7
    params.edge_correction_angle_samples = 720
    params.pair_h_min_over_r = 2.0
    params.pair_h_max_over_r = 15.0
    params.pair_dh_over_r = 0.35
    params.pair_smooth_window = 1
    params.contact_low_over_r = 2.0
    params.contact_high_over_r = 2.35

    # Candidate scoring: use more candidates rather than force one layout.
    params.nearest_distribution_weight = 0.70
    params.nn_hist_bin_width_over_r = 0.015
    params.nn_min_iqr_1 = 0.030
    params.nn_min_iqr_2 = 0.040
    params.nn_min_iqr_3 = 0.050
    params.nn_peak_limit_1 = 18.0
    params.nn_peak_limit_2 = 14.5
    params.nn_peak_limit_3 = 12.0

    params.valid_candidates_to_test = 6
    params.max_restarts = 150
    params.selection_mode = 'best_valid_randomness'
    params.tol_um = 1.0e-7
    params.seed = 2026
    params.output_prefix = 'fiber_centers_active_dccp_paper_faithful_v6'
    return params


if __name__ == "__main__":
    params = build_user_params()
    generate(params)
