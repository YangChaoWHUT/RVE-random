# -*- coding: utf-8 -*-
"""
Generate random periodic fiber centers for high-Vf RVE.

Key idea follows the paper:
distance(i,j) >= ri + rj + dis[i,j]
dis[i,j] = random.uniform(lmin, lmax)

This version does NOT use hexagonal initialization.
It starts from random centers and uses growth-relaxation + random shaking.

Modified version:
1. Add optional Monte-Carlo legal shuffling after relaxation to reduce near-contact pile-up.
2. Select several valid candidates using a randomness score instead of accepting the first valid layout.
3. Compute pair distribution function by direct annulus counting instead of differentiating Ripley's K.

Outputs:
1. fiber_centers_main.csv
2. fiber_centers_for_abaqus.csv
3. fiber_centers.xlsx
4. rve_preview.png
"""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


@dataclass
class RVEParams:
    Lx_um: float = 50.0
    Ly_um: float = 50.0

    target_vf: float = 0.80

    diameter_mode: str = "constant"
    diameter_mean_um: float = 7.0
    diameter_std_um: float = 0.317
    diameter_min_um: float = 5.0
    diameter_max_um: float = 8.0

    lmin_um: float = 0.10
    lmax_um: float = 0.15

    max_restarts: int = 200

    growth_steps: int = 45
    relax_iter_per_step: int = 1200
    final_relax_iter: int = 20000

    random_shake_um: float = 0.08
    tol_um: float = 1.0e-7

    # ------------------------------------------------------------
    # Additional options for paper-like randomness evaluation
    # ------------------------------------------------------------
    # first_valid: accept the first valid geometry.
    # best_valid_randomness: keep searching and select the valid geometry
    # with the lowest randomness score. This is usually better for
    # Pair distribution function.
    selection_mode: str = "best_valid_randomness"
    valid_candidates_to_test: int = 5

    # After a valid geometry is found, randomly move fibers inside the
    # feasible region. Only legal moves are accepted, so overlap constraints
    # are never violated. This helps reduce the artificial pile-up at h/r≈2.
    use_mc_shuffle: bool = True
    mc_steps_per_fiber: int = 800
    mc_move_amp_um: float = 0.06

    # Pair distribution settings. Direct annulus counting is used.
    pair_h_min_over_r: float = 2.0
    pair_h_max_over_r: float = 15.0
    pair_dh_over_r: float = 0.35
    pair_smooth_window: int = 5

    # Near-contact diagnostic interval. A very large fraction here usually
    # explains an excessively high first peak of g(h).
    contact_low_over_r: float = 2.0
    contact_high_over_r: float = 2.10

    # seed=None means every run will use a new random seed based on current time.
    # Set seed to a fixed integer, e.g. 2026, when you want to reproduce exactly the same RVE.
    seed: int = None
    output_prefix: str = "fiber_centers"


# ============================================================
# Basic functions
# ============================================================

def estimate_fiber_number(p):
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


def generate_radii(p, n, rng):
    if p.diameter_mode == "constant":
        d = np.ones(n) * p.diameter_mean_um

    elif p.diameter_mode == "normal":
        d = rng.normal(p.diameter_mean_um, p.diameter_std_um, n)
        d = np.clip(
            d,
            p.diameter_mean_um - 3.0 * p.diameter_std_um,
            p.diameter_mean_um + 3.0 * p.diameter_std_um
        )

    elif p.diameter_mode == "uniform":
        d = rng.uniform(p.diameter_min_um, p.diameter_max_um, n)

    else:
        raise ValueError("diameter_mode must be constant, normal, or uniform.")

    return d / 2.0


def make_gap_matrix(p, n, rng):
    gap = rng.uniform(p.lmin_um, p.lmax_um, (n, n))
    gap = np.triu(gap, 1)
    gap = gap + gap.T
    np.fill_diagonal(gap, 0.0)
    return gap


def actual_vf(radii, p):
    return float(np.sum(math.pi * radii ** 2) / (p.Lx_um * p.Ly_um))


def min_image(dx, dy, p):
    dx -= p.Lx_um * round(dx / p.Lx_um)
    dy -= p.Ly_um * round(dy / p.Ly_um)
    return dx, dy


def distance(c1, c2, p):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dx, dy = min_image(dx, dy, p)
    return math.sqrt(dx * dx + dy * dy)


def random_initial_centers(p, n, rng):
    centers = np.zeros((n, 2))
    centers[:, 0] = rng.uniform(0.0, p.Lx_um, n)
    centers[:, 1] = rng.uniform(0.0, p.Ly_um, n)
    return centers


# ============================================================
# Check functions
# ============================================================

def compute_metrics(centers, radii, gap, p):
    n = len(radii)

    min_surface_gap = 1.0e30
    min_constraint_margin = 1.0e30
    max_overlap = 0.0
    max_violation = 0.0
    energy = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(centers[i], centers[j], p)

            surface_gap = d - radii[i] - radii[j]
            required = radii[i] + radii[j] + gap[i, j]
            margin = d - required

            min_surface_gap = min(min_surface_gap, surface_gap)
            min_constraint_margin = min(min_constraint_margin, margin)

            if surface_gap < 0.0:
                max_overlap = max(max_overlap, -surface_gap)

            if margin < 0.0:
                v = -margin
                max_violation = max(max_violation, v)
                energy += v * v

    return {
        "min_surface_gap": min_surface_gap,
        "min_constraint_margin": min_constraint_margin,
        "max_overlap": max_overlap,
        "max_violation": max_violation,
        "energy": energy,
    }


def get_bad_pairs(centers, radii, gap, p):
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
# Random growth relaxation
# ============================================================

def relax_one_stage(centers, radii, gap, p, rng, alpha, n_iter, shake_amp):
    """
    alpha gradually increases from small value to 1.0.
    required distance = alpha * (ri + rj + disij)
    """
    centers = centers.copy()
    n = len(radii)

    for it in range(n_iter):
        force = np.zeros_like(centers)
        max_v = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dx, dy = min_image(dx, dy, p)

                d = math.sqrt(dx * dx + dy * dy)
                required = alpha * (radii[i] + radii[j] + gap[i, j])
                v = required - d

                if v > 0.0:
                    max_v = max(max_v, v)

                    if d < 1.0e-12:
                        theta = rng.uniform(0.0, 2.0 * math.pi)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                    else:
                        nx = dx / d
                        ny = dy / d

                    # Repulsive force
                    f = 0.5 * v
                    force[i, 0] += nx * f
                    force[i, 1] += ny * f
                    force[j, 0] -= nx * f
                    force[j, 1] -= ny * f

        # limit each move
        step_limit = 0.30 if alpha < 0.9 else 0.10
        move_norm = np.sqrt(np.sum(force * force, axis=1))

        for k in range(n):
            if move_norm[k] > step_limit:
                force[k, :] *= step_limit / move_norm[k]

        centers += 0.55 * force

        # random shaking: keep randomness, avoid hex locking
        if shake_amp > 0.0:
            centers[:, 0] += rng.normal(0.0, shake_amp, n)
            centers[:, 1] += rng.normal(0.0, shake_amp, n)

        centers[:, 0] = centers[:, 0] % p.Lx_um
        centers[:, 1] = centers[:, 1] % p.Ly_um

        if max_v < p.tol_um:
            break

    return centers


def growth_relax(centers, radii, gap, p, rng):
    """
    Start from random distribution.
    Gradually grow required distance.
    """
    alphas = np.linspace(0.25, 1.0, p.growth_steps)

    for k, alpha in enumerate(alphas):
        # shaking decreases during growth
        frac = 1.0 - float(k) / float(max(len(alphas) - 1, 1))
        shake = p.random_shake_um * frac

        centers = relax_one_stage(
            centers, radii, gap, p, rng,
            alpha=float(alpha),
            n_iter=p.relax_iter_per_step,
            shake_amp=shake
        )

    # final polishing without shaking
    centers = relax_one_stage(
        centers, radii, gap, p, rng,
        alpha=1.0,
        n_iter=p.final_relax_iter,
        shake_amp=0.0
    )

    return centers



# ============================================================
# Monte-Carlo legal shuffling and candidate scoring
# ============================================================

def is_one_fiber_valid(k, centers, radii, gap, p):
    """
    Check whether fiber k satisfies all pairwise distance constraints.
    Used by Monte-Carlo legal shuffling.
    """
    n = len(radii)
    for j in range(n):
        if j == k:
            continue
        d = distance(centers[k], centers[j], p)
        required = radii[k] + radii[j] + gap[k, j]
        if d < required - p.tol_um:
            return False
    return True


def monte_carlo_shuffle(centers, radii, gap, p, rng,
                        steps_per_fiber=None,
                        move_amp_um=None):
    """
    Randomly perturb fibers after a valid layout has been obtained.

    The move is accepted only if all distance constraints remain satisfied.
    Therefore, this step cannot create fiber overlap or insufficient spacing.
    Its purpose is to sample the feasible space more randomly and reduce the
    near-contact pile-up produced by pure relaxation.
    """
    if steps_per_fiber is None:
        steps_per_fiber = p.mc_steps_per_fiber
    if move_amp_um is None:
        move_amp_um = p.mc_move_amp_um

    centers = centers.copy()
    n = len(radii)
    n_steps = int(max(0, steps_per_fiber) * n)

    if n_steps <= 0 or move_amp_um <= 0.0:
        return centers

    accept = 0

    for step in range(n_steps):
        k = int(rng.integers(0, n))
        old = centers[k].copy()

        # Gaussian trial move. Periodic wrapping is applied immediately.
        centers[k, 0] = (centers[k, 0] + rng.normal(0.0, move_amp_um)) % p.Lx_um
        centers[k, 1] = (centers[k, 1] + rng.normal(0.0, move_amp_um)) % p.Ly_um

        if is_one_fiber_valid(k, centers, radii, gap, p):
            accept += 1
        else:
            centers[k] = old

    print("MC shuffle: steps = %d, accept ratio = %.4f" % (n_steps, accept / max(n_steps, 1)))
    return centers

# ============================================================
# Periodic copies for Abaqus
# ============================================================

def build_main_dataframe(centers, radii):
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


def build_for_abaqus_dataframe(centers, radii, p):
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
# Plot
# ============================================================

def plot_rve(df_for_abaqus, p, save_path="rve_preview.png"):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.add_patch(Rectangle((0, 0), p.Lx_um, p.Ly_um, fill=False, linewidth=2.0))

    for _, row in df_for_abaqus.iterrows():
        alpha = 0.35 if bool(row["is_periodic_copy"]) else 0.75

        ax.add_patch(
            Circle(
                (row["x_um"], row["y_um"]),
                row["r_um"],
                fill=True,
                alpha=alpha,
                linewidth=0.6
            )
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * p.Lx_um, 1.1 * p.Lx_um)
    ax.set_ylim(-0.1 * p.Ly_um, 1.1 * p.Ly_um)
    ax.set_xlabel("x / μm")
    ax.set_ylabel("y / μm")
    ax.set_title("Random periodic RVE, target Vf = %.3f" % p.target_vf)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_failed(centers, radii, gap, p):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((0, 0), p.Lx_um, p.Ly_um, fill=False, linewidth=2.0))

    for c, r in zip(centers, radii):
        ax.add_patch(Circle((c[0], c[1]), r, fill=True, alpha=0.6))

    bad = get_bad_pairs(centers, radii, gap, p)

    for item in bad[:30]:
        i = item[0] - 1
        j = item[1] - 1
        ax.plot([centers[i, 0], centers[j, 0]],
                [centers[i, 1], centers[j, 1]], linewidth=1.2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * p.Lx_um, 1.1 * p.Lx_um)
    ax.set_ylim(-0.1 * p.Ly_um, 1.1 * p.Ly_um)
    ax.set_title("Failed layout")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("failed_layout.png", dpi=300)
    plt.show()
# ============================================================
# Randomness evaluation, similar to the paper
# ============================================================

def periodic_pairwise_distance_angle(centers, p):
    """
    Calculate periodic pairwise distance and angle matrix.

    distance: center-to-center distance under minimum-image convention
    angle   : angle from fiber i to fiber j, degree, range [0, 360)
    """
    n = len(centers)
    dist = np.zeros((n, n), dtype=float)
    angle = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = np.inf
                angle[i, j] = np.nan
                continue

            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]

            # periodic minimum image
            dx -= p.Lx_um * round(dx / p.Lx_um)
            dy -= p.Ly_um * round(dy / p.Ly_um)

            d = math.sqrt(dx * dx + dy * dy)
            theta = math.degrees(math.atan2(dy, dx))

            if theta < 0.0:
                theta += 360.0

            dist[i, j] = d
            angle[i, j] = theta

    return dist, angle


def evaluate_nearest_neighbor_distance(centers, radii, p):
    """
    Evaluate 1st, 2nd, and 3rd nearest-neighbor distances.
    The paper plots h/r, so here distances are normalized by mean radius.
    """
    dist, angle = periodic_pairwise_distance_angle(centers, p)

    sorted_dist = np.sort(dist, axis=1)

    r_mean = float(np.mean(radii))

    nn1 = sorted_dist[:, 0] / r_mean
    nn2 = sorted_dist[:, 1] / r_mean
    nn3 = sorted_dist[:, 2] / r_mean

    df = pd.DataFrame({
        "fiber_id": np.arange(1, len(centers) + 1),
        "nearest_1_h_over_r": nn1,
        "nearest_2_h_over_r": nn2,
        "nearest_3_h_over_r": nn3,
    })

    return df


def evaluate_nearest_neighbor_orientation(centers, p):
    """
    Evaluate nearest-neighbor orientation CDF.
    For CSR, the theoretical CDF is F(theta) = theta / 360.
    """
    dist, angle = periodic_pairwise_distance_angle(centers, p)

    nearest_id = np.argmin(dist, axis=1)

    nn_angle = []
    for i in range(len(centers)):
        nn_angle.append(angle[i, nearest_id[i]])

    nn_angle = np.array(nn_angle)
    nn_angle_sorted = np.sort(nn_angle)

    cdf = np.arange(1, len(nn_angle_sorted) + 1) / float(len(nn_angle_sorted))
    csr_cdf = nn_angle_sorted / 360.0

    df = pd.DataFrame({
        "orientation_degree": nn_angle_sorted,
        "cdf_algorithm": cdf,
        "cdf_CSR": csr_cdf,
        "cdf_difference": cdf - csr_cdf,
    })

    return df


def evaluate_ripley_k(centers, p, h_min_over_r=2.0, h_max_over_r=15.0, n_h=120):
    """
    Ripley's K function under periodic boundary condition.

    For CSR:
        K_CSR(h) = pi * h^2

    Since this RVE is periodic, edge correction is not used here.
    This is suitable for periodic RVE evaluation.
    """
    n = len(centers)
    area = p.Lx_um * p.Ly_um

    # pairwise distance
    dist, angle = periodic_pairwise_distance_angle(centers, p)

    # use mean radius from average nearest scale
    # here radius is not passed, so estimate h range with geometry size
    # h values are passed in absolute scale outside this function.
    h_values = np.linspace(h_min_over_r, h_max_over_r, n_h)

    # This function receives h already normalized outside?
    # For clarity, use h_values as absolute h if caller passes absolute.
    K_values = []

    for h in h_values:
        count = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dist[i, j] <= h:
                    count += 1

        # ordered pair count. Common periodic estimator: A * count / [N * (N - 1)]
        K = area * count / float(n * (n - 1))
        K_values.append(K)

    K_values = np.array(K_values)
    K_csr = math.pi * h_values ** 2

    df = pd.DataFrame({
        "h_um": h_values,
        "K_algorithm": K_values,
        "K_CSR": K_csr,
        "K_minus_KCSR": K_values - K_csr,
    })

    return df


def evaluate_pair_distribution_from_k(df_k):
    """
    Pair distribution function from Ripley's K:

        g(h) = 1 / (2*pi*h) * dK(h)/dh

    For CSR, g(h) = 1.
    """
    h = df_k["h_um"].values
    K = df_k["K_algorithm"].values

    dK_dh = np.gradient(K, h)

    g = dK_dh / (2.0 * math.pi * h)

    df = pd.DataFrame({
        "h_um": h,
        "g_algorithm": g,
        "g_CSR": np.ones_like(g),
        "g_minus_1": g - 1.0,
    })

    return df




def evaluate_pair_distribution_shell(centers, radii, p,
                                     h_min_over_r=None,
                                     h_max_over_r=None,
                                     dh_over_r=None,
                                     smooth_window=None):
    """
    Pair distribution function by direct annulus counting.

    This is more stable than differentiating the step-like Ripley's K curve,
    especially for finite RVEs and high fiber volume fraction.

    For a completely spatially random point process, g(h) = 1.
    For high-Vf fibers, g(h) usually has a near-contact peak and then
    oscillates around 1.
    """
    n = len(centers)
    area = p.Lx_um * p.Ly_um
    r_mean = float(np.mean(radii))

    if h_min_over_r is None:
        h_min_over_r = p.pair_h_min_over_r
    if h_max_over_r is None:
        h_max_over_r = p.pair_h_max_over_r
    if dh_over_r is None:
        dh_over_r = p.pair_dh_over_r
    if smooth_window is None:
        smooth_window = p.pair_smooth_window

    h_min = h_min_over_r * r_mean
    h_max = h_max_over_r * r_mean

    # With minimum-image periodic distance, the reliable radius should not
    # exceed half of the smaller side length.
    h_limit = 0.5 * min(p.Lx_um, p.Ly_um)
    h_max = min(h_max, h_limit)

    if h_max <= h_min:
        raise ValueError(
            "h_max <= h_min. Increase the RVE size or reduce pair_h_max_over_r."
        )

    dist, _ = periodic_pairwise_distance_angle(centers, p)
    pair_dist = dist[np.triu_indices(n, k=1)]  # unordered pairs, no double counting

    dh = dh_over_r * r_mean
    edges = np.arange(h_min, h_max + 0.5 * dh, dh)
    if edges[-1] < h_max:
        edges = np.append(edges, h_max)

    count_unordered, _ = np.histogram(pair_dist, bins=edges)
    h_center = 0.5 * (edges[:-1] + edges[1:])

    shell_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    # Expected unordered pair number for CSR inside each annulus.
    # rho=(N-1)/A is used to be consistent with finite-N normalization.
    rho = (n - 1) / area
    expected_unordered = 0.5 * n * rho * shell_area

    g_raw = count_unordered / np.maximum(expected_unordered, 1.0e-12)

    if smooth_window is not None and smooth_window > 1:
        g = pd.Series(g_raw).rolling(
            window=int(smooth_window),
            center=True,
            min_periods=1
        ).mean().values
    else:
        g = g_raw

    df = pd.DataFrame({
        "h_um": h_center,
        "h_over_r": h_center / r_mean,
        "count_unordered": count_unordered,
        "g_raw": g_raw,
        "g_algorithm": g,
        "g_CSR": np.ones_like(g),
        "g_minus_1": g - 1.0,
    })

    return df


def contact_peak_diagnostics(centers, radii, p):
    """
    Count how many fiber pairs are located very close to h/r≈2.
    A large value means the first peak of pair distribution will be high.
    """
    n = len(centers)
    r_mean = float(np.mean(radii))
    dist, _ = periodic_pairwise_distance_angle(centers, p)
    pair_dist = dist[np.triu_indices(n, k=1)]
    h_over_r = pair_dist / r_mean

    low = p.contact_low_over_r
    high = p.contact_high_over_r
    count = int(np.sum((h_over_r >= low) & (h_over_r < high)))
    total = int(len(h_over_r))
    fraction = count / max(total, 1)

    return {
        "pair_total": total,
        "min_h_over_r": float(np.min(h_over_r)),
        "contact_interval_low": low,
        "contact_interval_high": high,
        "contact_pair_count": count,
        "contact_pair_fraction": fraction,
    }


def layout_randomness_score(centers, radii, p):
    """
    A quick scalar score used to choose among several valid candidates.

    The score is intentionally dominated by the direct pair-distribution RMSE
    because the user's current problem is the excessive first peak of g(h).
    """
    df_g = evaluate_pair_distribution_shell(centers, radii, p)
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)
    contact = contact_peak_diagnostics(centers, radii, p)

    g_rmse = math.sqrt(np.mean((df_g["g_algorithm"].values - 1.0) ** 2))
    orientation_ks = np.max(np.abs(df_ori["cdf_difference"].values))
    contact_fraction = contact["contact_pair_fraction"]

    # Weights are empirical. They favor reducing the excessive near-contact
    # peak while still keeping orientation reasonably random.
    score = g_rmse + 0.25 * orientation_ks + 20.0 * contact_fraction

    return score, {
        "score": score,
        "pair_distribution_RMSE_shell": g_rmse,
        "orientation_CDF_KS_distance": orientation_ks,
        "contact_pair_fraction": contact_fraction,
        "min_h_over_r": contact["min_h_over_r"],
    }

def plot_nearest_neighbor_distance(df_nn, save_path):
    plt.figure(figsize=(7, 5))

    bins = 18

    plt.hist(
        df_nn["nearest_1_h_over_r"].values,
        bins=bins,
        density=True,
        alpha=0.45,
        label="1st nearest"
    )

    plt.hist(
        df_nn["nearest_2_h_over_r"].values,
        bins=bins,
        density=True,
        alpha=0.45,
        label="2nd nearest"
    )

    plt.hist(
        df_nn["nearest_3_h_over_r"].values,
        bins=bins,
        density=True,
        alpha=0.45,
        label="3rd nearest"
    )

    plt.xlabel("h / r")
    plt.ylabel("PDF")
    plt.title("Nearest-neighbor distance distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_nearest_neighbor_orientation(df_ori, save_path):
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_ori["orientation_degree"].values,
        df_ori["cdf_algorithm"].values,
        linewidth=2.0,
        label="Generated RVE"
    )

    plt.plot(
        df_ori["orientation_degree"].values,
        df_ori["cdf_CSR"].values,
        linestyle="--",
        linewidth=2.0,
        label="CSR"
    )

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


def plot_ripley_k(df_k, radii, save_path):
    r_mean = float(np.mean(radii))

    plt.figure(figsize=(7, 5))

    plt.plot(
        df_k["h_um"].values / r_mean,
        df_k["K_algorithm"].values,
        linewidth=2.0,
        label="Generated RVE"
    )

    plt.plot(
        df_k["h_um"].values / r_mean,
        df_k["K_CSR"].values,
        linestyle="--",
        linewidth=2.0,
        label="CSR"
    )

    plt.xlabel("h / r")
    plt.ylabel("K(h)")
    plt.title("Ripley's K function")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_pair_distribution(df_g, radii, save_path):
    r_mean = float(np.mean(radii))

    if "h_over_r" in df_g.columns:
        x = df_g["h_over_r"].values
    else:
        x = df_g["h_um"].values / r_mean

    plt.figure(figsize=(7, 5))

    plt.plot(
        x,
        df_g["g_algorithm"].values,
        linewidth=2.0,
        label="Generated RVE"
    )

    plt.plot(
        x,
        df_g["g_CSR"].values,
        linestyle="--",
        linewidth=2.0,
        label="CSR"
    )

    plt.xlabel("h / r")
    plt.ylabel("g(h)")
    plt.title("Pair distribution function")
    plt.xlim(float(np.min(x)), float(np.max(x)))

    # Keep the plot readable. The raw values are still saved in Excel.
    ymax = max(3.0, min(8.0, 1.15 * float(np.nanmax(df_g["g_algorithm"].values))))
    plt.ylim(0.0, ymax)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def evaluate_randomness(centers, radii, p, output_prefix="fiber_centers"):
    """
    Run all randomness evaluations similar to the paper:
    1. nearest neighbor distance
    2. nearest neighbor orientation
    3. Ripley's K function
    4. pair distribution function
    """
    print("\n" + "=" * 80)
    print("Randomness evaluation starts...")
    print("=" * 80)

    r_mean = float(np.mean(radii))

    # 1. nearest neighbor distance
    df_nn = evaluate_nearest_neighbor_distance(centers, radii, p)

    # 2. nearest neighbor orientation
    df_ori = evaluate_nearest_neighbor_orientation(centers, p)

    # 3. Ripley's K
    h_min = 2.0 * r_mean
    h_max = min(15.0 * r_mean, 0.5 * min(p.Lx_um, p.Ly_um))
    df_k = evaluate_ripley_k(
        centers,
        p,
        h_min_over_r=h_min,
        h_max_over_r=h_max,
        n_h=120
    )

    # 4. pair distribution function
    # Direct annulus counting is used because differentiating K(h) strongly
    # amplifies finite-sample jumps near h/r≈2.
    df_g = evaluate_pair_distribution_shell(centers, radii, p)
    contact_diag = contact_peak_diagnostics(centers, radii, p)

    # Summary indices
    orientation_ks = np.max(np.abs(df_ori["cdf_difference"].values))

    k_rmse = math.sqrt(
        np.mean(
            (
                (df_k["K_algorithm"].values - df_k["K_CSR"].values)
                / np.maximum(df_k["K_CSR"].values, 1.0e-12)
            ) ** 2
        )
    )

    g_rmse = math.sqrt(
        np.mean(
            (df_g["g_algorithm"].values - 1.0) ** 2
        )
    )

    summary = pd.DataFrame([{
        "fiber_number": len(centers),
        "mean_radius_um": r_mean,
        "nearest_1_mean_h_over_r": df_nn["nearest_1_h_over_r"].mean(),
        "nearest_1_std_h_over_r": df_nn["nearest_1_h_over_r"].std(),
        "nearest_2_mean_h_over_r": df_nn["nearest_2_h_over_r"].mean(),
        "nearest_3_mean_h_over_r": df_nn["nearest_3_h_over_r"].mean(),
        "orientation_CDF_KS_distance": orientation_ks,
        "Ripley_K_relative_RMSE": k_rmse,
        "pair_distribution_RMSE": g_rmse,
        "contact_pair_fraction": contact_diag["contact_pair_fraction"],
        "contact_pair_count": contact_diag["contact_pair_count"],
        "min_h_over_r": contact_diag["min_h_over_r"],
    }])

    # Save figures
    plot_nearest_neighbor_distance(
        df_nn,
        save_path=output_prefix + "_eval_nearest_distance.png"
    )

    plot_nearest_neighbor_orientation(
        df_ori,
        save_path=output_prefix + "_eval_orientation_cdf.png"
    )

    plot_ripley_k(
        df_k,
        radii,
        save_path=output_prefix + "_eval_ripley_k.png"
    )

    plot_pair_distribution(
        df_g,
        radii,
        save_path=output_prefix + "_eval_pair_distribution.png"
    )

    # Save Excel
    eval_xlsx = output_prefix + "_randomness_evaluation.xlsx"

    with pd.ExcelWriter(eval_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df_nn.to_excel(writer, sheet_name="nearest_distance", index=False)
        df_ori.to_excel(writer, sheet_name="orientation_cdf", index=False)
        df_k.to_excel(writer, sheet_name="ripley_k", index=False)
        df_g.to_excel(writer, sheet_name="pair_distribution", index=False)

    print("Randomness evaluation finished.")
    print("Evaluation file saved:")
    print(eval_xlsx)
    print("=" * 80)

    print("Key evaluation indices:")
    print("orientation CDF KS distance = %.6e" % orientation_ks)
    print("Ripley K relative RMSE      = %.6e" % k_rmse)
    print("pair distribution RMSE      = %.6e" % g_rmse)
    print("contact pair fraction       = %.6e" % contact_diag["contact_pair_fraction"])
    print("min h/r                     = %.6e" % contact_diag["min_h_over_r"])
    print("=" * 80)

    return summary, df_nn, df_ori, df_k, df_g

# ============================================================
# Main
# ============================================================

def generate(p):
    # ------------------------------------------------------------
    # Random seed control
    # ------------------------------------------------------------
    # If p.seed is None, a new seed is generated from the current time.
    # Therefore, every run will generate a different RVE.
    # If p.seed is an integer, the result is fully reproducible.
    if p.seed is None:
        p.seed = int(time.time() * 1000000) % (2**32)

    print("=" * 80)
    print("Random seed used in this run: %d" % p.seed)
    print("=" * 80)

    random.seed(p.seed)
    rng = np.random.default_rng(p.seed)

    n = estimate_fiber_number(p)
    radii = generate_radii(p, n, rng)
    vf = actual_vf(radii, p)

    print("=" * 80)
    print("Random periodic RVE generation")
    print("Lx, Ly         = %.6f, %.6f um" % (p.Lx_um, p.Ly_um))
    print("target Vf      = %.6f" % p.target_vf)
    print("actual Vf      = %.6f" % vf)
    print("fiber number   = %d" % n)
    print("diameter mode  = %s" % p.diameter_mode)
    print("lmin, lmax     = %.6f, %.6f um" % (p.lmin_um, p.lmax_um))
    print("=" * 80)

    best_centers = None
    best_gap = None
    best_metrics = None
    best_score = None
    best_score_info = None

    # Backup for reporting if no valid layout is found.
    best_invalid_centers = None
    best_invalid_gap = None
    best_invalid_metrics = None

    valid_found = 0

    for restart in range(1, p.max_restarts + 1):
        print("\nrestart %03d starts..." % restart, flush=True)

        gap = make_gap_matrix(p, n, rng)
        centers = random_initial_centers(p, n, rng)

        centers = growth_relax(centers, radii, gap, p, rng)
        metrics = compute_metrics(centers, radii, gap, p)

        print(
            "restart %03d | min_gap = %.6e | min_margin = %.6e | "
            "max_overlap = %.6e | max_violation = %.6e | energy = %.6e"
            % (
                restart,
                metrics["min_surface_gap"],
                metrics["min_constraint_margin"],
                metrics["max_overlap"],
                metrics["max_violation"],
                metrics["energy"],
            )
        )

        # Track the best invalid result only for debugging/failure report.
        if best_invalid_metrics is None or metrics["energy"] < best_invalid_metrics["energy"]:
            best_invalid_centers = centers.copy()
            best_invalid_gap = gap.copy()
            best_invalid_metrics = metrics.copy()

        is_valid = (
            metrics["max_overlap"] <= p.tol_um and
            metrics["max_violation"] <= p.tol_um
        )

        if not is_valid:
            continue

        valid_found += 1
        print("Valid candidate %d found." % valid_found)

        candidate_centers = centers.copy()

        if p.use_mc_shuffle:
            candidate_centers = monte_carlo_shuffle(
                candidate_centers,
                radii,
                gap,
                p,
                rng,
                steps_per_fiber=p.mc_steps_per_fiber,
                move_amp_um=p.mc_move_amp_um
            )

        candidate_metrics = compute_metrics(candidate_centers, radii, gap, p)

        # Safety check after Monte-Carlo shuffle.
        if (
            candidate_metrics["max_overlap"] > p.tol_um or
            candidate_metrics["max_violation"] > p.tol_um
        ):
            print("Warning: candidate became invalid after MC shuffle. It is skipped.")
            continue

        if p.selection_mode == "first_valid":
            best_centers = candidate_centers.copy()
            best_gap = gap.copy()
            best_metrics = candidate_metrics.copy()
            best_score = 0.0
            best_score_info = {"score": 0.0}
            print("\nAccepted first valid layout.")
            break

        candidate_score, candidate_score_info = layout_randomness_score(
            candidate_centers,
            radii,
            p
        )

        print(
            "candidate randomness score = %.6e | g_RMSE = %.6e | "
            "contact_fraction = %.6e | min_h/r = %.6f"
            % (
                candidate_score_info["score"],
                candidate_score_info["pair_distribution_RMSE_shell"],
                candidate_score_info["contact_pair_fraction"],
                candidate_score_info["min_h_over_r"],
            )
        )

        if best_score is None or candidate_score < best_score:
            best_centers = candidate_centers.copy()
            best_gap = gap.copy()
            best_metrics = candidate_metrics.copy()
            best_score = candidate_score
            best_score_info = candidate_score_info.copy()
            print("This candidate is currently the best valid layout.")

        if valid_found >= p.valid_candidates_to_test:
            print("\nRequired number of valid candidates reached.")
            break

    # If no valid candidate survived the selection procedure, use the best
    # invalid result for failure reporting.
    if best_metrics is None:
        best_centers = best_invalid_centers
        best_gap = best_invalid_gap
        best_metrics = best_invalid_metrics

    if best_score_info is not None:
        print("\nBest valid candidate score information:")
        for key, value in best_score_info.items():
            if isinstance(value, float):
                print("  %s = %.6e" % (key, value))
            else:
                print("  %s = %s" % (key, value))

    if best_metrics["max_overlap"] > p.tol_um or best_metrics["max_violation"] > p.tol_um:
        print("\nGeneration failed.")
        print("Best metrics:")
        print(best_metrics)

        bad = get_bad_pairs(best_centers, radii, best_gap, p)
        print("Bad pair number:", len(bad))
        for item in bad[:20]:
            print(
                "fiber %d - %d | d = %.6f | required = %.6f | violation = %.6e"
                % item
            )

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
        "lmin_um": p.lmin_um,
        "lmax_um": p.lmax_um,
        "selection_mode": p.selection_mode,
        "valid_candidates_found": valid_found,
        "use_mc_shuffle": p.use_mc_shuffle,
        "mc_steps_per_fiber": p.mc_steps_per_fiber,
        "mc_move_amp_um": p.mc_move_amp_um,
        "best_randomness_score": best_score,
        "min_surface_gap_um": best_metrics["min_surface_gap"],
        "min_constraint_margin_um": best_metrics["min_constraint_margin"],
        "max_overlap_um": best_metrics["max_overlap"],
        "max_violation_um": best_metrics["max_violation"],
    }])

    main_csv = p.output_prefix + "_main.csv"
    abaqus_csv = p.output_prefix + "_for_abaqus.csv"
    xlsx_file = p.output_prefix + ".xlsx"

    df_main.to_csv(main_csv, index=False, encoding="utf-8-sig")
    df_for_abaqus.to_csv(abaqus_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="main_centers", index=False)
        df_for_abaqus.to_excel(writer, sheet_name="for_abaqus", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    plot_rve(df_for_abaqus, p)
    evaluate_randomness(best_centers, radii, p, output_prefix=p.output_prefix)
    print("\n" + "=" * 80)
    print("Generation succeeded.")
    print("Saved files:")
    print(main_csv)
    print(abaqus_csv)
    print(xlsx_file)
    print("rve_preview.png")
    print("=" * 80)
    print("actual Vf       = %.8f" % vf)
    print("min gap         = %.8e um" % best_metrics["min_surface_gap"])
    print("min margin      = %.8e um" % best_metrics["min_constraint_margin"])
    print("max overlap     = %.8e um" % best_metrics["max_overlap"])
    print("max violation   = %.8e um" % best_metrics["max_violation"])
    print("=" * 80)


if __name__ == "__main__":

    params = RVEParams(
        # 100 x 100 um gives much more stable statistics than 50 x 50 um.
        # If you need the full h/r = 15 range strictly, 120 x 120 um is better.
        Lx_um=100.0,
        Ly_um=100.0,

        target_vf=0.70,

        # Using a normal diameter distribution usually reduces the artificial
        # near-contact peak compared with perfectly identical fibers.
        diameter_mode="normal",
        diameter_mean_um=7.0,
        diameter_std_um=0.317,

        # If you must use the exact narrow paper setting, change this back to
        # 0.10-0.15. A wider gap interval helps avoid all near contacts being
        # concentrated at the same h/r value.
        lmin_um=0.1,
        lmax_um=0.15,

        max_restarts=200,

        growth_steps=45,
        relax_iter_per_step=1200,
        final_relax_iter=20000,

        # Random shaking during growth. Larger values keep more randomness
        # but may make convergence harder.
        random_shake_um=0.08,

        tol_um=1.0e-7,

        selection_mode="best_valid_randomness",
        valid_candidates_to_test=5,

        use_mc_shuffle=True,
        mc_steps_per_fiber=800,
        mc_move_amp_um=0.06,

        pair_h_min_over_r=2.0,
        pair_h_max_over_r=15.0,
        pair_dh_over_r=0.35,
        pair_smooth_window=5,

        contact_low_over_r=2.0,
        contact_high_over_r=2.10,

        # seed=None: every run generates a different RVE.
        # seed=2026: every run reproduces the same RVE.
        seed=2026,
        output_prefix="fiber_centers_100um_randomness_selected"
    )

    generate(params)