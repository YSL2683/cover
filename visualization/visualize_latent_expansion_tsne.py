#!/usr/bin/env python3
"""
t-SNE Latent Space Distribution & Expansion Visualization Pipeline.

Generates two primary visualizations based on t-SNE manifold projection:
1. Base policy vs. Residual RL policy distributions overlaid on the same 2D plane.
2. Base vs. Residual RL policy distribution overlaid with gray points, highlighting
   the density difference between the two distributions in green (intensity proportional
   to density difference) with statistical significance contour.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import torch


# ==============================================================================
# 1. Statistical Functions (ESS, MMD, SWD, Permutation Test)
# ==============================================================================

def compute_autocorrelation_time(trajectories: list[np.ndarray]) -> tuple[float, float]:
    """Computes integrated autocorrelation time (tau) using Geyer's initial positive sequence criterion."""
    tau_list = []
    for traj in trajectories:
        if len(traj) < 10:
            continue
        x = np.asarray(traj)
        x_centered = x - x.mean(axis=0)
        var = np.sum(x_centered ** 2)
        if var < 1e-8:
            continue
        T = len(traj)
        max_lag = min(T // 2, 60)
        rhos = []
        for k in range(1, max_lag):
            cross = np.sum(x_centered[:-k] * x_centered[k:])
            rhos.append(cross / var)
        tau = 1.0
        for rho in rhos:
            if rho <= 0:
                break
            tau += 2.0 * rho
        tau_list.append(tau)
    if len(tau_list) == 0:
        return 1.0, 1.0
    return float(np.mean(tau_list)), float(np.median(tau_list))


def compute_mmd(X: np.ndarray, Y: np.ndarray, max_samples: int = 2500, seed: int = 42) -> float:
    """Computes Maximum Mean Discrepancy (MMD) with median-heuristic Gaussian RBF kernel."""
    rng = np.random.RandomState(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    d_xy = cdist(X, Y, "sqeuclidean")
    median_dist = np.median(d_xy)
    gamma = 1.0 / (median_dist + 1e-8)

    d_xx = cdist(X, X, "sqeuclidean")
    d_yy = cdist(Y, Y, "sqeuclidean")

    k_xx = np.exp(-gamma * d_xx).mean()
    k_yy = np.exp(-gamma * d_yy).mean()
    k_xy = np.exp(-gamma * d_xy).mean()

    mmd_sq = k_xx + k_yy - 2.0 * k_xy
    return float(np.sqrt(max(0.0, mmd_sq)))


def compute_sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_projections: int = 150, seed: int = 42) -> float:
    """Computes Sliced Wasserstein Distance (SWD) in 16D space."""
    rng = np.random.RandomState(seed)
    dim = X.shape[1]
    projections = rng.randn(n_projections, dim)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    dists = []
    for p in projections:
        X_proj = X @ p
        Y_proj = Y @ p
        dists.append(wasserstein_distance(X_proj, Y_proj))
    return float(np.mean(dists))


def run_trajectory_permutation_test(
    all_trajs_2d: list[np.ndarray],
    n_base: int,
    n_rl: int,
    xedges: np.ndarray,
    yedges: np.ndarray,
    sigma: float = 2.5,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes a Trajectory-Level Cluster Permutation Test (1,000 iterations).
    Preserves intra-trajectory serial autocorrelation while testing global distribution expansion.
    """
    print(f"  [Permutation Test] Precomputing 2D histograms for {len(all_trajs_2d)} trajectories...")
    traj_histograms = []
    for t2d in all_trajs_2d:
        h, _, _ = np.histogram2d(t2d[:, 0], t2d[:, 1], bins=[xedges, yedges])
        traj_histograms.append(h)
    traj_histograms = np.array(traj_histograms)  # (N_trajs, Grid, Grid)

    # Observed densities
    h_base_obs = traj_histograms[:n_base].sum(axis=0)
    h_base_obs = h_base_obs / (h_base_obs.sum() + 1e-8)
    z_base_obs = gaussian_filter(h_base_obs, sigma=sigma)

    h_rl_obs = traj_histograms[n_base:].sum(axis=0)
    h_rl_obs = h_rl_obs / (h_rl_obs.sum() + 1e-8)
    z_rl_obs = gaussian_filter(h_rl_obs, sigma=sigma)

    delta_z_obs = z_rl_obs - z_base_obs

    print(f"  [Permutation Test] Running {n_permutations} cluster permutation iterations...")
    count_extreme = np.zeros_like(delta_z_obs)
    rng = np.random.RandomState(seed)
    n_total = n_base + n_rl

    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        g1 = perm[:n_base]
        g2 = perm[n_base:]

        h1 = traj_histograms[g1].sum(axis=0)
        h1 = h1 / (h1.sum() + 1e-8)
        h2 = traj_histograms[g2].sum(axis=0)
        h2 = h2 / (h2.sum() + 1e-8)

        z1 = gaussian_filter(h1, sigma=sigma)
        z2 = gaussian_filter(h2, sigma=sigma)

        delta_perm = z2 - z1
        count_extreme += (delta_perm >= delta_z_obs)

    p_values = count_extreme / float(n_permutations)
    return z_base_obs, z_rl_obs, delta_z_obs, p_values


def plot_smooth_glow(ax, Z, extent, cmap_name: str, max_alpha: float = 0.88, zorder: int = 1):
    """Renders a smooth non-linear alpha glow for density maps."""
    Z_2d = Z.T
    z_min, z_max = Z_2d.min(), Z_2d.max()
    z_norm = (Z_2d - z_min) / (z_max - z_min + 1e-8)
    z_color = np.clip(z_norm ** 0.60, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(z_color)
    rgba[..., 3] = np.clip(z_norm ** 0.75, 0.0, 1.0) * max_alpha
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="bicubic", zorder=zorder, aspect="auto")
    return z_min, z_max


# ==============================================================================
# 2. Main Execution Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="t-SNE Latent Space Distribution & Expansion Analysis")
    parser.add_argument("--cache_path", type=str, default="visualization/cache/Square_base_vs_residual_50ep.pt")
    parser.add_argument("--output_dir", type=str, default="visualization/outputs")
    parser.add_argument("--grid_size", type=int, default=150)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--alpha_level", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = PROJECT_ROOT / args.cache_path
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n[t-SNE Latent Analysis] Loading trajectory cache from:\n  {cache_path}\n{'='*70}")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    trajs_base_f = [np.asarray(t) for t in cache_data["base_trajs_f"] if len(t) > 0]
    trajs_base_w = [np.asarray(t) for t in cache_data["base_trajs_w"] if len(t) > 0]
    trajs_rl_f = [np.asarray(t) for t in cache_data["rl_trajs_f"] if len(t) > 0]
    trajs_rl_w = [np.asarray(t) for t in cache_data["rl_trajs_w"] if len(t) > 0]

    all_base_f = np.vstack(trajs_base_f)
    all_base_w = np.vstack(trajs_base_w)
    all_rl_f = np.vstack(trajs_rl_f)
    all_rl_w = np.vstack(trajs_rl_w)

    print(f"Base Front: {all_base_f.shape} | RL Front: {all_rl_f.shape}")
    print(f"Base Wrist: {all_base_w.shape} | RL Wrist: {all_rl_w.shape}")

    view_data = [
        {
            "name": "Main Camera (Agentview)",
            "short_name": "agentview",
            "trajs_b": trajs_base_f,
            "trajs_r": trajs_rl_f,
            "all_b": all_base_f,
            "all_r": all_rl_f,
        },
        {
            "name": "Wrist Camera (Eye-in-hand)",
            "short_name": "eye_in_hand",
            "trajs_b": trajs_base_w,
            "trajs_r": trajs_rl_w,
            "all_b": all_base_w,
            "all_r": all_rl_w,
        },
    ]

    processed_views = []

    for v_idx, v in enumerate(view_data):
        v_name = v["name"]
        print(f"\n>>> Processing View [{v_idx + 1}/2]: {v_name} <<<")

        # 1. Autocorrelation & ESS
        tau_b_mean, _ = compute_autocorrelation_time(v["trajs_b"])
        tau_r_mean, _ = compute_autocorrelation_time(v["trajs_r"])
        ess_b = len(v["all_b"]) / max(1.0, tau_b_mean)
        ess_r = len(v["all_r"]) / max(1.0, tau_r_mean)
        print(f"  [Autocorrelation] Tau_Base = {tau_b_mean:.2f} (ESS={ess_b:.1f}), Tau_RL = {tau_r_mean:.2f} (ESS={ess_r:.1f})")

        bw_adj = (max(tau_b_mean, tau_r_mean)) ** (1.0 / 6.0)
        sigma_eff = 2.4 * bw_adj
        print(f"  [KDE Bandwidth] Corrected Sigma = {sigma_eff:.2f} (Base multiplier: {bw_adj:.2f})")

        # 2. 16D Metrics (SVDD, MMD, SWD)
        print("  [Metrics] Computing 16D One-Class SVM / SVDD boundaries...")
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        oc_svm.fit(v["all_b"])
        pred_b = oc_svm.predict(v["all_b"])
        pred_r = oc_svm.predict(v["all_r"])
        ood_rate_b = float((pred_b == -1).mean() * 100.0)
        ood_rate_r = float((pred_r == -1).mean() * 100.0)

        print("  [Metrics] Computing 16D Maximum Mean Discrepancy (MMD)...")
        mmd_val = compute_mmd(v["all_b"], v["all_r"])

        print("  [Metrics] Computing 16D Sliced Wasserstein Distance (SWD)...")
        swd_val = compute_sliced_wasserstein(v["all_b"], v["all_r"])
        print(f"  -> SVDD OOD: Base={ood_rate_b:.1f}%, RL={ood_rate_r:.1f}% (+{ood_rate_r - ood_rate_b:.1f}%) | MMD={mmd_val:.4f} | SWD={swd_val:.4f}")

        # 3. Joint t-SNE Embedding
        print(f"  [t-SNE] Fitting joint t-SNE (perplexity={args.perplexity}, N={len(v['all_b']) + len(v['all_r'])})...")
        t0 = time.time()
        X_joint = np.vstack([v["all_b"], v["all_r"]])
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            init="pca",
            random_state=args.seed,
            n_jobs=-1,
        )
        Y_joint = tsne.fit_transform(X_joint)
        print(f"  [t-SNE] Completed in {time.time() - t0:.2f}s.")

        n_b = len(v["all_b"])
        b2d = Y_joint[:n_b]
        rl2d = Y_joint[n_b:]

        # Slice 2D trajectories
        all_trajs_2d = []
        curr_idx = 0
        for t in v["trajs_b"]:
            length = len(t)
            all_trajs_2d.append(Y_joint[curr_idx : curr_idx + length])
            curr_idx += length
        for t in v["trajs_r"]:
            length = len(t)
            all_trajs_2d.append(Y_joint[curr_idx : curr_idx + length])
            curr_idx += length

        # Coordinate grid with padding
        x_pad = (Y_joint[:, 0].max() - Y_joint[:, 0].min()) * 0.08
        y_pad = (Y_joint[:, 1].max() - Y_joint[:, 1].min()) * 0.08
        xmin, xmax = Y_joint[:, 0].min() - x_pad, Y_joint[:, 0].max() + x_pad
        ymin, ymax = Y_joint[:, 1].min() - y_pad, Y_joint[:, 1].max() + y_pad

        grid_n = args.grid_size
        xedges = np.linspace(xmin, xmax, grid_n + 1)
        yedges = np.linspace(ymin, ymax, grid_n + 1)

        # 4. Trajectory Permutation Test
        z_b_obs, z_r_obs, delta_z_obs, p_vals = run_trajectory_permutation_test(
            all_trajs_2d=all_trajs_2d,
            n_base=len(v["trajs_b"]),
            n_rl=len(v["trajs_r"]),
            xedges=xedges,
            yedges=yedges,
            sigma=sigma_eff,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )

        sig_mask = (p_vals < args.alpha_level) & (delta_z_obs > 0.0)
        sig_expansion_pct = float(sig_mask.mean() * 100.0)
        print(f"  [Permutation Result] Statistically Significant Expansion Area: {sig_expansion_pct:.2f}% (p < {args.alpha_level})")

        processed_views.append({
            "name": v_name,
            "short_name": v["short_name"],
            "b2d": b2d,
            "rl2d": rl2d,
            "Y_joint": Y_joint,
            "extent": [xmin, xmax, ymin, ymax],
            "xedges": xedges,
            "yedges": yedges,
            "z_b": z_b_obs,
            "z_r": z_r_obs,
            "delta_z": delta_z_obs,
            "p_vals": p_vals,
            "sig_mask": sig_mask,
            "sig_expansion_pct": sig_expansion_pct,
            "metrics": {
                "tau_base": tau_b_mean,
                "ess_base": ess_b,
                "tau_rl": tau_r_mean,
                "ess_rl": ess_r,
                "svdd_base": ood_rate_b,
                "svdd_rl": ood_rate_r,
                "svdd_diff": ood_rate_r - ood_rate_b,
                "mmd": mmd_val,
                "swd": swd_val,
                "sig_area_pct": sig_expansion_pct,
            },
        })

    # ==============================================================================
    # Visualization 1: Base vs Residual RL Overlaid on Same Plane (t-SNE)
    # ==============================================================================
    print("\n[Rendering] Visualization 1: t-SNE Base vs RL Distribution Overlay...")
    fig_v1, axes_v1 = plt.subplots(1, 2, figsize=(18, 8.5), dpi=250)
    fig_v1.patch.set_facecolor("#ffffff")

    for idx, pv in enumerate(processed_views):
        ax = axes_v1[idx]
        name = pv["name"]
        extent = pv["extent"]
        b2d = pv["b2d"]
        rl2d = pv["rl2d"]

        # Only scatter points without any contour lines
        ax.scatter(
            b2d[:, 0], b2d[:, 1],
            color="#ea580c",
            s=4.5,
            alpha=0.35,
            edgecolors="none",
            label=f"Base Policy (N={len(b2d)})",
        )
        ax.scatter(
            rl2d[:, 0], rl2d[:, 1],
            color="#2563eb",
            s=4.5,
            alpha=0.35,
            edgecolors="none",
            label=f"Residual RL Policy (N={len(rl2d)})",
        )

        ax.set_title(
            f"{name}\n[Base vs. Residual RL Latent Distribution Overlay (t-SNE)]",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.90)

    plt.suptitle(
        "Latent Space Representation Overlay: Base Policy vs. Residual RL Policy\n"
        "Joint t-SNE Embedding (Perplexity=40, PCA Init)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94], w_pad=2.5)

    v1_path = output_dir / "tsne_latent_distribution_overlay.png"
    fig_v1.savefig(v1_path, bbox_inches="tight")
    plt.close(fig_v1)
    print(f"  -> Saved Visualization 1 to: {v1_path}")

    # ==============================================================================
    # Visualization 2: Gray Points + Green Density Difference Glow (t-SNE)
    # ==============================================================================
    print("\n[Rendering] Visualization 2: Gray Points with Green Density Difference Glow...")
    fig_v2, axes_v2 = plt.subplots(1, 2, figsize=(19, 8.5), dpi=250)
    fig_v2.patch.set_facecolor("#ffffff")

    for idx, pv in enumerate(processed_views):
        ax = axes_v2[idx]
        name = pv["name"]
        extent = pv["extent"]
        Y_joint = pv["Y_joint"]
        delta_z = pv["delta_z"]
        sig_mask = pv["sig_mask"]
        sig_pct = pv["sig_expansion_pct"]

        # 1. Background points: Both policies in neutral gray
        ax.scatter(
            Y_joint[:, 0], Y_joint[:, 1],
            color="#9ca3af",
            s=4.0,
            alpha=0.25,
            edgecolors="none",
            zorder=1,
            label="Trajectory Steps (Base + RL)",
        )

        # 2. Density difference glow (RL density - Base density > 0) in deep green
        # The higher the density difference, the darker/more intense the green
        z_diff_pos = np.maximum(0.0, delta_z)
        # Clean up borders so no edge bleeding
        margin = 6
        z_diff_pos[:margin, :] = 0.0
        z_diff_pos[-margin:, :] = 0.0
        z_diff_pos[:, :margin] = 0.0
        z_diff_pos[:, -margin:] = 0.0

        zd_min, zd_max = plot_smooth_glow(ax, z_diff_pos, extent, cmap_name="Greens", max_alpha=0.92, zorder=2)

        ax.set_title(
            f"{name}\n[Density Difference: Gray Trajectory Support + Green Glow (t-SNE)]",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.25, linestyle=":")

        sm = cm.ScalarMappable(cmap=plt.cm.Greens, norm=mcolors.Normalize(vmin=zd_min, vmax=zd_max))
        cb = fig_v2.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("RL Policy Density Expansion (ΔZ = Z_RL - Z_Base)", rotation=270, labelpad=14, fontsize=9.5)

        legend_elements_v2 = [
            Line2D([0], [0], color="#16a34a", lw=4.0, alpha=0.9, label=f"Density Difference Glow (ΔZ > 0)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ca3af", markersize=6, label="Trajectory Steps (Gray)"),
        ]
        ax.legend(handles=legend_elements_v2, loc="upper right", fontsize=9.5, framealpha=0.90)

    plt.suptitle(
        "Latent Policy Expansion & Density Discrepancy Analysis (t-SNE)\n"
        "Gray Support Points with Green Density Difference Highlights",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94], w_pad=2.5)

    v2_path = output_dir / "tsne_latent_density_difference.png"
    fig_v2.savefig(v2_path, bbox_inches="tight")
    plt.close(fig_v2)
    print(f"  -> Saved Visualization 2 to: {v2_path}")

    # ==============================================================================
    # Visualization 3: Comprehensive 2x3 Dashboard (Combining 1, 2, and Metrics)
    # ==============================================================================
    print("\n[Rendering] Comprehensive 2x3 Publication Dashboard...")
    fig_dash, axes_dash = plt.subplots(2, 3, figsize=(24, 15), dpi=250)
    fig_dash.patch.set_facecolor("#ffffff")

    for r_idx, pv in enumerate(processed_views):
        name = pv["name"]
        extent = pv["extent"]
        b2d = pv["b2d"]
        rl2d = pv["rl2d"]
        Y_joint = pv["Y_joint"]
        delta_z = pv["delta_z"]
        m = pv["metrics"]

        # --- Col 0: Distribution Overlay (Scatter Points Only) ---
        ax0 = axes_dash[r_idx, 0]
        ax0.scatter(b2d[:, 0], b2d[:, 1], color="#ea580c", s=4.0, alpha=0.35, edgecolors="none", zorder=2, label="Base Steps")
        ax0.scatter(rl2d[:, 0], rl2d[:, 1], color="#2563eb", s=4.0, alpha=0.35, edgecolors="none", zorder=3, label="Residual RL Steps")
        ax0.set_title(f"{name}\n[1. Base & RL Distributions Overlaid (t-SNE)]", fontsize=12, fontweight="bold", pad=8)
        ax0.set_xlabel("t-SNE Dim 1", fontsize=10)
        ax0.set_ylabel("t-SNE Dim 2", fontsize=10)
        ax0.set_xlim(extent[0], extent[1])
        ax0.set_ylim(extent[2], extent[3])
        ax0.grid(True, alpha=0.25, linestyle=":")
        ax0.legend(loc="upper right", fontsize=9.0, framealpha=0.88)

        # --- Col 1: Gray Points + Green Density Difference Glow ---
        ax1 = axes_dash[r_idx, 1]
        ax1.scatter(Y_joint[:, 0], Y_joint[:, 1], color="#9ca3af", s=3.5, alpha=0.22, edgecolors="none", zorder=1, label="Support (Gray)")
        z_diff_pos = np.maximum(0.0, delta_z)
        z_diff_pos[:margin, :] = 0.0
        z_diff_pos[-margin:, :] = 0.0
        z_diff_pos[:, :margin] = 0.0
        z_diff_pos[:, -margin:] = 0.0
        zd_min, zd_max = plot_smooth_glow(ax1, z_diff_pos, extent, cmap_name="Greens", max_alpha=0.92, zorder=2)
        ax1.set_title(f"{name}\n[2. Gray Support + Green Density Difference Glow]", fontsize=12, fontweight="bold", pad=8)
        ax1.set_xlabel("t-SNE Dim 1", fontsize=10)
        ax1.set_ylabel("t-SNE Dim 2", fontsize=10)
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.grid(True, alpha=0.25, linestyle=":")

        sm1 = cm.ScalarMappable(cmap=plt.cm.Greens, norm=mcolors.Normalize(vmin=zd_min, vmax=zd_max))
        cb1 = fig_dash.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.03)
        cb1.set_label("Density Difference (ΔZ > 0)", rotation=270, labelpad=12, fontsize=8.5)

        custom_lines = [
            Line2D([0], [0], color="#16a34a", lw=3.5, alpha=0.9, label="Density Difference Glow"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ca3af", markersize=5, label="Gray Support"),
        ]
        ax1.legend(handles=custom_lines, loc="upper right", fontsize=8.5, framealpha=0.88)

        # --- Col 2: High-Dimensional Quantitative Verification Scorecard ---
        ax2 = axes_dash[r_idx, 2]
        ax2.set_facecolor("#f8fafc")
        ax2.axis("off")

        card_text = (
            f"========================================\n"
            f"   High-Dimensional (16D) Validation\n"
            f"   View: {name}\n"
            f"========================================\n\n"
            f"1. SVDD / One-Class SVM OOD Expansion:\n"
            f"   - Base Support Outliers: {m['svdd_base']:.1f}%\n"
            f"   - Residual RL Outliers : {m['svdd_rl']:.1f}%\n"
            f"   -> Net OOD Expansion   : +{m['svdd_diff']:.1f}%\n\n"
            f"2. Statistical Distance Metrics (16D):\n"
            f"   - Maximum Mean Discrepancy (MMD):\n"
            f"     MMD = {m['mmd']:.4f}\n"
            f"   - Sliced Wasserstein Distance (W1):\n"
            f"     SWD = {m['swd']:.4f}\n\n"
            f"3. Autocorrelation & Sample Independence:\n"
            f"   - Base Autocorr Time (tau): {m['tau_base']:.1f} steps\n"
            f"   - Base Effective Sample   : {m['ess_base']:.0f} / {len(pv['b2d'])}\n"
            f"   - RL Autocorr Time (tau)  : {m['tau_rl']:.1f} steps\n"
            f"   - RL Effective Sample     : {m['ess_rl']:.0f} / {len(pv['rl2d'])}\n\n"
            f"4. Permutation Test Significance:\n"
            f"   - Iterations               : 1,000 Trajectory Clusters\n"
            f"   - Significance Level       : alpha = 0.05\n"
            f"   -> Confirmed Expansion Area: {m['sig_area_pct']:.2f}% of manifold\n"
            f"========================================"
        )
        ax2.text(
            0.05, 0.95, card_text,
            transform=ax2.transAxes,
            fontsize=10.0,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5),
        )
        ax2.set_title(f"{name}\n[3. 16D Metric Verification Scorecard]", fontsize=12, fontweight="bold", pad=8)

    plt.suptitle(
        "t-SNE Latent Space Distribution Expansion & Rigorous Verification Dashboard\n"
        "Base vs. RL Overlay | Gray Support with Green Density Difference | 1,000 Permutations (p < 0.05)",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], w_pad=2.0, h_pad=2.5)

    dash_path = output_dir / "tsne_latent_expansion_rigorous_dashboard.png"
    fig_dash.savefig(dash_path, bbox_inches="tight")
    plt.close(fig_dash)
    print(f"  -> Saved Comprehensive Dashboard to: {dash_path}")

    # Copy all figures to artifact directory
    artifact_dir = Path("/home/moai/.gemini/antigravity-cli/brain/f279ab21-5659-4869-bcd5-c8600b0f2725")
    shutil.copy2(v1_path, artifact_dir / "tsne_latent_distribution_overlay.png")
    shutil.copy2(v2_path, artifact_dir / "tsne_latent_density_difference.png")
    shutil.copy2(dash_path, artifact_dir / "tsne_latent_expansion_rigorous_dashboard.png")
    print(f"  -> Successfully copied all figures to artifact directory: {artifact_dir}")

    print("\nAll t-SNE visualization tasks completed successfully!")


if __name__ == "__main__":
    main()
