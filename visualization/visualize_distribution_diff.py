#!/usr/bin/env python3
"""
Visualization of Base Policy vs. Residual Policy Distributions and Expansion Difference (Glow Style).

Panels (2 rows x 3 columns):
- Row 0: Main Camera (Agentview)
    1. Base Policy: Orange Density Glow + Base Policy scatter points.
    2. Residual RL Policy: Blue Density Glow + Residual RL scatter points (solid blue).
    3. Expansion Difference: Green Density Glow showing positive difference (Residual - Base)
       with BOTH Base (Orange) and Residual (Blue) scatter points overlaid.
- Row 1: Wrist Camera (Eye-in-hand)
    1. Base Policy: Orange Density Glow + Base Policy scatter points.
    2. Residual RL Policy: Blue Density Glow + Residual RL scatter points (solid blue).
    3. Expansion Difference: Green Density Glow showing positive difference (Residual - Base)
       with BOTH Base (Orange) and Residual (Blue) scatter points overlaid.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import torch


def _to_np(z):
    if z is None:
        return np.array([])
    if isinstance(z, list):
        if len(z) == 0:
            return np.array([])
        arrs = []
        for x in z:
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            elif hasattr(x, "cpu"):
                x = x.cpu().numpy()
            else:
                x = np.asarray(x)
            arrs.append(x.reshape(-1))
        return np.stack(arrs, axis=0)
    else:
        if hasattr(z, "detach"):
            z = z.detach().cpu().numpy()
        elif hasattr(z, "cpu"):
            z = z.cpu().numpy()
        else:
            z = np.asarray(z)
        if z.ndim == 3:
            if z.shape[0] == 1:
                z = z.squeeze(0)
            elif z.shape[1] == 1:
                z = z.squeeze(1)
            else:
                z = z.reshape(-1, z.shape[-1])
        elif z.ndim == 1:
            z = np.expand_dims(z, 0)
        return z


def _plot_density_glow(ax, Z, xmin, xmax, ymin, ymax, cmap_name: str, max_alpha: float = 0.88, zorder: int = 1):
    """Renders a continuous, smooth density glow using bicubic imshow with power-law alpha fade."""
    Z_2d = Z.T  # Transpose so rows correspond to Y and columns to X
    Z_min = Z_2d.min()
    Z_max = Z_2d.max()
    Z_norm = (Z_2d - Z_min) / (Z_max - Z_min + 1e-8)

    # Non-linear color mapping: boost medium/low densities into rich vibrant tones
    Z_color = np.clip(Z_norm ** 0.58, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(Z_color)

    # Smooth alpha fade: zero density fades seamlessly to transparent
    rgba[..., 3] = np.clip(Z_norm ** 0.70, 0.0, 1.0) * max_alpha

    im = ax.imshow(
        rgba,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        interpolation="bicubic",
        zorder=zorder,
        aspect="auto",
    )
    return Z_min, Z_max


def render_distribution_diff_glow(
    cache_path: Path | str,
    save_path: Path | str,
    point_size: float = 6.0,
    point_alpha: float = 0.55,
    task_name: str = "Square",
    diff_style: str = "glow_over_gray",
):
    print(f"[Loading] Cache data from: {cache_path}")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    base_trajs_f = cache_data["base_trajs_f"]
    base_trajs_w = cache_data["base_trajs_w"]
    rl_trajs_f = cache_data["rl_trajs_f"]
    rl_trajs_w = cache_data["rl_trajs_w"]

    all_z_base_f = np.concatenate([_to_np(z) for z in base_trajs_f], axis=0)
    all_z_base_w = np.concatenate([_to_np(z) for z in base_trajs_w], axis=0)

    rl_trajs_f_clean = [_to_np(z) for z in rl_trajs_f if len(_to_np(z)) > 0]
    rl_trajs_w_clean = [_to_np(z) for z in rl_trajs_w if len(_to_np(z)) > 0]
    all_z_rl_f = np.vstack(rl_trajs_f_clean)
    all_z_rl_w = np.vstack(rl_trajs_w_clean)

    # 1. Fit PCA on Base Policy representations (consistent common frame of reference)
    pca_f = PCA(n_components=2).fit(all_z_base_f)
    pca_w = PCA(n_components=2).fit(all_z_base_w)

    base_2d_f = pca_f.transform(all_z_base_f)
    base_2d_w = pca_w.transform(all_z_base_w)

    rl_2d_f = pca_f.transform(all_z_rl_f)
    rl_2d_w = pca_w.transform(all_z_rl_w)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13), dpi=200)

    view_configs = [
        {
            "row": 0,
            "view_name": "Main Camera (Agentview)",
            "base_2d": base_2d_f,
            "rl_2d": rl_2d_f,
        },
        {
            "row": 1,
            "view_name": "Wrist Camera (Eye-in-hand)",
            "base_2d": base_2d_w,
            "rl_2d": rl_2d_w,
        },
    ]

    for cfg in view_configs:
        r = cfg["row"]
        view_name = cfg["view_name"]
        base_2d = cfg["base_2d"]
        rl_2d = cfg["rl_2d"]

        # Unified coordinate grid for strict 1:1 comparison
        x_all = np.concatenate([base_2d[:, 0], rl_2d[:, 0]])
        y_all = np.concatenate([base_2d[:, 1], rl_2d[:, 1]])
        xmin, xmax = x_all.min() - 0.5, x_all.max() + 0.5
        ymin, ymax = y_all.min() - 0.5, y_all.max() + 0.5

        X, Y = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Fit Gaussian KDE for Base Policy
        kernel_base = gaussian_kde(base_2d.T)
        Z_base = np.reshape(kernel_base(positions).T, X.shape)

        # Fit Gaussian KDE for Residual Policy
        kernel_rl = gaussian_kde(rl_2d.T)
        Z_rl = np.reshape(kernel_rl(positions).T, X.shape)

        # Compute Difference: Positive Expansion of Residual over Base
        Z_diff = np.maximum(0.0, Z_rl - Z_base)

        # -------------------------------------------------------------
        # Col 0: Base Policy (Orange Density Glow + Scatter Points)
        # -------------------------------------------------------------
        ax0 = axes[r, 0]
        z_b_min, z_b_max = _plot_density_glow(ax0, Z_base, xmin, xmax, ymin, ymax, "Oranges", max_alpha=0.88, zorder=1)
        ax0.scatter(
            base_2d[:, 0],
            base_2d[:, 1],
            color="#c2410c",
            s=point_size,
            alpha=point_alpha * 0.75,
            edgecolors="none",
            zorder=3,
            label="Base Trajectory Steps",
        )
        ax0.set_title(f"{view_name}\n[1. Base Policy Distribution (Orange Glow)]", fontsize=13, fontweight="bold", pad=10)
        ax0.set_xlabel("Principal Component 1", fontsize=11)
        ax0.set_ylabel("Principal Component 2", fontsize=11)
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        ax0.grid(True, alpha=0.25, linestyle="--")

        sm0 = cm.ScalarMappable(cmap=plt.cm.Oranges, norm=mcolors.Normalize(vmin=z_b_min, vmax=z_b_max))
        cbar0 = fig.colorbar(sm0, ax=ax0, fraction=0.046, pad=0.04)
        cbar0.set_label("Base Policy Density", rotation=270, labelpad=16, fontsize=10)
        ax0.legend(loc="upper right", fontsize=9, framealpha=0.85)

        # -------------------------------------------------------------
        # Col 1: Residual RL Policy (Blue Density Glow + Scatter Points)
        # -------------------------------------------------------------
        ax1 = axes[r, 1]
        z_rl_min, z_rl_max = _plot_density_glow(ax1, Z_rl, xmin, xmax, ymin, ymax, "Blues", max_alpha=0.88, zorder=1)
        ax1.scatter(
            rl_2d[:, 0],
            rl_2d[:, 1],
            color="#1d4ed8",
            s=point_size,
            alpha=point_alpha * 0.75,
            edgecolors="none",
            zorder=3,
            label="Residual RL Trajectory Steps",
        )
        ax1.set_title(f"{view_name}\n[2. Residual RL Policy Distribution (Blue Glow)]", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel("Principal Component 1", fontsize=11)
        ax1.set_ylabel("Principal Component 2", fontsize=11)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.grid(True, alpha=0.25, linestyle="--")

        sm1 = cm.ScalarMappable(cmap=plt.cm.Blues, norm=mcolors.Normalize(vmin=z_rl_min, vmax=z_rl_max))
        cbar1 = fig.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Residual RL Density", rotation=270, labelpad=16, fontsize=10)
        ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)

        # -------------------------------------------------------------
        # Col 2: Difference / Expansion
        # Combine Base and RL points into unified background, both in gray,
        # and visualize their expansion difference in green glow on top.
        # -------------------------------------------------------------
        ax2 = axes[r, 2]
        if diff_style == "glow_over_gray":
            # Combine Base and RL points into a single unified point cloud
            combined_2d = np.vstack([base_2d, rl_2d])

            # Display BOTH as neutral gray dots like a background
            ax2.scatter(
                combined_2d[:, 0],
                combined_2d[:, 1],
                color="#9ca3af",
                s=point_size * 0.75,
                alpha=0.30,
                edgecolors="none",
                zorder=1,
                label="Trajectory Steps (Base & RL Background)",
            )
            # Visualize the difference in green glow on top
            z_d_min, z_d_max = _plot_density_glow(
                ax2, Z_diff, xmin, xmax, ymin, ymax, "Greens", max_alpha=0.90, zorder=2
            )
            ax2.set_title(
                f"{view_name}\n[3. Difference (Green Glow) over Combined Steps (Gray Background)]",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
        else:
            # Green Density Glow in background
            z_d_min, z_d_max = _plot_density_glow(ax2, Z_diff, xmin, xmax, ymin, ymax, "Greens", max_alpha=0.88, zorder=1)
            # Base steps in neutral gray dots
            ax2.scatter(
                base_2d[:, 0],
                base_2d[:, 1],
                color="#94a3b8",
                s=point_size * 0.8,
                alpha=0.45,
                edgecolors="none",
                zorder=2,
                label="Base Steps (Gray)",
            )
            # Residual RL steps in green dots on top
            ax2.scatter(
                rl_2d[:, 0],
                rl_2d[:, 1],
                color="#16a34a",
                s=point_size * 0.85,
                alpha=0.55,
                edgecolors="none",
                zorder=3,
                label="Residual RL Steps (Green)",
            )
            ax2.set_title(f"{view_name}\n[3. Expansion: Base (Gray) + Residual (Green Dots)]", fontsize=13, fontweight="bold", pad=10)

        ax2.set_xlabel("Principal Component 1", fontsize=11)
        ax2.set_ylabel("Principal Component 2", fontsize=11)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.grid(True, alpha=0.25, linestyle="--")

        sm2 = cm.ScalarMappable(cmap=plt.cm.Greens, norm=mcolors.Normalize(vmin=z_d_min, vmax=z_d_max))
        cbar2 = fig.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Expansion Density: max(0, RL - Base)", rotation=270, labelpad=16, fontsize=10)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.85)

    suptitle = (
        f"Latent Manifold Expansion Analysis: Base Policy vs. Residual RL Policy\n"
        f"[Task: {task_name} | 50 Successful Trajectories per Policy | Col 1: Base (Orange) | Col 2: Residual RL (Blue) | Col 3: Expansion Difference (Green)]"
    )
    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Done] Successfully generated distribution difference comparison figure: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Base vs Residual Distributions and Expansion Difference (Glow Style)")
    parser.add_argument(
        "--cache_path",
        type=str,
        default="visualization/cache/Square_base_vs_residual_50ep.pt",
        help="Path to cached trajectories .pt file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualization/outputs/base_vs_residual_distribution_expansion.png",
        help="Path to save output figure",
    )
    parser.add_argument("--point_size", type=float, default=6.5, help="Scatter point size")
    parser.add_argument("--point_alpha", type=float, default=0.6, help="Scatter point alpha")
    parser.add_argument("--task", type=str, default="Square", help="Task name")
    parser.add_argument(
        "--diff_style",
        type=str,
        default="glow_over_gray",
        choices=["glow_over_gray", "green_rl_dots"],
        help="Panel 3 style: 'glow_over_gray' (gray points with green glow on top) or 'green_rl_dots' (gray base points with green RL points)",
    )
    args = parser.parse_args()

    cache_path = PROJECT_ROOT / args.cache_path
    output_path = PROJECT_ROOT / args.output_path

    render_distribution_diff_glow(
        cache_path=cache_path,
        save_path=output_path,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        task_name=args.task,
        diff_style=args.diff_style,
    )


if __name__ == "__main__":
    main()
