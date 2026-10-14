import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
import matplotlib.gridspec as gridspec

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MUJOCO_GL"] = "egl"
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix

def main(results_dict=None):
    print("Capturing agentview image from robosuite...")
    env = suite.make(
        env_name='NutAssemblySquare',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview'],
        camera_heights=1024,
        camera_widths=1024,
    )
    obs = env.reset()
    img = obs['agentview_image'] # [1024, 1024, 3] uint8
    H, W = 1024, 1024

    T_cam = get_camera_transform_matrix(env.sim, 'agentview', H, W)

    def project_to_img(x, y, z=0.825):
        p_h = np.array([x, y, z, 1.0])
        p_proj = T_cam @ p_h
        u = p_proj[0] / p_proj[2]
        v = p_proj[1] / p_proj[2]
        return u, H - v

    def get_quad(xb, yb, z=0.825):
        c1 = project_to_img(xb[0], yb[0], z)
        c2 = project_to_img(xb[1], yb[0], z)
        c3 = project_to_img(xb[1], yb[1], z)
        c4 = project_to_img(xb[0], yb[1], z)
        return np.array([c1, c2, c3, c4])

    peg1_pos = env.sim.data.body_xpos[env.sim.model.body_name2id('peg1')]
    peg2_pos = env.sim.data.body_xpos[env.sim.model.body_name2id('peg2')]
    u_peg1, v_peg1 = project_to_img(peg1_pos[0], peg1_pos[1], peg1_pos[2])
    u_peg2, v_peg2 = project_to_img(peg2_pos[0], peg2_pos[1], peg2_pos[2])

    if results_dict is None:
        results_dict = {
            "id": 72.0,
            "forward_8cm": 30.0,
            "midway_center": 30.0,
            "true_center": 0.0,
            "lateral_center": 56.0,
        }

    # Exact size: dx = 0.005m (0.5cm), dy = 0.115m (11.5cm)
    # Default ID center: (-0.1125, +0.1675)
    scenarios = [
        {
            "id": "id",
            "name": "1. ID Baseline (Default)",
            "short": "ID Default (Baseline)",
            "x_bounds": [-0.115, -0.110],
            "y_bounds": [0.110, 0.225],
            "center": [-0.1125, 0.1675],
            "color": "#00E676",       # Vibrant Green
            "edge": "#00B0FF",
            "sr": results_dict.get("id", 72.0),
            "z": 0.83,
        },
        {
            "id": "forward_8cm",
            "name": "2. Forward Shift (+8cm, +X)",
            "short": "Forward +8cm",
            "x_bounds": [-0.035, -0.030],
            "y_bounds": [0.110, 0.225],
            "center": [-0.0325, 0.1675],
            "color": "#FF9100",       # Vibrant Orange
            "edge": "#E65100",
            "sr": results_dict.get("forward_8cm", 30.0),
            "z": 0.83,
        },
        {
            "id": "midway_center",
            "name": "3. Toward Center (60% Midway Shift)",
            "short": "Midway Center (60% Shift)",
            "x_bounds": [-0.048, -0.043],
            "y_bounds": [0.010, 0.125],
            "center": [-0.0455, 0.0675],
            "color": "#FFD600",       # Amber / Yellow
            "edge": "#FF6D00",
            "sr": results_dict.get("midway_center", 30.0),
            "z": 0.83,
        },
        {
            "id": "true_center",
            "name": "4. TRUE Table Center (0, 0)",
            "short": "TRUE Table Center (0, 0)",
            "x_bounds": [-0.0025, 0.0025],
            "y_bounds": [-0.0575, 0.0575],
            "center": [0.0, 0.0],
            "color": "#FF1744",       # Crimson Red
            "edge": "#B71C1C",
            "sr": results_dict.get("true_center", 0.0),
            "z": 0.83,
        },
        {
            "id": "lateral_center",
            "name": "5. Lateral Centerline (-8cm in Y)",
            "short": "Lateral Center (-8cm Y)",
            "x_bounds": [-0.115, -0.110],
            "y_bounds": [0.030, 0.145],
            "center": [-0.1125, 0.0875],
            "color": "#D500F9",       # Purple / Magenta
            "edge": "#4A148C",
            "sr": results_dict.get("lateral_center", 56.0),
            "z": 0.83,
        },
    ]

    out_dir = PROJECT_ROOT / "visualization" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_dir = Path("/home/moai/.gemini/antigravity-cli/brain/f279ab21-5659-4869-bcd5-c8600b0f2725")

    # ==========================================
    # 1. Standalone High-Res Agentview Perspective View
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax.imshow(img)

    for sc in scenarios:
        quad = get_quad(sc["x_bounds"], sc["y_bounds"], z=sc["z"])
        poly = Polygon(
            quad,
            closed=True,
            facecolor=sc["color"],
            edgecolor=sc.get("edge", "white"),
            linewidth=2.8,
            alpha=0.6,
            zorder=5,
        )
        ax.add_patch(poly)

        # Plot projected center point
        u_c, v_c = project_to_img(sc["center"][0], sc["center"][1], z=sc["z"])
        ax.scatter([u_c], [v_c], color="white", edgecolor="black", s=60, zorder=6)

    # Annotate Pegs
    ax.scatter([u_peg1], [v_peg1], color="#00E5FF", s=180, marker="*", edgecolor="black", zorder=7)
    ax.text(u_peg1, v_peg1 - 25, "SquarePeg (Target)", color="#00E5FF", fontsize=11, fontweight="bold",
            ha="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.85, edgecolor="#00E5FF"))

    ax.scatter([u_peg2], [v_peg2], color="#B0BEC5", s=110, marker="o", edgecolor="black", zorder=7)
    ax.text(u_peg2, v_peg2 - 20, "RoundPeg", color="#B0BEC5", fontsize=9.5,
            ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor="#9E9E9E"))

    # Labels for scenarios
    label_offsets = {
        "id": (-90, 50),
        "forward_8cm": (100, 10),
        "midway_center": (-115, -40),
        "true_center": (110, -50),
        "lateral_center": (-115, 65),
    }

    for sc in scenarios:
        u_c, v_c = project_to_img(sc["center"][0], sc["center"][1], z=sc["z"])
        dx_off, dy_off = label_offsets[sc["id"]]
        ax.annotate(
            f"{sc['short']}\nSR: {sc['sr']:.1f}%",
            xy=(u_c, v_c),
            xytext=(u_c + dx_off, v_c + dy_off),
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=2.0),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="black", edgecolor=sc["color"], lw=2.0, alpha=0.9),
            color="white",
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )

    # Shift Direction Arrows from ID Center
    u_id, v_id = project_to_img(scenarios[0]["center"][0], scenarios[0]["center"][1], z=scenarios[0]["z"])
    for sc in scenarios[1:]:
        u_tgt, v_tgt = project_to_img(sc["center"][0], sc["center"][1], z=sc["z"])
        ax.annotate(
            "",
            xy=(u_tgt, v_tgt),
            xytext=(u_id, v_id),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.6, ls="--", alpha=0.8),
            zorder=7,
        )

    ax.set_title(
        "Robosuite Square: Comparison of True Center vs Forward vs ID Shifts\n"
        "[Exact Base Size Maintained: 0.5cm (X) x 11.5cm (Y)]",
        fontsize=14, fontweight="bold", pad=15, color="black"
    )
    ax.axis("off")
    plt.tight_layout()

    out_cam = out_dir / "scenario_exact_size_shifts_agentview.png"
    plt.savefig(out_cam, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_cam}")

    if brain_dir.exists():
        import shutil
        shutil.copy(out_cam, brain_dir / "scenario_exact_size_shifts_agentview.png")

    # ==========================================
    # 2. Comprehensive 3-Panel Composite Dashboard
    # ==========================================
    fig = plt.figure(figsize=(19, 10), dpi=180)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.28)

    # Left: Camera View
    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.imshow(img)
    for sc in scenarios:
        quad = get_quad(sc["x_bounds"], sc["y_bounds"], z=sc["z"])
        poly = Polygon(quad, closed=True, facecolor=sc["color"], edgecolor=sc.get("edge", "white"),
                       linewidth=2.2, alpha=0.6, zorder=5)
        ax_cam.add_patch(poly)
        u_c, v_c = project_to_img(sc["center"][0], sc["center"][1], z=sc["z"])
        ax_cam.scatter([u_c], [v_c], color="white", edgecolor="black", s=40, zorder=6)

    ax_cam.scatter([u_peg1], [v_peg1], color="#00E5FF", s=140, marker="*", edgecolor="black", zorder=7)
    ax_cam.text(u_peg1, v_peg1 - 25, "SquarePeg (Target)", color="#00E5FF", fontsize=10, fontweight="bold",
                ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.85, edgecolor="#00E5FF"))

    for sc in scenarios:
        u_c, v_c = project_to_img(sc["center"][0], sc["center"][1], z=sc["z"])
        dx_off, dy_off = label_offsets[sc["id"]]
        ax_cam.annotate(
            f"{sc['short']}\nSR: {sc['sr']:.1f}%",
            xy=(u_c, v_c),
            xytext=(u_c + dx_off, v_c + dy_off),
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor=sc["color"], lw=1.5, alpha=0.85),
            color="white", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=8,
        )

    ax_cam.set_title("(A) Agentview Camera Perspective (3D Mujoco Projected)", fontsize=12, fontweight="bold", pad=10)
    ax_cam.axis("off")

    # Top-Right: Metric 2D Table Map (X vs Y)
    ax_map = fig.add_subplot(gs[0, 1])
    ax_map.add_patch(Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor="#F5F5F5", edgecolor="#9E9E9E", lw=1.5, ls="--", label="Table (0.8x0.8m)"))
    ax_map.axhline(0, color="gray", lw=0.8, ls=":")
    ax_map.axvline(0, color="gray", lw=0.8, ls=":")

    # Pegs
    ax_map.scatter([peg1_pos[0]], [peg1_pos[1]], color="#00E5FF", s=130, marker="*", edgecolor="black", zorder=6, label="SquarePeg (+0.23, +0.10)")
    ax_map.scatter([peg2_pos[0]], [peg2_pos[1]], color="#78909C", s=80, marker="o", edgecolor="black", zorder=6, label="RoundPeg (+0.23, -0.10)")

    # Table Physical Center Marker
    ax_map.scatter([0.0], [0.0], color="red", s=80, marker="x", linewidth=2.5, zorder=7, label="Table Center (0, 0)")

    # Boxes
    for sc in scenarios:
        xb = sc["x_bounds"]
        yb = sc["y_bounds"]
        w = xb[1] - xb[0]
        h = yb[1] - yb[0]
        rect = Rectangle(
            (xb[0], yb[0]), w, h,
            facecolor=sc["color"],
            edgecolor=sc.get("edge", "black"),
            linewidth=2.0,
            alpha=0.65,
            zorder=5,
            label=f"{sc['short']} ({sc['sr']:.1f}%)"
        )
        ax_map.add_patch(rect)
        ax_map.scatter([sc["center"][0]], [sc["center"][1]], color="black", s=25, zorder=6)

    # Shift vectors from ID Center
    x_id_c, y_id_c = scenarios[0]["center"]
    for sc in scenarios[1:]:
        xc, yc = sc["center"]
        ax_map.annotate(
            "",
            xy=(xc, yc),
            xytext=(x_id_c, y_id_c),
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=1.8, ls="--"),
            zorder=7,
        )

    ax_map.set_xlim(-0.25, 0.35)
    ax_map.set_ylim(-0.15, 0.30)
    ax_map.set_xlabel("X (Robot -> Forward Table Center, m)", fontsize=10, fontweight="bold")
    ax_map.set_ylabel("Y (Table Lateral Axis, m)", fontsize=10, fontweight="bold")
    ax_map.set_title("(B) Physical 2D Top-Down Metric Map (Exact Size: 0.5cm x 11.5cm)", fontsize=12, fontweight="bold")
    ax_map.grid(True, linestyle="--", alpha=0.4)
    ax_map.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # Bottom-Right: Success Rate Comparison Bar Chart
    ax_bar = fig.add_subplot(gs[1, 1])
    names = [sc["short"] for sc in scenarios]
    srs = [sc["sr"] for sc in scenarios]
    colors = [sc["color"] for sc in scenarios]
    edges = [sc.get("edge", "black") for sc in scenarios]

    bars = ax_bar.barh(names[::-1], srs[::-1], color=colors[::-1], edgecolor=edges[::-1], height=0.55, linewidth=1.5, alpha=0.85)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Zero-Shot Success Rate (%) [50 Episodes]", fontsize=10, fontweight="bold")
    ax_bar.set_title("(C) 200-Demo Base Policy Zero-Shot Degradation", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, sr in zip(bars, srs[::-1]):
        width = bar.get_width()
        ax_bar.text(
            width + 2.0, bar.get_y() + bar.get_height() / 2,
            f"{sr:.1f}%",
            va="center", ha="left", fontsize=10, fontweight="bold", color="black"
        )

    plt.suptitle("Robosuite Square: Exact Base Size Shifts & Collapse at TRUE Table Center (0, 0)",
                 fontsize=14, fontweight="bold", y=0.98)

    out_comp = out_dir / "scenario_exact_size_shifts_composite.png"
    plt.savefig(out_comp, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_comp}")

    if brain_dir.exists():
        import shutil
        shutil.copy(out_comp, brain_dir / "scenario_exact_size_shifts_composite.png")

if __name__ == "__main__":
    main()
