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

def main():
    print("Initializing environment and capturing agentview image...")
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

    def project_to_img(x, y, z=0.82):
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

    scenarios = [
        {
            "name": "1. ID Baseline (Reference)",
            "short": "ID Demonstration",
            "x_bounds": [-0.115, -0.110],
            "y_bounds": [0.110, 0.115],
            "color": "#00E5FF",       # Bright Cyan
            "edge": "#00838F",
            "sr": 72.0,
            "z": 0.83,
        },
        {
            "name": "Previous Expanded OOD (4.875x15.875)",
            "short": "Previous Expanded OOD",
            "x_bounds": [-0.136875, -0.088125],
            "y_bounds": [0.088125, 0.246875],
            "color": "#FFD600",       # Amber / Yellow
            "edge": "#FF6D00",
            "sr": 56.0,
            "z": 0.825,
            "linestyle": "--",
            "fill": False,
        },
        {
            "name": "2. Table Center Shift (Recommended)",
            "short": "Table Center Shift",
            "x_bounds": [-0.040, -0.005],
            "y_bounds": [0.010, 0.045],
            "color": "#FF1744",       # Crimson Red
            "edge": "#B71C1C",
            "sr": 2.0,
            "z": 0.83,
        },
        {
            "name": "3. Diagonal Shift",
            "short": "Diagonal Shift",
            "x_bounds": [-0.040, -0.005],
            "y_bounds": [0.190, 0.225],
            "color": "#D500F9",       # Magenta / Purple
            "edge": "#4A148C",
            "sr": 16.0,
            "z": 0.83,
        },
        {
            "name": "4. Pure Forward Shift (+X)",
            "short": "Pure Forward Shift",
            "x_bounds": [-0.040, -0.005],
            "y_bounds": [0.100, 0.125],
            "color": "#00E676",       # Bright Spring Green
            "edge": "#1B5E20",
            "sr": 32.0,
            "z": 0.83,
        },
    ]

    fig = plt.figure(figsize=(22, 11), dpi=200)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.25, 0.75], height_ratios=[1.15, 0.85], hspace=0.22, wspace=0.12)

    # -------------------------------------------------------------
    # Panel 1: Perspective Agentview Camera with Projected Quads
    # -------------------------------------------------------------
    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.imshow(img)
    ax_cam.set_title("Agentview Camera Perspective View (Projected Spatial Regions)", fontsize=16, fontweight='bold', pad=12)

    # Draw coordinate frame indicator on table in camera view
    u_origin, v_origin = project_to_img(0.0, 0.0, 0.82)
    u_x_axis, v_x_axis = project_to_img(0.14, 0.0, 0.82) # +X forward (up in image)
    u_y_axis, v_y_axis = project_to_img(0.0, 0.14, 0.82) # +Y lateral (right in image)

    # Plot Peg markers
    ax_cam.scatter([u_peg1], [v_peg1], color='#FFD700', edgecolor='black', s=200, zorder=10, marker='s')
    ax_cam.text(u_peg1 + 20, v_peg1 - 15, "SquarePeg (Target Goal)", color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='gold'))

    ax_cam.scatter([u_peg2], [v_peg2], color='#B0BEC5', edgecolor='black', s=180, zorder=10, marker='o')
    ax_cam.text(u_peg2 - 130, v_peg2 - 15, "RoundPeg", color='white', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none'))

    # Plot coordinate arrows on the image
    ax_cam.annotate('', xy=(u_x_axis, v_x_axis), xytext=(u_origin, v_origin),
                    arrowprops=dict(facecolor='#FF5252', edgecolor='black', width=3.5, headwidth=10, shrink=0.02))
    ax_cam.text(u_x_axis - 5, v_x_axis - 18, "+X (Forward / Table Front)", color='#D50000', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

    ax_cam.annotate('', xy=(u_y_axis, v_y_axis), xytext=(u_origin, v_origin),
                    arrowprops=dict(facecolor='#2979FF', edgecolor='black', width=3.5, headwidth=10, shrink=0.02))
    ax_cam.text(u_y_axis + 10, v_y_axis - 5, "+Y (Lateral / Table Side)", color='#0D47A1', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='blue'))

    # Draw bounding polygons for each scenario
    for sc in scenarios:
        quad = get_quad(sc["x_bounds"], sc["y_bounds"], z=sc.get("z", 0.825))
        is_fill = sc.get("fill", True)
        ls = sc.get("linestyle", "-")
        
        if is_fill:
            poly = Polygon(quad, closed=True, facecolor=sc["color"], edgecolor=sc["edge"],
                           alpha=0.50, linewidth=2.5, linestyle=ls, label=f"{sc['short']} [SR: {sc['sr']:.1f}%]")
        else:
            poly = Polygon(quad, closed=True, facecolor='none', edgecolor=sc["edge"],
                           alpha=0.95, linewidth=2.5, linestyle=ls, label=f"{sc['short']} [SR: {sc['sr']:.1f}%]")
        ax_cam.add_patch(poly)

        # Compute centroid for label placement
        cen_u = np.mean(quad[:, 0])
        cen_v = np.mean(quad[:, 1])
        
        # Add crisp text callouts
        if "Table Center" in sc["name"]:
            ax_cam.text(cen_u - 40, cen_v - 25, f"Table Center Shift\n(SR: {sc['sr']:.0f}%)", color='white',
                        fontsize=11, fontweight='heavy', ha='center',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor=sc["color"], alpha=0.95, edgecolor='black', lw=1.5))
        elif "Diagonal" in sc["name"]:
            ax_cam.text(cen_u + 75, cen_v - 15, f"Diagonal Shift\n(SR: {sc['sr']:.0f}%)", color='white',
                        fontsize=11, fontweight='heavy', ha='center',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor=sc["color"], alpha=0.95, edgecolor='black', lw=1.5))
        elif "Pure Forward" in sc["name"]:
            ax_cam.text(cen_u, cen_v + 42, f"Pure Forward Shift\n(SR: {sc['sr']:.0f}%)", color='black',
                        fontsize=10, fontweight='heavy', ha='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=sc["color"], alpha=0.9, edgecolor='black', lw=1.2))
        elif "ID" in sc["name"]:
            ax_cam.text(cen_u + 75, cen_v + 30, f"ID Base Area\n(SR: {sc['sr']:.0f}%)", color='black',
                        fontsize=11, fontweight='heavy', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=sc["color"], alpha=0.95, edgecolor='black', lw=1.5))

    ax_cam.set_xlim(0, W)
    ax_cam.set_ylim(H, 0)
    ax_cam.axis('off')
    # Put legend at top-left to avoid any overlap with SquarePeg on top-right
    ax_cam.legend(loc='upper left', fontsize=11, framealpha=0.92, facecolor='#FAFAFA', edgecolor='#BDBDBD')

    # -------------------------------------------------------------
    # Panel 2: Top-down 2D Metric Cartesian Map (X-Y plane in cm)
    # -------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 1])
    ax_map.set_facecolor('#F8F9FA')

    # Draw table boundaries: X: [-40, 40] cm, Y: [-40, 40] cm
    table_rect = Rectangle((-40, -40), 80, 80, facecolor='#ECEFF1', edgecolor='#90A4AE', linewidth=1.5, linestyle=':')
    ax_map.add_patch(table_rect)

    # Peg markers (in cm)
    ax_map.scatter([peg1_pos[1]*100], [peg1_pos[0]*100], color='#FFD700', edgecolor='black', s=200, marker='s', zorder=10, label='SquarePeg (Target)')
    ax_map.scatter([peg2_pos[1]*100], [peg2_pos[0]*100], color='#B0BEC5', edgecolor='black', s=160, marker='o', zorder=10, label='RoundPeg')

    # Draw scenarios on 2D map
    for sc in scenarios:
        x_min, x_max = sc["x_bounds"][0]*100, sc["x_bounds"][1]*100
        y_min, y_max = sc["y_bounds"][0]*100, sc["y_bounds"][1]*100
        w = y_max - y_min
        h = x_max - x_min
        
        is_fill = sc.get("fill", True)
        ls = sc.get("linestyle", "-")

        if is_fill:
            rect = Rectangle((y_min, x_min), w, h, facecolor=sc["color"], edgecolor=sc["edge"],
                             alpha=0.45, linewidth=2, linestyle=ls)
        else:
            rect = Rectangle((y_min, x_min), w, h, facecolor='none', edgecolor=sc["edge"],
                             alpha=0.9, linewidth=2, linestyle=ls)
        ax_map.add_patch(rect)

        # Plot shift displacement vectors from ID center
        if "Table Center" in sc["name"] or "Diagonal" in sc["name"]:
            id_cx, id_cy = -11.25, 11.25
            sc_cx, sc_cy = (x_min + x_max)/2, (y_min + y_max)/2
            ax_map.annotate('', xy=(sc_cy, sc_cx), xytext=(id_cy, id_cx),
                            arrowprops=dict(arrowstyle="->", color=sc["edge"], lw=2.2, linestyle="--"))
            dist = np.hypot(sc_cx - id_cx, sc_cy - id_cy)
            mid_y = (sc_cy + id_cy)/2
            mid_x = (sc_cx + id_cx)/2
            ax_map.text(mid_y, mid_x, f"Δ={dist:.1f}cm", fontsize=9.5, fontweight='bold', color=sc["edge"],
                        ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.92, edgecolor='none'))

    ax_map.set_xlabel("Y: Lateral Across Table (cm)", fontsize=11, fontweight='bold')
    ax_map.set_ylabel("X: Forward Into Table (cm)", fontsize=11, fontweight='bold')
    ax_map.set_title("Metric Top-Down Layout & Shift Displacements", fontsize=13, fontweight='bold', pad=10)
    ax_map.set_xlim(-25, 30)
    ax_map.set_ylim(-20, 30)
    ax_map.grid(True, linestyle='--', alpha=0.6)
    ax_map.legend(loc='lower left', fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 3: Base Policy Success Rate Comparison Bar Chart
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor('#F8F9FA')

    names = ["ID Base", "Prev OOD\n(Expanded)", "Pure Forward\nShift", "Diagonal\nShift", "Table Center\nShift"]
    srs = [72.0, 56.0, 32.0, 16.0, 2.0]
    colors = ["#00E5FF", "#FFD600", "#00E676", "#D500F9", "#FF1744"]
    edges = ["#00838F", "#FF6D00", "#1B5E20", "#4A148C", "#B71C1C"]

    bars = ax_bar.bar(names, srs, color=colors, edgecolor=edges, width=0.55, linewidth=1.5, alpha=0.85)
    ax_bar.set_ylabel("Zero-Shot Success Rate (%)", fontsize=11, fontweight='bold')
    ax_bar.set_title("200-Demo Base Policy Evaluation (N=50 each)", fontsize=13, fontweight='bold', pad=10)
    ax_bar.set_ylim(0, 85)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, sr in zip(bars, srs):
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{sr:.1f}%",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    output_path = "/home/moai/ysl_ws/cover/visualization/outputs/scenario_initialization_agentview_composite.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Polished composite visualization saved to {output_path}")

if __name__ == "__main__":
    main()
