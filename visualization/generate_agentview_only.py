import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MUJOCO_GL"] = "egl"
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix

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
img = obs['agentview_image']
H, W = 1024, 1024
T_cam = get_camera_transform_matrix(env.sim, 'agentview', H, W)

def project_to_img(x, y, z=0.82):
    p_h = np.array([x, y, z, 1.0])
    p_proj = T_cam @ p_h
    return p_proj[0] / p_proj[2], H - (p_proj[1] / p_proj[2])

def get_quad(xb, yb, z=0.825):
    return np.array([
        project_to_img(xb[0], yb[0], z),
        project_to_img(xb[1], yb[0], z),
        project_to_img(xb[1], yb[1], z),
        project_to_img(xb[0], yb[1], z),
    ])

fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
ax.imshow(img)
ax.set_title("Agentview Camera: Scenario Initialization Bounds & Zero-Shot Success Rates", fontsize=15, fontweight='bold', pad=15)

peg1_pos = env.sim.data.body_xpos[env.sim.model.body_name2id('peg1')]
peg2_pos = env.sim.data.body_xpos[env.sim.model.body_name2id('peg2')]
u_peg1, v_peg1 = project_to_img(peg1_pos[0], peg1_pos[1], peg1_pos[2])
u_peg2, v_peg2 = project_to_img(peg2_pos[0], peg2_pos[1], peg2_pos[2])

ax.scatter([u_peg1], [v_peg1], color='#FFD700', edgecolor='black', s=220, zorder=10, marker='s')
ax.text(u_peg1 + 22, v_peg1 - 15, "SquarePeg (Target Goal)", color='white', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75, edgecolor='gold'))

ax.scatter([u_peg2], [v_peg2], color='#B0BEC5', edgecolor='black', s=200, zorder=10, marker='o')
ax.text(u_peg2 - 130, v_peg2 - 15, "RoundPeg", color='white', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none'))

# Coordinate arrows
u_origin, v_origin = project_to_img(0.0, 0.0, 0.82)
u_x_axis, v_x_axis = project_to_img(0.14, 0.0, 0.82)
u_y_axis, v_y_axis = project_to_img(0.0, 0.14, 0.82)

ax.annotate('', xy=(u_x_axis, v_x_axis), xytext=(u_origin, v_origin),
            arrowprops=dict(facecolor='#FF5252', edgecolor='black', width=3.5, headwidth=10, shrink=0.02))
ax.text(u_x_axis - 5, v_x_axis - 18, "+X (Forward / Table Front)", color='#D50000', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax.annotate('', xy=(u_y_axis, v_y_axis), xytext=(u_origin, v_origin),
            arrowprops=dict(facecolor='#2979FF', edgecolor='black', width=3.5, headwidth=10, shrink=0.02))
ax.text(u_y_axis + 10, v_y_axis - 5, "+Y (Lateral / Table Side)", color='#0D47A1', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='blue'))

scenarios = [
    {"name": "1. ID Baseline", "xb": [-0.115, -0.110], "yb": [0.110, 0.115], "c": "#00E5FF", "e": "#00838F", "sr": 72.0, "fill": True, "ls": "-"},
    {"name": "Previous Expanded OOD", "xb": [-0.136875, -0.088125], "yb": [0.088125, 0.246875], "c": "#FFD600", "e": "#FF6D00", "sr": 56.0, "fill": False, "ls": "--"},
    {"name": "2. Table Center Shift", "xb": [-0.040, -0.005], "yb": [0.010, 0.045], "c": "#FF1744", "e": "#B71C1C", "sr": 2.0, "fill": True, "ls": "-"},
    {"name": "3. Diagonal Shift", "xb": [-0.040, -0.005], "yb": [0.190, 0.225], "c": "#D500F9", "e": "#4A148C", "sr": 16.0, "fill": True, "ls": "-"},
    {"name": "4. Pure Forward Shift", "xb": [-0.040, -0.005], "yb": [0.100, 0.125], "c": "#00E676", "e": "#1B5E20", "sr": 32.0, "fill": True, "ls": "-"},
]

for sc in scenarios:
    quad = get_quad(sc["xb"], sc["yb"])
    if sc["fill"]:
        poly = Polygon(quad, closed=True, facecolor=sc["c"], edgecolor=sc["e"], alpha=0.5, linewidth=2.5, linestyle=sc["ls"], label=f"{sc['name']} [SR: {sc['sr']:.1f}%]")
    else:
        poly = Polygon(quad, closed=True, facecolor='none', edgecolor=sc["e"], alpha=0.95, linewidth=2.5, linestyle=sc["ls"], label=f"{sc['name']} [SR: {sc['sr']:.1f}%]")
    ax.add_patch(poly)

    cen_u = np.mean(quad[:, 0])
    cen_v = np.mean(quad[:, 1])
    if "Center" in sc["name"]:
        ax.text(cen_u - 40, cen_v - 25, f"Table Center Shift\n(SR: {sc['sr']:.0f}%)", color='white', fontsize=11, fontweight='heavy', ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=sc["c"], alpha=0.95, edgecolor='black', lw=1.5))
    elif "Diagonal" in sc["name"]:
        ax.text(cen_u + 75, cen_v - 15, f"Diagonal Shift\n(SR: {sc['sr']:.0f}%)", color='white', fontsize=11, fontweight='heavy', ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=sc["c"], alpha=0.95, edgecolor='black', lw=1.5))
    elif "Forward" in sc["name"]:
        ax.text(cen_u, cen_v + 42, f"Pure Forward Shift\n(SR: {sc['sr']:.0f}%)", color='black', fontsize=10, fontweight='heavy', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=sc["c"], alpha=0.9, edgecolor='black', lw=1.2))
    elif "ID" in sc["name"]:
        ax.text(cen_u + 75, cen_v + 30, f"ID Base Area\n(SR: {sc['sr']:.0f}%)", color='black', fontsize=11, fontweight='heavy', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=sc["c"], alpha=0.95, edgecolor='black', lw=1.5))

ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')
ax.legend(loc='upper left', fontsize=11, framealpha=0.92, facecolor='#FAFAFA', edgecolor='#BDBDBD')

plt.tight_layout()
out_path = "/home/moai/ysl_ws/cover/visualization/outputs/scenario_initialization_agentview_only.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved to {out_path}")
