import sys
import os
from pathlib import Path
sys.path.append(f"{PROJECT_ROOT}")

import time
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for numpy array rendering
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
from tqdm import tqdm

from lane.e2c import MLPE2C

import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load E2C models
    print("Loading Pretrained E2C Encoders...")
    e2c_main = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to(device)
    e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to(device)
    
    e2c_main.load_state_dict(torch.load(f"{PROJECT_ROOT}/lane/pretrained_e2c/lift/e2c_front.pt", map_location=device))
    e2c_wrist.load_state_dict(torch.load(f"{PROJECT_ROOT}/lane/pretrained_e2c/lift/e2c_wrist.pt", map_location=device))
    e2c_main.eval()
    e2c_wrist.eval()
    
    # 2. Load DINOv2
    print("Loading DINOv2...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dino.eval()
    
    def get_dino(img):
        # img: [B, C, H, W]
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return dino(img)
    
    # 3. Load Original Demo Data from lane/demo
    print("Loading lane/demo original data...")
    demo_dir = f"{PROJECT_ROOT}/lane/demo/robosuite_lift/20"
    pt_files = glob.glob(f"{demo_dir}/*.pt")
    if not pt_files:
        print("No .pt files found in demo directory.")
        return
    pt_file = pt_files[0]
    demo_data = torch.load(pt_file, weights_only=False)
    starts = np.load(f"{demo_dir}/demo_starts.npy")
    ends = np.load(f"{demo_dir}/demo_ends.npy")
    
    # images are in demo_data[0] of shape (N, 6, 128, 128)
    all_images = torch.from_numpy(demo_data[0]).float() / 255.0  # Normalize to [0, 1]
    
    z_demo_main = []
    z_demo_wrist = []
    one_step_dist_main = []
    one_step_dist_wrist = []
    
    playback_frames = []
    
    with torch.no_grad():
        # Iterate over each episode based on starts and ends
        for ep_idx, (start, end) in enumerate(tqdm(zip(starts, ends), total=len(starts))):
            ep_z_m = []
            ep_z_w = []
            
            for i in range(start, end):
                # Channel 0:3 is main, 3:6 is wrist (or vice versa, but we treat 0:3 as main)
                img_main = all_images[i:i+1, :3].to(device)
                img_wrist = all_images[i:i+1, 3:].to(device)
                
                dm = get_dino(img_main)
                dw = get_dino(img_wrist)
                
                zm = e2c_main.enc(dm)[0]
                zw = e2c_wrist.enc(dw)[0]
                
                ep_z_m.append(zm)
                ep_z_w.append(zw)
                
                playback_frames.append({
                    "ep_idx": ep_idx,
                    "img_m": img_main.cpu().squeeze(0),
                    "img_w": img_wrist.cpu().squeeze(0),
                    "z_m": zm.clone(),
                    "z_w": zw.clone()
                })
                
            if len(ep_z_m) > 1:
                z_m_tensor = torch.stack(ep_z_m, dim=0)
                z_w_tensor = torch.stack(ep_z_w, dim=0)
                z_demo_main.append(z_m_tensor)
                z_demo_wrist.append(z_w_tensor)
                
                diff_m = ((z_m_tensor[1:] - z_m_tensor[:-1])**2).sum(dim=1).mean().item()
                diff_w = ((z_w_tensor[1:] - z_w_tensor[:-1])**2).sum(dim=1).mean().item()
                one_step_dist_main.append(diff_m)
                one_step_dist_wrist.append(diff_w)
            
    ref_dist_m = np.mean(one_step_dist_main)
    ref_dist_w = np.mean(one_step_dist_wrist)
    print(f"Computed ref_one_step_dist_main: {ref_dist_m:.4f}")
    print(f"Computed ref_one_step_dist_wrist: {ref_dist_w:.4f}")
    
    # Setup interactive plots via canvas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    canvas = agg.FigureCanvasAgg(fig)
    
    history_m = []
    history_w = []
    
    print("\n=============================================")
    print("Demo Playback Ready!")
    print("Playing directly from lane/demo original files.")
    print("Distance should be EXACTLY 0.0")
    print("Press ESC to Quit")
    print("=============================================\n")
    
    current_ep_playing = playback_frames[0]["ep_idx"]
    
    for i, frame_data in enumerate(playback_frames):
        ep_idx = frame_data["ep_idx"]
        
        if ep_idx != current_ep_playing:
            history_m.clear()
            history_w.clear()
            current_ep_playing = ep_idx
            print(f"Playing Episode {ep_idx}...")
            
        # Images are [C, H, W] in 0-1 range
        img_m_np = frame_data["img_m"].numpy().transpose(1, 2, 0)
        img_w_np = frame_data["img_w"].numpy().transpose(1, 2, 0)
        
        # BGR Conversion
        # In the original datasets, images are usually RGB. We swap to BGR for cv2
        img_m_cv = cv2.cvtColor((img_m_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_w_cv = cv2.cvtColor((img_w_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        img_m_cv = cv2.resize(img_m_cv, (300, 300))
        img_w_cv = cv2.resize(img_w_cv, (300, 300))
        cams_stacked = np.vstack([img_m_cv, img_w_cv])
        
        # Compute real-time min_dist
        zm = frame_data["z_m"]
        zw = frame_data["z_w"]
        
        min_d_m = float('inf')
        min_d_w = float('inf')
        
        # Compute distance against all cached demos
        for z_demo in z_demo_main:
            dists = ((z_demo - zm)**2).sum(dim=1)
            min_d_m = min(min_d_m, dists.min().item())
            
        for z_demo in z_demo_wrist:
            dists = ((z_demo - zw)**2).sum(dim=1)
            min_d_w = min(min_d_w, dists.min().item())
            
        history_m.append(min_d_m)
        history_w.append(min_d_w)
        
        # Keep history length manageable
        if len(history_m) > 400:
            history_m.pop(0)
            history_w.pop(0)
            
        # Update Plot
        ax1.clear()
        ax1.plot(history_m, label="min_dist_main", color="blue")
        ax1.axhline(ref_dist_m, color="red", linestyle="--", label="ref_one_step_dist")
        ax1.set_ylim(-0.01, max(ref_dist_m * 2.0, 0.01)) # Zoom in
        ax1.set_title(f"Front Camera (Ep {ep_idx}) Latent Dist: {min_d_m:.6f}")
        ax1.legend(loc="upper right")
        
        ax2.clear()
        ax2.plot(history_w, label="min_dist_wrist", color="green")
        ax2.axhline(ref_dist_w, color="red", linestyle="--", label="ref_one_step_dist")
        ax2.set_ylim(-0.01, max(ref_dist_w * 2.0, 0.01))
        ax2.set_title(f"Wrist Camera (Ep {ep_idx}) Latent Dist: {min_d_w:.6f}")
        ax2.legend(loc="upper right")
        
        canvas.draw()
        plot_img = np.asarray(canvas.buffer_rgba())[..., :3]
        plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        # Combine into Dashboard
        h_cams = cams_stacked.shape[0]
        aspect = plot_img_bgr.shape[1] / plot_img_bgr.shape[0]
        plot_img_resized = cv2.resize(plot_img_bgr, (int(h_cams * aspect), h_cams))
        
        dashboard = np.hstack([cams_stacked, plot_img_resized])
        
        # Add text overlay showing it's perfectly 0
        status_color = (0, 255, 0) if min_d_m < 1e-4 and min_d_w < 1e-4 else (0, 0, 255)
        cv2.putText(dashboard, "PERFECT MATCH" if status_color == (0, 255, 0) else "MISMATCH", 
                    (cams_stacked.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
        cv2.imshow("Demo Playback Dashboard", dashboard)
        
        # 30fps playback speed
        key = cv2.waitKey(33) & 0xFF
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()
