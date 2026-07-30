import sys
import os
from pathlib import Path
sys.path.append("/home/moai/ysl_ws/cover")

import time
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
from resfit.dexmg.environments.dexmg import create_vectorized_env
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load E2C models
    print("Loading Pretrained E2C Encoders...")
    e2c_main = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to(device)
    e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=7, z_dimension=16).to(device)
    
    e2c_main.load_state_dict(torch.load("/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift/e2c_front.pt", map_location=device))
    e2c_wrist.load_state_dict(torch.load("/home/moai/ysl_ws/cover/lane/pretrained_e2c/lift/e2c_wrist.pt", map_location=device))
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
    
    # 3. Load Offline Dataset and Compute z_demo & ref_one_step_dist
    print("Precomputing z_demo from offline dataset...")
    dataset_name = "ysl2683/lane_lift_id_20_aligned"
    dataset_path = f"/home/moai/ysl_ws/cover/resfit/my_lerobot_data/{dataset_name}"
    dataset = LeRobotDataset(dataset_name, root=dataset_path)
    
    z_demo_main = []
    z_demo_wrist = []
    one_step_dist_main = []
    one_step_dist_wrist = []
    
    with torch.no_grad():
        current_ep = -1
        ep_z_m = []
        ep_z_w = []
        
        # Load up to 5 episodes to be fast
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            ep_idx = sample["episode_index"].item()
            if ep_idx > 4:
                break
                
            if ep_idx != current_ep:
                if len(ep_z_m) > 1:
                    z_m_tensor = torch.cat(ep_z_m, dim=0)
                    z_w_tensor = torch.cat(ep_z_w, dim=0)
                    z_demo_main.append(z_m_tensor)
                    z_demo_wrist.append(z_w_tensor)
                    
                    diff_m = ((z_m_tensor[1:] - z_m_tensor[:-1])**2).sum(dim=1).mean().item()
                    diff_w = ((z_w_tensor[1:] - z_w_tensor[:-1])**2).sum(dim=1).mean().item()
                    one_step_dist_main.append(diff_m)
                    one_step_dist_wrist.append(diff_w)
                ep_z_m = []
                ep_z_w = []
                current_ep = ep_idx
                
            img_main = sample["observation.images.frontview"].unsqueeze(0).to(device)
            img_wrist = sample["observation.images.robot0_eye_in_hand"].unsqueeze(0).to(device)
            
            dm = get_dino(img_main)
            dw = get_dino(img_wrist)
            
            zm = e2c_main.enc(dm)[0]
            zw = e2c_wrist.enc(dw)[0]
            
            ep_z_m.append(zm)
            ep_z_w.append(zw)
            
        # Add the last episode
        if len(ep_z_m) > 1:
            z_m_tensor = torch.cat(ep_z_m, dim=0)
            z_w_tensor = torch.cat(ep_z_w, dim=0)
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
    
    # 4. Instantiate Environment
    print("Initializing Lift environment...")
    env = create_vectorized_env("Lift", num_envs=1, device=device)
    obs, _ = env.reset()
    env.vec_env.call('set_wrapper_attr', 'horizon', 1000000)
    
    # Setup interactive plots via canvas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    canvas = agg.FigureCanvasAgg(fig)
    
    history_m = []
    history_w = []
    
    print("\n=============================================")
    print("Interactive Control Ready!")
    print("Controls:")
    print("  W / S : Move +X / -X")
    print("  A / D : Move +Y / -Y")
    print("  Q / E : Move +Z / -Z")
    print("  Z / C : Rotate (Yaw) Left / Right")
    print("  Space : Toggle Gripper (Open/Close)")
    print("  R     : Reset Environment (Randomize cube)")
    print("  ESC   : Quit")
    print("=============================================\n")
    
    gripper_state = -1.0 # Open
    action_scale = 1.0
    
    while True:
        action = np.zeros((1, 7), dtype=np.float32)
        
        # Display Camera Feeds
        img_m_np = obs["observation.images.frontview"][0].cpu().numpy().transpose(1, 2, 0)
        img_w_np = obs["observation.images.robot0_eye_in_hand"][0].cpu().numpy().transpose(1, 2, 0)
        
        img_m_cv = cv2.cvtColor((img_m_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_w_cv = cv2.cvtColor((img_w_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Display in slightly larger window for ease of viewing
        img_m_cv = cv2.resize(img_m_cv, (300, 300))
        img_w_cv = cv2.resize(img_w_cv, (300, 300))
        
        # Only show dataset images
        cv2.putText(img_m_cv, "Front View (Model)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_w_cv, "Wrist View (Model)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cams_stacked = np.vstack([img_m_cv, img_w_cv])
        
        # Keyboard Input
        key = cv2.waitKey(50) & 0xFF
        
        if key == 27: # ESC
            break
        elif key == ord('r'):
            obs, _ = env.reset()
            history_m.clear()
            history_w.clear()
            print("Environment Reset (Cube randomized).")
            continue
            
        elif key == ord('w'): action[0, 0] = action_scale
        elif key == ord('s'): action[0, 0] = -action_scale
        elif key == ord('a'): action[0, 1] = action_scale
        elif key == ord('d'): action[0, 1] = -action_scale
        elif key == ord('q'): action[0, 2] = action_scale
        elif key == ord('e'): action[0, 2] = -action_scale
        elif key == ord('z'): action[0, 5] = action_scale
        elif key == ord('c'): action[0, 5] = -action_scale
        elif key == ord(' '): 
            gripper_state *= -1.0
            print("Gripper toggled.")
            
        action[0, 6] = gripper_state
        
        # Compute real-time min_dist
        with torch.no_grad():
            img_m_tensor = obs["observation.images.frontview"].to(device)
            img_w_tensor = obs["observation.images.robot0_eye_in_hand"].to(device)
            
            dm = get_dino(img_m_tensor)
            dw = get_dino(img_w_tensor)
            
            zm = e2c_main.enc(dm)[0]
            zw = e2c_wrist.enc(dw)[0]
            
            min_d_m = float('inf')
            min_d_w = float('inf')
            
            for z_demo in z_demo_main:
                dists = ((z_demo - zm)**2).sum(dim=1)
                min_d_m = min(min_d_m, dists.min().item())
                
            for z_demo in z_demo_wrist:
                dists = ((z_demo - zw)**2).sum(dim=1)
                min_d_w = min(min_d_w, dists.min().item())
                
            history_m.append(min_d_m)
            history_w.append(min_d_w)
            
            # Keep history from growing too long (keep last 100 steps)
            if len(history_m) > 100:
                history_m.pop(0)
                history_w.pop(0)
                
            # Update Plot
            ax1.clear()
            ax1.plot(history_m, label="min_dist_main", color="blue")
            ax1.axhline(ref_dist_m, color="red", linestyle="--", label="ref_one_step_dist")
            ax1.set_title(f"Front Camera Latent Distance (Current: {min_d_m:.2f})")
            ax1.legend(loc="upper right")
            
            ax2.clear()
            ax2.plot(history_w, label="min_dist_wrist", color="green")
            ax2.axhline(ref_dist_w, color="red", linestyle="--", label="ref_one_step_dist")
            ax2.set_title(f"Wrist Camera Latent Distance (Current: {min_d_w:.2f})")
            ax2.legend(loc="upper right")
            
            canvas.draw()
            plot_img = np.asarray(canvas.buffer_rgba())[..., :3]
            plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            
            # Combine into Dashboard
            h_cams = cams_stacked.shape[0]
            # Resize plot_img_bgr height to match cams_stacked
            aspect = plot_img_bgr.shape[1] / plot_img_bgr.shape[0]
            plot_img_resized = cv2.resize(plot_img_bgr, (int(h_cams * aspect), h_cams))
            
            dashboard = np.hstack([cams_stacked, plot_img_resized])
            cv2.imshow("Dashboard", dashboard)
            
        # Step Environment
        # Ignore done flag to prevent auto-reset so user can control freely
        obs, reward, terminated, truncated, info = env.step(action)
            
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()
