import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_latent_distances():
    latent_path = "/home/ysl2683/cover/lane/pretrained_e2c/lift/demo_latents.pt"
    if not os.path.exists(latent_path):
        print("Latent file not found!")
        return
        
    data = torch.load(latent_path, map_location="cpu", weights_only=False)
    z_f_list = data["z_demo_front"]
    z_w_list = data["z_demo_wrist"]
    
    plt.figure(figsize=(12, 6))
    
    num_demos_to_plot = min(5, len(z_f_list))
    
    # Plot Front Camera Latent Distance to Goal
    plt.subplot(1, 2, 1)
    for i in range(num_demos_to_plot):
        z_f = z_f_list[i].squeeze(0) # Shape: (T, 16)
        goal_z_f = z_f[-1] # Last frame is goal
        
        # Calculate L2 distance from each frame to goal
        dist = torch.norm(z_f - goal_z_f, p=2, dim=-1).numpy()
        
        # Normalize time to 0-1 for plotting different length demos
        time_steps = np.linspace(0, 1, len(dist))
        plt.plot(time_steps, dist, label=f'Demo {i+1}')
        
    plt.title("Front Camera (16D) - Distance to Goal")
    plt.xlabel("Normalized Time")
    plt.ylabel("L2 Distance")
    plt.grid(True)
    plt.legend()
    
    # Plot Wrist Camera Latent Distance to Goal
    plt.subplot(1, 2, 2)
    for i in range(num_demos_to_plot):
        z_w = z_w_list[i].squeeze(0) # Shape: (T, 16)
        goal_z_w = z_w[-1] # Last frame is goal
        
        # Calculate L2 distance from each frame to goal
        dist = torch.norm(z_w - goal_z_w, p=2, dim=-1).numpy()
        
        # Normalize time to 0-1 for plotting different length demos
        time_steps = np.linspace(0, 1, len(dist))
        plt.plot(time_steps, dist, label=f'Demo {i+1}')
        
    plt.title("Wrist Camera (16D) - Distance to Goal")
    plt.xlabel("Normalized Time")
    plt.ylabel("L2 Distance")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = "/home/ysl2683/.gemini/antigravity-cli/brain/0e2a66f7-0466-4792-b66f-af0935b365fb/e2c_latent_distance.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    visualize_latent_distances()
