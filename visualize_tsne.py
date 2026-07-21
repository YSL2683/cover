import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE

def visualize_tsne():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    latent_path = os.path.join(SCRIPT_DIR, "lane/pretrained_e2c/lift/demo_latents.pt")
    if not os.path.exists(latent_path):
        print("Latent file not found!")
        return
        
    data = torch.load(latent_path, map_location="cpu", weights_only=False)
    z_f_list = data["z_demo_front"]
    z_w_list = data["z_demo_wrist"]
    
    num_demos_to_plot = len(z_f_list)
    
    # Collect all points to fit TSNE
    all_z_f = []
    all_z_w = []
    times_f = []
    times_w = []
    demo_ids_f = []
    
    for i in range(num_demos_to_plot):
        z_f = z_f_list[i].squeeze(0).numpy() # (T, 16)
        z_w = z_w_list[i].squeeze(0).numpy() # (T, 16)
        
        all_z_f.append(z_f)
        all_z_w.append(z_w)
        
        T_len = len(z_f)
        times_f.extend(np.linspace(0, 1, T_len))
        times_w.extend(np.linspace(0, 1, T_len))
        demo_ids_f.extend([i] * T_len)
        
    all_z_f = np.concatenate(all_z_f, axis=0)
    all_z_w = np.concatenate(all_z_w, axis=0)
    
    # Run TSNE
    tsne_f = TSNE(n_components=2, perplexity=15, random_state=42)
    tsne_w = TSNE(n_components=2, perplexity=15, random_state=42)
    
    z_f_2d = tsne_f.fit_transform(all_z_f)
    z_w_2d = tsne_w.fit_transform(all_z_w)
    
    plt.figure(figsize=(14, 6))
    
    # Custom colormaps for each demo to distinguish them
    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
    
    # Plot Front
    plt.subplot(1, 2, 1)
    idx = 0
    for i in range(num_demos_to_plot):
        T_len = len(z_f_list[i].squeeze(0))
        pts = z_f_2d[idx:idx+T_len]
        t_vals = np.linspace(0.2, 1.0, T_len) # 0.2 to 1.0 for alpha/color intensity
        
        plt.scatter(pts[:, 0], pts[:, 1], c=t_vals, cmap=cmaps[i % len(cmaps)], 
                    edgecolor='k', linewidth=0.5, s=50, label=f'Demo {i+1}')
        
        # Draw line connecting trajectory
        plt.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.3, linewidth=1)
        
        # Mark goal
        plt.scatter(pts[-1, 0], pts[-1, 1], color='red', marker='*', s=150, zorder=5)
        
        idx += T_len
        
    plt.title("Front Camera (16D -> 2D t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True, alpha=0.3)
    
    # Plot Wrist
    plt.subplot(1, 2, 2)
    idx = 0
    for i in range(num_demos_to_plot):
        T_len = len(z_w_list[i].squeeze(0))
        pts = z_w_2d[idx:idx+T_len]
        t_vals = np.linspace(0.2, 1.0, T_len)
        
        plt.scatter(pts[:, 0], pts[:, 1], c=t_vals, cmap=cmaps[i % len(cmaps)], 
                    edgecolor='k', linewidth=0.5, s=50, label=f'Demo {i+1}')
        
        # Draw line connecting trajectory
        plt.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.3, linewidth=1)
        
        # Mark goal
        plt.scatter(pts[-1, 0], pts[-1, 1], color='red', marker='*', s=150, zorder=5)
        
        idx += T_len
        
    plt.title("Wrist Camera (16D -> 2D t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "outputs/tsne_20_aligned.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    visualize_tsne()
