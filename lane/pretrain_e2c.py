import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import argparse
import torchvision.transforms as T

# Import MLPE2C from LaNE
from e2c import MLPE2C

def random_crop(images, output_size=112):
    """
    images: tensor shape (B, C, H, W)
    """
    n, c, h, w = images.shape
    crop_max = h - output_size + 1
    w1 = torch.randint(0, crop_max, (n,))
    h1 = torch.randint(0, crop_max, (n,))
    cropped = torch.empty((n, c, output_size, output_size), dtype=images.dtype, device=images.device)
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = image[:, h11 : h11 + output_size, w11 : w11 + output_size]
    return cropped

def center_crop(images, output_size=112):
    n, c, h, w = images.shape
    assert h >= output_size
    top = (h - output_size) // 2
    left = (w - output_size) // 2
    return images[:, :, top : top + output_size, left : left + output_size]


def load_demos(demo_dir):
    # Load .pt payload
    files = os.listdir(demo_dir)
    pt_files = [f for f in files if f.endswith(".pt")]
    if len(pt_files) == 0:
        raise ValueError("No .pt file found in demo directory")
    
    pt_path = os.path.join(demo_dir, pt_files[0])
    payload = torch.load(pt_path, weights_only=False)
    
    obs_list = payload[0]      # [N, 6, 128, 128] uint8
    next_obs_list = payload[1] # [N, 6, 128, 128] uint8
    action_list = torch.tensor(payload[2], dtype=torch.float32)   # [N, action_dim]
    
    # Load demo_starts and demo_ends
    demo_starts = np.load(os.path.join(demo_dir, "demo_starts.npy"))
    demo_ends = np.load(os.path.join(demo_dir, "demo_ends.npy"))
    
    return torch.tensor(obs_list), torch.tensor(next_obs_list), action_list, demo_starts, demo_ends

def get_dino_features(images, dino, device, is_training=True):
    # images: [B, 6, 128, 128] uint8
    images = images.to(device)
    if is_training:
        cropped = random_crop(images, 112)
    else:
        cropped = center_crop(images, 112)
        
    cropped = cropped.float() / 255.0
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    front = normalize(cropped[:, :3])
    wrist = normalize(cropped[:, 3:6])
    
    with torch.no_grad():
        feat_f = dino(front)
        feat_w = dino(wrist)
        
    return feat_f, feat_w


def train_e2c(obs, next_obs, actions, dino, device="cuda", n_iter=5000, mse_tol=1e-2, mode="decoupled"):
    action_dim = actions.shape[1]
    
    if mode == "unified":
        e2c_unified = MLPE2C(obs_shape=(768,), action_dim=action_dim, z_dimension=16).to(device)
        opt_unified = torch.optim.Adam(e2c_unified.parameters(), lr=1e-4)
    else:
        e2c_front = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
        e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
        
        opt_f = torch.optim.Adam(e2c_front.parameters(), lr=1e-4)
        opt_w = torch.optim.Adam(e2c_wrist.parameters(), lr=1e-4)
    
    dataset = TensorDataset(obs, next_obs, actions)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print(f"Training E2C (Early stopping if MSE < {mse_tol})...")
    
    global_step = 0
    early_stop = False
    
    for epoch in range(n_iter):
        if early_stop:
            break
            
        total_loss_f = 0
        total_loss_w = 0
        total_loss_u = 0
        
        if mode == "unified":
            e2c_unified.train()
        else:
            e2c_front.train()
            e2c_wrist.train()
        
        for b_obs, b_nobs, b_act in loader:
            b_act = b_act.to(device).float()
            
            # Extract features on the fly with random crop
            b_obs_f, b_obs_w = get_dino_features(b_obs, dino, device, is_training=True)
            b_nobs_f, b_nobs_w = get_dino_features(b_nobs, dino, device, is_training=True)
            
            if mode == "unified":
                b_obs_u = torch.cat([b_obs_f, b_obs_w], dim=1)
                b_nobs_u = torch.cat([b_nobs_f, b_nobs_w], dim=1)
                
                dkl_u, mse_u, ref_kl_u, _ = e2c_unified(b_obs_u, b_act, b_nobs_u, None, None)
                loss_u = dkl_u + mse_u * 768 + ref_kl_u
                
                opt_unified.zero_grad()
                loss_u.backward()
                opt_unified.step()
                
                total_loss_u += loss_u.item()
                
                global_step += 1
                
                if mse_tol is not None and mse_u.item() < mse_tol:
                    print(f"Early stopping at epoch {epoch+1}, global step {global_step} (MSE U={mse_u.item():.4f})")
                    early_stop = True
                    break
            else:
                dkl_f, mse_f, ref_kl_f, _ = e2c_front(b_obs_f, b_act, b_nobs_f, None, None)
                loss_f = dkl_f + mse_f * 384 + ref_kl_f
                
                dkl_w, mse_w, ref_kl_w, _ = e2c_wrist(b_obs_w, b_act, b_nobs_w, None, None)
                loss_w = dkl_w + mse_w * 384 + ref_kl_w
                
                opt_f.zero_grad()
                loss_f.backward()
                opt_f.step()
                
                opt_w.zero_grad()
                loss_w.backward()
                opt_w.step()
                
                total_loss_f += loss_f.item()
                total_loss_w += loss_w.item()
                
                global_step += 1
                
                if mse_tol is not None and mse_f.item() < mse_tol and mse_w.item() < mse_tol:
                    print(f"Early stopping at epoch {epoch+1}, global step {global_step} (MSE F={mse_f.item():.4f}, MSE W={mse_w.item():.4f})")
                    early_stop = True
                    break
            
        if not early_stop and (epoch+1) % 100 == 0:
            if mode == "unified":
                print(f"Epoch {epoch+1}: Loss U = {total_loss_u/len(loader):.4f}")
            else:
                print(f"Epoch {epoch+1}: Loss F = {total_loss_f/len(loader):.4f}, Loss W = {total_loss_w/len(loader):.4f}")
            
    if mode == "unified":
        return e2c_unified
    else:
        return e2c_front, e2c_wrist

def precompute_demo_latents(e2c_models, obs, dino, demo_starts, demo_ends, device="cuda", mode="decoupled"):
    if mode == "unified":
        e2c_models.eval()
    else:
        e2c_models[0].eval()
        e2c_models[1].eval()
    
    z_demo_front = []
    z_demo_wrist = []
    z_demo_unified = []
    demo_lengths = []
    
    batch_size = 32
    
    with torch.no_grad():
        for start, end in zip(demo_starts, demo_ends):
            traj_obs = obs[start:end]
            
            zf_list = []
            zw_list = []
            zu_list = []
            
            for i in range(0, len(traj_obs), batch_size):
                b_obs = traj_obs[i:i+batch_size]
                feat_f, feat_w = get_dino_features(b_obs, dino, device, is_training=False) # center crop
                
                if mode == "unified":
                    feat_u = torch.cat([feat_f, feat_w], dim=1)
                    zu, _ = e2c_models.enc(feat_u)
                    zu_list.append(zu.cpu())
                else:
                    zf, _ = e2c_models[0].enc(feat_f)
                    zw, _ = e2c_models[1].enc(feat_w)
                    zf_list.append(zf.cpu())
                    zw_list.append(zw.cpu())
                
            if mode == "unified":
                z_demo_unified.append(torch.cat(zu_list, dim=0).unsqueeze(0))
            else:
                z_demo_front.append(torch.cat(zf_list, dim=0).unsqueeze(0))
                z_demo_wrist.append(torch.cat(zw_list, dim=0).unsqueeze(0))
            demo_lengths.append(end - start)
            
    if mode == "unified":
        return z_demo_unified, demo_lengths
    else:
        return z_demo_front, z_demo_wrist, demo_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain E2C latent space on demos")
    parser.add_argument("--demo_dir", type=str, required=True, help="Path to demo directory (e.g. demo/robosuite_nut_assembly_square/20/)")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save pretrained E2C models")
    parser.add_argument("--mode", type=str, default="decoupled", choices=["decoupled", "unified"], help="Mode for latent space (decoupled or unified)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_dir = args.demo_dir
    
    print("Loading demos...")
    obs, next_obs, actions, starts, ends = load_demos(demo_dir)
    
    print("Loading DINOv2...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    dino.eval()
    
    print("Training E2C with on-the-fly random crop augmentation...")
    e2c_models = train_e2c(obs, next_obs, actions, dino, device=device, n_iter=1000, mode=args.mode)
    
    print("Precomputing demo latents (with center crop)...")
    if args.mode == "unified":
        z_du, t_lens = precompute_demo_latents(e2c_models, obs, dino, starts, ends, device, mode=args.mode)
    else:
        z_df, z_dw, t_lens = precompute_demo_latents(e2c_models, obs, dino, starts, ends, device, mode=args.mode)
    
    print("Saving artifacts...")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.mode == "unified":
        torch.save(e2c_models.state_dict(), os.path.join(save_dir, "e2c_unified.pt"))
        torch.save({
            "z_demo_unified": z_du,
            "demo_lengths": t_lens
        }, os.path.join(save_dir, "demo_latents.pt"))
    else:
        torch.save(e2c_models[0].state_dict(), os.path.join(save_dir, "e2c_front.pt"))
        torch.save(e2c_models[1].state_dict(), os.path.join(save_dir, "e2c_wrist.pt"))
        torch.save({
            "z_demo_front": z_df,
            "z_demo_wrist": z_dw,
            "demo_lengths": t_lens
        }, os.path.join(save_dir, "demo_latents.pt"))
    
    print("Done!")
