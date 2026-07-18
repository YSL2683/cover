import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Import MLPE2C from LaNE
from e2c import MLPE2C

def load_demos(demo_dir):
    # Load .pt payload
    files = os.listdir(demo_dir)
    pt_files = [f for f in files if f.endswith(".pt")]
    if len(pt_files) == 0:
        raise ValueError("No .pt file found in demo directory")
    
    pt_path = os.path.join(demo_dir, pt_files[0])
    payload = torch.load(pt_path, weights_only=False)
    
    obs_list = payload[0]      # [N, 6, 128, 128]
    next_obs_list = payload[1] # [N, 6, 128, 128]
    action_list = torch.tensor(payload[2], dtype=torch.float32)   # [N, action_dim]
    
    # Load demo_starts and demo_ends
    demo_starts = np.load(os.path.join(demo_dir, "demo_starts.npy"))
    demo_ends = np.load(os.path.join(demo_dir, "demo_ends.npy"))
    
    return obs_list, next_obs_list, action_list, demo_starts, demo_ends

def extract_dino_features(obs_list, dino, device, batch_size=32):
    # obs_list is shape [N, C, H, W], uint8 from 0 to 255.
    # We should convert to float and scale. 
    # Usually images in LaNE demo are saved. Let's see how DINO is used in LaNE.
    # In sac.py L720 dino_embed, it converts to float and interpolates to 112 -> 128 -> etc.
    # Let's assume standard normalization.
    
    features_front = []
    features_wrist = []
    
    # Simple normalization / preprocessing matching LaNE
    # sac.py: preprocess = transforms.Compose([transforms.Resize(pre_transform_image_size), ...])
    # DINO expects 14-divisible sizes, usually 224 or 128. LaNE uses 112 pre-crop then 112.
    
    import torchvision.transforms as T
    # Simplified DINO transform
    transform = T.Compose([
        T.Resize((112, 112)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dino.eval()
    
    N = len(obs_list)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.tensor(obs_list[i:i+batch_size], device=device).float() / 255.0
            front = batch[:, :3]
            wrist = batch[:, 3:6]
            
            front = transform(front)
            wrist = transform(wrist)
            
            feat_f = dino(front)
            feat_w = dino(wrist)
            
            features_front.append(feat_f.cpu())
            features_wrist.append(feat_w.cpu())
            
    return torch.cat(features_front, dim=0), torch.cat(features_wrist, dim=0)

def train_e2c(obs_f, obs_w, next_obs_f, next_obs_w, actions, device="cuda", n_iter=5000):
    action_dim = actions.shape[1]
    
    e2c_front = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=16).to(device)
    # Wrist uses higher dim as per design review
    e2c_wrist = MLPE2C(obs_shape=(384,), action_dim=action_dim, z_dimension=32).to(device)
    
    opt_f = torch.optim.Adam(e2c_front.parameters(), lr=1e-4)
    opt_w = torch.optim.Adam(e2c_wrist.parameters(), lr=1e-4)
    
    dataset = TensorDataset(obs_f, obs_w, actions, next_obs_f, next_obs_w)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("Training E2C...")
    for step in range(n_iter):
        total_loss_f = 0
        total_loss_w = 0
        
        for b_obs_f, b_obs_w, b_act, b_nobs_f, b_nobs_w in loader:
            b_obs_f, b_obs_w = b_obs_f.to(device), b_obs_w.to(device)
            b_act = b_act.to(device).float()
            b_nobs_f, b_nobs_w = b_nobs_f.to(device), b_nobs_w.to(device)
            
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
            
        if (step+1) % 100 == 0:
            print(f"Step {step+1}: Loss F = {total_loss_f/len(loader):.4f}, Loss W = {total_loss_w/len(loader):.4f}")
            
    return e2c_front, e2c_wrist

def precompute_demo_latents(e2c_front, e2c_wrist, obs_f, obs_w, demo_starts, demo_ends, device="cuda"):
    e2c_front.eval()
    e2c_wrist.eval()
    
    z_demo_front = []
    z_demo_wrist = []
    demo_lengths = []
    
    with torch.no_grad():
        for start, end in zip(demo_starts, demo_ends):
            traj_f = obs_f[start:end].to(device)
            traj_w = obs_w[start:end].to(device)
            
            zf, _ = e2c_front.enc(traj_f)
            zw, _ = e2c_wrist.enc(traj_w)
            
            z_demo_front.append(zf.unsqueeze(0).cpu())
            z_demo_wrist.append(zw.unsqueeze(0).cpu())
            demo_lengths.append(end - start)
            
    return z_demo_front, z_demo_wrist, demo_lengths

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo/robosuite_pick_place_can/20/")
    
    print("Loading demos...")
    obs, next_obs, actions, starts, ends = load_demos(demo_dir)
    
    print("Loading DINOv2...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    
    print("Extracting DINO features...")
    obs_f, obs_w = extract_dino_features(obs, dino, device)
    next_obs_f, next_obs_w = extract_dino_features(next_obs, dino, device)
    
    e2c_f, e2c_w = train_e2c(obs_f, obs_w, next_obs_f, next_obs_w, actions, device=device, n_iter=1000)
    
    print("Precomputing demo latents...")
    z_df, z_dw, t_lens = precompute_demo_latents(e2c_f, e2c_w, obs_f, obs_w, starts, ends, device)
    
    print("Saving artifacts...")
    os.makedirs("pretrained_e2c", exist_ok=True)
    torch.save(e2c_f.state_dict(), "pretrained_e2c/e2c_front.pt")
    torch.save(e2c_w.state_dict(), "pretrained_e2c/e2c_wrist.pt")
    torch.save({
        "z_demo_front": z_df,
        "z_demo_wrist": z_dw,
        "demo_lengths": t_lens
    }, "pretrained_e2c/demo_latents.pt")
    
    print("Done!")
