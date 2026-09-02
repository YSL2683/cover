import torch
import numpy as np
import sys
import wandb
from pathlib import Path
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Add lane to sys.path to import MLPE2C
lane_dir = Path(__file__).resolve().parents[3] / "lane"
sys.path.append(str(lane_dir))
from e2c import MLPE2C

class LaNERewardShaper:
    def __init__(self, device, action_dim, offline_rb, online_rb=None, p_reward=1.0, action_l2_reg_weight=0.0, reward_type="reward_2",
                 beta=0.5, alpha=0.98, w_m=0.3, w_w=0.7, gamma=0.99, e2c_mode="decoupled", ref_horizon=30.0):
        self.device = device
        self.p_reward = p_reward
        self.action_l2_reg_weight = action_l2_reg_weight
        self.reward_type = reward_type
        self.beta = beta
        self.alpha = alpha
        self.w_m = w_m
        self.w_w = w_w
        self.gamma = gamma
        self.ref_horizon = ref_horizon
        self.offline_rb = offline_rb
        self.online_rb = online_rb
        self.e2c_mode = e2c_mode
        
        # Determine main camera key dynamically from offline buffer
        try:
            obs_keys = offline_rb[0]["obs"].keys()
            self.main_cam_key = "observation.images.agentview" if "observation.images.agentview" in obs_keys else "observation.images.frontview"
        except:
            self.main_cam_key = "observation.images.frontview"
        
        print("Loading DINOv2 model...")
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
        self.dino.eval()
        print("DINOv2 loaded.")
        
        if self.e2c_mode == "unified":
            self.e2c_unified = MLPE2C(
                obs_shape=(768,), action_dim=action_dim, z_dimension=16, crop_shape=None
            ).to(device)
            self.e2c_unified_opt = torch.optim.Adam(self.e2c_unified.parameters(), lr=1e-4)
            self.z_demo_unified_cache = {}
            self.ref_one_step_dist_unified = None
        else:
            # Two cameras: front and wrist, 384 dim each for DINOv2 ViT-S
            self.e2c_main = MLPE2C(
                obs_shape=(384,), action_dim=action_dim, z_dimension=16, crop_shape=None
            ).to(device)
            self.e2c_wrist = MLPE2C(
                obs_shape=(384,), action_dim=action_dim, z_dimension=16, crop_shape=None
            ).to(device)
            
            self.e2c_main_opt = torch.optim.Adam(self.e2c_main.parameters(), lr=1e-4)
            self.e2c_wrist_opt = torch.optim.Adam(self.e2c_wrist.parameters(), lr=1e-4)
            
            self.z_demo_main_cache = {}
            self.z_demo_wrist_cache = {}
            self.ref_one_step_dist_main = None
            self.ref_one_step_dist_wrist = None
            
        self.initialized = False
        
        dones = offline_rb["next", "done"].squeeze().cpu().numpy()
        self.demo_ends = np.where(dones)[0]
        self.demo_starts = np.zeros_like(self.demo_ends)
        self.demo_starts[1:] = self.demo_ends[:-1] + 1
        
        self.dino_cache_offline = None
        
    def dino_embed(self, obs):
        with torch.no_grad():
            image1, image2 = torch.split(obs, [3, 3], dim=1)
            image1 = TF.center_crop(image1, output_size=112)
            image2 = TF.center_crop(image2, output_size=112)
            image1 = TF.normalize(image1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image2 = TF.normalize(image2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            dino_emb1 = self.dino(image1)
            dino_emb2 = self.dino(image2)
        return torch.cat([dino_emb1, dino_emb2], dim=1)

    def precompute_offline_dino(self):
        print("Precomputing DINOv2 embeddings for offline buffer...")
        dino_obs_list = []
        dino_next_list = []
        batch_size = 128
        for i in range(0, len(self.offline_rb), batch_size):
            batch = self.offline_rb[i:i+batch_size].to(self.device)
            img_main = batch["obs", self.main_cam_key]
            img_wrist = batch["obs", "observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([img_main, img_wrist], dim=1).float() / 255.0
            
            next_img_main = batch["next", "obs", self.main_cam_key]
            next_img_wrist = batch["next", "obs", "observation.images.robot0_eye_in_hand"]
            next_obs_img = torch.cat([next_img_main, next_img_wrist], dim=1).float() / 255.0
            
            dino_obs = self.dino_embed(obs_img).cpu()
            dino_next = self.dino_embed(next_obs_img).cpu()
            dino_obs_list.append(dino_obs)
            dino_next_list.append(dino_next)
            
        dino_obs_tensor = torch.cat(dino_obs_list, dim=0)
        dino_next_tensor = torch.cat(dino_next_list, dim=0)
        
        storage_size = self.offline_rb._storage._storage.shape[0]
        
        full_dino_obs = torch.zeros((storage_size, dino_obs_tensor.shape[-1]), device=self.offline_rb._storage._storage.device)
        full_dino_obs[:dino_obs_tensor.shape[0]] = dino_obs_tensor.to(full_dino_obs.device)
        
        full_dino_next = torch.zeros((storage_size, dino_next_tensor.shape[-1]), device=self.offline_rb._storage._storage.device)
        full_dino_next[:dino_next_tensor.shape[0]] = dino_next_tensor.to(full_dino_next.device)
        
        self.offline_rb._storage._storage.unlock_()
        self.offline_rb._storage._storage.set("dino", full_dino_obs)
        self.offline_rb._storage._storage.set(("next", "dino"), full_dino_next)
        self.offline_rb._storage._storage.lock_()
        print("Precomputing done.")

    def precompute_online_dino(self, online_rb):
        print("Precomputing DINOv2 embeddings for warmup online buffer...")
        dino_obs_list = []
        dino_next_list = []
        batch_size = 128
        for i in range(0, len(online_rb), batch_size):
            batch = online_rb[i:i+batch_size].to(self.device)
            img_main = batch["obs", self.main_cam_key]
            img_wrist = batch["obs", "observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([img_main, img_wrist], dim=1).float() / 255.0
            
            next_img_main = batch["next", "obs", self.main_cam_key]
            next_img_wrist = batch["next", "obs", "observation.images.robot0_eye_in_hand"]
            next_obs_img = torch.cat([next_img_main, next_img_wrist], dim=1).float() / 255.0
            
            dino_obs = self.dino_embed(obs_img).cpu()
            dino_next = self.dino_embed(next_obs_img).cpu()
            dino_obs_list.append(dino_obs)
            dino_next_list.append(dino_next)
            
        full_dino_obs = torch.zeros((online_rb._storage._storage.shape[0], 768), dtype=torch.float32, device=online_rb._storage._storage.device)
        full_dino_next = torch.zeros((online_rb._storage._storage.shape[0], 768), dtype=torch.float32, device=online_rb._storage._storage.device)
        
        if len(dino_obs_list) > 0:
            cat_dino_obs = torch.cat(dino_obs_list, dim=0).to(online_rb._storage._storage.device)
            cat_dino_next = torch.cat(dino_next_list, dim=0).to(online_rb._storage._storage.device)
            full_dino_obs[:len(online_rb)] = cat_dino_obs
            full_dino_next[:len(online_rb)] = cat_dino_next
            
        online_rb._storage._storage.unlock_()
        online_rb._storage._storage.set("dino", full_dino_obs)
        online_rb._storage._storage.set(("next", "dino"), full_dino_next)
        online_rb._storage._storage.lock_()
        print("Online precomputing done.")

    def add_dino_to_tensordict(self, td):
        # td is the tensordict collected from env
        # add "dino" and "next", "dino"
        img_main = td["obs", self.main_cam_key]
        img_wrist = td["obs", "observation.images.robot0_eye_in_hand"]
        is_unbatched = img_main.ndim == 3
        if is_unbatched:
            img_main = img_main.unsqueeze(0)
            img_wrist = img_wrist.unsqueeze(0)
            
        obs_img = torch.cat([img_main, img_wrist], dim=1).float() / 255.0
        
        next_img_main = td["next", "obs", self.main_cam_key]
        next_img_wrist = td["next", "obs", "observation.images.robot0_eye_in_hand"]
        if is_unbatched:
            next_img_main = next_img_main.unsqueeze(0)
            next_img_wrist = next_img_wrist.unsqueeze(0)
            
        next_obs_img = torch.cat([next_img_main, next_img_wrist], dim=1).float() / 255.0
        
        dino_obs = self.dino_embed(obs_img.to(self.device)).to(img_main.device)
        dino_next = self.dino_embed(next_obs_img.to(self.device)).to(img_main.device)
        
        if is_unbatched:
            td["dino"] = dino_obs.squeeze(0)
            td["next", "dino"] = dino_next.squeeze(0)
        else:
            td["dino"] = dino_obs
            td["next", "dino"] = dino_next
        return td

    def _sample_e2c(self, batch_size=256):
        if self.online_rb is not None and len(self.online_rb) > batch_size // 2:
            off_batch_size = batch_size // 2
            on_batch_size = batch_size - off_batch_size
            
            idx_off = np.random.randint(0, len(self.offline_rb), size=off_batch_size)
            batch_off = self.offline_rb[idx_off]
            
            batch_on = self.online_rb.sample(on_batch_size)
            batch = torch.cat([batch_off, batch_on], dim=0).to(self.device)
        else:
            idx = np.random.randint(0, len(self.offline_rb), size=batch_size)
            batch = self.offline_rb[idx].to(self.device)
            
        dino_obs = batch["dino"]
        dino_next_obs = batch["next", "dino"]
        action = batch["action"]
        return dino_obs, action, dino_next_obs

    def update_e2c(self, num_updates=1000, mse_tol=0.2):
        for i in range(num_updates):
            dino_obs, action, dino_next_obs = self._sample_e2c()
            
            if self.e2c_mode == "unified":
                dkl_u, mse_u, ref_kl_u, _ = self.e2c_unified(dino_obs, action, dino_next_obs, None, None)
                loss_u = dkl_u + mse_u * 768 + ref_kl_u
                
                self.e2c_unified_opt.zero_grad()
                loss_u.backward()
                self.e2c_unified_opt.step()
                
                if mse_tol is not None and mse_u.item() < mse_tol:
                    break
            else:
                dino_obs_m, dino_obs_w = dino_obs[:, :384], dino_obs[:, 384:]
                dino_next_obs_m, dino_next_obs_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
                
                mse_w_mult = 384
                
                dkl_m, mse_m, ref_kl_m, _ = self.e2c_main(dino_obs_m, action, dino_next_obs_m, None, None)
                dkl_w, mse_w, ref_kl_w, _ = self.e2c_wrist(dino_obs_w, action, dino_next_obs_w, None, None)
                
                loss_m = dkl_m + mse_m * 384 + ref_kl_m
                loss_w = dkl_w + mse_w * mse_w_mult + ref_kl_w
                loss = loss_m + loss_w
                
                self.e2c_main_opt.zero_grad()
                self.e2c_wrist_opt.zero_grad()
                loss.backward()
                self.e2c_main_opt.step()
                self.e2c_wrist_opt.step()
                
                if mse_tol is not None and ((mse_m + mse_w)/2).item() < mse_tol:
                    break
                
        if self.e2c_mode == "unified":
            return {
                "lane/e2c_loss_u": loss_u.item(),
                "lane/e2c_mse_u": mse_u.item(),
                "lane/e2c_updates": i + 1
            }
        else:
            return {
                "lane/e2c_loss_m": loss_m.item(),
                "lane/e2c_loss_w": loss_w.item(),
                "lane/e2c_mse_m": mse_m.item(),
                "lane/e2c_mse_w": mse_w.item(),
                "lane/e2c_updates": i + 1
            }
                
    def initialize_demos(self):
        one_step_dist_list_main = []
        one_step_dist_list_wrist = []
        one_step_dist_list_unified = []
        
        flat_z_m_list = []
        flat_rem_t_m_list = []
        flat_z_w_list = []
        flat_rem_t_w_list = []
        flat_z_u_list = []
        flat_rem_t_u_list = []
        
        for i, (start, end) in enumerate(zip(self.demo_starts, self.demo_ends)):
            batch = self.offline_rb[start:end+1].to(self.device)
            dino_next_obs = batch["next", "dino"]
            
            if self.e2c_mode == "unified":
                z_u = self.e2c_unified.enc(dino_next_obs)[0].detach() # [T, latent_dim]
                self.z_demo_unified_cache[i] = z_u.unsqueeze(0).cpu().numpy() # keep old cache for safety
                
                T = z_u.shape[0]
                rem_t = (T - torch.arange(T, device=self.device)) / T
                flat_z_u_list.append(z_u)
                flat_rem_t_u_list.append(rem_t)
                
                if T > 1:
                    one_step_dist_list_unified.append(((z_u[1:] - z_u[:-1]) ** 2).sum(axis=1).mean().item())
            else:
                dino_f, dino_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
                
                z_m = self.e2c_main.enc(dino_f)[0].detach() # [T, latent_dim]
                z_w = self.e2c_wrist.enc(dino_w)[0].detach() # [T, latent_dim]
                
                self.z_demo_main_cache[i] = z_m.unsqueeze(0).cpu().numpy()
                self.z_demo_wrist_cache[i] = z_w.unsqueeze(0).cpu().numpy()
                
                T = z_m.shape[0]
                rem_t = (T - torch.arange(T, device=self.device, dtype=torch.float32)) / T
                flat_z_m_list.append(z_m)
                flat_z_w_list.append(z_w)
                flat_rem_t_m_list.append(rem_t)
                flat_rem_t_w_list.append(rem_t)
                
                if T > 1:
                    one_step_dist_list_main.append(((z_m[1:] - z_m[:-1]) ** 2).sum(dim=1).mean().item())
                    one_step_dist_list_wrist.append(((z_w[1:] - z_w[:-1]) ** 2).sum(dim=1).mean().item())
        
        if self.e2c_mode == "unified":
            self.ref_one_step_dist_unified = sum(one_step_dist_list_unified) / len(one_step_dist_list_unified)
            self.flat_z_u = torch.cat(flat_z_u_list, dim=0)
            self.flat_rem_t_u = torch.cat(flat_rem_t_u_list, dim=0)
        else:
            self.ref_one_step_dist_main = sum(one_step_dist_list_main) / len(one_step_dist_list_main)
            self.ref_one_step_dist_wrist = sum(one_step_dist_list_wrist) / len(one_step_dist_list_wrist)
            self.flat_z_m = torch.cat(flat_z_m_list, dim=0)
            self.flat_z_w = torch.cat(flat_z_w_list, dim=0)
            self.flat_rem_t_m = torch.cat(flat_rem_t_m_list, dim=0)
            self.flat_rem_t_w = torch.cat(flat_rem_t_w_list, dim=0)
            
        self.initialized = True
    def _compute_potential_unified(self, dino_tensor):
        z_pred_u = self.e2c_unified.enc(dino_tensor)[0].unsqueeze(1).detach().cpu().numpy()
        
        N = len(dino_tensor)
        min_dist_u = np.ones(N) * 10000
        idx_u_best = np.zeros(N)
        T_demos_u = np.zeros(N)
        
        for i in range(len(self.demo_starts)):
            z_demo_u = self.z_demo_unified_cache[i]
            z_dist_u = ((z_demo_u - z_pred_u) ** 2).sum(axis=2)
            z_dist_min_u = z_dist_u.min(axis=1)
            update_min_u = z_dist_min_u < min_dist_u
            min_dist_u[update_min_u] = z_dist_min_u[update_min_u]
            idx_u_best[update_min_u] = z_dist_u.argmin(axis=1)[update_min_u]
            T_demos_u[update_min_u] = z_dist_u.shape[1]
            
        gamma_u = self.beta / ((self.ref_one_step_dist_unified ** 2) + 1e-8)
        
        # 4th power kernel
        S_unified = np.exp(-gamma_u * (min_dist_u ** 2))
        rem_t_u_norm = self.ref_horizon * (T_demos_u - idx_u_best) / np.maximum(T_demos_u, 1)
        
        Phi = np.power(self.alpha, rem_t_u_norm) * S_unified
        
        return Phi, S_unified, min_dist_u, rem_t_u_norm

    def _compute_potential(self, dino_tensor):
        """Computes the visual potential function Phi(s) for a batch of DINO embeddings."""
        dino_m, dino_w = dino_tensor[:, :384], dino_tensor[:, 384:]
        
        z_pred_m = self.e2c_main.enc(dino_m)[0].detach()
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].detach()
        
        # Calculate explicit squared Euclidean distance to avoid torch.cdist precision issues (e.g. catastrophic cancellation)
        dist_m = torch.sum((z_pred_m.unsqueeze(1) - self.flat_z_m.unsqueeze(0)) ** 2, dim=2)
        min_dist_m, min_idx_m = dist_m.min(dim=1)
        rem_t_m = self.flat_rem_t_m[min_idx_m]
        
        dist_w = torch.sum((z_pred_w.unsqueeze(1) - self.flat_z_w.unsqueeze(0)) ** 2, dim=2)
        min_dist_w, min_idx_w = dist_w.min(dim=1)
        rem_t_w = self.flat_rem_t_w[min_idx_w]
        
        gamma_m = self.beta / ((self.ref_one_step_dist_main ** 2) + 1e-8)
        gamma_w = self.beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
        
        S_main = torch.exp(-gamma_m * (min_dist_m ** 2))
        S_wrist = torch.exp(-gamma_w * (min_dist_w ** 2))
        
        rem_t_m_norm = self.ref_horizon * rem_t_m
        rem_t_w_norm = self.ref_horizon * rem_t_w
        
        Phi = self.w_m * S_main * (self.alpha ** rem_t_m_norm) + self.w_w * S_wrist * (self.alpha ** rem_t_w_norm)
        
        return Phi.cpu().numpy(), S_main.cpu().numpy(), S_wrist.cpu().numpy(), min_dist_m.cpu().numpy(), min_dist_w.cpu().numpy(), rem_t_m_norm.cpu().numpy(), rem_t_w_norm.cpu().numpy()
    def _compute_potential_sync(self, dino_tensor):
        """Computes the visual potential function Phi(s) using the Max-Similarity timestep synchronization."""
        dino_m, dino_w = dino_tensor[:, :384], dino_tensor[:, 384:]
        
        z_pred_m = self.e2c_main.enc(dino_m)[0].unsqueeze(1).detach().cpu().numpy()
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
        
        N = len(dino_tensor)
        min_dist_m = np.ones(N) * 10000
        min_dist_w = np.ones(N) * 10000
        idx_m_best = np.zeros(N)
        idx_w_best = np.zeros(N)
        T_demos_m = np.zeros(N)
        T_demos_w = np.zeros(N)
        
        for i in range(len(self.demo_starts)):
            z_demo_m = self.z_demo_main_cache[i]
            z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
            z_dist_min_m = z_dist_m.min(axis=1)
            update_min_m = z_dist_min_m < min_dist_m
            min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
            idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
            T_demos_m[update_min_m] = z_dist_m.shape[1]
            
            z_demo_w = self.z_demo_wrist_cache[i]
            z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
            z_dist_min_w = z_dist_w.min(axis=1)
            update_min_w = z_dist_min_w < min_dist_w
            min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
            idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
            T_demos_w[update_min_w] = z_dist_w.shape[1]
            
        gamma_m = self.beta / ((self.ref_one_step_dist_main ** 2) + 1e-8)
        gamma_w = self.beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
        
        # 4th power kernel: exp(-gamma * d^4)
        S_main = np.exp(-gamma_m * (min_dist_m ** 2))
        S_wrist = np.exp(-gamma_w * (min_dist_w ** 2))
        
        rem_t_m_norm = self.ref_horizon * (T_demos_m - idx_m_best) / np.maximum(T_demos_m, 1)
        rem_t_w_norm = self.ref_horizon * (T_demos_w - idx_w_best) / np.maximum(T_demos_w, 1)
        
        # Max-Similarity Time Sync
        rem_t_sync_norm = np.where(S_main >= S_wrist, rem_t_m_norm, rem_t_w_norm)
        
        # PBRS Potential with synced time discounting
        Phi = (self.w_m * S_main + self.w_w * S_wrist) * np.power(self.alpha, rem_t_sync_norm)
        
        return Phi, S_main, S_wrist, min_dist_m, min_dist_w, rem_t_m_norm, rem_t_w_norm, rem_t_sync_norm

    def _compute_potential_softsync(self, dino_tensor):
        """Computes the visual potential function Phi(s) using Softmax-Similarity timestep synchronization."""
        dino_m, dino_w = dino_tensor[:, :384], dino_tensor[:, 384:]
        
        z_pred_m = self.e2c_main.enc(dino_m)[0].unsqueeze(1).detach().cpu().numpy()
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
        
        N = len(dino_tensor)
        min_dist_m = np.ones(N) * 10000
        min_dist_w = np.ones(N) * 10000
        idx_m_best = np.zeros(N)
        idx_w_best = np.zeros(N)
        T_demos_m = np.zeros(N)
        T_demos_w = np.zeros(N)
        
        for i in range(len(self.demo_starts)):
            z_demo_m = self.z_demo_main_cache[i]
            z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
            z_dist_min_m = z_dist_m.min(axis=1)
            update_min_m = z_dist_min_m < min_dist_m
            min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
            idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
            T_demos_m[update_min_m] = z_dist_m.shape[1]
            
            z_demo_w = self.z_demo_wrist_cache[i]
            z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
            z_dist_min_w = z_dist_w.min(axis=1)
            update_min_w = z_dist_min_w < min_dist_w
            min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
            idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
            T_demos_w[update_min_w] = z_dist_w.shape[1]
            
        gamma_m = self.beta / ((self.ref_one_step_dist_main ** 2) + 1e-8)
        gamma_w = self.beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
        
        # 4th power kernel: exp(-gamma * d^4)
        S_main = np.exp(-gamma_m * (min_dist_m ** 2))
        S_wrist = np.exp(-gamma_w * (min_dist_w ** 2))
        
        rem_t_m_norm = self.ref_horizon * (T_demos_m - idx_m_best) / np.maximum(T_demos_m, 1)
        rem_t_w_norm = self.ref_horizon * (T_demos_w - idx_w_best) / np.maximum(T_demos_w, 1)
        
        # Soft-Similarity Time Sync
        sum_S = S_main + S_wrist + 1e-8
        rem_t_softsync_norm = (S_main * rem_t_m_norm + S_wrist * rem_t_w_norm) / sum_S
        
        # PBRS Potential with soft-synced time discounting
        Phi = (self.w_m * S_main + self.w_w * S_wrist) * np.power(self.alpha, rem_t_softsync_norm)
        
        return Phi, S_main, S_wrist, min_dist_m, min_dist_w, rem_t_m_norm, rem_t_w_norm, rem_t_softsync_norm


    def _compute_potential_max(self, dino_tensor):
        """Computes the visual potential function Phi(s) using the maximum of the two remaining timesteps."""
        dino_m, dino_w = dino_tensor[:, :384], dino_tensor[:, 384:]
        
        z_pred_m = self.e2c_main.enc(dino_m)[0].unsqueeze(1).detach().cpu().numpy()
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
        
        N = len(dino_tensor)
        min_dist_m = np.ones(N) * 10000
        min_dist_w = np.ones(N) * 10000
        idx_m_best = np.zeros(N)
        idx_w_best = np.zeros(N)
        T_demos_m = np.zeros(N)
        T_demos_w = np.zeros(N)
        
        for i in range(len(self.demo_starts)):
            z_demo_m = self.z_demo_main_cache[i]
            z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
            z_dist_min_m = z_dist_m.min(axis=1)
            update_min_m = z_dist_min_m < min_dist_m
            min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
            idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
            T_demos_m[update_min_m] = z_dist_m.shape[1]
            
            z_demo_w = self.z_demo_wrist_cache[i]
            z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
            z_dist_min_w = z_dist_w.min(axis=1)
            update_min_w = z_dist_min_w < min_dist_w
            min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
            idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
            T_demos_w[update_min_w] = z_dist_w.shape[1]
            
        gamma_m = self.beta / ((self.ref_one_step_dist_main ** 2) + 1e-8)
        gamma_w = self.beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
        
        # 4th power kernel: exp(-gamma * d^4)
        S_main = np.exp(-gamma_m * (min_dist_m ** 2))
        S_wrist = np.exp(-gamma_w * (min_dist_w ** 2))
        
        rem_t_m_norm = self.ref_horizon * (T_demos_m - idx_m_best) / np.maximum(T_demos_m, 1)
        rem_t_w_norm = self.ref_horizon * (T_demos_w - idx_w_best) / np.maximum(T_demos_w, 1)
        
        # Max Time Sync (Conservative approach)
        rem_t_max_norm = np.maximum(rem_t_m_norm, rem_t_w_norm)
        
        # PBRS Potential with max time discounting
        Phi = (self.w_m * S_main + self.w_w * S_wrist) * np.power(self.alpha, rem_t_max_norm)
        
        return Phi, S_main, S_wrist, min_dist_m, min_dist_w, rem_t_m_norm, rem_t_w_norm, rem_t_max_norm

    def _compute_potential_2squared(self, dino_tensor):
        """Computes visual potential Phi(s) using a 2nd-power (squared-distance) RBF kernel."""
        dino_m, dino_w = dino_tensor[:, :384], dino_tensor[:, 384:]
        
        z_pred_m = self.e2c_main.enc(dino_m)[0].detach() # [N, latent_dim]
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].detach()
        
        # cdist computes euclidean distance, we want squared euclidean distance
        dist_m = torch.cdist(z_pred_m, self.flat_z_m, p=2.0) ** 2 # [N, total_frames]
        min_dist_m, min_idx_m = dist_m.min(dim=1) # [N]
        rem_t_m = self.flat_rem_t_m[min_idx_m] # [N]
        
        dist_w = torch.cdist(z_pred_w, self.flat_z_w, p=2.0) ** 2
        min_dist_w, min_idx_w = dist_w.min(dim=1)
        rem_t_w = self.flat_rem_t_w[min_idx_w]
        
        gamma_m = self.beta / (self.ref_one_step_dist_main + 1e-8)
        gamma_w = self.beta / (self.ref_one_step_dist_wrist + 1e-8)
        
        S_main = torch.exp(-gamma_m * min_dist_m)
        S_wrist = torch.exp(-gamma_w * min_dist_w)
        
        rem_t_m_norm = self.ref_horizon * rem_t_m
        rem_t_w_norm = self.ref_horizon * rem_t_w
        
        Phi = self.w_m * S_main * (self.alpha ** rem_t_m_norm) + self.w_w * S_wrist * (self.alpha ** rem_t_w_norm)
        
        return Phi.cpu().numpy(), S_main.cpu().numpy(), S_wrist.cpu().numpy(), min_dist_m.cpu().numpy(), min_dist_w.cpu().numpy(), rem_t_m_norm.cpu().numpy(), rem_t_w_norm.cpu().numpy()
    def shape_reward(self, batch, step):
        if self.p_reward == 0 or self.reward_type.lower() == "none":
            return {}
            
        if not self.initialized:
            self.initialize_demos()
            
        if self.reward_type == "reward_1":
            dino_next_obs = batch["next", "dino"]
            not_done = ~batch["nonterminal"].squeeze()
            
            dino_m, dino_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
            
            z_pred_m = self.e2c_main.enc(dino_m)[0].unsqueeze(1).detach().cpu().numpy()
            z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
            
            N = len(dino_next_obs)
            min_dist_m = np.ones(N) * 10000
            min_dist_w = np.ones(N) * 10000
            idx_m_best = np.zeros(N)
            idx_w_best = np.zeros(N)
            T_demos_m = np.zeros(N)
            T_demos_w = np.zeros(N)
            
            for i in range(len(self.demo_starts)):
                z_demo_m = self.z_demo_main_cache[i]
                z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
                z_dist_min_m = z_dist_m.min(axis=1)
                update_min_m = z_dist_min_m < min_dist_m
                min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
                idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
                T_demos_m[update_min_m] = z_dist_m.shape[1]
                
                z_demo_w = self.z_demo_wrist_cache[i]
                z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
                z_dist_min_w = z_dist_w.min(axis=1)
                update_min_w = z_dist_min_w < min_dist_w
                min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
                idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
                T_demos_w[update_min_w] = z_dist_w.shape[1]

            not_done_np = not_done.detach().cpu().numpy().flatten()
            mask_m = (min_dist_m < self.ref_one_step_dist_main) & not_done_np
            mask_w = (min_dist_w < self.ref_one_step_dist_wrist) & not_done_np
            
            prog_m = idx_m_best / np.maximum(T_demos_m, 1)
            prog_w = idx_w_best / np.maximum(T_demos_w, 1)
            
            final_reward_mask = np.zeros_like(mask_m, dtype=bool)
            final_discount_power = np.zeros_like(mask_m, dtype=np.float32)
            
            idx_11 = mask_m & mask_w
            final_reward_mask[idx_11] = True
            min_prog = np.minimum(prog_m[idx_11], prog_w[idx_11])
            final_discount_power[idx_11] = np.maximum(T_demos_m[idx_11], T_demos_w[idx_11]) * (1 - min_prog)
            
            idx_01 = (~mask_m) & mask_w
            final_reward_mask[idx_01] = True
            final_discount_power[idx_01] = T_demos_w[idx_01] * (1 - prog_w[idx_01])
            
            demo_reward_discount = 0.98
            additional_reward = (
                np.power(demo_reward_discount, final_discount_power)
                * final_reward_mask
                * self.p_reward
            )
            
            add_rew = torch.as_tensor(additional_reward, device=self.device).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # Action L2 penalty: penalize residual action magnitude in ID (1,1) states
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                idx_11_torch = torch.as_tensor(idx_11, device=self.device, dtype=torch.float32)
                penalty = self.action_l2_reg_weight * idx_11_torch * action_l2
                penalty = penalty.view(batch["next", "reward"].shape)
                batch["next", "reward"] -= penalty
                action_l2_penalty_mean = penalty.mean().item()
            
            return {
                "lane/avg_discount": (final_discount_power * final_reward_mask).sum() / max(final_reward_mask.sum(), 1),
                "lane/num_additional_reward": final_reward_mask.sum(),
                "lane/num_11_reward": idx_11.sum(),
                "lane/num_01_reward": idx_01.sum(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
            }
            
        elif self.reward_type == "reward_2":
            dino_next_obs = batch["next", "dino"]
            dino_m, dino_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
            z_pred_m = self.e2c_main.enc(dino_m)[0].unsqueeze(1).detach().cpu().numpy()
            z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
            
            N = len(dino_next_obs)
            min_dist_m = np.ones(N) * 10000
            min_dist_w = np.ones(N) * 10000
            idx_m_best = np.zeros(N)
            idx_w_best = np.zeros(N)
            T_demos_m = np.zeros(N)
            T_demos_w = np.zeros(N)
            
            for i in range(len(self.demo_starts)):
                z_demo_m = self.z_demo_main_cache[i]
                z_dist_m = ((z_demo_m - z_pred_m) ** 2).sum(axis=2)
                z_dist_min_m = z_dist_m.min(axis=1)
                update_min_m = z_dist_min_m < min_dist_m
                min_dist_m[update_min_m] = z_dist_min_m[update_min_m]
                idx_m_best[update_min_m] = z_dist_m.argmin(axis=1)[update_min_m]
                T_demos_m[update_min_m] = z_dist_m.shape[1]
                
                z_demo_w = self.z_demo_wrist_cache[i]
                z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
                z_dist_min_w = z_dist_w.min(axis=1)
                update_min_w = z_dist_min_w < min_dist_w
                min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
                idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
                T_demos_w[update_min_w] = z_dist_w.shape[1]
            
            # -------------------------------------------------------------
            # Reward 2: Continuous RBF Kernel with 4th power distance
            # -------------------------------------------------------------
            # Note: min_dist_m is already the SQUARED distance (L2 norm squared)
            # To get 4th power distance, we simply square min_dist_m again.
            # epsilon is self.ref_one_step_dist (which is also a squared distance)
            
            
            # gamma = beta / (epsilon^2) so that exp(-gamma * (epsilon^2)) = exp(-beta)
            gamma_m = self.beta / ((self.ref_one_step_dist_main ** 2) + 1e-8)
            gamma_w = self.beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
            
            # Similarity scores S_main and S_wrist (using 4th power of distance)
            S_main = np.exp(-gamma_m * (min_dist_m ** 2))
            S_wrist = np.exp(-gamma_w * (min_dist_w ** 2))
            
            # Remaining timesteps: T_i^* - t^*
            rem_t_m = T_demos_m - idx_m_best
            rem_t_w = T_demos_w - idx_w_best
            
            # Dense reward computation
            r_dense = (self.w_m * np.power(self.alpha, rem_t_m) * S_main) + (self.w_w * np.power(self.alpha, rem_t_w) * S_wrist)
            r_dense = r_dense * self.p_reward
            
            # Add dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # Action regularization term
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                # S_main * S_wrist
                S_joint = torch.as_tensor(S_main * S_wrist, device=self.device, dtype=torch.float32)
                
                # r_reg = lambda * (S_main * S_wrist) * ||a_res||^2
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/S_main_avg": S_main.mean(),
                "lane/S_main_hist": wandb.Histogram(S_main),
                "lane/S_wrist_avg": S_wrist.mean(),
                "lane/S_wrist_hist": wandb.Histogram(S_wrist),
                "lane/r_dense_avg": r_dense.mean(),
                "lane/r_dense_hist": wandb.Histogram(r_dense),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/min_dist_main_avg": min_dist_m.mean(),
                "lane/min_dist_main_hist": wandb.Histogram(min_dist_m),
                "lane/min_dist_wrist_avg": min_dist_w.mean(),
                "lane/min_dist_wrist_hist": wandb.Histogram(min_dist_w),
                "lane/rem_t_main_avg": rem_t_m.mean(),
                "lane/rem_t_main_hist": wandb.Histogram(rem_t_m),
                "lane/rem_t_wrist_avg": rem_t_w.mean(),
                "lane/rem_t_wrist_hist": wandb.Histogram(rem_t_w),
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }
            
        elif self.reward_type == "reward_pbrs_sync":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS) with Max-Similarity Time Sync
            # -------------------------------------------------------------
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next, rem_t_sync_next = self._compute_potential_sync(batch["next", "dino"])
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr, rem_t_sync_curr = self._compute_potential_sync(batch["dino"])
            
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/rem_t_sync_next_avg": rem_t_sync_next.mean(),
                "lane/rem_t_sync_next_hist": wandb.Histogram(rem_t_sync_next),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_softsync":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS) with Soft-Similarity Time Sync
            # -------------------------------------------------------------
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next, rem_t_sync_next = self._compute_potential_softsync(batch["next", "dino"])
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr, rem_t_sync_curr = self._compute_potential_softsync(batch["dino"])
            
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/rem_t_softsync_next_avg": rem_t_sync_next.mean(),
                "lane/rem_t_softsync_next_hist": wandb.Histogram(rem_t_sync_next),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_max":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS) with Max Time Sync
            # -------------------------------------------------------------
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next, rem_t_max_next = self._compute_potential_max(batch["next", "dino"])
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr, rem_t_max_curr = self._compute_potential_max(batch["dino"])
            
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/rem_t_max_next_avg": rem_t_max_next.mean(),
                "lane/rem_t_max_next_hist": wandb.Histogram(rem_t_max_next),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS)
            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using self.gamma)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            # batch["nonterminal"] is True when episode is ongoing, False when done.
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_no_mask":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping WITHOUT Terminal Masking
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using self.gamma)
            # NO terminal masking: Phi(s_{terminal}) is evaluated normally
            gamma_env = self.gamma
            
            r_dense = (gamma_env * Phi_next - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_nstep":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS) for N-Step Returns
            # F(s, a, s_n) = gamma^n * Phi(s_n) - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state in n-step jump)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using batch["gamma"] which is gamma^n)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            if "gamma" in batch.keys():
                gamma_env = batch["gamma"].squeeze().detach().cpu().numpy()
            else:
                gamma_env = self.gamma # fallback just in case
            
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_no_mask_nstep":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping WITHOUT Terminal Masking for N-Step Returns
            # F(s, a, s_n) = gamma^n * Phi(s_n) - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state in n-step jump)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using batch["gamma"] which is gamma^n)
            # NO terminal masking: Phi(s_{terminal}) is evaluated normally
            if "gamma" in batch.keys():
                gamma_env = batch["gamma"].squeeze().detach().cpu().numpy()
            else:
                gamma_env = self.gamma # fallback just in case
            
            r_dense = (gamma_env * Phi_next - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }


        elif self.reward_type == "reward_pbrs_no_step_penalty":
            # 0. Cancel base -1.0 step penalty
            batch["next", "reward"] = torch.where(
                batch["next", "reward"] < 0.0,
                batch["next", "reward"] + 1.0,
                batch["next", "reward"]
            )
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS)
            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using self.gamma)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            # batch["nonterminal"] is True when episode is ongoing, False when done.
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_no_step_penalty_success1000":
            # 0. Cancel base -1.0 step penalty
            batch["next", "reward"] = torch.where(
                batch["next", "reward"] < 0.0,
                batch["next", "reward"] + 1.0,
                batch["next", "reward"]
            )
            # 0.5. Inflate success reward from 100.0 to 1000.0 to overcome PBRS terminal drop
            batch["next", "reward"] = torch.where(
                batch["next", "reward"] == 100.0,
                torch.tensor(1000.0, device=self.device, dtype=torch.float32),
                batch["next", "reward"]
            )
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS)
            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using self.gamma)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            # batch["nonterminal"] is True when episode is ongoing, False when done.
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_no_step_penalty_success1":
            # 0. Cancel base -1.0 step penalty
            batch["next", "reward"] = torch.where(
                batch["next", "reward"] < 0.0,
                batch["next", "reward"] + 1.0,
                batch["next", "reward"]
            )
            # 0.5. Downscale success reward from 100.0 to 1.0
            batch["next", "reward"] = torch.where(
                batch["next", "reward"] == 100.0,
                torch.tensor(1.0, device=self.device, dtype=torch.float32),
                batch["next", "reward"]
            )
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS)
            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state)
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state)
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential(batch["dino"])
            
            # 3. PBRS Difference (using self.gamma)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            # batch["nonterminal"] is True when episode is ongoing, False when done.
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_2squared":
            # -------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS) with 2nd-power (squared) distance kernel
            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            # Similarity: exp(-gamma * d^2)  vs reward_pbrs which uses exp(-gamma * d^4)
            # A 2nd-power kernel gives a wider, smoother potential landscape.
            # -------------------------------------------------------------
            # 1. Compute Potential for s' (next state) using 2squared kernel
            Phi_next, S_main_next, S_wrist_next, min_dist_m_next, min_dist_w_next, rem_t_m_next, rem_t_w_next = self._compute_potential_2squared(batch["next", "dino"])
            
            # 2. Compute Potential for s (current state) using 2squared kernel
            Phi_curr, S_main_curr, S_wrist_curr, min_dist_m_curr, min_dist_w_curr, rem_t_m_curr, rem_t_w_curr = self._compute_potential_2squared(batch["dino"])
            
            # 3. PBRS Difference (using batch["gamma"] which is gamma^n)
            # Apply terminal masking: Phi(s_{terminal}) = 0
            if "gamma" in batch.keys():
                gamma_env = batch["gamma"].squeeze().detach().cpu().numpy()
            else:
                gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            # Add PBRS dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # 4. Action regularization term (using S_next as reference for ID boundary)
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_main_curr * S_wrist_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_main_next_avg": S_main_next.mean(),
                "lane/S_main_next_hist": wandb.Histogram(S_main_next),
                "lane/S_wrist_next_avg": S_wrist_next.mean(),
                "lane/S_wrist_next_hist": wandb.Histogram(S_wrist_next),
                "lane/min_dist_main_next_avg": min_dist_m_next.mean(),
                "lane/min_dist_wrist_next_avg": min_dist_w_next.mean(),
                "lane/rem_t_main_next_avg": rem_t_m_next.mean(),
                "lane/rem_t_wrist_next_avg": rem_t_w_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_main": self.ref_one_step_dist_main,
                "lane/ref_one_step_dist_wrist": self.ref_one_step_dist_wrist
            }

        elif self.reward_type == "reward_pbrs_unified":
            Phi_next, S_unified_next, min_dist_u_next, rem_t_u_next = self._compute_potential_unified(batch["next", "dino"])
            Phi_curr, S_unified_curr, min_dist_u_curr, rem_t_u_curr = self._compute_potential_unified(batch["dino"])
            
            gamma_env = self.gamma
            nonterminal_mask = batch["nonterminal"].squeeze().detach().cpu().numpy()
            
            r_dense = (gamma_env * Phi_next * nonterminal_mask - Phi_curr) * self.p_reward
            
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                a_total = batch["action"]
                a_base = batch["obs", "observation.base_action"]
                a_res = a_total - a_base
                action_l2 = (a_res ** 2).sum(dim=-1)
                
                S_joint = torch.as_tensor(S_unified_curr, device=self.device, dtype=torch.float32)
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/Phi_next_avg": Phi_next.mean(),
                "lane/Phi_curr_avg": Phi_curr.mean(),
                "lane/Phi_next_hist": wandb.Histogram(Phi_next),
                "lane/PBRS_dense_avg": r_dense.mean(),
                "lane/PBRS_dense_min": r_dense.min(),
                "lane/PBRS_dense_max": r_dense.max(),
                "lane/PBRS_dense_hist": wandb.Histogram(r_dense),
                "lane/S_unified_next_avg": S_unified_next.mean(),
                "lane/S_unified_next_hist": wandb.Histogram(S_unified_next),
                "lane/min_dist_unified_next_avg": min_dist_u_next.mean(),
                "lane/rem_t_unified_next_avg": rem_t_u_next.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
                "lane/ref_one_step_dist_unified": self.ref_one_step_dist_unified
            }
