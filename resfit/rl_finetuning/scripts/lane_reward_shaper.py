import torch
import numpy as np
import sys
from pathlib import Path
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Add lane to sys.path to import MLPE2C
lane_dir = Path(__file__).resolve().parents[3] / "lane"
sys.path.append(str(lane_dir))
from e2c import MLPE2C

class LaNERewardShaper:
    def __init__(self, device, action_dim, offline_rb, p_reward=1.0, action_l2_reg_weight=0.0, reward_type="reward_2"):
        self.device = device
        self.p_reward = p_reward
        self.action_l2_reg_weight = action_l2_reg_weight
        self.reward_type = reward_type
        self.offline_rb = offline_rb
        
        print("Loading DINOv2 model...")
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
        self.dino.eval()
        print("DINOv2 loaded.")
        
        # Two cameras: front and wrist, 384 dim each for DINOv2 ViT-S
        self.e2c_front = MLPE2C(
            obs_shape=(384,), action_dim=action_dim, z_dimension=16, crop_shape=None
        ).to(device)
        self.e2c_wrist = MLPE2C(
            obs_shape=(384,), action_dim=action_dim, z_dimension=16, crop_shape=None
        ).to(device)
        
        self.e2c_front_opt = torch.optim.Adam(self.e2c_front.parameters(), lr=1e-4)
        self.e2c_wrist_opt = torch.optim.Adam(self.e2c_wrist.parameters(), lr=1e-4)
        
        self.z_demo_front_cache = {}
        self.z_demo_wrist_cache = {}
        self.ref_one_step_dist_front = None
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
            image1 = F.interpolate(image1, size=(224, 224), mode="bilinear", align_corners=False)
            image2 = F.interpolate(image2, size=(224, 224), mode="bilinear", align_corners=False)
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
            img_front = batch["obs", "observation.images.frontview"]
            img_wrist = batch["obs", "observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([img_front, img_wrist], dim=1).float() / 255.0
            
            next_img_front = batch["next", "obs", "observation.images.frontview"]
            next_img_wrist = batch["next", "obs", "observation.images.robot0_eye_in_hand"]
            next_obs_img = torch.cat([next_img_front, next_img_wrist], dim=1).float() / 255.0
            
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
            img_front = batch["obs", "observation.images.frontview"]
            img_wrist = batch["obs", "observation.images.robot0_eye_in_hand"]
            obs_img = torch.cat([img_front, img_wrist], dim=1).float() / 255.0
            
            next_img_front = batch["next", "obs", "observation.images.frontview"]
            next_img_wrist = batch["next", "obs", "observation.images.robot0_eye_in_hand"]
            next_obs_img = torch.cat([next_img_front, next_img_wrist], dim=1).float() / 255.0
            
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
        img_front = td["obs", "observation.images.frontview"]
        img_wrist = td["obs", "observation.images.robot0_eye_in_hand"]
        is_unbatched = img_front.ndim == 3
        if is_unbatched:
            img_front = img_front.unsqueeze(0)
            img_wrist = img_wrist.unsqueeze(0)
            
        obs_img = torch.cat([img_front, img_wrist], dim=1).float() / 255.0
        
        next_img_front = td["next", "obs", "observation.images.frontview"]
        next_img_wrist = td["next", "obs", "observation.images.robot0_eye_in_hand"]
        if is_unbatched:
            next_img_front = next_img_front.unsqueeze(0)
            next_img_wrist = next_img_wrist.unsqueeze(0)
            
        next_obs_img = torch.cat([next_img_front, next_img_wrist], dim=1).float() / 255.0
        
        dino_obs = self.dino_embed(obs_img.to(self.device)).to(img_front.device)
        dino_next = self.dino_embed(next_obs_img.to(self.device)).to(img_front.device)
        
        if is_unbatched:
            td["dino"] = dino_obs.squeeze(0)
            td["next", "dino"] = dino_next.squeeze(0)
        else:
            td["dino"] = dino_obs
            td["next", "dino"] = dino_next
        return td

    def _sample_e2c(self, batch_size=256):
        idx = np.random.randint(0, len(self.offline_rb), size=batch_size)
        batch = self.offline_rb[idx].to(self.device)
        dino_obs = batch["dino"]
        dino_next_obs = batch["next", "dino"]
        action = batch["action"]
        return dino_obs, action, dino_next_obs

    def update_e2c(self, num_updates, mse_tol=0.2):
        for i in range(num_updates):
            dino_obs, action, dino_next_obs = self._sample_e2c()
            
            dino_obs_f, dino_obs_w = dino_obs[:, :384], dino_obs[:, 384:]
            dino_next_obs_f, dino_next_obs_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
            
            mse_w_mult = 384
            
            dkl_f, mse_f, ref_kl_f, _ = self.e2c_front(dino_obs_f, action, dino_next_obs_f, None, None)
            dkl_w, mse_w, ref_kl_w, _ = self.e2c_wrist(dino_obs_w, action, dino_next_obs_w, None, None)
            
            loss_f = dkl_f + mse_f * 384 + ref_kl_f
            loss_w = dkl_w + mse_w * mse_w_mult + ref_kl_w
            loss = loss_f + loss_w
            
            self.e2c_front_opt.zero_grad()
            self.e2c_wrist_opt.zero_grad()
            loss.backward()
            self.e2c_front_opt.step()
            self.e2c_wrist_opt.step()
            
            if mse_tol is not None and ((mse_f + mse_w)/2).item() < mse_tol:
                break
                
    def initialize_demos(self):
        one_step_dist_list_front = []
        one_step_dist_list_wrist = []
        
        for i, (start, end) in enumerate(zip(self.demo_starts, self.demo_ends)):
            batch = self.offline_rb[start:end+1].to(self.device)
            dino_next_obs = batch["next", "dino"]
            dino_f, dino_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
            
            z_f = self.e2c_front.enc(dino_f)[0].unsqueeze(0).detach().cpu().numpy()
            z_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(0).detach().cpu().numpy()
            
            self.z_demo_front_cache[i] = z_f
            self.z_demo_wrist_cache[i] = z_w
            
            if z_f.shape[1] > 1:
                one_step_dist_list_front.append(((z_f[0, 1:] - z_f[0, :-1]) ** 2).sum(axis=1).mean())
                one_step_dist_list_wrist.append(((z_w[0, 1:] - z_w[0, :-1]) ** 2).sum(axis=1).mean())
                
        self.ref_one_step_dist_front = np.mean(one_step_dist_list_front)
        self.ref_one_step_dist_wrist = np.mean(one_step_dist_list_wrist)
        self.initialized = True

    def shape_reward(self, batch, step):
        if self.p_reward == 0:
            return batch
            
        if not self.initialized:
            self.update_e2c(1000, mse_tol=0.2)
            self.initialize_demos()
        elif step % 300 == 0:
            self.update_e2c(1000, mse_tol=0.2)
            self.initialize_demos()
            
        # compute for the current batch
        dino_next_obs = batch["next", "dino"]
        not_done = ~batch["nonterminal"].squeeze()
        
        dino_f, dino_w = dino_next_obs[:, :384], dino_next_obs[:, 384:]
        
        z_pred_f = self.e2c_front.enc(dino_f)[0].unsqueeze(1).detach().cpu().numpy()
        z_pred_w = self.e2c_wrist.enc(dino_w)[0].unsqueeze(1).detach().cpu().numpy()
        
        N = len(dino_next_obs)
        min_dist_f = np.ones(N) * 10000
        min_dist_w = np.ones(N) * 10000
        idx_f_best = np.zeros(N)
        idx_w_best = np.zeros(N)
        T_demos = np.zeros(N)
        
        for i in range(len(self.demo_starts)):
            z_demo_f = self.z_demo_front_cache[i]
            z_dist_f = ((z_demo_f - z_pred_f) ** 2).sum(axis=2)
            z_dist_min_f = z_dist_f.min(axis=1)
            update_min_f = z_dist_min_f < min_dist_f
            min_dist_f[update_min_f] = z_dist_min_f[update_min_f]
            idx_f_best[update_min_f] = z_dist_f.argmin(axis=1)[update_min_f]
            
            z_demo_w = self.z_demo_wrist_cache[i]
            z_dist_w = ((z_demo_w - z_pred_w) ** 2).sum(axis=2)
            z_dist_min_w = z_dist_w.min(axis=1)
            update_min_w = z_dist_min_w < min_dist_w
            min_dist_w[update_min_w] = z_dist_min_w[update_min_w]
            idx_w_best[update_min_w] = z_dist_w.argmin(axis=1)[update_min_w]
            
            updated_any = update_min_f | update_min_w
            T_demos[updated_any] = z_dist_f.shape[1]
            
        if self.reward_type == "reward_1":
            not_done_np = not_done.detach().cpu().numpy().flatten()
            mask_f = (min_dist_f < self.ref_one_step_dist_front) & not_done_np
            mask_w = (min_dist_w < self.ref_one_step_dist_wrist) & not_done_np
            
            prog_f = idx_f_best / np.maximum(T_demos, 1)
            prog_w = idx_w_best / np.maximum(T_demos, 1)
            
            final_reward_mask = np.zeros_like(mask_f, dtype=bool)
            final_discount_power = np.zeros_like(mask_f, dtype=np.float32)
            
            idx_11 = mask_f & mask_w
            final_reward_mask[idx_11] = True
            min_prog = np.minimum(prog_f[idx_11], prog_w[idx_11])
            final_discount_power[idx_11] = T_demos[idx_11] * (1 - min_prog)
            
            idx_01 = (~mask_f) & mask_w
            final_reward_mask[idx_01] = True
            final_discount_power[idx_01] = T_demos[idx_01] * (1 - prog_w[idx_01])
            
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
                action = batch["action"]
                action_l2 = (action ** 2).sum(dim=-1)
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
            # -------------------------------------------------------------
            # Reward 2: Continuous RBF Kernel with 4th power distance
            # -------------------------------------------------------------
            # Note: min_dist_f is already the SQUARED distance (L2 norm squared)
            # To get 4th power distance, we simply square min_dist_f again.
            # epsilon is self.ref_one_step_dist (which is also a squared distance)
            
            beta = 1.0
            
            # gamma = beta / (epsilon^2) so that exp(-gamma * (epsilon^2)) = exp(-beta)
            gamma_m = beta / ((self.ref_one_step_dist_front ** 2) + 1e-8)
            gamma_w = beta / ((self.ref_one_step_dist_wrist ** 2) + 1e-8)
            
            # Similarity scores S_main and S_wrist (using 4th power of distance)
            S_main = np.exp(-gamma_m * (min_dist_f ** 2))
            S_wrist = np.exp(-gamma_w * (min_dist_w ** 2))
            
            w_m = 0.3
            w_w = 0.7
            alpha = 0.98
            
            # Remaining timesteps: T_i^* - t^*
            rem_t_f = T_demos - idx_f_best
            rem_t_w = T_demos - idx_w_best
            
            # Dense reward computation
            r_dense = (w_m * np.power(alpha, rem_t_f) * S_main) + (w_w * np.power(alpha, rem_t_w) * S_wrist)
            r_dense = r_dense * self.p_reward
            
            # Add dense reward to batch
            add_rew = torch.as_tensor(r_dense, device=self.device, dtype=torch.float32).view(batch["next", "reward"].shape)
            batch["next", "reward"] += add_rew
            
            # Action regularization term
            action_l2_penalty_mean = 0.0
            if self.action_l2_reg_weight > 0:
                action = batch["action"]
                action_l2 = (action ** 2).sum(dim=-1)
                
                # S_main * S_wrist
                S_joint = torch.as_tensor(S_main * S_wrist, device=self.device, dtype=torch.float32)
                
                # r_reg = lambda * (S_main * S_wrist) * ||a_res||^2
                r_reg = self.action_l2_reg_weight * S_joint * action_l2
                r_reg = r_reg.view(batch["next", "reward"].shape)
                
                batch["next", "reward"] -= r_reg
                action_l2_penalty_mean = r_reg.mean().item()
                
            return {
                "lane/S_main_avg": S_main.mean(),
                "lane/S_wrist_avg": S_wrist.mean(),
                "lane/r_dense_avg": r_dense.mean(),
                "lane/action_l2_penalty": action_l2_penalty_mean,
            }
