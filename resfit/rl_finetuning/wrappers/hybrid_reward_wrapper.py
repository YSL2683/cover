import torch
import torch.nn as nn
import numpy as np

class LatentDistanceModule(nn.Module):
    def __init__(self, e2c_front, e2c_wrist, z_demo_front, z_demo_wrist, demo_lengths, p_reward=1.0, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load DINOv2
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(self.device)
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
            
        # Frozen E2C models
        self.e2c_front = e2c_front.to(self.device)
        self.e2c_wrist = e2c_wrist.to(self.device)
        self.e2c_front.eval()
        self.e2c_wrist.eval()
        
        for param in self.e2c_front.parameters():
            param.requires_grad = False
        for param in self.e2c_wrist.parameters():
            param.requires_grad = False
            
        # Demo latents: list of tensors, each [1, T_i, z_dim]
        self.z_demo_front = [z.to(self.device) for z in z_demo_front]
        self.z_demo_wrist = [z.to(self.device) for z in z_demo_wrist]
        
        # Demo lengths T_i*
        self.demo_lengths = torch.tensor(demo_lengths, dtype=torch.float32, device=self.device)
        
        self.p_reward = p_reward
        
        # Compute reference one-step distances (ID/OOD threshold)
        self.ref_dist_front = self._compute_ref_dist(self.z_demo_front)
        self.ref_dist_wrist = self._compute_ref_dist(self.z_demo_wrist)

    def _compute_ref_dist(self, z_demos):
        # z_demos: list of [1, T, z_dim]
        dists = []
        for z in z_demos:
            if z.shape[1] > 1:
                diff = z[:, 1:] - z[:, :-1]
                dist = (diff ** 2).sum(dim=-1).mean()
                dists.append(dist)
        if len(dists) == 0:
            return torch.tensor(1.0, device=self.device)
        return torch.stack(dists).mean()
        
    def extract_dino_features(self, images):
        # images: [B, C, H, W]
        with torch.no_grad():
            import torch.nn.functional as F
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.dino(images)
        return features

    @torch.no_grad()
    def compute_dense_reward(self, obs_front, obs_wrist):
        """
        obs_front: [B, 3, H, W]
        obs_wrist: [B, 3, H, W]
        Returns:
            dense_reward: [B]
            is_state_11: [B] boolean mask for (1,1) state
        """
        B = obs_front.shape[0]
        
        # Extract DINO features
        feat_front = self.extract_dino_features(obs_front) # [B, 384]
        feat_wrist = self.extract_dino_features(obs_wrist) # [B, 384]
        
        # Encode to latent space
        z_pred_front, _ = self.e2c_front.enc(feat_front) # [B, z_dim_main]
        z_pred_wrist, _ = self.e2c_wrist.enc(feat_wrist) # [B, z_dim_wrist]
        
        z_pred_front = z_pred_front.unsqueeze(1) # [B, 1, z_dim_main]
        z_pred_wrist = z_pred_wrist.unsqueeze(1) # [B, 1, z_dim_wrist]
        
        # Find nearest demo step
        # Since demos have different lengths, iterate over demos
        min_dist_front = torch.full((B,), float('inf'), device=self.device)
        min_dist_wrist = torch.full((B,), float('inf'), device=self.device)
        
        best_prog_front = torch.zeros(B, device=self.device)
        best_prog_wrist = torch.zeros(B, device=self.device)
        best_T_front = torch.zeros(B, device=self.device)
        best_T_wrist = torch.zeros(B, device=self.device)
        
        for i, (z_df, z_dw, T_i) in enumerate(zip(self.z_demo_front, self.z_demo_wrist, self.demo_lengths)):
            # z_df: [1, T_i, z_dim_main]
            dist_f = ((z_df - z_pred_front) ** 2).sum(dim=-1) # [B, T_i]
            dist_w = ((z_dw - z_pred_wrist) ** 2).sum(dim=-1) # [B, T_i]
            
            min_df, idx_f = dist_f.min(dim=1) # [B]
            min_dw, idx_w = dist_w.min(dim=1) # [B]
            
            # Update front best
            update_f = min_df < min_dist_front
            min_dist_front[update_f] = min_df[update_f]
            best_prog_front[update_f] = idx_f[update_f].float() / T_i
            best_T_front[update_f] = T_i
            
            # Update wrist best
            update_w = min_dw < min_dist_wrist
            min_dist_wrist[update_w] = min_dw[update_w]
            best_prog_wrist[update_w] = idx_w[update_w].float() / T_i
            best_T_wrist[update_w] = T_i

        # ID/OOD masking
        mask_f = min_dist_front < self.ref_dist_front
        mask_w = min_dist_wrist < self.ref_dist_wrist
        
        # Quadrant classification
        idx_11 = mask_f & mask_w
        idx_10 = mask_f & (~mask_w)
        idx_01 = (~mask_f) & mask_w
        # idx_00 = (~mask_f) & (~mask_w)
        
        dense_reward = torch.zeros(B, device=self.device)
        
        # (1,1) Case
        # Reward is given, and residual action is restricted
        if idx_11.any():
            # Normalized progress: min of main and wrist
            min_prog = torch.min(best_prog_front[idx_11], best_prog_wrist[idx_11])
            # Restore to temporal distance using front's T as reference (or mean, but user said "convert back using T_i*")
            # We will use best_T_front since they are in (1,1). Both have a T_i.
            # Using T_front for restoration
            restored_dist = best_T_front[idx_11] * (1.0 - min_prog)
            discount = (0.98 ** restored_dist)
            dense_reward[idx_11] = discount * self.p_reward

        # (1,0) Case: Main ID, Wrist OOD
        if idx_10.any():
            w_main = 0.5
            restored_dist = best_T_front[idx_10] * (1.0 - best_prog_front[idx_10])
            discount = (0.98 ** restored_dist)
            dense_reward[idx_10] = w_main * discount * self.p_reward
            
        # (0,1) Case: Main OOD, Wrist ID
        if idx_01.any():
            w_wrist = 1.0
            restored_dist = best_T_wrist[idx_01] * (1.0 - best_prog_wrist[idx_01])
            discount = (0.98 ** restored_dist)
            dense_reward[idx_01] = w_wrist * discount * self.p_reward
            
        return dense_reward, idx_11


class HybridRewardVecEnvWrapper:
    """
    Wraps a Vectorized Environment to inject dense reward and state flags.
    Should be applied AFTER BasePolicyVecEnvWrapper.
    """
    def __init__(self, env, latent_reward_module: LatentDistanceModule, main_cam_key, wrist_cam_key):
        self.env = env
        self.latent_reward_module = latent_reward_module
        self.main_cam_key = main_cam_key
        self.wrist_cam_key = wrist_cam_key
        
    def __getattr__(self, name):
        return getattr(self.env, name)
        
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Assume next_obs has image tensors [B, C, H, W] in [0, 255]
        # DINO expects images normalized with ImageNet stats, shape [B, 3, 224, 224] typically, 
        # but LaNE might use its own transform. Here we assume pre-transformed or we add transform.
        # This is a simplified transformation for demonstration.
        
        front_img = next_obs[self.main_cam_key].float() / 255.0
        wrist_img = next_obs[self.wrist_cam_key].float() / 255.0
        
        r_dense, is_state_11 = self.latent_reward_module.compute_dense_reward(front_img, wrist_img)
        
        # Map base reward: 1.0 (success) -> 100.0, 0.0 (step) -> -1.0
        base_reward = torch.where(reward > 0.5, torch.tensor(100.0, device=reward.device), torch.tensor(-1.0, device=reward.device))
        
        # Add dense reward to environment reward
        # reward is typically [B, 1] or [B]
        r_dense = r_dense.view_as(reward)
        reward = base_reward + r_dense
        
        # Inject state flag into observation for L2 penalty later
        next_obs["observation.is_state_11"] = is_state_11.unsqueeze(-1)
        
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        front_img = obs[self.main_cam_key].float() / 255.0
        wrist_img = obs[self.wrist_cam_key].float() / 255.0
        
        _, is_state_11 = self.latent_reward_module.compute_dense_reward(front_img, wrist_img)
        obs["observation.is_state_11"] = is_state_11.unsqueeze(-1)
        
        return obs, info
