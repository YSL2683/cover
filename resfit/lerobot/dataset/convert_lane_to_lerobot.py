import os
import sys
import torch
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    demo_dir = "/home/ysl2683/LaNE/demo/robosuite_pick_place_can/20/"
    pt_files = [f for f in os.listdir(demo_dir) if f.endswith(".pt")]
    pt_path = os.path.join(demo_dir, pt_files[0])
    payload = torch.load(pt_path, weights_only=False)
    
    obs_list = payload[0]      # [N, 6, 128, 128]
    next_obs_list = payload[1] # [N, 6, 128, 128]
    action_list = payload[2]   # [N, action_dim]
    reward_list = payload[3]
    not_done_list = payload[4]
    state_list = payload[5]    # [N, 9]
    
    demo_starts = np.load(os.path.join(demo_dir, "demo_starts.npy"))
    demo_ends = np.load(os.path.join(demo_dir, "demo_ends.npy"))
    
    features = {
        "observation.images.frontview": {"dtype": "video", "shape": (3, 128, 128), "names": ["channel", "height", "width"]},
        "observation.images.robot0_eye_in_hand": {"dtype": "video", "shape": (3, 128, 128), "names": ["channel", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (9,), "names": ["state_" + str(i) for i in range(9)]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action_" + str(i) for i in range(7)]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    
    dataset = LeRobotDataset.create(
        repo_id="ysl2683/lane_can",
        fps=10,
        root="/home/ysl2683/residual-offpolicy-rl/resfit/my_lerobot_data",
        features=features
    )
    
    for ep_idx, (start, end) in enumerate(zip(demo_starts, demo_ends)):
        for i in range(start, end):
            # obs_list is uint8
            front_img = obs_list[i][:3].transpose(1, 2, 0)
            wrist_img = obs_list[i][3:6].transpose(1, 2, 0)
            
            frame = {
                "observation.images.frontview": Image.fromarray(front_img),
                "observation.images.robot0_eye_in_hand": Image.fromarray(wrist_img),
                "observation.state": torch.tensor(state_list[i], dtype=torch.float32),
                "action": torch.tensor(action_list[i], dtype=torch.float32),
                "next.reward": torch.tensor([reward_list[i][0]], dtype=torch.float32),
                "next.done": torch.tensor([not not_done_list[i][0]], dtype=torch.bool)
            }
            dataset.add_frame(frame, task="PickPlaceCan")
        dataset.save_episode()
        
    dataset.consolidate()
    print("Dataset converted to LeRobot format!")

if __name__ == "__main__":
    main()
