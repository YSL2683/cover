import torch
import numpy as np
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

def main():
    pt_path = "/home/ysl2683/cover/lane/demo/robosuite_lift/20/0_400.pt" # 0_400 might be different but glob catches it
    starts_path = "/home/ysl2683/cover/lane/demo/robosuite_lift/20/demo_starts.npy"
    ends_path = "/home/ysl2683/cover/lane/demo/robosuite_lift/20/demo_ends.npy"
    
    # Try alternate path if it doesn't exist
    if not os.path.exists(pt_path):
        import glob
        files = glob.glob("/home/ysl2683/cover/lane/demo/robosuite_lift/20/*.pt")
        if files:
            pt_path = files[0]
            
    payload = torch.load(pt_path, weights_only=False)
    obs_list = payload[0]      # [N, 6, 128, 128]
    action_list = payload[2]   # [N, 7]
    state_list = payload[5]    # [N, state_dim]
    
    starts = np.load(starts_path)
    ends = np.load(ends_path)
    
    state_dim = state_list.shape[1] if hasattr(state_list, 'shape') else len(state_list[0])

    features = {
        "observation.images.frontview": {"dtype": "image", "shape": (3, 128, 128), "names": ["c", "h", "w"]},
        "observation.images.robot0_eye_in_hand": {"dtype": "image", "shape": (3, 128, 128), "names": ["c", "h", "w"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    }
    
    dataset = LeRobotDataset.create(
        repo_id="ysl2683/lane_lift_id_20_aligned",
        fps=10,
        features=features,
        root="/home/ysl2683/cover/resfit/my_lerobot_data/ysl2683/lane_lift_id_20_aligned"
    )
    
    # Process episodes
    for ep_idx in range(len(starts)):
        ep_start = starts[ep_idx]
        ep_end = ends[ep_idx]
        
        for i in range(ep_start, ep_end):
            img = obs_list[i]
            front_img = img[:3]
            wrist_img = img[3:]
            
            act = action_list[i]
            state = state_list[i]
            
            front_img_np = np.clip(front_img * 255.0, 0, 255).astype(np.uint8) if front_img.dtype != torch.uint8 else front_img.numpy()
            wrist_img_np = np.clip(wrist_img * 255.0, 0, 255).astype(np.uint8) if wrist_img.dtype != torch.uint8 else wrist_img.numpy()
            
            frame_dict = {
                "observation.images.frontview": front_img_np,
                "observation.images.robot0_eye_in_hand": wrist_img_np,
                "observation.state": state.numpy() if hasattr(state, 'numpy') else np.array(state, dtype=np.float32),
                "action": act.numpy() if hasattr(act, 'numpy') else np.array(act, dtype=np.float32)
            }
            dataset.add_frame(frame_dict, task="Lift")
            
        dataset.save_episode()
    print("Conversion complete!")

if __name__ == "__main__":
    main()
