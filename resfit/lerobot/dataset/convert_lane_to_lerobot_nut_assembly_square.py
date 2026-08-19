import argparse
import os
import shutil
from pathlib import Path
import glob

import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from tqdm import tqdm


def convert_lane_to_lerobot(
    demo_dir: str,
    output_dir: str,
    repo_id: str | None = None,
    train_ratio: float = 1.0,
):
    """
    Convert LaNE demo (.pt + .npy) dataset to LeRobot format.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    demo_dir = Path(demo_dir)
    starts_path = demo_dir / "demo_starts.npy"
    ends_path = demo_dir / "demo_ends.npy"
    
    pt_files = list(demo_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {demo_dir}")
    pt_path = pt_files[0]
    
    print(f"Loading payload from {pt_path}")
    payload = torch.load(pt_path, weights_only=False)
    obs_list = payload[0]      # [N, 6, 128, 128]
    action_list = payload[2]   # [N, 7]
    state_list = payload[5]    # [N, state_dim]
    
    starts = np.load(starts_path)
    ends = np.load(ends_path)
    
    state_dim = state_list.shape[1] if hasattr(state_list, 'shape') else len(state_list[0])

    action_names = [
        "eef_delta_pos_x", "eef_delta_pos_y", "eef_delta_pos_z", 
        "eef_delta_rot_rx", "eef_delta_rot_ry", "eef_delta_rot_rz", 
        "gripper_action"
    ]
    
    state_names = [
        "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
        "robot0_eef_quat_w", "robot0_eef_quat_x", "robot0_eef_quat_y", "robot0_eef_quat_z",
        "robot0_gripper_qpos_0", "robot0_gripper_qpos_1"
    ]

    features = {
        "observation.images.agentview": {"dtype": "video", "shape": (3, 128, 128), "names": ["c", "h", "w"]},
        "observation.images.robot0_eye_in_hand": {"dtype": "video", "shape": (3, 128, 128), "names": ["c", "h", "w"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,), "names": action_names},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    
    dataset_repo_id = repo_id if repo_id else output_dir.name
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=10,
        features=features,
        root=str(output_dir),
        use_videos=True,
    )
    
    print(f"Processing {len(starts)} episodes...")
    for ep_idx in tqdm(range(len(starts))):
        ep_start = starts[ep_idx]
        ep_end = ends[ep_idx]
        
        for i in range(ep_start, ep_end):
            img = obs_list[i]
            front_img = img[:3]
            wrist_img = img[3:]
            
            act = action_list[i]
            state = state_list[i]
            is_done = (i == ep_end - 1)
            
            # Correct normalization: only scale if it's not uint8 already
            front_img_np = front_img if front_img.dtype == np.uint8 else np.clip(front_img * 255.0, 0, 255).astype(np.uint8)
            wrist_img_np = wrist_img if wrist_img.dtype == np.uint8 else np.clip(wrist_img * 255.0, 0, 255).astype(np.uint8)
            
            frame_dict = {
                "observation.images.agentview": torch.from_numpy(front_img_np),
                "observation.images.robot0_eye_in_hand": torch.from_numpy(wrist_img_np),
                "observation.state": state.numpy() if hasattr(state, 'numpy') else np.array(state, dtype=np.float32),
                "action": act.numpy() if hasattr(act, 'numpy') else np.array(act, dtype=np.float32),
                "next.done": torch.tensor([is_done], dtype=torch.bool),
            }
            dataset.add_frame(frame_dict, task="NutAssemblySquare")
            
        dataset.save_episode()
        
    # Create train/test split if needed
    total_episodes = len(starts)
    if train_ratio < 1.0:
        num_train = int(total_episodes * train_ratio)
        train_range = f"0:{num_train}"
        test_range = f"{num_train}:{total_episodes}"
        dataset.meta.info["splits"] = {"train": train_range, "test": test_range}
        write_info(dataset.meta.info, dataset.root)

    print(f"Successfully created LeRobot dataset at {output_dir}")

    # Upload to HF Hub if requested
    if repo_id:
        from huggingface_hub import create_repo
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        # Assign exactly the repo_id the user asked for
        dataset.repo_id = repo_id
        dataset.push_to_hub()
        print(f"Dataset successfully uploaded to {repo_id}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LaNE demo to LeRobot format.")
    parser.add_argument(
        "--demo_dir",
        type=str,
        required=True,
        help="Path to directory containing demo .pt and .npy files",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the LeRobot dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID for uploading the dataset (e.g., YSL2683/lane_lift_id_20_aligned)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Ratio of trajectories to assign to train split (0.0-1.0)",
    )

    args = parser.parse_args()

    convert_lane_to_lerobot(
        args.demo_dir,
        args.output_dir,
        args.repo_id,
        args.train_ratio,
    )
