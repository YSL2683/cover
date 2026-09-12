import argparse
import os
import shutil
from pathlib import Path
import glob
import sys

# Ensure conda env bin is in PATH so ffmpeg can be found
conda_bin = "/home/moai/miniconda3/envs/cover/bin"
if conda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from tqdm import tqdm
from huggingface_hub import create_repo, HfApi

TASK_CONFIGS = {
    "coffee": {
        "default_demo_dir": "lane/demo/mimicgen_coffee/50",
        "dataset_name": "mimicgen_coffee_D0_50",
        "task_name": "Coffee",
    },
    "mug_cleanup": {
        "default_demo_dir": "lane/demo/mimicgen_mug_cleanup/50",
        "dataset_name": "mimicgen_mug_cleanup_D0_50",
        "task_name": "MugCleanup",
    },
    "three_piece_assembly": {
        "default_demo_dir": "lane/demo/mimicgen_three_piece_assembly/100",
        "dataset_name": "mimicgen_three_piece_assembly_D0_100",
        "task_name": "ThreePieceAssembly",
    },
}


def convert_lane_mimicgen_to_lerobot(
    demo_dir: str,
    output_dir: str,
    task_name: str = "MimicGenTask",
    repo_id: str | None = None,
    push_to_hub: bool = False,
    train_ratio: float = 1.0,
):
    """
    Convert rendered MimicGen (.pt + .npy) dataset to LeRobot v2.1 format.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    demo_dir = Path(demo_dir)
    starts_path = demo_dir / "demo_starts.npy"
    ends_path = demo_dir / "demo_ends.npy"

    pt_files = sorted(list(demo_dir.glob("*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {demo_dir}")
    pt_path = pt_files[0]

    print(f"\n==========================================")
    print(f"Loading payload from {pt_path}")
    payload = torch.load(pt_path, weights_only=False)
    obs_list = payload[0]      # [N, 6, 128, 128]
    action_list = payload[2]   # [N, 7]
    state_list = payload[5]    # [N, state_dim]

    starts = np.load(starts_path)
    ends = np.load(ends_path)

    state_dim = state_list.shape[1] if hasattr(state_list, "shape") else len(state_list[0])

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
    print(f"Creating LeRobot dataset with repo_id={dataset_repo_id} at {output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=20,
        features=features,
        root=str(output_dir),
        use_videos=True,
    )

    print(f"Processing {len(starts)} episodes ({ends[-1]} total frames)...")
    for ep_idx in tqdm(range(len(starts)), desc=f"Converting {task_name}"):
        ep_start = int(starts[ep_idx])
        ep_end = int(ends[ep_idx])

        for i in range(ep_start, ep_end):
            img = obs_list[i]
            front_img = img[:3]
            wrist_img = img[3:]

            act = action_list[i]
            state = state_list[i]
            is_done = (i == ep_end - 1)

            front_img_np = front_img if front_img.dtype == np.uint8 else np.clip(front_img * 255.0, 0, 255).astype(np.uint8)
            wrist_img_np = wrist_img if wrist_img.dtype == np.uint8 else np.clip(wrist_img * 255.0, 0, 255).astype(np.uint8)

            frame_dict = {
                "observation.images.agentview": torch.from_numpy(front_img_np),
                "observation.images.robot0_eye_in_hand": torch.from_numpy(wrist_img_np),
                "observation.state": state.numpy() if hasattr(state, "numpy") else np.array(state, dtype=np.float32),
                "action": act.numpy() if hasattr(act, "numpy") else np.array(act, dtype=np.float32),
                "next.done": torch.tensor([is_done], dtype=torch.bool),
            }
            dataset.add_frame(frame_dict, task=task_name)

        dataset.save_episode()

    total_episodes = len(starts)
    if train_ratio < 1.0:
        num_train = int(total_episodes * train_ratio)
        train_range = f"0:{num_train}"
        test_range = f"{num_train}:{total_episodes}"
        dataset.meta.info["splits"] = {"train": train_range, "test": test_range}
        write_info(dataset.meta.info, dataset.root)
    else:
        dataset.meta.info["splits"] = {"train": f"0:{total_episodes}"}
        write_info(dataset.meta.info, dataset.root)

    print(f"Successfully created LeRobot dataset at {output_dir}")

    if push_to_hub and repo_id:
        print(f"Uploading dataset to Hugging Face Hub: {repo_id} ...")
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        dataset.repo_id = repo_id
        dataset.push_to_hub()
        print(f"Dataset successfully uploaded to {repo_id}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert MimicGen datasets to LeRobot format and upload to Hub.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["coffee", "mug_cleanup", "three_piece_assembly", "all"],
        help="Task to convert",
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default=None,
        help="Custom demo directory path (optional)",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="resfit/my_lerobot_data/ysl2683",
        help="Base directory for saving datasets",
    )
    parser.add_argument(
        "--hub_owner",
        type=str,
        default="YSL2683",
        help="Hugging Face organization or user account",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=True,
        help="Whether to upload to Hugging Face Hub",
    )
    parser.add_argument(
        "--no_push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not upload to Hugging Face Hub",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Ratio of trajectories to assign to train split",
    )

    args = parser.parse_args()

    tasks_to_run = (
        ["coffee", "mug_cleanup", "three_piece_assembly"]
        if args.task == "all"
        else [args.task]
    )

    base_dir = Path("/home/moai/ysl_ws/cover")

    for task_key in tasks_to_run:
        cfg = TASK_CONFIGS[task_key]
        demo_dir = args.demo_dir if args.demo_dir else str(base_dir / cfg["default_demo_dir"])
        output_dir = str(base_dir / args.output_base_dir / cfg["dataset_name"])
        repo_id = f"{args.hub_owner}/{cfg['dataset_name']}" if args.hub_owner else None

        print(f"\n========================================================")
        print(f"Task: {task_key}")
        print(f"Demo dir: {demo_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Repo ID: {repo_id}")
        print(f"Push to hub: {args.push_to_hub}")
        print(f"========================================================")

        convert_lane_mimicgen_to_lerobot(
            demo_dir=demo_dir,
            output_dir=output_dir,
            task_name=cfg["task_name"],
            repo_id=repo_id,
            push_to_hub=args.push_to_hub,
            train_ratio=args.train_ratio,
        )

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
