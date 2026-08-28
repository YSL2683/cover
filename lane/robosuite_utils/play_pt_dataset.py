import os
import glob
import torch
import numpy as np
import cv2
import sys

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def play_dataset(data_dir):
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not pt_files:
        print(f"No .pt files found in {data_dir}!")
        return
    
    pt_path = pt_files[0]
    starts_path = os.path.join(data_dir, "demo_starts.npy")
    ends_path = os.path.join(data_dir, "demo_ends.npy")
    
    print(f"Loading {pt_path}...")
    payload = torch.load(pt_path, weights_only=False)
    
    obs_list = payload[0]  # [N, 6, 128, 128] uint8
    starts = np.load(starts_path)
    ends = np.load(ends_path)
    
    print(f"Loaded {len(starts)} demos. Playing on display (10Hz)...")
    print("Press 'q' to quit, 'n' to skip to the next episode.")
    
    for ep_idx, (start, end) in enumerate(zip(starts, ends)):
        print(f"Playing Episode {ep_idx + 1}/{len(starts)} (Frames: {end - start})")
        for i in range(start, end):
            obs = obs_list[i]  # shape: (6, 128, 128)
            
            # Split into Agentview (first 3) and Eye-in-hand (last 3)
            # Transpose from (C, H, W) to (H, W, C)
            agentview = obs[:3].transpose(1, 2, 0)
            wristview = obs[3:].transpose(1, 2, 0)
            
            # Convert RGB to BGR for OpenCV
            agentview_bgr = cv2.cvtColor(agentview, cv2.COLOR_RGB2BGR)
            wristview_bgr = cv2.cvtColor(wristview, cv2.COLOR_RGB2BGR)
            
            # Resize for better visibility (2x scale)
            agentview_bgr = cv2.resize(agentview_bgr, (256, 256))
            wristview_bgr = cv2.resize(wristview_bgr, (256, 256))
            
            # Concatenate horizontally
            frame = np.concatenate([agentview_bgr, wristview_bgr], axis=1)
            
            # Add text
            cv2.putText(frame, f"Ep: {ep_idx+1}/{len(starts)} | Step: {i-start}/{end-start}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Agentview", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Eye-in-Hand", (266, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Dataset Playback (Press 'q' to quit, 'n' for next)", frame)
            
            # 10Hz = 100ms per frame
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("Playback stopped by user.")
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break # Skip to next episode
            
    cv2.destroyAllWindows()
    print("Playback finished.")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else f"{PROJECT_ROOT}/lane/demo/robomimic_square/50"
    play_dataset(target_dir)
