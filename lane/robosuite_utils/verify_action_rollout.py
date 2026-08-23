import os
import glob
import torch
import numpy as np
import h5py
import robosuite as suite
from robosuite import load_controller_config
import cv2

def verify_rollout():
    hdf5_path = "/home/moai/ysl_ws/cover/lane/demo/demo_v15.hdf5"
    pt_dir = "/home/moai/ysl_ws/cover/lane/demo/robomimic_square/50"
    
    # 1. Load .pt Actions
    pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
    if not pt_files:
        print("No .pt file found!")
        return
    pt_path = pt_files[0]
    
    print(f"Loading {pt_path}...")
    payload = torch.load(pt_path, weights_only=False)
    pt_actions = payload[2]  # [N, 7]
    starts = np.load(os.path.join(pt_dir, "demo_starts.npy"))
    ends = np.load(os.path.join(pt_dir, "demo_ends.npy"))
    
    # 2. Open HDF5 to get original initial states
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    demos = sorted(demos, key=lambda x: int(x.split("_")[1]))[:50]
    
    # 3. Setup Robosuite Env
    config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        controller_configs=config,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        horizon=300,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    env.sim.model.site_rgba[:, 3] = 0.0
    
    print("Environment loaded. Starting Action Rollout verification...")
    print("Press 'q' to quit, 'n' to skip to next episode.")
    
    for i in range(len(demos)):
        demo_name = demos[i]
        ep_actions = pt_actions[starts[i]:ends[i]]
        
        # Get exact initial state from original HDF5
        initial_state = f["data"][demo_name]["states"][0]
        
        env.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        
        print(f"\n--- Episode {i+1} ({demo_name}) ---")
        print(f"Rolling out {len(ep_actions)} actions...")
        
        success = False
        for step_idx, action in enumerate(ep_actions):
            # Rollout the action in the physics engine
            obs, reward, done, info = env.step(action)
            
            if env._check_success():
                success = True
            
            # Render and display
            agentview = obs["agentview_image"][::-1] # robosuite images are upside down by default
            wristview = obs["robot0_eye_in_hand_image"][::-1]
            
            agentview_bgr = cv2.cvtColor(agentview, cv2.COLOR_RGB2BGR)
            wristview_bgr = cv2.cvtColor(wristview, cv2.COLOR_RGB2BGR)
            
            agentview_bgr = cv2.resize(agentview_bgr, (256, 256))
            wristview_bgr = cv2.resize(wristview_bgr, (256, 256))
            
            frame = np.concatenate([agentview_bgr, wristview_bgr], axis=1)
            
            cv2.putText(frame, f"Rollout Ep: {i+1} | Step: {step_idx}/{len(ep_actions)}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Agentview", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Eye-in-Hand", (266, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Action Rollout Verification", frame)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("Verification stopped by user.")
                cv2.destroyAllWindows()
                f.close()
                return
            elif key == ord('n'):
                break
                
        print(f"Episode {i+1} Result: {'SUCCESS' if success else 'FAILED'}")
        
    cv2.destroyAllWindows()
    f.close()
    print("Verification finished.")

if __name__ == "__main__":
    verify_rollout()
