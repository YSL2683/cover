# Project Agents Guide: ResFiT (Residual Off-Policy RL for Finetuning)

This document provides technical context, architecture details, and coding conventions for AI agents and developers working on the **ResFiT** repository.

## 1. Project Overview & Architecture

**ResFiT** is a framework for **Residual Off-Policy RL for Finetuning Behavior Cloning Policies**. The system leverages a pre-trained base Behavior Cloning (BC) policy (like ACT or Diffusion models) and trains a residual Reinforcement Learning (RL) policy (based on TD3) to predict corrective action offsets.

### High-Level Architecture
- **Robot Simulator**: Uses `gymnasium`, `robosuite`, and custom dexterous manipulation wrappers (`dexmg`).
- **Base Policy (BC)**: Handled via the `lerobot` framework. Pre-trained on demonstration datasets.
- **Residual Policy (RL)**: Implemented in `resfit/rl_finetuning`. A Q-learning agent (TD3) outputs action residuals.
- **Environment Wrappers**: Combine the base policy and the RL environment (e.g., `BasePolicyVecEnvWrapper`).

### Directory Structure
- `resfit/lerobot/`: BC policy training, loading, and dataset utilities (integration with HuggingFace LeRobot).
- `resfit/rl_finetuning/`: Core RL codebase. Contains TD3 algorithms, replay buffers, config (`hydra`), and training scripts.
- `resfit/rl_finetuning/wrappers/`: Crucial gym wrappers (e.g., `residual_env_wrapper.py`).
- `resfit/dexmg/`: Dexterous manipulation environments and vectorized environment builders.
- `lane/`: Components related to Representation Learning and E2C (Embed to Control) architecture.

## 2. Environments & Data Pipelines

### Observation and Action Spaces
- **Observations**: Dictionary format containing:
  - `observation.state`: Low-dimensional proprioceptive/state data.
  - `observation.images.*`: Camera image observations.
  - `observation.base_action`: Augmented into the state by the `BasePolicyVecEnvWrapper`.
- **Actions**: The final action sent to the simulator is the sum of the **base action** and the **residual action**.
- **Normalization**: Handled by `ActionScaler` and `StateStandardizer`.
  - Base actions are normalized to `[-1, 1]`.
  - Residual actions are predicted as `action_scale * [-1, 1]` and added to the base normalized action.

### Simulator Interaction Loop
- Vectorized environments return `dict[str, torch.Tensor]`.
- `BasePolicyVecEnvWrapper.step()`:
  1. Combines `base_naction + residual_naction`.
  2. Unscales combined action to simulator bounds.
  3. Steps underlying `vec_env`.
  4. Runs base policy inference on the *new* observation to get the next `base_naction`.
  5. Augments the new observation with the next `base_naction`.
- Handles `final_obs` padding (terminal states have zeroed-out base actions).

### Data Pipelines & Replay Buffers
- Uses `torchrl` `TensorDictReplayBuffer` with `LazyMemmapStorage` for memory efficiency.
- **Offline Buffer**: Pre-populated from `LeRobotDataset` demonstrations.
- **Online Buffer**: Populated during environment rollouts.
- Images are converted to `uint8` before being stored to save memory.
- Strong caching mechanism based on Hugging Face repos. Hashes config parameters to load/save buffer states.

## 3. Standard Execution Commands

### Environment Setup
```bash
conda create -n cover python=3.10 -y
conda activate cover
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
pip install wandb draccus==0.10.0 torchrl==0.9.2 hydra-core serial deepdiff matplotlib
```

### BC Policy Training
```bash
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset <hf_dataset_id> \
    --policy act \
    --wandb_project <project>
```

### Residual RL Training
```bash
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_coffee_config \
    wandb.project=<project> \
    debug=false
```

### Rendering & Logging
- **Headless Rendering**: Handled via EGL. Script sets `os.environ["MUJOCO_GL"] = "egl"`.
- **Logging**: WandB integration is standard for tracking metrics. TensorBoard is also used.

## 4. Technical Invariants & Gotchas

- **Thread Limiting**: RL training scripts explicitly cap BLAS/OpenMP threads (e.g., `OMP_NUM_THREADS="1"`) at the very top of the file to prevent CPU contention during vectorized environment steps. **Do not remove these lines.**
- **Seeding & Reproducibility**: Strict deterministic seeding is enforced for `random`, `numpy`, `torch`, `cuda`, and the environment `reset(seed=cfg.seed)`.
- **Device Placement**: Replay buffers typically reside on CPU memory/disk (via `MemmapStorage`). Ensure tensors are moved to the correct device (GPU) before passing to policy networks or calculating losses.
- **Action Safety**: Residual actions are bounded by `agent.actor.action_scale`. Combined actions must be properly unscaled before hitting the physics engine.
- **Replay Buffer Cache Invalidations**: Changing state dimensions, image keys, or normalization logic will invalidate the replay buffer cache hash. Be aware that the buffer will need to be rebuilt from scratch if these change.

## 5. Agent Guardrails & Conventions

- **File Modification Boundaries**: Do not alter `robosuite` physics parameters, underlying `dexmg` reward functions, or core simulation loop mechanics unless explicitly instructed.
- **Code Standards**: 
  - Adhere strictly to PyTorch and TorchRL conventions. 
  - Maintain the existing type annotation style (e.g., `dict[str, torch.Tensor]`).
  - Use the minimal diff principle: only modify the specific functions or classes requested by the user.
- **Self-Verification**:
  - Before finalizing changes to wrappers or replay buffers, trace the tensor shapes and device placements conceptually.
  - Verify that offline normalization (`ActionScaler`, `StateStandardizer`) perfectly mirrors the online environment loop logic.
