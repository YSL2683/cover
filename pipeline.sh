#!/bin/bash
# ==============================================================================
# Full Pipeline: E2C Pretraining -> Base Policy Training -> Residual RL
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "==============================================================="
echo " Step 1: Pretrain Latent Space (E2C / VAE) for Visual Similarity "
echo "==============================================================="
# This script extracts DINOv2 features from the offline demo data and trains the 
# E2C VAE encoder for both frontview and wrist cameras.
# Pre-trained weights will be saved to resfit/lane/pretrained_e2c/
conda run -n cover python resfit/lane/pretrain_e2c.py

echo "==============================================================="
echo " Step 2: Train Base Policy (Diffusion Policy) via Behavior Cloning "
echo "==============================================================="
# This script uses the standard LeRobot pipeline to train a Diffusion Policy 
# on the offline dataset (e.g. ysl2683/lane_can). 
# Note: Ensure that the dataset name matches your HuggingFace/local dataset.
conda run -n cover python resfit/lerobot/scripts/train.py \
  policy=diffusion \
  env=robomimic \
  env.name=can \
  dataset_repo_id=ysl2683/lane_can \
  training.offline_steps=100000 \
  eval.n_episodes=50 \
  wandb.enable=false

echo "==============================================================="
echo " Step 3: Train Residual RL Policy (TD3) "
echo "==============================================================="
# Once the base policy and the E2C encoders are trained, we run the Residual RL 
# script. It injects the HybridRewardVecEnvWrapper (using the E2C models) and 
# trains a TD3 agent to output residual actions on top of the frozen base policy.
# Make sure to update base_policy.wt_version if you trained a new base policy!
PYTHONPATH=$(pwd) CACHE_DIR=$(pwd)/resfit/my_lerobot_data/ysl2683/lane_can \
conda run -n cover --no-capture-output python resfit/rl_finetuning/scripts/train_residual_td3.py \
  offline_data.name=ysl2683/lane_can \
  task=PickPlaceCan \
  wandb.mode=disabled \
  algo.learning_starts=200 \
  algo.total_timesteps=500000 \
  algo.critic_warmup_steps=100 \
  eval_interval_every_steps=5000 \
  algo.buffer_size=10000 \
  rl_camera="[observation.images.frontview,observation.images.robot0_eye_in_hand]" \
  video_key=observation.images.frontview

echo "==============================================================="
echo " Pipeline Finished Successfully! "
echo "==============================================================="
