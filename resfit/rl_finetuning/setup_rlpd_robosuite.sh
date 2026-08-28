#!/usr/bin/env bash
set -euo pipefail

./resfit/lerobot/setup_lerobot.sh
./resfit/dexmg/setup_dexmg.sh

python3.10 -m pip install wandb einops psutil
