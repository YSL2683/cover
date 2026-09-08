# Cover (ResFiT & LaNE) 환경 설정 가이드 (SETUP_README.md)

이 문서는 새로운 서버 PC(Ubuntu / NVIDIA GPU)에서 **`cover` Conda 가상환경**을 구축하고, **Behavior Cloning (Diffusion)**, **LaNE E2C**, **Residual RL (TD3)** 파이프라인을 오류 없이 곧바로 실행할 수 있도록 실제 검증된 모든 설정 단계와 트러블슈팅 해결책을 집대성한 가이드입니다.,

---

## 목차
1. [시스템 요구사항 및 패키지](#1-시스템-요구사항-및-패키지)
2. [Conda 가상환경 생성 및 기본 도구 준비](#2-conda-가상환경-생성-및-기본-도구-준비)
3. [프로젝트 의존성 설치 (setup 스크립트)](#3-프로젝트-의존성-설치-setup-스크립트)
4. [핵심 패키지 버전 정렬 및 필수 트러블슈팅 (중요)](#4-핵심-패키지-버전-정렬-및-필수-트러블슈팅-중요)
5. [로컬 경로 및 매크로 등록](#5-로컬-경로-및-매크로-등록)
6. [계정 연동 (Git, HuggingFace, WandB)](#6-계정-연동-git-huggingface-wandb)
7. [환경 검증 테스트](#7-환경-검증-테스트)
8. [단계별 학습 실행 워크플로우](#8-단계별-학습-실행-워크플로우)

---

## 1. 시스템 요구사항 및 패키지

- **OS**: Ubuntu 20.04 / 22.04 LTS (x86_64)
- **GPU**: NVIDIA RTX 3090 / 4090 / A100 등 (VRAM 24GB 권장)
- **CUDA 드라이버**: 535 이상 (CUDA 12.x / 13.x 지원)

### 시스템 라이브러리 설치
헤드리스(Headless) 렌더링(EGL, OSMesa) 및 컴파일 의존성을 설치합니다:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf ffmpeg
```

---

## 2. Conda 가상환경 생성 및 기본 도구 준비

Python 3.10 버전으로 `cover` 가상환경을 생성하고 활성화합니다:

```bash
# 1. 가상환경 생성
conda create -n cover python=3.10 -y

# 2. 가상환경 활성화
conda activate cover

# 3. 비디오 인코딩/디코딩용 FFmpeg 설치 (conda-forge)
conda install -c conda-forge "ffmpeg>=6,<8" -y
```

> [!TIP]
> 프로젝트의 기존 설치 스크립트(`setup_dexmg.sh`)에는 `micromamba install -n residual ...` 명령이 하드코딩되어 있습니다. 스크립트 수정 없이 호환되도록 Conda 환경 디렉터리에 심볼릭 링크 및 `micromamba` 래퍼를 생성해 두면 오류를 방지할 수 있습니다:
> ```bash
> # residual 환경명을 cover로 링크
> ln -s $(conda info --base)/envs/cover $(conda info --base)/envs/residual 2>/dev/null || true
> 
> # micromamba 명령어 호출 시 conda를 대신 호출하도록 래퍼 생성
> cat << 'EOF' > $(conda info --base)/envs/cover/bin/micromamba
> #!/bin/bash
> exec $(conda info --base)/bin/conda "$@"
> EOF
> chmod +x $(conda info --base)/envs/cover/bin/micromamba
> ```

---

## 3. 프로젝트 의존성 설치 (setup 스크립트)

저장소 루트 디렉터리(`cover`)로 이동한 후, 제공된 통합 셋업 스크립트를 실행합니다:

```bash
cd ~/ysl_ws/cover
export PATH=$(conda info --base)/envs/cover/bin:$PATH

# deps/ 디렉터리에 lerobot, robosuite, dexmimicgen, mimicgen 클론 및 기본 설치 진행
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
```

---

## 4. 핵심 패키지 버전 정렬 및 필수 트러블슈팅 (중요)

스크립트 실행 후 최신 라이브러리 간의 C++ ABI 및 API 변경으로 인한 불일치를 해결하기 위해 **반드시 아래 패키지들을 순서대로 설치/재설치**해야 합니다.

### (1) Robosuite 버전 다운그레이드 (`1.4.1`)
- **원인**: `deps/robosuite` 최신 커밋(v1.5.1)은 `load_controller_config` API가 누락되고 composite controller 아키텍처로 변경되어 `Can`, `Square` 등 단일 arm 환경 생성 시 `AssertionError: OSC_POSE controller is specified, but not imported or loaded` 에러가 발생합니다.
- **해결책**:
  ```bash
  pip install robosuite==1.4.1
  ```

### (2) TorchCodec 버전 업그레이드 (`0.15.0`)
- **원인**: 기본 설치되는 torchcodec 0.4.0은 최신 PyTorch(2.14+)와 C++ ABI 심볼(`materialize_cow_storage`)이 맞지 않아 비디오 디코더(`VideoDecoder`) 로드에 실패하고 데이터셋 읽기 단계에서 `RuntimeError: Could not load libtorchcodec` 에러가 발생합니다.
- **해결책**:
  ```bash
  pip install torchcodec==0.15.0
  ```

### (3) scikit-learn 설치 (잠재공간 시각화)
- **원인**: Residual RL 학습 중 평가(Evaluation) 단계(`evaluate_dexmg.py` -> `visualize_rl_latents.py`)에서 t-SNE 잠재공간 시각화를 위해 `from sklearn.manifold import TSNE`를 임포트합니다. 미설치 시 10,000 스텝 후 첫 평가에서 학습이 중단됩니다.
- **해결책**:
  ```bash
  pip install scikit-learn
  ```

### (4) RL 파인튜닝 & 모니터링 필수 패키지
```bash
pip install draccus==0.10.0 hydra-core matplotlib tensorboard wandb deepdiff serial
```

---

## 5. 로컬 경로 및 매크로 등록

### (1) `resfit` 모듈 자동 인식 설정 (.pth 파일)
소스 코드 변경 없이 파이썬 인터프리터가 어디서든 `resfit` 패키지를 찾을 수 있도록 가상환경의 `site-packages`에 경로 파일을 등록합니다:

```bash
COVER_ROOT=$(pwd)
echo "${COVER_ROOT}" > $(conda info --base)/envs/cover/lib/python3.10/site-packages/resfit.pth
```

### (2) Robosuite 개인 매크로 파일 생성
```bash
python $(conda info --base)/envs/cover/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

---

## 6. 계정 연동 (Git, HuggingFace, WandB)

### (1) Git 계정 설정 (현재 레포 전용)
전역 설정과 분리하여 `cover` 레포에만 계정을 등록합니다:
```bash
git config --local user.name "YSL2683"
git config --local user.email "dbstjd55@seoultech.ac.kr"
```

### (2) Hugging Face 로그인 (데이터셋 접근)
```bash
hf auth login
```

### (3) Weights & Biases 로그인 (학습 지표 로깅)
```bash
wandb login
```

---

## 7. 환경 검증 테스트

설정이 정상적으로 완료되었는지 확인하기 위해 아래 원클릭 검증 커맨드를 실행합니다:

```bash
python -c "
import torch
import robosuite
import dexmimicgen
from torchcodec.decoders import VideoDecoder
from sklearn.manifold import TSNE
from resfit.dexmg.environments.dexmg import create_vectorized_env

assert torch.cuda.is_available(), 'CUDA is not available!'
assert hasattr(robosuite, 'load_controller_config'), 'robosuite load_controller_config missing!'

print('=' * 50)
print('✓ PyTorch CUDA:', torch.cuda.get_device_name(0))
print('✓ Robosuite Version:', robosuite.__version__)
print('✓ TorchCodec VideoDecoder: Ready')
print('✓ Scikit-Learn TSNE: Ready')
print('=' * 50)

# 시뮬레이션 환경 생성 테스트
env = create_vectorized_env(env_name='Can', num_envs=2, camera_size=128)
obs, _ = env.reset()
assert 'observation.images.agentview' in obs
env.close()
print('✓ Robosuite Can Environment: Reset Success!')
print('>>> All tests passed successfully! <<<')
"
```

---

## 8. 단계별 학습 실행 워크플로우

### Step 1. Base BC Policy (Diffusion) 학습
데모 데이터셋(`ysl2683/robomimic_can_v15_10`)을 이용해 Diffusion 정책을 사전학습합니다:

```bash
# 쉘 스크립트 실행
bash resfit/lerobot/shell/train_diffusion_can.sh
```
- **주요 파라미터**:
  - `--dataset`: `ysl2683/robomimic_can_v15_10`
  - `--eval_env`: `Can`
  - `--policy`: `diffusion` (`crop_shape: [112, 112]`)
  - `--rollout_freq`: `1000`
- **저장 경로**: `resfit/my_lerobot_data/bc_run_<timestamp>_robomimic_can_v15_10_diffusion/`
  - 최고 성능 체크포인트: `best/policy`

---

### Step 2. LaNE E2C 잠재 공간 사전학습
데모 궤적 영상에서 잠재 공간(Latent Space)을 학습하여 보상 함수(PBRS)에 사용할 모델을 생성합니다:

```bash
export PYTHONPATH=$(pwd)/lane:${PYTHONPATH:-}

python lane/pretrain_e2c.py \
    --demo_dir lane/demo/robomimic_can/10 \
    --save_dir lane/pretrained_e2c/can \
    --mode decoupled
```
- **출력 아티팩트**:
  - `lane/pretrained_e2c/can/e2c_main.pt`: 메인 뷰 E2C 모델
  - `lane/pretrained_e2c/can/e2c_wrist.pt`: 손목 뷰 E2C 모델
  - `lane/pretrained_e2c/can/demo_latents.pt`: 데모 Latent 임베딩

---

### Step 3. Residual TD3 강화학습 (PBRS 기반 Finetuning)
Step 1의 베이스 정책 가중치와 Step 2의 E2C 모델을 결합하여 Residual RL 학습을 수행합니다:

```bash
# Can 태스크 실행 스크립트
bash resfit/shell_paper/ID_finetuninig/can/run_can_pbrs_no_mask_nstep_ID_preward0.1.sh
```
- **스크립트 내 핵심 설정 확인**:
  - `BASE_POLICY_PATH`: Step 1에서 학습된 `.../best/policy` 디렉터리 경로
  - `E2C_DIR`: Step 2에서 생성된 `lane/pretrained_e2c/can` 경로
  - `OFFLINE_DATA_DIR`: 오프라인 데모 버퍼 경로
  - `DIFFUSION_DDIM_STEPS`: `20` (빠른 추론을 위한 DDIM 패치 적용)
