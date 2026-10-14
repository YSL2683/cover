# Cover RL & DexMG 다른 PC(RTX 3090 / 5090) Docker 이전 및 실행 가이드

본 문서는 현재 PC에서 구축된 강화학습/모방학습 환경(ResFiT, PBRS, LeRobot, Robosuite, DexMimicGen)을 Docker 컨테이너로 빌드하고, 로컬 네트워크를 통해 다른 PC(RTX 3090 또는 RTX 5090)로 이전하여 동일하게 학습을 진행하는 전체 과정을 안내합니다.

---

## 0. 전체 이전 워크플로우 요약

1. **[현재 PC]** Docker 이미지 빌드 및 아카이브 저장 (`cover_image.tar.gz`)
2. **[현재 PC $\rightarrow$ 타겟 PC]** 로컬 네트워크(`rsync`)로 필수 파일 전송 (~40GB: 코드 + 38GB 데모 데이터)
   *(주의: 375GB 과거 산출물 `resfit/outputs` 및 82GB 캐시 `scratch`는 전송에서 자동 제외)*
3. **[타겟 PC]** NVIDIA Container Toolkit 설치 (최초 1회)
4. **[타겟 PC]** Docker 이미지 로드 (`docker load`)
5. **[타겟 PC]** 컨테이너 실행 및 환경 검증 (`test_container_env.py`)
6. **[타겟 PC]** 기존 학습 셸 스크립트 그대로 실행 (`run_*.sh`)

---

## 1. [현재 PC] Docker 이미지 빌드 및 저장

### 1-1. Docker 이미지 빌드
워크스페이스 루트(`/home/moai/ysl_ws/cover`)에서 아래 명령어를 실행합니다:

```bash
cd /home/moai/ysl_ws/cover

# Dockerfile을 사용하여 cover:latest 이미지 빌드
docker build -t cover:latest -f docker/Dockerfile .
```

### 1-2. 빌드된 이미지를 tar.gz 파일로 저장
```bash
# 이미지를 압축 파일로 내보내기 (용량 약 8~10GB)
docker save cover:latest | gzip > cover_image.tar.gz
```

---

## 2. [현재 PC $\rightarrow$ 타겟 PC] 로컬 네트워크 전송 (rsync)

불필요한 375GB 과거 산출물(`resfit/outputs`)과 82GB 캐시(`scratch`)를 제외하고, 필수적인 코드, 데모 데이터셋(`my_lerobot_data`, 38GB), 사전학습 E2C 가중치(`lane/pretrained_e2c`, 171MB), 빌드된 도커 이미지만 고속으로 전송합니다.

> **참고**: `<TARGET_USER>`와 `<TARGET_IP>`는 타겟 PC의 사용자명과 IP 주소로 변경하세요. (예: `moai@192.168.0.50`)

### 2-1. Docker 이미지 파일 전송
```bash
scp cover_image.tar.gz <TARGET_USER>@<TARGET_IP>:~/
```

### 2-2. 워크스페이스 코드 및 데이터 전송
```bash
# 타겟 PC의 홈 디렉토리(또는 원하는 경로)에 cover 폴더 동기화
rsync -avzP \
  --exclude 'resfit/outputs' \
  --exclude 'scratch' \
  --exclude '*.tar.gz' \
  --exclude '.git' \
  /home/moai/ysl_ws/cover/ <TARGET_USER>@<TARGET_IP>:~/cover/
```
*(기본적인 `resfit/outputs`와 `scratch` 폴더는 타겟 PC에서 학습 시작 시 자동으로 생성되므로 전송할 필요가 없습니다.)*

### 2-3. WandB 인증 키 전송 (선택 사항)
WandB 로그 연동을 유지하려면 `~/.netrc` 파일도 전송합니다:
```bash
scp ~/.netrc <TARGET_USER>@<TARGET_IP>:~/
```

---

## 3. [타겟 PC] 사전 준비 (최초 1회만 수행)

타겟 PC(Ubuntu)에서 GPU 컨테이너를 실행하기 위해 **NVIDIA Container Toolkit**이 설치되어 있어야 합니다.

### 3-1. NVIDIA 드라이버 확인
```bash
nvidia-smi
```
- **RTX 3090**: 드라이버 버전 525 이상 권장 (535, 550, 560 등)
- **RTX 5090**: 드라이버 버전 570 / 595 이상 필수

### 3-2. NVIDIA Container Toolkit 설치
이미 설치되어 있다면 이 단계를 건너뜁니다. 미설치 시:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 4. [타겟 PC] Docker 이미지 로드 및 검증

### 4-1. 전송받은 이미지 로드
```bash
cd ~
docker load < cover_image.tar.gz
```
로드가 완료되면 `docker images` 명령어로 `cover:latest`를 확인할 수 있습니다.

### 4-2. 환경 종합 검증 (Smoke Test)
워크스페이스로 이동한 뒤 제공된 검증 스크립트를 도커로 실행합니다:

```bash
cd ~/cover

# test_container_env.py 실행
./docker/run_container.sh python docker/test_container_env.py
```

성공 시 아래와 같이 5가지 항목이 모두 `[PASS]`로 출력됩니다:
```text
============================================================
Cover RL Container Environment Verification Test
============================================================
[1/5] Testing PyTorch & CUDA...
  [PASS] Found 1 GPU(s). Device 0: NVIDIA GeForce RTX ...
[2/5] Testing MuJoCo Headless EGL Rendering...
  [PASS] Rendered offscreen frame successfully: shape=(128, 128, 3)
[3/5] Testing TorchRL & TensorDict...
  [PASS] TensorDict GPU allocation succeeded.
[4/5] Testing Robosuite & DexMimicGen...
  [PASS] Robosuite & DexMimicGen imports verified.
[5/5] Testing LeRobot & Hugging Face...
  [PASS] LeRobot and DiffusionPolicy loader imported successfully.
============================================================
ALL VERIFICATION CHECKS PASSED! Container is ready for training.
============================================================
```

---

## 5. [타겟 PC] 학습 실행 방법

기존에 사용하시던 셸 스크립트를 **컨테이너 내부에서 그대로** 실행하실 수 있습니다.

### 방법 1: 컨테이너 내부에서 AGY(AI 어시스턴트) 실행 (가장 추천)
`./docker/run_container.sh`는 호스트 PC의 `agy` 바이너리와 로그인 세션(`~/.gemini`)을 컨테이너 내부에 **자동으로 감지하여 마운트**합니다.
따라서 컨테이너 안에서 별도의 설치나 재로그인 없이 일반 PC처럼 `agy`를 즉시 사용할 수 있습니다.

```bash
cd ~/cover

# 1. GPU 컨테이너 셸 진입 (기본적으로 agy와 세션 자동 마운트됨)
./docker/run_container.sh

# 2. 컨테이너 내부에서 바로 agy 실행
agy
```

또는 컨테이너 시작과 동시에 바로 `agy` 실행:
```bash
./docker/run_container.sh agy
```

컨테이너 안의 `agy`에게:
> *"이어서 `run_square_resfit_position_ood_1000k.sh` 학습 시작하고 로그 모니터링해줘"*

라고 평소처럼 지시하시면 컨테이너 내부 환경에서 완벽하게 학습 및 모니터링을 수행합니다.

### 방법 2: 대화형(Interactive) 셸 진입 후 직접 스크립트 실행
```bash
cd ~/cover

# 1. GPU 컨테이너 셸 진입
./docker/run_container.sh bash

# 2. 컨테이너 내부에서 원하는 학습 스크립트 실행
# 예: 50 데모 Position OOD ResFiT 학습
./resfit/shell_paper/test/run_square_resfit_position_ood_1000k.sh

# 예: 200 데모 Position OOD PBRS 학습
./resfit/shell_paper/test/run_square_pbrs_no_mask_nstep_weighted_time_position_ood_preward0.1_200demos_1000k.sh
```

### 방법 3: 백그라운드 단일 명령어로 실행 (nohup / tmux / screen)
tmux나 screen 세션 안에서 바로 실행하거나, 백그라운드로 실행할 수 있습니다:

```bash
cd ~/cover

# 컨테이너에 명령어 직접 전달
./docker/run_container.sh ./resfit/shell_paper/test/run_square_resfit_position_ood_1000k.sh
```

### 방법 4: docker-compose 사용
```bash
cd ~/cover/docker
docker compose run --rm cover ./resfit/shell_paper/test/run_square_resfit_position_ood_1000k.sh
```

---

## 6. 자주 묻는 질문 및 트러블슈팅 (FAQ)

### Q1. 타겟 PC가 여러 대의 GPU(예: GPU 0, GPU 1)를 가지고 있는데, 특정 GPU만 지정해서 학습할 수 있나요?
네, `GPU_DEVICE` 환경변수로 특정 GPU를 지정할 수 있습니다:
```bash
# GPU 1번만 할당하여 실행
GPU_DEVICE='"device=1"' ./docker/run_container.sh bash

# GPU 0, 1번 모두 할당하여 실행 (기본값)
./docker/run_container.sh bash
```

### Q2. 컨테이너 내부에서 생성된 산출물(`resfit/outputs`)이 호스트 타겟 PC에도 저장되나요?
네, 워크스페이스 디렉토리 전체(`~/cover`)가 컨테이너 내부의 `/workspace/cover`로 마운트되어 있으므로, 학습 도중 저장되는 모델 가중치(`models/agent_*.pt`), Q-궤적 플롯, 텐서보드 로그는 실시간으로 호스트 PC의 디스크에 안전하게 저장됩니다.

### Q3. RTX 3090과 RTX 5090에서 별도의 이미지 설정 변경이 필요한가요?
필요 없습니다. 빌드된 PyTorch와 CUDA 환경은 Ampere(`sm_86`, RTX 3090)부터 Blackwell(`sm_120`, RTX 5090)까지 모두 지원하는 공용 커널 아키텍처를 내장하고 있으므로, 동일한 이미지를 두 기종 모두에서 그대로 사용하실 수 있습니다.
