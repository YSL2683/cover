# moai-pobi 머신용 agy CLI Agent 지침서

> 본 문서는 `moai-pobi` 머신에서 agy CLI Agent가 환경을 `moai-rupy`와 동일하게 동기화하고, 50 데모 조건에서 동일한 수렴 경향성(`lane` 메트릭 하강 및 성공률 상승)을 재현하기 위한 공식 지침서입니다.

---

## 1. 배경 및 문제 현상 요약 (Background & Context)

### 1.1 관찰된 현상
`moai-rupy`와 `moai-pobi`에서 동일 task(`Square`)로 학습을 진행했을 때, WandB의 `lane` 섹션 메트릭에서 뚜렷한 경향성 차이가 관찰되었습니다:
- **`moai-rupy`**:
  - `lane/rem_t_wrist_next_avg`: 12.0 $\to$ **8.5 수준으로 점진적 하강**
  - `lane/min_dist_wrist_next_avg`: 0.15 $\to$ **0.03~0.04 수준으로 급감**
  - `lane/S_wrist_next_avg`: 0.79 $\to$ **0.96 이상으로 급상승**
  - $\implies$ **물리적 의미**: 로봇이 너트를 쥐고 페그 결합(Insertion) 구간에 성공적으로 진입하여 온라인 버퍼에 결합 궤적들을 대량으로 축적하고 있다는 증거.
- **과거 `moai-pobi` (50 데모 구형 수식 실험들)**:
  - `lane/rem_t_wrist_next_avg`: **16.0 부근에서 수평 정체**
  - `lane/min_dist_wrist_next_avg`: **0.20~0.25 부근에서 수평 정체**
  - `lane/S_wrist_next_avg`: **0.60~0.67 정체**
  - $\implies$ **물리적 의미**: 페그 하강에 실패하고 상공에서 헛도는(호버링) 궤적만 버퍼에 채워져 로컬 미니멈에 고착됨.

### 1.2 근본 원인 규명
1. **보상 수식의 상공 호버링 로컬 미니멈 (가장 핵심적인 원인)**:
   - 과거 Pobi의 대표 실행(`pnyte50o`, `ujvx4mzn`)은 구형 수식(`reward_pbrs_no_mask_nstep`)을 사용했습니다.
   - 로봇이 페그로 하강할 때 3인칭(main) 카메라가 로봇 팔에 가려져 $S_{\text{main}}$이 하락하는데, 구형 수식에서는 이것이 보상 감점($\gamma \Phi' - \Phi < 0$)으로 작용하여 에이전트가 감점을 피하고자 상공에서 호버링하는 정책을 학습해 버렸습니다.
   - 반면 최신 수식인 `reward_pbrs_no_mask_nstep_weighted_time` (`Ours`)은 손목 유사도($S_{\text{wrist}}$) 기반으로 시간 감쇄를 가중 통합하여 이 하강 장벽을 해소했습니다. (실제로 Pobi 최신 실행 `1d8q7w4r`에서 이 수식을 적용해 **성공률 94%** 달성 확인).
2. **라이브러리 버전 불일치 (환경 파편화)**:
   - `moai-rupy` (RTX 5090): `numpy == 1.26.4`, `robosuite == 1.4.0` (`deps/robosuite`), `torchrl == 0.13.3`, `tensordict == 0.13.0`, `gymnasium == 1.2.2`
   - `moai-pobi` (RTX 4090): `numpy == 2.2.6`, `robosuite == 1.4.1` (PyPI), `torchrl == 0.8.0`, `tensordict == 0.8.2`, `gymnasium == 1.1.1`
   - 사각 너트와 페그 간의 간극(tolerance)이 1~2mm인 고난도 강체 접촉 시뮬레이션에서 `numpy` 2.x의 타입 프로모션/정밀도 차이 및 `robosuite` 버전 간의 물리 접촉 거동 차이가 존재합니다.
3. **데모 수 차이 (200 데모 vs 50 데모)**:
   - 현재 Pobi에서 돌고 있는 최신 실행(`1d8q7w4r`)은 **200 데모** 세팅입니다. 200 데모는 작업 공간의 잠재 공간(Latent Space)을 4배 촘촘하게 덮고 있으므로, 결합 시에도 기준 데모 거리가 0.12로 항상 가까워 Rupy의 50 데모 실험 대비 메트릭 감소 폭이 작게 나타납니다.

---

## 2. 절대 안전 규칙 (Safety Guardrails)

> [!CAUTION]
> **현재 Pobi에서 실행 중인 학습 태스크(`1d8q7w4r` 등)는 사용자가 명시적으로 중단하라고 요청하기 전까지 절대 kill하거나 중단하지 마십시오.**
> 현재 실행은 28만 스텝 시점에서 성공률 94~96%로 수렴하고 있는 가치 있는 200 데모 기준선 데이터입니다.
> 충분히 수렴을 확인한 후, 사용자의 명시적 승인을 얻어 중단하거나 완료된 후 다음 단계로 넘어가십시오.

---

## 3. Step-by-Step 실행 지침 (Agent Execution Steps)

### Step 1. 현재 학습 상태 확인 및 수렴 대기
1. 현재 실행 중인 프로세스 및 WandB 런(`1d8q7w4r`)의 진행 상태를 확인합니다.
2. 수렴 여부를 확인하고 사용자가 중단 또는 후속 작업을 승인하면 다음 단계로 진입합니다.

### Step 2. 라이브러리 통일 (moai-rupy 스택으로 일치)
Pobi의 `cover` 가상환경에서 라이브러리를 Rupy와 동일하게 맞춥니다:

```bash
conda activate cover

# 1. 기존 환경 패키지 백업
pip freeze > /home/moai/ysl_ws/cover/scratch/requirements_pobi_backup_$(date +%Y%m%d).txt

# 2. robosuite를 레포 내부 submodule (1.4.0)로 editable 설치
pip install -e /home/moai/ysl_ws/cover/deps/robosuite

# 3. 핵심 연산 및 RL 라이브러리 통일 (NumPy 1.x 강제 고정, Gymnasium, TorchRL, TensorDict)
# POBI의 NumPy 2.2.6을 1.26.4로 다운그레이드하여 물리/배열 연산 일치
pip install numpy==1.26.4 gymnasium==1.2.2 numba==0.66.0 llvmlite==0.48.0 torchrl==0.13.3 tensordict==0.13.0 daqp==0.8.7

# 4. 설치 버전 검증
python -c "
import numpy as np, torchrl, tensordict, robosuite, gymnasium, numba
print('numpy     :', np.__version__, '(1.26.4 필수)')
print('gymnasium :', gymnasium.__version__, '(1.2.2 필수)')
print('torchrl   :', torchrl.__version__, '(0.13.3 필수)')
print('tensordict:', tensordict.__version__, '(0.13.0 필수)')
print('robosuite :', robosuite.__version__, robosuite.__file__)
"
```
- **요구 확인 조건**:
  - `numpy` == `1.26.4` (NumPy 2.x가 아닌 1.26.4 필수)
  - `gymnasium` == `1.2.2`
  - `torchrl` $\ge$ `0.13.3`
  - `tensordict` $\ge$ `0.13.0`
  - `robosuite` == `1.4.0` (경로가 `/home/moai/ysl_ws/cover/deps/robosuite/...` 이어야 함)

---

### Step 3. 현재 200 데모 실험 쉘 스크립트 재실행
현재 Pobi에서 실행 중인 200 데모 학습(`1d8q7w4r`)을 실행했던 **동일한 쉘 스크립트**를 다시 실행합니다.

1. **현재 실행 스크립트 확인**:
   - 현재 실행 중인 프로세스의 커맨드라인 또는 스크립트를 확인합니다:
     ```bash
     ps aux | grep train_residual_td3
     ```
   - 당시 Pobi에서 실행했던 원본 쉘 스크립트 파일이 있다면 해당 스크립트를 실행합니다.

2. **200 데모 전용 쉘 스크립트 직접 실행**:
   - 동일한 조건(200 데모, `weighted_time`, seed 42)으로 준비된 레포 공식 스크립트를 실행합니다:
     ```bash
     bash resfit/shell_paper/test/run_square_pbrs_no_mask_nstep_weighted_time_ID_preward0.1_200demos_1000k.sh \
         --wandb_name "Square_reward_pbrs_no_mask_nstep_weighted_time_beta1.0_scale0.1_200demos_1000k_pobi_reunified_seed42"
     ```
   *(참고: 별도 인자 없이 스크립트만 실행해도 기본값으로 200 데모, `Square_200` E2C, `weighted_time`, `seed=42`가 모두 세팅되어 실행됩니다.)*

---

### Step 4. 검증 판정 기준 (Evaluation & Diagnosis)
재실험 진행 중 WandB 로그에서 기존 실행(`1d8q7w4r`)과 다음 항목을 비교합니다:

1. **Case A: `lane/rem_t_wrist_next_avg`가 8.5 수준으로 감소하고 `min_dist_wrist_next_avg`가 0.04 수준으로 급감하는 경우**:
   - $\implies$ 이전의 15.34 정체는 `numpy 2.x` 및 `robosuite 1.4.1` 등의 **라이브러리 버전 불일치로 인한 것임이 최종 입증**됨.
2. **Case B: 여전히 15 부근을 유지하면서 성공률이 94%+로 빠르게 치솟는 경우**:
   - $\implies$ 200 데모의 경우 시연 데이터가 4배 촘촘하여 작업 공간 전역에서 기준 데모가 가깝게 매칭되는 **200 데모 고유의 정상적인 잠재 공간 특성임이 입증**됨.

어느 결과가 나오든, 현재 사용자분께서 갖고 계신 의문에 대해 가장 명확하고 결정적인 근거를 확보할 수 있습니다.
