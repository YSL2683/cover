# Latent Space Visualization: Base Policy vs. Residual RL Policy

이 폴더(`visualization/`)는 학습이 완료된 베이스 정책(Diffusion/ACT)과 잔차 강화학습(Residual TD3) 모델의 잠재 공간(Latent Space)을 비교·시각화하기 위한 사후 분석 도구입니다.

---

## 1. 시각화 방식 요약

- **배경 (Background)**: 베이스 정책 단독 실행 시 수집된 **50개 성공 궤적**의 잠재 공간 2D PCA 투영 및 가우시안 커널 밀도 추정(Gaussian KDE) **주황색 등고선 (`cm.Oranges`)**.
- **전경 (Foreground)**: 잔차 RL이 결합된 전체 정책의 **50개 성공 궤적**을 동일한 2D PCA 평면에 점(Scatter)으로 투영.
- **Cross-View 유사도 컬러링**: 주황색 등고선과 명확히 대비되는 **Deep Blue 단일 색조 그라디언트** (`#80d8ff` 연한 하늘색 $\to$ `#081b4b` 짙은 네이비)를 적용:
  - **Main Camera 패널**: 손목 카메라 관점의 유사도 $S_{\text{wrist}}(s)$로 채색.
  - **Wrist Camera 패널**: 메인 카메라 관점의 유사도 $S_{\text{main}}(s)$로 채색.
- **마커 제거**: 이전 시각화에 포함되었던 시작점($\triangle$) 및 목표점($\star$) 마커를 완전히 제거하여 궤적의 순수한 분포와 흐름만 나타납니다.

---

## 2. 파일 구성

- **`visualize_base_vs_residual.py`**: 롤아웃 수집, 잠재 벡터 추출, 캐싱, PCA/KDE 계산 및 플롯 생성을 담당하는 파이썬 스크립트.
- **`run_visualize_square_pbrs.sh`**: `Square` 환경 및 지정 모델(`Square_reward_pbrs_no_mask_nstep_beta1.0_scale0.1`)에 대해 사전 설정된 실행 스크립트.
- **`cache/`**: 수집된 50개 성공 궤적(`base_trajs`, `rl_trajs`)을 `.pt`로 자동 저장하는 디렉토리.
- **`outputs/`**: 최종 렌더링된 고해상도(200 DPI) 이미지(`base_vs_residual_*.png`)가 저장되는 디렉토리.

---

## 3. 실행 방법

### 기본 실행 (50개 성공 궤적 수집 및 시각화)
```bash
bash visualization/run_visualize_square_pbrs.sh
```

### 캐시 활용 (이미 수집된 궤적이 있을 때 렌더링만 2초 만에 재실행)
```bash
bash visualization/run_visualize_square_pbrs.sh --use_cache
```

### Dry-Run 테스트 (학습 중 GPU 리소스 부하 없이 파이프라인 검증)
```bash
bash visualization/run_visualize_square_pbrs.sh --dry_run
```

---

## 4. 다른 태스크 및 체크포인트로 확장하는 방법

명령행 인자를 통해 임의의 체크포인트와 베이스 정책으로 쉽게 교체하여 실행할 수 있습니다:

```bash
python visualization/visualize_base_vs_residual.py \
    --task Can \
    --checkpoint_path resfit/outputs/<YOUR_RUN_DIR>/models/agent_best.pt \
    --base_policy_path resfit/my_lerobot_data/<YOUR_BC_POLICY_DIR> \
    --e2c_dir lane/pretrained_e2c/can \
    --offline_data_name resfit/my_lerobot_data/ankile/robomimic-mh-can-image \
    --num_episodes 50 \
    --output_path visualization/outputs/base_vs_residual_can_50ep.png
```
