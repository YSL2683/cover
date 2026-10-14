# ResFiT Square Task Final Evaluation Shell Scripts (`shell_final`)

Square Task의 전체 논문 실험(Exp 1 ID, Exp 2 OOD, Exp 3 Ablation)을 위한 독립 실행형 쉘 스크립트 모음입니다.
각 run은 독립된 1개의 쉘 스크립트로 분리되어 있으며, PC 1과 PC 2에서 분산 실행할 수 있도록 큐 스크립트가 구성되어 있습니다.

---

## 1. 디렉토리 구조 (Directory Structure)

```
resfit/shell_final/square/
├── exp1_id/                             # [실험 1] In-Distribution 성능 평가
│   ├── weak_50d/                        # Weak Prior (50 demos BC Policy)
│   │   ├── run_square_id_weak_resfit_seed{42,52,62}.sh
│   │   ├── run_square_id_weak_unified_seed{42,52,62}.sh
│   │   └── run_square_id_weak_decoupled_seed{42,52,62}.sh
│   └── strong_200d/                      # Strong Prior (200 demos BC Policy)
│       ├── run_square_id_strong_resfit_seed{42,52,62}.sh
│       ├── run_square_id_strong_unified_seed{42,52,62}.sh
│       └── run_square_id_strong_decoupled_seed{42,52,62}.sh
│
├── exp2_ood/                            # [실험 2] Out-of-Distribution (Position) 평가
│   ├── weak_50d/                        # Weak Prior (50 demos BC Policy)
│   │   ├── run_square_ood_weak_resfit_seed{42,52,62}.sh
│   │   ├── run_square_ood_weak_unified_seed{42,52,62}.sh
│   │   └── run_square_ood_weak_decoupled_seed{42,52,62}.sh
│   └── strong_200d/                      # Strong Prior (200 demos BC Policy)
│       ├── run_square_ood_strong_resfit_seed{42,52,62}.sh
│       ├── run_square_ood_strong_unified_seed{42,52,62}.sh
│       └── run_square_ood_strong_decoupled_seed{42,52,62}.sh
│
├── exp3_ablation/                       # [실험 3] 보상 구성 요소 Ablation Study (Weak ID)
│   ├── run_square_ablation_weak_no_pbrs_seed{42,52,62}.sh        # w/o PBRS (단순 유사도 잠재 보상)
│   ├── run_square_ablation_weak_no_action_reg_seed{42,52,62}.sh  # w/o Action L2 Regularization (reg=0.0)
│   └── run_square_ablation_weak_no_time_weight_seed{42,52,62}.sh # w/o Similarity-Weighted Time Decay
│
└── queues/                              # 배치 실행 큐 스크립트 (Queue Runners)
    ├── queue_pc1_exp1_id.sh             # PC 1 전용: Exp 1 ID 전체 순차 실행 (18 runs)
    ├── queue_pc1_exp1_id_weak.sh        # PC 1: Exp 1 ID Weak Prior 9 runs
    ├── queue_pc1_exp1_id_strong.sh      # PC 1: Exp 1 ID Strong Prior 9 runs
    ├── queue_pc2_exp2_ood.sh            # PC 2 전용: Exp 2 OOD 전체 순차 실행 (18 runs)
    ├── queue_pc2_exp2_ood_weak.sh       # PC 2: Exp 2 OOD Weak Prior 9 runs
    ├── queue_pc2_exp2_ood_strong.sh     # PC 2: Exp 2 OOD Strong Prior 9 runs
    └── queue_exp3_ablation.sh           # Exp 3 Ablation 9 runs
```

---

## 2. 공통 하이퍼파라미터 및 규격 (Global Specifications)

- **Total RL Timesteps**: `1,000,000` steps (`algo.total_timesteps=1000000`)
- **Evaluation Interval**: `10,000` steps (`eval_interval_every_steps=10000`)
- **Diffusion DDIM Steps**: `20` (`base_policy.diffusion_ddim_steps=20`)
- **Residual Action Scale**: `0.1` (`agent.actor.action_scale=0.1`)
- **Camera Views**: `['observation.images.agentview', 'observation.images.robot0_eye_in_hand']`
- **WandB Project**: `paper_square`
- **Seeds**: `[42, 52, 62]`

---

## 3. 방법론별 파라미터 세부사항 (Method Specifications)

| 구분 | 알고리즘 / 모델 | Reward Type | `p_reward` | Action L2 Reg | E2C 모델 경로 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Vanilla ResFiT** | Sparse RL | `none` | `0.0` | `0.0005` | N/A |
| **Unified PBRS** | Unified Latent PBRS | `reward_pbrs_unified_no_mask_nstep` | `0.1` | `0.00005` | `lane/pretrained_e2c_unified/square[_200]` |
| **Decoupled PBRS (Ours)** | Decoupled Latent PBRS | `reward_pbrs_no_mask_nstep_weighted_time` | `0.1` | `0.00005` | `lane/pretrained_e2c/square_z8[_200]` (`z_dim=8`) |

### [Exp 3] Ablation 파라미터 (Weak Base 50d)
1. **w/o PBRS Formulation**:
   - `algo.reward_type="reward_similarity_potential"`
   - `algo.p_reward=0.01` (누적 발산 방지를 위한 스케일링)
2. **w/o Action Regularization**:
   - `agent.actor.action_l2_reg_weight=0.0`
3. **w/o Time-Aware Decay**:
   - `algo.reward_type="reward_pbrs_no_mask_nstep"`
   - `algo.p_reward=0.1`

---

## 4. 실행 방법 (How to Run)

### 1) 단일 Run 개별 실행
필요한 실험의 스크립트를 직접 실행합니다:
```bash
# 예시: In-Distribution Weak Decoupled seed 42 실행
./resfit/shell_final/square/exp1_id/weak_50d/run_square_id_weak_decoupled_seed42.sh
```

### 2) PC 1 / PC 2 분산 큐 실행

#### PC 1 (In-Distribution Task 전체 - 18 runs):
```bash
./resfit/shell_final/square/queues/queue_pc1_exp1_id.sh
```

#### PC 2 (Out-of-Distribution Task 전체 - 18 runs):
```bash
./resfit/shell_final/square/queues/queue_pc2_exp2_ood.sh
```

#### Ablation Study (9 runs):
```bash
./resfit/shell_final/square/queues/queue_exp3_ablation.sh
```

---

## 5. 특징 및 주의사항 (Key Highlights)

1. **완전한 머신 호환성**: 스크립트 실행 위치 및 Conda 경로 (`/home/moai/` 및 `/home/ysl2683/`)를 자동 감지하므로 어떤 PC에서도 별도 수정 없이 실행 가능합니다.
2. **독립된 Scratch 캐시**: 동시 실행 시 GPU/Memory contention 및 캐시 충돌을 방지하기 위해 run마다 독립된 `CACHE_DIR`을 생성합니다.
3. **학습 중단 방지 규칙 준수**: 스크립트는 준비만 완료되어 있으며, 사용자의 직접 명령 전까지 자동 실행되지 않습니다.
