# Experiment Report: GRPO on 3-Digit Addition with Qwen2.5-1.5B

> **Date**: 2026-08-06
> **Model**: Qwen2.5-1.5B-Instruct
> **Task**: 3-digit integer addition (a, b ∈ [100, 999])
> **Hardware**: RTX 4090 (48GB), Docker (CUDA 12.4.1)
> **Framework**: TRL 1.9.2, Transformers 5.14.1, PyTorch 2.11.0

---

## 1. Objective

Train Qwen2.5-1.5B-Instruct on 3-digit addition using GRPO to improve arithmetic accuracy.

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Group size (G) | 8 |
| Temperature | 0.8 |
| Learning rate | 5e-6 |
| Max steps | 500 |
| Warmup steps | 30 |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Beta (KL penalty) | 0.1 |
| Epsilon (PPO clip) | 0.2 |
| Max grad norm | 0.5 |
| Max completion length | 32 |
| Reward functions | correctness (1.0) + format (0.2) |
| vLLM | Disabled (version incompatibility) |
| Precision | bf16 |

## 3. Evaluation Method

### Initial (flawed) evaluation

- 20 questions, single seed (seed=999), greedy decoding
- Result reported: 90% → 100% (+10%)

### Revised (rigorous) evaluation

- 500 questions per seed, 5 seeds [42, 123, 456, 789, 999]
- Total: 2,500 samples per model
- Greedy decoding, batch size 64, max new tokens 16
- Reports mean ± std with 95% Wilson confidence intervals

## 4. Results

### 4.1 Revised Evaluation (500 × 5)

| Model | Accuracy (mean±std) | 95% CI (Wilson) | Correct/Total |
|-------|---------------------|-----------------|---------------|
| Base model | 94.92% ± 1.06% | [93.99%, 95.71%] | 2373/2500 |
| Trained model | 91.72% ± 1.59% | [90.57%, 92.74%] | 2293/2500 |
| **Change** | **-3.20%** | **CIs do not overlap** | -80/2500 |

**The trained model is statistically significantly worse than the base model.**

### 4.2 Per-Seed Breakdown

| Seed | Base | Trained | Diff |
|------|------|---------|------|
| 42 | 95.80% (479/500) | 91.40% (457/500) | -4.40% |
| 123 | 93.20% (466/500) | 93.40% (467/500) | +0.20% |
| 456 | 95.80% (479/500) | 91.00% (455/500) | -4.80% |
| 789 | 95.00% (475/500) | 89.60% (448/500) | -5.40% |
| 999 | 94.80% (474/500) | 93.20% (466/500) | -1.60% |

4 out of 5 seeds show regression. The single seed (999) used in the initial evaluation happened to have the smallest regression (-1.6%), which was flipped into an apparent +10% improvement by the small sample size (n=20).

### 4.3 Why the n=20 Result Was Misleading

With n=20, the 95% Wilson confidence intervals are extremely wide:

| Proportion | 95% CI | Width |
|------------|--------|-------|
| 18/20 (90%) | [70.1%, 97.2%] | ±13.6% |
| 20/20 (100%) | [83.2%, 100%] | ±8.4% |

These intervals heavily overlap. The "improvement" from 90% to 100% (2 additional correct answers out of 20) is well within sampling noise.

## 5. Training Process Analysis

### 5.1 Phase Breakdown

| Phase | Steps | Avg Reward | Avg KL | Avg Entropy | Avg Grad | zero_std% | Non-perfect steps |
|-------|-------|-----------|--------|-------------|----------|-----------|-------------------|
| Warmup | 1-50 | 0.90 | 0.78 | 0.075 | 26.9 | 78% | 20/50 |
| **Collapse** | **51-100** | **0.53** | **2.96** | **0.332** | **70.9** | **54%** | **38/50** |
| Recovery | 101-200 | 0.89 | 1.09 | 0.249 | 16.9 | 71% | 46/100 |
| Stabilizing | 201-300 | 0.97 | 0.82 | 0.081 | 7.5 | 83% | 35/100 |
| Converging | 301-400 | 1.14 | 0.12 | 0.028 | 4.7 | 94% | 10/100 |
| Final | 401-500 | 1.09 | 0.40 | 0.044 | 4.0 | 95% | 14/100 |

### 5.2 The Collapse (Steps 51-100)

The critical event was a model collapse during steps 51-100:

- **Reward dropped from 0.90 to 0.53** (-41%)
- **KL spiked from 0.78 to 2.96** (max 11.03)
- **Entropy surged from 0.075 to 0.332** (model became uncertain)
- **4 steps had reward=0** (entire group wrong): steps 46, 47, 48, 71
- **Gradient maxed at 313.8** (step 82), with 24 steps having gradient > 100

Root cause: During warmup (steps 1-50), the model occasionally encountered problems it couldn't solve. These triggered massive gradients (200-313), which `max_grad_norm=0.5` truncated in magnitude but not direction. Accumulated directional updates pushed the model away from its original distribution, causing catastrophic forgetting.

### 5.3 Key Gradient Explosion Events

| Step | Raw Gradient | Reward | KL | What Happened |
|------|-------------|--------|-----|---------------|
| 17 | 266.5 | 0.95 | 0.0 | First wrong answer, large update |
| 31 | 302.3 | 0.30 | 3.22 | Model starts collapsing |
| 70 | 244.2 | 0.09 | 9.44 | Worst step: reward near 0, KL near 10 |
| 82 | 313.8 | 0.11 | 6.98 | Maximum gradient: model almost entirely wrong |

### 5.4 The Recovery Was Not a True Recovery

Although reward returned to 1.2 (perfect) by step 300+, this was deceptive:

1. **Training reward ≠ test accuracy**: Training generates random problems each step; the model may have memorized patterns rather than generalized
2. **KL history was severe**: Max KL = 11.03 means the model's internal representations were fundamentally altered
3. **zero_std = 95% in final phase**: The model answers all 8 group members correctly on training problems, but this is because the task is too easy (95% base accuracy), not because GRPO improved anything

## 6. Root Cause Analysis

### Primary cause: Task too easy for the model

The 1.5B model already solves 3-digit addition at 94.92% accuracy. This leaves almost no room for GRPO improvement:

- `frac_reward_zero_std` averaged 82% across all 500 steps (78% of the time, all 8 answers in a group are correct or all wrong)
- Only 91/500 steps had any learning signal (zero_std < 0.99)
- When the model does encounter a wrong answer, the gradient is enormous (200+) because such events are rare and the model is far from the optimal direction

### Secondary cause: Insufficient gradient control

- `max_grad_norm=0.5` truncated gradients from 300+ to 0.5, but the direction was still destructive
- `beta=0.1` (KL penalty) was insufficient to prevent the KL from reaching 11.03
- Full-parameter fine-tuning meant every weight was susceptible to drift

### Why the initial evaluation was misleading

- n=20 has ±10-20% confidence intervals — any result within that range is noise
- seed=999 happened to show the smallest regression (-1.6%), which n=20 flipped to +10%
- A single seed provides no variance estimate

## 7. Comparison with Stage 2 (0.5B model)

| Dimension | Stage 2 (0.5B, 3-digit) | Stage 4 (1.5B, 3-digit) |
|-----------|------------------------|------------------------|
| Base accuracy | ~50% | ~95% |
| After training | ~80% | ~92% |
| Change | +30% | **-3%** |
| Task difficulty | Appropriate (30-70% range) | Too easy (>90% range) |
| Learning signal | Good (zero_std < 70%) | Almost none (zero_std ≈ 82%) |
| Gradient stability | Moderate | Poor (max 313.8) |
| Evaluation rigor | n=20, 1 seed | n=500, 5 seeds |

## 8. Lessons Learned

1. **Statistical rigor is non-negotiable**: n=20 with 1 seed can flip a -3% regression into a +10% apparent improvement. Always use n≥200 with multiple seeds.

2. **Task difficulty must match model level**: GRPO requires base accuracy in the 30-70% range. At 95%, there is no learning signal — the model is already correct, and rare wrong answers trigger destructive gradients.

3. **High base accuracy + full fine-tuning = catastrophic forgetting risk**: When the model is already good, any gradient update can only make it worse. The rare wrong-answer events produce enormous gradients that push the model away from its optimal state.

4. **Training reward is not test accuracy**: The model reached reward=1.2 (perfect) on training, but test accuracy dropped. This is because training problems are randomly generated and the task is easy — "getting them right" doesn't mean the model improved.

5. **Gradient clipping prevents magnitude but not direction**: `max_grad_norm=0.5` limits step size but the accumulated direction over many steps can still be destructive.

## 9. Recommendations for Next Experiment

### Option A: Harder task (recommended)

Change to 5-digit addition (a, b ∈ [10000, 99999]) to bring base accuracy into the 30-70% range where GRPO is effective.

### Option B: LoRA fine-tuning

Use LoRA adapters instead of full fine-tuning to freeze base weights and prevent catastrophic forgetting.

### Option C: More conservative parameters

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| learning_rate | 5e-6 | 1e-6 |
| max_grad_norm | 0.5 | 0.1 |
| beta (KL) | 0.1 | 0.3 |
| warmup_steps | 30 | 50 |

## 10. Artifacts

| File | Description |
|------|-------------|
| `output/eval_results_500.json` | Full evaluation results (per-seed, summary, CI) |
| `output/stage4_trainer_state.json` | Complete 500-step training log (all metrics per step) |
| `output/stage4_runs/` | TensorBoard event files |
| `stage4_eval.py` | Evaluation script (500q × 5 seeds, batch inference) |
| `stage4_train.py` | Training script |

---

*This report documents a failed experiment. The failure itself is the most valuable outcome — it demonstrates why statistical rigor matters and why task difficulty must match model capability in GRPO training.*
