# Stage 4 Evaluation Results

> 2026-08-06 | 4090 server | Qwen2.5-1.5B-Instruct + GRPO
> **Revised evaluation: 500 questions × 5 seeds (replaces earlier n=20, single-seed evaluation)**

## Why the Previous Evaluation Was Invalid

The initial evaluation used only 20 questions with a single seed (seed=999), reporting "90% → 100%". This had two critical issues:

1. **Sample size too small**: With n=20, the 95% Wilson CI for 90% is [70.1%, 97.2%] and for 100% is [83.2%, 100%] — heavily overlapping, no statistical significance
2. **Single seed**: Only one test set was tried, providing no variance estimate

The revised evaluation uses 500 questions × 5 seeds = 2,500 total samples per model.

## Evaluation Method

- **Test set**: 500 three-digit addition problems per seed (a,b ∈ [100,999])
- **Seeds**: [42, 123, 456, 789, 999] — 5 independent test sets
- **Decoding**: Greedy (temperature=0, deterministic)
- **Batch size**: 64
- **Max new tokens**: 16
- **Script**: `stage4_eval.py`

## Results

| Model | Accuracy (mean±std) | 95% CI (Wilson) | Correct/Total |
|-------|---------------------|-----------------|---------------|
| Base model | 94.92% ± 1.06% | [93.99%, 95.71%] | 2373/2500 |
| Trained model | 91.72% ± 1.59% | [90.57%, 92.74%] | 2293/2500 |
| **Change** | **-3.20%** | CIs do not overlap | -80/2500 |

**Conclusion**: The trained model is **statificantly worse** than the base model on this task. The 95% confidence intervals do not overlap, confirming this is not due to random variation.

## Per-Seed Detail

| Seed | Base | Trained | Diff |
|------|------|---------|------|
| 42 | 95.80% (479/500) | 91.40% (457/500) | -4.40% |
| 123 | 93.20% (466/500) | 93.40% (467/500) | +0.20% |
| 456 | 95.80% (479/500) | 91.00% (455/500) | -4.80% |
| 789 | 95.00% (475/500) | 89.60% (448/500) | -5.40% |
| 999 | 94.80% (474/500) | 93.20% (466/500) | -1.60% |

4 out of 5 seeds show regression; 1 seed shows essentially no change.

## Analysis: Why Did the Model Regress?

### 1. Base model was already very strong (94.92%)

The 1.5B model already solves 3-digit addition at ~95% accuracy. This leaves almost no room for improvement — the task is too easy for this model size. This is the same lesson from Stage 2: when base accuracy is >90%, GRPO has almost no learning signal because `frac_reward_zero_std` ≈ 1.0 (all answers in a group are correct).

### 2. Training may have caused catastrophic forgetting

The GRPO training with 500 steps may have slightly degraded the model's general arithmetic ability while optimizing for the specific reward function. The KL penalty (beta=0.1) may not have been sufficient to prevent this.

### 3. The n=20 "90%→100%" result was pure noise

Seed 999 happened to be the seed with the smallest regression (-1.6%). With only 20 questions, the sampling variance was large enough to flip a -1.6% regression into an apparent +10% improvement. This demonstrates why small-sample evaluations are misleading.

## Comparison with Stage 2 (0.5B model)

| | Stage 2 (0.5B) | Stage 4 (1.5B) |
|--|----------------|----------------|
| Base accuracy | ~50% | ~95% |
| After training | ~80% | ~92% |
| Change | +30% | **-3%** |
| Task difficulty | Appropriate (30-70% range) | Too easy (>90% range) |
| Learning signal | Good (zero_std < 70%) | Almost none (zero_std ≈ 100%) |

## Lessons Learned

1. **Always evaluate with sufficient sample size** (n≥200) and multiple seeds
2. **Task difficulty must match model level** — GRPO needs base accuracy in 30%-70% range
3. **A 1.5B model already does 3-digit addition at 95%** — need harder tasks (4-digit, multiplication, word problems) to see GRPO improvement
4. **Small-sample evaluations can flip regression into apparent improvement** — this is why statistical rigor matters

## File Reference

| File | Description |
|------|-------------|
| `output/eval_results_500.json` | Complete evaluation results (config, per-seed, summary) |
| `output/stage4_runs/` | TensorBoard logs from training |
| `output/stage4_trainer_state.json` | Full 500-step training history |
