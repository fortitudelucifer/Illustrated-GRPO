# Experiment Report: GRPO on Multiplication with Qwen2.5-1.5B

> **Date**: 2026-08-06
> **Model**: Qwen2.5-1.5B-Instruct
> **Task**: Integer multiplication
> **Hardware**: RTX 4090 (48GB), Docker (CUDA 12.4.1)
> **Framework**: TRL 1.9.2, Transformers 5.14.1, PEFT, PyTorch 2.11.0
> **Status**: Base accuracy probing — determining the right difficulty level

---

## 1. Motivation

Four addition experiments (3-digit, 5-digit, 6-digit full FT, 6-digit LoRA) all suffered from insufficient learning signal because the base model's addition accuracy was too high (81-95%). GRPO requires base accuracy in the **30-60% sweet spot** where groups frequently have mixed correct/incorrect answers.

Multiplication is fundamentally harder for LLMs than addition:
- Addition is linear (digit-by-digit with carry) — LLMs learn the pattern and generalize
- Multiplication requires intermediate products + multi-row addition + carry propagation — a multi-step algorithm that 1.5B models don't reliably perform

## 2. Base Accuracy Probing

### 2.1 3-digit × 3-digit multiplication (a, b ∈ [100, 999])

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 42 | 4.00% | 8/200 |
| 123 | 4.50% | 9/200 |
| 456 | 5.50% | 11/200 |
| 789 | 5.00% | 10/200 |
| 999 | 5.50% | 11/200 |
| **Mean** | **4.90% ± 0.65%** | **49/1000** |

**Verdict: Too difficult.** At 4.9% accuracy, virtually all 8 answers in a group will be wrong (`zero_std ≈ 1.0`), giving GRPO no learning signal. This is the mirror image of the addition problem — instead of "all correct, no signal," it's "all wrong, no signal."

### 2.2 Difficulty scaling

| Task | a range | b range | Max result | Base accuracy | In sweet spot? |
|------|---------|---------|------------|--------------|----------------|
| 3-digit × 3-digit | [100,999] | [100,999] | 998001 (6 digits) | 4.90% (measured) | No — too low |
| **2-digit × 2-digit** | **[10,99]** | **[10,99]** | **9801 (4 digits)** | **63.80% (measured)** | **Yes — in sweet spot** |
| 1-digit × 3-digit | [2,9] | [100,999] | 8991 (4 digits) | not tested | — |
| 1-digit × 2-digit | [2,9] | [10,99] | 891 (3 digits) | not tested | — |

### 2.3 2-digit × 2-digit results

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 42 | 62.00% | 124/200 |
| 123 | 59.50% | 119/200 |
| 456 | 64.00% | 128/200 |
| 789 | 68.50% | 137/200 |
| 999 | 65.00% | 130/200 |
| **Mean** | **63.80% ± 3.37%** | **638/1000** |

**Verdict: In the GRPO sweet spot.** At 63.8% accuracy, roughly 5/8 answers in a group will be correct and 3/8 will be wrong. This gives GRPO strong learning signal — `zero_std` should be well below 50%, meaning most steps will produce useful gradients.

This is the first task where the base accuracy is genuinely in the ideal range for GRPO. Combined with the LoRA configuration that prevents catastrophic forgetting, this experiment has the best chance of showing meaningful improvement.

## 3. Why Multiplication Is the Right Task Family

### 3.1 LLMs can't multiply

Unlike addition, multiplication is not a simple sequential algorithm for LLMs. It requires:
1. Computing partial products (e.g., 34 × 12 → 34×2=68, 34×10=340)
2. Shifting and adding partial products (68 + 340 = 408)
3. Managing carries across multiple intermediate steps

A 1.5B model trained on general text has seen multiplication tables but hasn't internalized the algorithm. This means there's genuine room for improvement via RL — the model can learn to be more systematic.

### 3.2 Reward function compatibility

The existing reward function (`correctness_reward`) works unchanged for multiplication:
- Extract the last number from the response
- Compare to the ground truth (a × b)
- Reward 1.0 for correct, 0.0 for incorrect

Only `format_reward` needs a minor update (digit count range for the format bonus).

### 3.3 Output length

| Task | Max output digits | Max new tokens needed |
|------|-------------------|----------------------|
| 6-digit addition | 7 | 16 |
| 3-digit × 3-digit | 6 | 32 |
| 2-digit × 2-digit | 4 | 16 |
| 1-digit × 3-digit | 4 | 16 |

2-digit × 2-digit and 1-digit × 3-digit both have short outputs, keeping `max_completion_length` small and inference fast.

## 4. Experimental Plan

### Phase 1: Probe base accuracy (complete)
- [x] 3-digit × 3-digit: 4.90% (too low)
- [x] 2-digit × 2-digit: 63.80% (in sweet spot — selected for training)

### Phase 2: Train with LoRA GRPO
Once we find a difficulty level with 30-60% base accuracy:
- Use the LoRA configuration from the successful 6-digit addition experiment
- r=32, alpha=64, 7 target modules, lr=1e-5, beta=0.04, temp=0.9
- 500 steps initial, extend to 1000 if improvement is ongoing
- Evaluate with 500 questions × 5 seeds

### Phase 3: Evaluate and report
- Same rigorous evaluation protocol (500q × 5 seeds, Wilson CI)
- Compare base vs trained accuracy
- Analyze training stability (collapse check, KL, gradient norms)

## 5. Artifacts

| File | Description |
|------|-------------|
| `experiment_report_3digit.md` | 3-digit addition full FT failure report |
| `experiment_report_6digit_lora.md` | 6-digit addition LoRA success report (+2.00%) |
| `output/eval_results_*.json` | All evaluation result files |

---

*This report is a living document. It will be updated as base accuracy probing continues and training experiments are conducted.*
