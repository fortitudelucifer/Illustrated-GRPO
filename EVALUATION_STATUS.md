# Evaluation Status

> Last audited: 2026-10-05
>
> Applies to the Stage 4 arithmetic experiments and derived figures.

## Current interpretation

The saved Stage 4 accuracy values are **legacy exploratory observations under the original protocol**. They remain useful for reconstructing the experiment history and training dynamics, but they are not confirmed held-out generalization effects.

The reported two-digit multiplication difference (`67.44% → 73.80%`, `+6.36` percentage points) must not be described as statistically significant or as a validated generalization gain.

## Verified protocol limitations

### Train/evaluation overlap

The multiplication training script generates 1,000 operand pairs with `seed=42`. The evaluation script generates 500 pairs for each of `[42, 123, 456, 789, 999]` with the same pair-generation sequence.

Reproducing that logic gives:

| Quantity | Value |
|---|---:|
| Unique ordered evaluation pairs | 2,149 |
| Evaluation rows overlapping ordered training pairs | 722/2,500 |
| Evaluation rows overlapping training modulo commutativity | 892/2,500 |
| Seed-42 evaluation rows identical to the training prefix | 500/500 |

The multiplication evaluation is therefore not a disjoint held-out test. The Stage 4 addition protocols also use training seed 42 and include a seed-42 evaluation partition generated in the same order.

### Meaning of the five seeds

The five values are **question-generation seeds evaluated against one trained checkpoint per configuration**. They are evaluation partitions, not five independent training runs. Positive differences in all five multiplication partitions do not establish robustness to training randomness.

### Statistical analysis

The evaluation script reports separate Wilson intervals for the Base and Trained marginal accuracies. It does not save per-question paired predictions or run McNemar's test, a paired bootstrap, or an equivalent paired comparison. Marginal CI non-overlap is therefore not reported as a computed `p < 0.05` result.

### Full-FT and LoRA comparison

The six-digit Full-FT and LoRA runs also differ in learning rate, KL coefficient, warmup, sampling temperature, and gradient clipping. Their trajectories are a descriptive comparison between configurations, not an isolated LoRA ablation.

## What the repository still supports

- **Training logs and reward/KL/gradient trajectories** remain valid descriptive evidence of what happened in the saved runs.
- **Base/Trained before-after values** remain legacy descriptive observations under the original protocol.
- **Held-out generalization** is not currently established.
- **Statistical significance of Base/Trained differences** is not currently established.
- **A single-factor causal effect of LoRA** is not currently established because multiple settings changed together.

## Evidence labels

- **Observed/descriptive result**: a value reproduced from the saved original protocol.
- **Preliminary legacy result**: an observed value with a known protocol limitation.
- **Confirmed held-out generalization**: reserved for disjoint splits, independent training replications, saved paired predictions, and a prespecified analysis.

No current Stage 4 result has the third label. A disjoint-split, independently replicated evaluation is planned but has not yet been completed.
