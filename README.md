# The Illustrated GRPO: Reproduction & Learning Guide

> **English** | [中文](README-cn.md)

> Paper: [The Illustrated GRPO (Towards AI)](https://towardsai.com/p/l/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation)
> Original algorithm paper: [DeepSeekMath (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300)

---

## 0. Prerequisites for Beginners

> Your math background (calculus, linear algebra, probability & statistics) is more than enough. Below maps what you already know to GRPO concepts.

### 0.1 From Your Math Knowledge to GRPO Concepts

| What You Already Know (Math Class) | GRPO Concept | Simple Explanation |
|-------------------------------------|--------------|---------------------|
| Probability distribution P(X=x) | Policy π(a\|s) | Probability of outputting action a given input s. In LLMs, "action" = next token |
| Expectation E[X], Variance Var(X) | Reward mean and std | Score a group of outputs, compute mean and std — the statistics you already know |
| Standardization (X-μ)/σ | Advantage A_i | Standardize rewards so "better than average" is positive, "worse" is negative |
| Gradient descent ∇f(θ) | Policy gradient | Update model parameters θ along gradient direction — increase good outputs, decrease bad ones |
| KL Divergence D(P‖Q) | KL penalty | Measures difference between two distributions, prevents trained model from drifting too far from original |
| Clipping function clip(x, a, b) | PPO Clipping | Limits update magnitude, prevents training collapse from too-large steps |
| Token | Token | Smallest unit of text an LLM processes, roughly a word or character |
| LoRA | Low-Rank Adaptation | Don't modify all parameters, only train a small low-rank matrix — saves VRAM |
| bf16 | Bfloat16 | 16-bit float, uses half the VRAM of float32 with the same dynamic range |

### 0.2 Understanding GRPO with a Real-Life Analogy

> Imagine a student (model) solving math problems (prompts), and a teacher (reward function) grading them.

```
1. Teacher gives a math problem
2. Student writes 4 different solutions at their current level (group sampling G=4)
3. Teacher grades each: r1=1, r2=0, r3=1, r4=0 (reward computation)
4. Compute class average mean=0.5, std=0.5
5. Each solution's "relative performance" = (own score - average) / std
   → A1=+1 (above average), A2=-1 (below average), A3=+1, A4=-1 (advantage computation)
6. Student reflects: practice good solutions, avoid bad ones (policy update)
7. But don't overfit — problem-solving style shouldn't drift too far from original (KL penalty)
8. Repeat — student gets better and better at this type of problem
```

This is the entire idea behind GRPO. The formulas just describe this process precisely.

### 0.3 Key Terminology Table (Refer Back Anytime)

| Term | English | Understanding via Math Knowledge |
|------|---------|----------------------------------|
| Policy | Policy π_θ | A function parameterized by θ, takes a question, outputs a probability distribution over answers |
| Reward | Reward r | A scalar score — higher is better (like the opposite of a loss function) |
| Advantage | Advantage A | Standardized reward: (r - mean) / std, represents "relative goodness" |
| Value Network | Critic / Value Model | An extra model in PPO that estimates "average score" — GRPO removes it |
| Reference Model | Reference Model π_ref | Frozen original model, used to compute KL divergence, prevents drift |
| Old Policy | Old Policy π_θold | Snapshot of model from last step, used to compute probability ratio, synced periodically |
| Probability Ratio | Ratio π_θ/π_θold | New model's probability of a token / old model's probability of the same token |
| KL Divergence | KL Divergence | D(P‖Q) = Σ P(x) log(P(x)/Q(x)) from probability theory — measures distribution difference |
| Clipping | Clipping | Constrains probability ratio to [1-ε, 1+ε], prevents excessive updates |
| Token | Token | Smallest unit of text an LLM processes, roughly a word or character |
| LoRA | Low-Rank Adaptation | Don't modify all parameters, only train a small low-rank matrix — saves VRAM |
| bf16 | Bfloat16 | 16-bit float, uses half the VRAM of float32 with the same dynamic range |

---

## 1. Hardware Environment

| Machine | GPU | VRAM | Role |
|---------|-----|------|------|
| Local | RTX 5070 Ti | 16GB | Development, debugging, teaching notebooks (Stages 1-3) |
| Headless server | RTX 4090 | 48GB | Formal training (Stages 4-6, Docker isolated) |

## 2. Model Files

| Model | Local Path | 4090 Path | Size | Usage |
|-------|-----------|-----------|------|-------|
| Qwen2.5-0.5B-Instruct | `/data/models/Qwen2.5-0.5B-Instruct` | — | 954MB | Stage 2: local teaching training |
| Qwen2.5-1.5B-Instruct | `/data/models/Qwen2.5-1.5B-Instruct` | `/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct` | 2.9GB | Stage 4: 4090 advanced training |
| Qwen2.5-3B-Instruct | `/data/models/Qwen2.5-3B-Instruct` | — | 5.8GB | Stage 5: 4090 larger model experiments |

## 3. Environment Management Strategy

- **Stages 1-3 (Learning & Exploration)**: Conda environment, flexible package installation
- **Stages 4-6 (Formal Training)**: Docker container, isolated and reproducible, doesn't pollute the 4090 server

## 4. GRPO Algorithm Deep Dive (From Intuition to Formulas)

### 4.1 Step 1: Understanding "What is Policy Gradient" (via Probability Theory)

From probability theory: if X ~ P(x) and you want to maximize E[f(X)], use gradient ascent:

```
∇_θ E[f(X)] = E[f(X) * ∇_θ log P(X)]
```

In GRPO:
- P(X) → Policy π_θ (probability of model outputting some text)
- f(X) → Advantage A (how much better this text is than average)
- Gradient ascent → increase probability of good answers, decrease bad ones

**In one sentence**: Policy gradient = "do good behaviors more, bad behaviors less", implemented via gradient ascent.

### 4.2 Step 2: Understanding "Why Advantage" (via Statistical Standardization)

What's wrong with using raw reward r for gradient ascent? If all answers get positive scores, the model blindly increases all probabilities — no discrimination.

The solution is **standardization** (which you know):

```
A_i = (r_i - mean) / std
```

- r_i > mean → A_i > 0 → increase this answer's probability
- r_i < mean → A_i < 0 → decrease this answer's probability
- Dividing by std → removes scale effects, comparable across different problems

**In one sentence**: Advantage = standardized reward, the z-score you learned in probability theory.

### 4.3 Step 3: Understanding "Why Clipping" (Preventing Excessive Updates)

If a single update step is too aggressive, the model can collapse (like "variance explosion" in probability).

PPO's solution: compute the probability ratio r = π_new/π_old, clip it to [1-ε, 1+ε]:

```
clip(r, 1-ε, 1+ε)  →  e.g. ε=0.2, ratio constrained to [0.8, 1.2]
```

**In one sentence**: Clipping = step-size protection, prevents the model from going too far in one step, similar to learning rate decay.

### 4.4 Step 4: Understanding "Why KL Penalty" (Preventing Forgetting)

After training, the model might learn "shortcuts" — e.g., only outputting specific formats to score, losing language ability.

KL divergence (from your probability course) measures distribution difference:

```
KL(π_θ || π_ref) = Σ π_θ(x) * log(π_θ(x) / π_ref(x))
```

Subtracting β * KL from the loss forces the new model not to drift too far from the reference model.

**In one sentence**: KL penalty = "don't learn the wrong things", quantified via KL divergence.

### 4.5 Complete GRPO Objective Function (Now You Can Read It)

```
J_GRPO(θ) = E[ 1/G * Σ min( π_θ/π_θold * A_i, clip(π_θ/π_θold, 1-ε, 1+ε) * A_i ) - β * KL(π_θ || π_ref) ]
```

Term-by-term breakdown:

| Formula Part | Meaning | Your Math Knowledge |
|--------------|---------|---------------------|
| E[...] | Expectation (average over all prompts) | E[X] from probability |
| 1/G * Σ | Average over G outputs | Sum divided by count |
| π_θ/π_θold | New/old policy probability ratio | Ratio of two probabilities |
| A_i | Advantage of i-th output | Standardized z-score |
| min(..., clip(...)) | Clipped policy gradient | Take smaller value = more conservative update |
| β * KL(...) | KL divergence penalty | D(P‖Q) from probability theory |

### 4.6 Three Model Roles

| Model | Symbol | Role | Analogy |
|-------|--------|------|---------|
| Policy Model | π_θ | The model being trained, parameters updated | Student currently learning |
| Old Policy | π_θold | Frozen parameters for advantage computation, synced periodically | Student's level from last exam |
| Reference Model | π_ref | Baseline for KL penalty, prevents drift | Student's "original personality", don't overfit |

### 4.7 GRPO vs PPO: Why Remove the Value Network

| Dimension | PPO | GRPO | Why the Change |
|-----------|-----|------|----------------|
| Value network | Requires critic model | Not needed | Group mean already estimates "average score" — no extra model needed |
| VRAM usage | 2x (policy + critic) | 1x (policy only) | One fewer model = half the VRAM |
| Advantage estimation | GAE + value function | Group mean/std normalization | Direct sample statistics, simpler |
| KL penalty | In reward | In loss | Simplifies computation, advantage unaffected by KL |

**Core insight**: If you've already sampled G outputs and scored them, their mean is a natural estimate of "average score" — no need to train a value network to predict it. This is the essence of GRPO.

---

## 5. Stage-by-Stage Reproduction Roadmap

### Stage 0: Environment Setup (Done ✅)

- [x] Models downloaded to `/data/models`
- [x] Documentation created

### Stage 1: Toy Algorithm Understanding (Local CPU, ~1 hour)

> **You need**: Python basics, numpy operations, probability concepts (mean/std)
> **You don't need**: GPU, deep learning frameworks, any RL knowledge

- **Goal**: Implement every step of GRPO by hand on CPU with a number sorting task
- **Tools**: Conda + Jupyter Notebook
- **Material**: [djemec/rl_grpo_explainer.ipynb](https://github.com/djemec/descriptive_notebooks/blob/main/rl_grpo_explainer.ipynb)
- **You'll learn**:
  - How to implement a simple "policy" with numpy (output probability distribution)
  - How to sample multiple outputs (group sampling)
  - How to compute reward and advantage (just mean and std standardization)
  - How to update policy parameters via gradient
- **Understanding checkpoints**:
  - [ ] Can explain "why divide advantage by standard deviation"
  - [ ] Can explain "why use advantage instead of raw reward"
  - [ ] Can manually compute advantage for a simple example

### Stage 2: Minimal LLM GRPO (Local 5070Ti 16GB, ~2 hours)

> **You need**: Stage 1 conceptual understanding, Python basics
> **You don't need**: Deep Transformer internals (treat it as a black box)

- **Goal**: Run GRPO training with a real LLM, observe TensorBoard curves
- **Model**: `/data/models/Qwen2.5-0.5B-Instruct`
- **Data**: `trl-lib/DeepMath-103K` or `openai/gsm8k`
- **Framework**: TRL GRPOTrainer (a library, just a few lines of code)
- **Key config**: `num_generations=4, max_completion_length=64, bf16=True`
- **You'll learn**:
  - What LLM inputs/outputs look like (prompt → token → probability → text)
  - How TRL library wraps the GRPO algorithm
  - How to visualize training with TensorBoard
- **Understanding checkpoints**:
  - [ ] Can explain `num_generations` parameter (it's group size G)
  - [ ] Can find the reward curve in TensorBoard and explain the trend
  - [ ] Can explain changes in model output before/after training

### Stage 3: Reading Source Code (Local, ~2 hours)

> **You need**: Stage 1-2 conceptual understanding, ability to read Python code
> **You don't need**: Ability to write everything from scratch

- **Goal**: Line-by-line understanding of GRPO implementation
- **Material**: [transparent-grpo](https://github.com/siyuan-harry/transparent-grpo) (~400-line single file)
- **Flow**: `Generate → Reward → Advantage → Update` linear code, read top to bottom
- **You'll learn**:
  - How probability ratio π_θ/π_θold is computed in code (log subtraction)
  - How KL divergence is implemented (Schulman estimator)
  - How clipping works in PyTorch (`torch.clamp`)
- **Understanding checkpoints**:
  - [ ] Can point to which line computes advantage
  - [ ] Can point to which line computes KL divergence
  - [ ] Can explain why log probabilities are used instead of raw probabilities

### Stage 4: Formal Training (Headless 4090, Docker, ~3-5 hours)

> **You need**: Stage 2-3 understanding, basic Docker commands
> **You don't need**: Deep Docker knowledge (complete Dockerfile and commands provided)

- **Goal**: Larger model + larger group + vLLM acceleration, observe more pronounced training effects
- **Model**: `/data/models/Qwen2.5-1.5B-Instruct` or `3B`
- **Environment**: Docker container (nvidia/cuda base image)
- **Key config**: `num_generations=8, max_completion_length=256, use_vllm=True`
- **You'll learn**:
  - How larger group (G=8) stabilizes advantage estimation (statistics: more samples = better estimate)
  - How vLLM accelerates generation (batched inference engine)
  - How Docker isolates environments
- **Understanding checkpoints**:
  - [ ] Can explain why G=8 is more stable than G=4 (sample size vs estimation variance)
  - [ ] Can compare training effects between 0.5B and 1.5B models

### Stage 5: Custom Reward Functions + Comparison Experiments (4090, ~3 hours)

> **You need**: Stage 4 training experience, Python function writing
> **You don't need**: Reward model training knowledge (we use rule-based rewards)

- **Goal**: Implement format_reward + accuracy_reward from the paper, compare different reward combinations
- **Experiment design**:
  - Experiment A: correctness reward only → observe if model learns to answer
  - Experiment B: correctness + format reward → observe if format improves
  - Experiment C: correctness + format + length reward → observe answer length changes
- **You'll learn**:
  - How reward function design affects model behavior
  - How to weight multi-objective rewards
  - "Reward hacking" — model exploiting loopholes
- **Understanding checkpoints**:
  - [ ] Can explain why correctness-only reward may underperform
  - [ ] Can observe at least one reward hacking phenomenon
  - [ ] Can design your own reward function

### Stage 6: Evaluation & Visualization (~1 hour)

> **You need**: Understanding from all previous stages
> **You don't need**: Additional knowledge

- **Goal**: Before/after model output comparison + TensorBoard metric analysis
- **Tools**: TensorBoard, [UNIPO interactive visualization](https://poloclub.github.io/unipo/)
- **You'll learn**:
  - How to scientifically evaluate model improvement (comparison testing)
  - How to interpret abnormal signals in training curves
- **Understanding checkpoints**:
  - [ ] Can use before/after models to answer 10 questions and compare accuracy
  - [ ] Can explain the relationship between reward, KL, and loss curves in TensorBoard

---

## 6. Dependencies

### Stage 1 (Toy Learning)

```bash
conda create -n grpo-learn python=3.10 -y
conda activate grpo-learn
pip install numpy matplotlib jupyter
```

### Stages 2-3 (TRL Training + Source Reading)

```bash
conda create -n grpo-tutorial python=3.10 -y
conda activate grpo-tutorial
pip install torch transformers trl datasets accelerate peft tensorboard
```

### Stages 4-6 (Docker Formal Training)

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3-pip git
RUN pip install torch transformers trl datasets accelerate peft \
    tensorboard vllm deepspeed
WORKDIR /workspace
```

```bash
docker build -t grpo-train .
docker run --gpus all --shm-size 16g -p 6006:6006 -v $(pwd):/workspace -it grpo-train
```

---

## 7. Key Code Templates

### Minimal GRPO Training Script

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

training_args = GRPOConfig(
    output_dir="output",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=True,
    max_completion_length=64,
    num_generations=4,
    logging_steps=10,
    report_to=["tensorboard"],
)

trainer = GRPOTrainer(
    model="/data/models/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("output")
```

### Custom Reward Functions

```python
import re

def format_reward(completions, **kwargs):
    """Reward for <think>...<answer>...</answer> format"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in completions]

def accuracy_reward(completions, answers, **kwargs):
    """Reward for correct answer"""
    rewards = []
    for completion, answer in zip(completions, answers):
        try:
            extracted = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if extracted and extracted.group(1).strip() == str(answer).strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards
```

### TensorBoard Visualization

```bash
tensorboard --logdir output/runs --port 6006
```

Key metrics:
- `reward`: Average reward (should increase)
- `reward_std`: Within-group reward std (reflects exploration)
- `kl`: KL divergence (should stay in reasonable range)
- `loss`: Total loss

---

## 7.5. GRPO Tuning Lessons from Practice

> The following lessons come from multiple rounds of experiments training Qwen2.5-0.5B on addition tasks with TRL on an RTX 5070 Ti (16GB).

### Experiment Environment

- **GPU**: RTX 5070 Ti (16GB)
- **Model**: Qwen2.5-0.5B-Instruct (494M parameters)
- **Framework**: TRL GRPOTrainer
- **Task**: Integer addition ("Answer with just the number")

### Three Training Rounds Comparison

| Round | Task | G | temp | lr | beta | max_steps | batch | accum | Result |
|-------|------|---|------|-----|------|-----------|-------|-------|--------|
| 1st | 2-digit | 4 | 1.0 | 1e-5 | 0.04 | 50 | 4 | 2 | 90%→65% (**regression**) |
| 2nd | 2-digit | 4 | 0.7 | 5e-6 | 0.1 | 300 | 4 | 2 | Almost no learning signal |
| 3rd | 3-digit | 6 | 0.8 | 5e-6 | 0.1 | 300 | 2 | 3 | 50%→80% (**success**) |

### Core Lessons

#### 1. `frac_reward_zero_std` is the Most Important Metric

> **If all G answers in a group are correct or all wrong, advantages are all 0, and the model learns nothing.**

- 2nd round: `frac_reward_zero_std` ≈ 100%, 300 steps essentially wasted
- 3rd round: `frac_reward_zero_std` = 63%, 37% of steps had learning signal, accuracy improved 30%

**The primary tuning goal is to bring `frac_reward_zero_std` down** — otherwise no other parameter matters.

#### 2. Task Difficulty Must Match Model Level

| Task | Base accuracy | Within-group diversity | Effect |
|------|--------------|----------------------|--------|
| 2-digit addition | ~90% | Very low (almost all correct) | No learning |
| 3-digit addition | ~50% | Higher (mix of correct/wrong) | Effective learning |

**Rule of thumb**: GRPO works best when base model accuracy is 30%-70%. Too high (all correct) or too low (all wrong) won't work.

#### 3. Temperature Choice is a Balancing Act

| Temperature | Effect |
|------------|--------|
| 0.7 | Too deterministic, answers nearly identical, high `zero_std` |
| 0.8 | Balanced, some within-group diversity ✅ |
| 1.0 | Too random, model generates long nonsense, entropy spikes |

**Experience**: 0.8 is a good starting point for 0.5B model on addition tasks. Adjust per task.

#### 4. Gradient Clipping is Mandatory

- Without `max_grad_norm`: gradient spikes to 388~524, training unstable
- With `max_grad_norm=0.5`: gradient truncated, but sometimes too aggressive (500+ → 0.5, information loss)

**Experience**: 0.5 is conservative, 1.0 may be better. When unsure, start with 0.5.

#### 5. KL Penalty Needs Dynamic Adjustment

| beta | KL behavior |
|------|-------------|
| 0.04 | KL失控 to 4.3+ |
| 0.1 | KL avg 1.6, max 7.8 (still high) |

**Experience**: If KL consistently >2.0, increase `beta`. 0.1 may not be enough for 0.5B, try 0.2. But too high limits learning.

#### 6. VRAM Management

**Key formula**: Sequences per step = `per_device_train_batch_size × num_generations`

| Config | Sequences | VRAM |
|--------|-----------|------|
| batch=4, G=4 | 16 | ~13 GB |
| batch=2, G=6 | 12 | ~10 GB |

**Tip**: When increasing G, decrease batch proportionally to keep sequence count within VRAM. Use `gradient_accumulation_steps` to compensate effective batch size.

#### 7. TRL Divisibility Constraint

> `generation_batch_size` (= `batch_size × gradient_accumulation_steps`) must be divisible by `num_generations`.

| batch | accum | product | Divisible by G=6? |
|-------|-------|---------|-------------------|
| 2 | 4 | 8 | ❌ Error |
| 2 | 3 | 6 | ✅ |

**Experience**: First determine G, then work backwards to make batch × accum = multiple of G.

### Tuning Decision Flow

```
1. Choose task difficulty → base accuracy in 30%-70%
2. Choose G and temp → get frac_reward_zero_std < 70%
3. Adjust batch and accum → satisfy VRAM + divisibility constraint
4. Set max_grad_norm → prevent gradient explosion
5. Set beta → keep KL below 1.0
6. Set lr and warmup → work with above parameters
7. Run 50 steps, observe logs → confirm learning signal
8. Run full training → compare before/after accuracy
```

### Common Issues Quick Reference

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| `frac_reward_zero_std=1.0` | Task too easy or temperature too low | Harder tasks / increase temp / increase G |
| KL keeps rising | beta too small | Increase beta |
| Gradient explosion | No gradient clipping | Set max_grad_norm=0.5~1.0 |
| OOM | Too many sequences | Reduce batch or G, compensate with accum |
| Reward not increasing | Insufficient learning signal | Check zero_std, adjust task difficulty |
| Answers too long/nonsensical | Temperature too high | Lower temp + limit max_completion_length |
| `ValueError: generation_batch_size` | Divisibility constraint | Adjust accum so batch×accum is divisible by G |

---

## 8. References

| Resource | Link | Description |
|----------|------|-------------|
| Towards AI article | [Link](https://towardsai.com/p/l/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation) | Original article |
| DeepSeekMath paper | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | Original GRPO paper |
| TRL GRPO docs | [HF Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) | Official implementation |
| HF Course Ch12 | [Link](https://huggingface.co/docs/course/main/en/chapter12/4) | GRPO tutorial |
| HF Cookbook | [Link](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) | Complete notebook |
| transparent-grpo | [GitHub](https://github.com/siyuan-harry/transparent-grpo) | ~400-line single-file implementation |
| mini-grpo | [GitHub](https://github.com/JFan5/mini-grpo) | ~500-line pure PyTorch |
| GRPO Explainer | [GitHub](https://github.com/djemec/descriptive_notebooks/blob/main/rl_grpo_explainer.ipynb) | CPU teaching notebook |
| UNIPO visualization | [poloclub.github.io/unipo](https://poloclub.github.io/unipo/) | Interactive algorithm visualization |
| HF Blog: GRPO = PPO without critic | [Link](https://huggingface.co/blog/garg-aayush/derive-grpo-loss) | Step-by-step GRPO derivation |

---

## 9. Progress Tracking

- [x] Stage 0: Environment setup (model download + documentation)
- [x] Stage 1: Toy algorithm understanding (`stage1_toy_grpo.ipynb`)
- [x] Stage 2: Minimal LLM GRPO training (`stage2_trl_grpo.ipynb`)
- [x] Stage 3: Source code reading (`stage3_source_code.ipynb`)
- [x] Stage 4: 4090 formal training (`stage4_notebook.ipynb` + `stage4_train.py` + `stage4_eval.py` + `stage4_deploy.sh` + `Dockerfile`)
- [x] Stage 4: Training records & evaluation results (`output/stage4_runs/` + `output/stage4_trainer_state.json` + `output/stage4_eval_results.md`)
- [x] Complete training summary (`stage4_summary.md`)
- [ ] Stage 5: Custom reward function experiments
- [ ] Stage 6: Evaluation & visualization

---

## 10. Repository Structure

```
illustrated-grpo/
├── README.md                      # Main doc: GRPO theory + hardware + progress (English)
├── README-cn.md                   # Main doc (Chinese)
├── stage4_summary.md              # Complete training summary (parameters/results/changes per run)
├── 4090_agent_上手指南.md          # 4090 server environment guide
├── requirements.txt               # Dependencies
├── Dockerfile                     # Stage 4 Docker image definition
├── .gitignore                     # Excludes model weights/checkpoints
│
├── stage1_toy_grpo.ipynb          # Stage 1: numpy GRPO from scratch
├── stage1_training_visualization.png  # Stage 1 training visualization
├── stage2_trl_grpo.ipynb          # Stage 2: TRL + 0.5B training
├── stage3_source_code.ipynb       # Stage 3: TRL source code walkthrough
├── stage3_experiments.py          # Stage 3: K1/K3, epochs, clip comparison experiments
├── stage4_notebook.ipynb          # Stage 4: teaching notebook
├── stage4_train.py                # Stage 4: 1.5B training script
├── stage4_eval.py                 # Stage 4: evaluation script
├── stage4_deploy.sh               # Stage 4: Docker deployment script
│
└── output/                        # Training records (no model weights)
    ├── README.md                  # Output directory description
    ├── stage4_eval_results.md     # Stage 4 evaluation results summary
    ├── stage4_trainer_state.json  # Stage 4 complete 500-step training history
    ├── stage4_runs/               # Stage 4 TensorBoard logs
    ├── runs/                      # Stage 2 TensorBoard logs
    └── completions/               # Stage 2 model-generated answer records
```

---

## 11. Learning Tips

1. **Don't skip stages**: Each stage builds on the previous one — skipping leads to confusion later
2. **Intuition before formulas**: Understand each concept with an analogy first, then math, then code
3. **Experiment with parameters**: After getting it running, change parameters (e.g. G=4 to G=8) and observe changes
4. **Unfamiliar terms**: Refer back to section 0.3 terminology table
5. **Unfamiliar formulas**: Refer back to section 4 for analogies and math knowledge mapping
6. **Pass every understanding checkpoint**: Make sure you can answer those questions before moving on
7. **For the complete training journey**: See `stage4_summary.md` — records every training run's parameters, results, reasons for changes, and improvements
