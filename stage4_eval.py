"""
Stage 4b Evaluation: Compare 1.5B model accuracy before/after GRPO training
- 500 held-out 5-digit addition problems
- 5 random seeds for test set generation
- Batch inference (batch_size=64)
- Reports mean±std and 95% confidence intervals
"""
import torch
import re
import random
import time
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/models/Qwen2.5-1.5B-Instruct"
TRAINED_PATH = "/output/grpo_1.5b_5digit_addition"
NUM_QUESTIONS = 500
BATCH_SIZE = 64
SEEDS = [42, 123, 456, 789, 999]
MAX_NEW_TOKENS = 16


def generate_test_questions(seed, n=NUM_QUESTIONS):
    """Generate n 3-digit addition problems with the given seed."""
    rng = random.Random(seed)
    questions = []
    for _ in range(n):
        a = rng.randint(10000, 99999)
        b = rng.randint(10000, 99999)
        questions.append((a, b, str(a + b)))
    return questions


def test_model_batched(model, tokenizer, questions, batch_size=BATCH_SIZE):
    """Run batched greedy inference and return accuracy."""
    prompts = []
    for a, b, ans in questions:
        messages = [{"role": "user", "content": f"What is {a}+{b}? Answer with just the number."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    correct = 0
    total = len(questions)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        batch_answers = [q[2] for q in questions[start:end]]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            numbers = re.findall(r'\d+', response)
            if numbers and numbers[-1] == batch_answers[i]:
                correct += 1

    return correct / total


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def main():
    print("=" * 70)
    print("Stage 4b Evaluation: 5-digit addition, 500 questions x 5 seeds x 2 models")
    print("=" * 70)
    print(f"Questions per seed: {NUM_QUESTIONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    base_results = []
    for seed in SEEDS:
        questions = generate_test_questions(seed)
        t0 = time.time()
        acc = test_model_batched(base_model, tokenizer, questions)
        elapsed = time.time() - t0
        base_results.append(acc)
        print(f"  Base model  | seed={seed:>4d} | acc={acc:.4f} ({int(acc*NUM_QUESTIONS)}/{NUM_QUESTIONS}) | {elapsed:.1f}s")

    del base_model
    torch.cuda.empty_cache()

    # Load trained model
    print("\nLoading trained model...")
    t0 = time.time()
    trained_model = AutoModelForCausalLM.from_pretrained(
        TRAINED_PATH, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    trained_results = []
    for seed in SEEDS:
        questions = generate_test_questions(seed)
        t0 = time.time()
        acc = test_model_batched(trained_model, tokenizer, questions)
        elapsed = time.time() - t0
        trained_results.append(acc)
        print(f"  Trained     | seed={seed:>4d} | acc={acc:.4f} ({int(acc*NUM_QUESTIONS)}/{NUM_QUESTIONS}) | {elapsed:.1f}s")

    del trained_model
    torch.cuda.empty_cache()

    # Summary
    base_mean = np.mean(base_results)
    base_std = np.std(base_results, ddof=1)
    trained_mean = np.mean(trained_results)
    trained_std = np.std(trained_results, ddof=1)

    base_ci = wilson_ci(base_mean, NUM_QUESTIONS * len(SEEDS))
    trained_ci = wilson_ci(trained_mean, NUM_QUESTIONS * len(SEEDS))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Base model:    {base_mean:.4f} +/- {base_std:.4f}  (95% CI: [{base_ci[0]:.4f}, {base_ci[1]:.4f}])")
    print(f"  Trained model: {trained_mean:.4f} +/- {trained_std:.4f}  (95% CI: [{trained_ci[0]:.4f}, {trained_ci[1]:.4f}])")
    print(f"  Improvement:   {trained_mean - base_mean:+.4f}")
    print()

    # Per-seed detail
    print("Per-seed detail:")
    print(f"  {'Seed':>6s} | {'Base':>8s} | {'Trained':>8s} | {'Diff':>8s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")
    for i, seed in enumerate(SEEDS):
        diff = trained_results[i] - base_results[i]
        print(f"  {seed:>6d} | {base_results[i]:>8.4f} | {trained_results[i]:>8.4f} | {diff:>+8.4f}")

    # Save results as JSON
    results = {
        "config": {
            "num_questions": NUM_QUESTIONS,
            "seeds": SEEDS,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "model_path": MODEL_PATH,
            "trained_path": TRAINED_PATH,
        },
        "base_model": {
            "per_seed": base_results,
            "mean": float(base_mean),
            "std": float(base_std),
            "ci_95": [float(base_ci[0]), float(base_ci[1])],
        },
        "trained_model": {
            "per_seed": trained_results,
            "mean": float(trained_mean),
            "std": float(trained_std),
            "ci_95": [float(trained_ci[0]), float(trained_ci[1])],
        },
        "improvement": float(trained_mean - base_mean),
    }

    output_path = "/output/eval_results_5digit_500.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
