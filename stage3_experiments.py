"""
阶段 3 动手实验：在 transparent_grpo.py 基础上做 3 个对比实验
适配 Qwen2.5-0.5B + 加法任务，可在 5070Ti (16GB) 上运行
"""
import os
import sys
import gc
import copy
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re
import random
from typing import List, Dict

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/data/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# ============================================================
# 加法任务（和阶段 2 一致）
# ============================================================
def generate_addition_dataset(n=200):
    random.seed(42)
    data = []
    for _ in range(n):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        data.append((f"What is {a}+{b}? Answer with just the number.", str(a + b)))
    return data

def compute_reward_addition(texts: List[str], answers: List[str]) -> List[float]:
    rewards = []
    for text, ans in zip(texts, answers):
        numbers = re.findall(r'\d+', text)
        is_correct = bool(numbers and numbers[-1] == ans)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards

class AdditionDataset(Dataset):
    def __init__(self, data, size=200):
        self.data = data
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        prompt, answer = self.data[idx % len(self.data)]
        return {"prompt": prompt, "answer": answer}

# ============================================================
# 简化版 GRPO 训练（从 transparent_grpo.py 提取核心逻辑）
# ============================================================
def run_grpo_experiment(
    experiment_name,
    group_size=4,
    learning_rate=5e-6,
    beta=0.1,
    clip_epsilon=0.2,
    inner_update_epochs=3,
    num_steps=30,
    max_new_tokens=16,
    kl_mode="K1",  # "K1" = KL in reward, "K3" = KL in loss
    log_every=5,
):
    """
    运行一次 GRPO 实验，返回指标字典
    """
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"  group_size={group_size}, lr={learning_rate}, beta={beta}")
    print(f"  clip_epsilon={clip_epsilon}, inner_epochs={inner_update_epochs}")
    print(f"  kl_mode={kl_mode}, num_steps={num_steps}")
    print(f"{'='*60}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, trust_remote_code=True, device_map={"": DEVICE}
    )
    model.gradient_checkpointing_enable()
    model.train()

    # 参考模型（冻结）
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, trust_remote_code=True, device_map={"": DEVICE}
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 数据
    data = generate_addition_dataset(200)
    dataset = AdditionDataset(data, size=num_steps * 2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    data_iter = iter(dataloader)

    # 记录指标
    metrics = {"loss": [], "reward": [], "kl": [], "raw_reward": [], "grad_norm": []}

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt = batch[0]["prompt"]
        answer = batch[0]["answer"]

        # --- 收集阶段 ---
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]

        # 组采样：生成 G 个回答
        model.eval()
        with torch.no_grad():
            input_ids_repeated = inputs.input_ids.repeat(group_size, 1)
            attention_mask_repeated = inputs.attention_mask.repeat(group_size, 1)
            generated_ids = model.generate(
                input_ids=input_ids_repeated,
                attention_mask=attention_mask_repeated,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        model.train()

        # 计算奖励
        completion_ids = generated_ids[:, prompt_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards_list = compute_reward_addition(completions, [answer] * group_size)
        rewards = torch.tensor(rewards_list, dtype=torch.float32).to(DEVICE)

        # 计算 log 概率
        attention_mask = (generated_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = generated_ids[:, 1:]
            log_probs_all = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)

            ref_outputs = ref_model(input_ids=generated_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)

        # loss mask：排除 prompt 部分
        loss_mask = attention_mask[:, 1:].clone().float()
        loss_mask[:, :prompt_len - 1] = 0.0

        # KL 计算
        per_token_kl = token_log_probs.detach() - ref_token_log_probs.detach()
        kl_penalty = (per_token_kl * loss_mask).sum(dim=1)

        if kl_mode == "K1":
            # K1: KL 放在 reward 里
            rewards_with_kl = rewards - beta * kl_penalty
        else:
            # K3: KL 不放在 reward 里，后面放在 loss 里
            rewards_with_kl = rewards

        # 优势计算
        mean_r = rewards_with_kl.mean()
        std_r = rewards_with_kl.std()
        advantages = (rewards_with_kl - mean_r) / (std_r + 1e-4)

        # --- 训练阶段 ---
        old_log_probs = token_log_probs.detach()
        step_losses, step_kls = [], []

        for epoch in range(inner_update_epochs):
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = generated_ids[:, 1:]

            new_log_probs_all = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(new_log_probs_all, -1, targets.unsqueeze(-1)).squeeze(-1)

            # 概率比
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            # PPO 裁剪
            adv_per_token = advantages.unsqueeze(1).expand_as(new_log_probs)
            surr1 = ratio * adv_per_token
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_per_token
            pg_loss = -torch.min(surr1, surr2)

            loss = (pg_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)

            # K3 模式：KL 放在 loss 里
            if kl_mode == "K3":
                kl_loss = beta * (per_token_kl.detach() * loss_mask).sum() / (loss_mask.sum() + 1e-6)
                loss = loss + kl_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            step_losses.append(loss.item())
            step_kls.append(kl_penalty.mean().item())

        # 记录指标
        avg_loss = np.mean(step_losses)
        avg_reward = rewards.mean().item()
        avg_kl = np.mean(step_kls)
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        metrics["loss"].append(avg_loss)
        metrics["reward"].append(avg_reward)
        metrics["kl"].append(avg_kl)
        metrics["raw_reward"].append(avg_reward)
        metrics["grad_norm"].append(grad_norm)

        if step % log_every == 0 or step == num_steps - 1:
            print(f"  Step {step:3d} | Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | "
                  f"KL: {avg_kl:.4f} | GradNorm: {grad_norm:.2f}")

    # 清理显存
    del model, ref_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return metrics


# ============================================================
# 运行 3 个实验
# ============================================================
if __name__ == "__main__":
    results = {}

    # 实验 1：K1 vs K3 KL 估计器
    print("\n" + "=" * 60)
    print("实验 1：K1 (KL in reward) vs K3 (KL in loss)")
    print("=" * 60)

    results["K1"] = run_grpo_experiment(
        "K1: KL 放在 reward 里",
        kl_mode="K1", beta=0.1, num_steps=30, group_size=4,
    )
    gc.collect()
    torch.cuda.empty_cache()
    results["K3"] = run_grpo_experiment(
        "K3: KL 放在 loss 里",
        kl_mode="K3", beta=0.1, num_steps=30, group_size=4,
    )

    # 实验 2：inner_update_epochs = 1 vs 3 vs 10
    print("\n" + "=" * 60)
    print("实验 2：inner_update_epochs = 1 vs 3 vs 10")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    results["epoch_1"] = run_grpo_experiment(
        "inner_update_epochs=1",
        inner_update_epochs=1, num_steps=30, group_size=4,
    )
    gc.collect()
    torch.cuda.empty_cache()
    results["epoch_3"] = run_grpo_experiment(
        "inner_update_epochs=3",
        inner_update_epochs=3, num_steps=30, group_size=4,
    )
    gc.collect()
    torch.cuda.empty_cache()
    results["epoch_10"] = run_grpo_experiment(
        "inner_update_epochs=10",
        inner_update_epochs=10, num_steps=30, group_size=4,
    )

    # 实验 3：clip_epsilon = 0.1 vs 0.2 vs 0.3
    print("\n" + "=" * 60)
    print("实验 3：clip_epsilon = 0.1 vs 0.2 vs 0.3")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    results["clip_0.1"] = run_grpo_experiment(
        "clip_epsilon=0.1（更保守）",
        clip_epsilon=0.1, num_steps=30, group_size=4,
    )
    gc.collect()
    torch.cuda.empty_cache()
    results["clip_0.2"] = run_grpo_experiment(
        "clip_epsilon=0.2（默认）",
        clip_epsilon=0.2, num_steps=30, group_size=4,
    )
    gc.collect()
    torch.cuda.empty_cache()
    results["clip_0.3"] = run_grpo_experiment(
        "clip_epsilon=0.3（更激进）",
        clip_epsilon=0.3, num_steps=30, group_size=4,
    )

    # ============================================================
    # 汇总对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    def summarize(metrics, name):
        r = metrics["reward"]
        k = metrics["kl"]
        l = metrics["loss"]
        g = metrics["grad_norm"]
        # 对比前 10 步和后 10 步
        early_r = np.mean(r[:10])
        late_r = np.mean(r[-10:])
        max_kl = max(k)
        mean_loss = np.mean(l)
        mean_grad = np.mean(g)
        print(f"  {name:25s} | "
              f"Reward: {early_r:.2f}→{late_r:.2f} | "
              f"MaxKL: {max_kl:.2f} | "
              f"MeanLoss: {mean_loss:.4f} | "
              f"MeanGrad: {mean_grad:.2f}")

    print("\n--- 实验 1: K1 vs K3 ---")
    summarize(results["K1"], "K1 (KL in reward)")
    summarize(results["K3"], "K3 (KL in loss)")

    print("\n--- 实验 2: inner_update_epochs ---")
    summarize(results["epoch_1"], "epochs=1")
    summarize(results["epoch_3"], "epochs=3")
    summarize(results["epoch_10"], "epochs=10")

    print("\n--- 实验 3: clip_epsilon ---")
    summarize(results["clip_0.1"], "clip=0.1")
    summarize(results["clip_0.2"], "clip=0.2")
    summarize(results["clip_0.3"], "clip=0.3")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
