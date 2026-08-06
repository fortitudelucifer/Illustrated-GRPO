"""
阶段 4d：正式训练脚本
在 4090 (48GB) 上用 Qwen2.5-1.5B + LoRA + G=8 训练六位数加法

实验背景：
- 三位数加法基座 95%，全参数微调退化 -3.2%（显著）
- 五位数加法基座 83%，全参数微调退化 -1.28%（不显著）
- 六位数加法基座 81%，全参数微调退化 -2.04%（不显著）
- 三次实验均出现 collapse-recovery 模式：step 51-100 reward 暴跌，恢复后比原来差
- 根因：全参数微调导致灾难性遗忘 + lr 过高
- 本次改用 LoRA 适配器（冻结原始权重）+ 更保守的参数

使用方法：
  python stage4_train.py

或者用 accelerate：
  accelerate launch stage4_train.py
"""
import os
import re
import random
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/models/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/output/grpo_1.5b_6digit_lora"
LOG_DIR = "/output/logs/stage4d"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 数据集：六位数加法（a, b ∈ [100000, 999999]，结果为 6-7 位数）
# ============================================================
def make_addition_dataset(n=1000, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        a = random.randint(100000, 999999)
        b = random.randint(100000, 999999)
        data.append({"prompt": f"What is {a}+{b}? Answer with just the number.",
                      "answer": str(a + b)})
    return data

def format_prompt(example):
    messages = [{"role": "user", "content": example["prompt"]}]
    return {"prompt": messages}

# ============================================================
# 奖励函数
# ============================================================
def extract_text(completion):
    """从 completion 中提取文本（兼容 str 和 list[dict] 格式）"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
            else:
                parts.append(str(msg))
        return " ".join(parts)
    return str(completion)

def correctness_reward(completions, **kwargs):
    """正确性奖励：答案正确得 1.0，否则 0.0"""
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, answer in zip(completions, answers):
        text = extract_text(completion)
        numbers = re.findall(r'\d+', text)
        is_correct = bool(numbers and numbers[-1] == answer)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """格式奖励：回答简短（只含数字）得 0.2，过长扣分"""
    rewards = []
    for completion in completions:
        text = extract_text(completion)
        stripped = text.strip()
        if re.match(r'^\d{6,7}$', stripped):
            rewards.append(0.2)
        elif re.match(r'^\d+$', stripped):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards

# ============================================================
# GRPO 配置
# ============================================================
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    logging_steps=1,
    save_steps=100,
    save_total_limit=3,

    # 模型
    bf16=True,
    gradient_checkpointing=True,

    # 生成
    num_generations=8,              # G=8（4090 显存够）
    max_completion_length=64,       # 6位数加法结果最多7位，留余量
    generation_kwargs={
        "temperature": 0.9,         # 略高温以增加组内多样性
        "do_sample": True,
    },

    # vLLM 加速（暂时关闭，TRL 1.9.2 + vLLM 版本兼容性问题）
    use_vllm=False,

    # 训练（LoRA + 保守参数）
    learning_rate=1e-5,             # LoRA 需要更高 lr（只训练少量参数）
    num_train_epochs=1,
    max_steps=500,                  # 500 步
    warmup_steps=100,               # 更长预热，防止早期崩溃
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # 4×2=8，能被 G=8 整除

    # 稳定性
    beta=0.04,                      # LoRA 下 KL 惩罚可以小一些（权重冻结，不会跑偏太远）
    max_grad_norm=1.0,              # LoRA 梯度通常较小，放宽裁剪
    epsilon=0.2,                    # PPO 裁剪

    # 其他
    report_to="tensorboard",
    seed=42,
)

# ============================================================
# LoRA 配置
# ============================================================
peft_config = LoraConfig(
    r=32,                           # 秩：32（平衡容量和参数量）
    lora_alpha=64,                  # alpha = 2*r 是常用设置
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("阶段 4d：LoRA + 6-digit addition")
    print(f"  模型: {MODEL_PATH}")
    print(f"  Task: 6-digit addition")
    print(f"  LoRA: r=32, alpha=64, dropout=0.05")
    print(f"  G={config.num_generations}, temp=0.9")
    print(f"  lr={config.learning_rate}, beta={config.beta}")
    print(f"  max_steps={config.max_steps}, warmup={config.warmup_steps}")
    print(f"  vLLM: {config.use_vllm}")
    print("=" * 60)

    # 准备数据
    raw_data = make_addition_dataset(1000)
    dataset = Dataset.from_list(raw_data).map(format_prompt)
    print(f"数据集大小: {len(dataset)}")

    # 创建 Trainer（使用 LoRA）
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=config,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
        peft_config=peft_config,
    )

    # 训练
    print("\n开始训练...")
    trainer.train()
    print("\n训练完成！")

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    print(f"模型已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
