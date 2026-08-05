"""
阶段 4：正式训练脚本
在 4090 (48GB) 上用 Qwen2.5-1.5B + vLLM 加速 + G=8 训练三位数加法

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

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/mnt/hdd_wd_4tb/grpo_output/grpo_1.5b_addition"
LOG_DIR = "/mnt/hdd_wd_4tb/grpo_output/logs/stage4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 数据集：三位数加法（和阶段 2 相同任务，但数据量更大）
# ============================================================
def make_addition_dataset(n=1000, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
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
        if re.match(r'^\d{3,4}$', stripped):
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
    max_completion_length=32,       # 加法回答短，32 够用
    generation_kwargs={
        "temperature": 0.8,
        "do_sample": True,
    },

    # vLLM 加速（暂时关闭，TRL 1.9.2 + vLLM 版本兼容性问题）
    use_vllm=False,

    # 训练
    learning_rate=5e-6,
    num_train_epochs=1,
    max_steps=500,                  # 500 步
    warmup_steps=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # 4×2=8，能被 G=8 整除

    # 稳定性
    beta=0.1,                       # KL 惩罚
    max_grad_norm=0.5,              # 梯度裁剪
    epsilon=0.2,                    # PPO 裁剪（TRL 1.9.x 用 epsilon）

    # 其他
    report_to="tensorboard",
    seed=42,
)

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("阶段 4：正式训练")
    print(f"  模型: {MODEL_PATH}")
    print(f"  G={config.num_generations}, temp={config.temperature}")
    print(f"  lr={config.learning_rate}, beta={config.beta}")
    print(f"  max_steps={config.max_steps}")
    print(f"  vLLM: {config.use_vllm}")
    print("=" * 60)

    # 准备数据
    raw_data = make_addition_dataset(1000)
    dataset = Dataset.from_list(raw_data).map(format_prompt)
    print(f"数据集大小: {len(dataset)}")

    # 创建 Trainer
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=config,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
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
