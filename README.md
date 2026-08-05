# The Illustrated GRPO 复现与学习指南

> 论文: [The Illustrated GRPO (Towards AI)](https://towardsai.com/p/l/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation)
> 原始算法论文: [DeepSeekMath (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300)

---

## 零、写给新手：你需要知道的前置概念

> 你的数学基础（高数、线代、概率统计）完全够用。下面把你会的数学和 GRPO 中用到的概念对应起来。

### 0.1 从你的数学知识到 GRPO 的概念映射

| 你已经会的（数学课） | GRPO 中对应的概念 | 简单解释 |
|---------------------|-------------------|---------|
| 概率分布 P(X=x) | 策略 π(a\|s) | 模型在给定输入 s 时，输出动作 a 的概率。LLM 中"动作"就是下一个 token |
| 期望 E[X]、方差 Var(X) | 奖励的均值和标准差 | 对一组输出打分，算均值和标准差——就是你熟悉的统计量 |
| 标准化 (X-μ)/σ | 优势函数 A_i | 把奖励做标准化，让"比平均好"的为正，"比平均差"的为负 |
| 梯度下降 ∇f(θ) | 策略梯度 | 沿梯度方向更新模型参数 θ，让好输出概率变高、差输出概率变低 |
| KL 散度 D(P‖Q) | KL 惩罚 | 衡量两个概率分布的差异，防止训练后的模型偏离原始模型太远 |
| 截断函数 clip(x, a, b) | PPO Clipping | 限制参数更新幅度，防止更新幅度过大导致训练崩溃 |
| Token | Token | LLM 处理文本的最小单位，大致等于一个词或一个字 |
| LoRA | Low-Rank Adaptation | 不改全部参数，只训练一个小的低秩矩阵（线代中的秩），省显存 |
| bf16 | Bfloat16 | 16位浮点数，比 float32 省一半显存，数值范围和 float32 一样 |

### 0.2 用一个生活类比理解 GRPO 全过程

> 想象一个学生（模型）在做数学题（prompt），老师（奖励函数）批改打分。

```
1. 老师出了一道数学题
2. 学生用当前的水平，尝试写出 4 份不同的解答（组采样 G=4）
3. 老师批改每份解答，给分 r1=1, r2=0, r3=1, r4=0（奖励计算）
4. 算出班级平均分 mean=0.5，标准差 std=0.5
5. 每份解答的"相对表现" = (自己的分 - 平均分) / 标准差
   → A1=+1（比平均好），A2=-1（比平均差），A3=+1，A4=-1（优势计算）
6. 学生反思：好的解法多学学，差的解法别再犯了（策略更新）
7. 但不能学歪了——解题风格不能偏离原来的自己太多（KL 惩罚）
8. 重复以上步骤，学生越来越擅长做这类题
```

这就是 GRPO 的全部思想。后面的公式只是把这个过程用数学精确描述。

### 0.3 关键术语表（遇到不懂的随时回来查）

| 术语 | 英文 | 用你的数学知识理解 |
|------|------|-------------------|
| 策略 | Policy π_θ | 一个参数为 θ 的函数，输入问题，输出回答的概率分布 |
| 奖励 | Reward r | 一个标量打分，回答越好分越高（类似损失函数的相反数） |
| 优势 | Advantage A | 标准化后的奖励：(r - 均值) / 标准差，表示"相对好坏" |
| 价值网络 | Critic / Value Model | PPO 中用来估计"平均能得多少分"的额外模型，GRPO 把它去掉了 |
| 参考模型 | Reference Model π_ref | 冻结的原始模型，用来计算 KL 散度，防止训练跑偏 |
| 旧策略 | Old Policy π_θold | 上一步的模型快照，用来计算概率比值，定期同步 |
| 概率比 | Ratio π_θ/π_θold | 新模型输出某 token 的概率 / 旧模型输出同一 token 的概率 |
| KL 散度 | KL Divergence | 你在概率论学过的 D(P‖Q) = Σ P(x) log(P(x)/Q(x))，衡量两个分布差异 |
| 裁剪 | Clipping | 把概率比限制在 [1-ε, 1+ε] 范围内，防止更新幅度过大 |
| Token | Token | LLM 处理文本的最小单位，大致等于一个词或一个字 |
| LoRA | Low-Rank Adaptation | 不改全部参数，只训练一个小的低秩矩阵（线代中的秩），省显存 |
| bf16 | Bfloat16 | 16位浮点数，比 float32 省一半显存，数值范围和 float32 一样 |

---

## 一、硬件环境

| 机器 | GPU | 显存 | 角色 |
|------|-----|------|------|
| 本机 | RTX 5070 Ti | 16GB | 开发、调试、教学 notebook（阶段 1-3） |
| 无头服务器 | RTX 4090 | 48GB | 正式训练（阶段 4-6，Docker 隔离） |

## 二、模型文件

| 模型 | 本机路径 | 4090 路径 | 大小 | 用途 |
|------|---------|---------|------|------|
| Qwen2.5-0.5B-Instruct | `/data/models/Qwen2.5-0.5B-Instruct` | — | 954MB | 阶段2：本机教学训练 |
| Qwen2.5-1.5B-Instruct | `/data/models/Qwen2.5-1.5B-Instruct` | `/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct` | 2.9GB | 阶段4：4090 进阶训练 |
| Qwen2.5-3B-Instruct | `/data/models/Qwen2.5-3B-Instruct` | — | 5.8GB | 阶段5：4090 更大模型实验 |

## 三、环境管理策略

- **阶段 1-3（学习探索）**：Conda 环境，灵活装包调试
- **阶段 4-6（正式训练）**：Docker 容器，环境隔离可复现，不污染 4090 服务器

## 四、GRPO 算法原理详解（从直觉到公式）

### 4.1 第一步：理解"什么是策略梯度"（用概率论理解）

你在概率论学过：如果 X ~ P(x)，想最大化 E[f(X)]，可以用梯度上升：

```
∇_θ E[f(X)] = E[f(X) * ∇_θ log P(X)]
```

在 GRPO 中：
- P(X) → 策略 π_θ（模型输出某段文字的概率）
- f(X) → 优势 A（这段文字比平均水平好多少）
- 梯度上升 → 让好回答出现概率变高，差回答出现概率变低

**一句话**：策略梯度 = "好的行为多做，差的行为少做"，用你学过的梯度上升实现。

### 4.2 第二步：理解"为什么需要优势"（用统计学的标准化理解）

直接用奖励 r 做梯度上升有什么问题？如果所有回答都得正分，模型会无脑增加所有回答的概率——没有区分度。

解决方案就是你熟悉的**标准化**：

```
A_i = (r_i - mean) / std
```

- r_i > mean → A_i > 0 → 增加这个回答的概率
- r_i < mean → A_i < 0 → 降低这个回答的概率
- 除以 std → 消除量纲影响，不同题目之间可比较

**一句话**：优势 = 标准化后的奖励，就是你在概率论学过的 z-score。

### 4.3 第三步：理解"为什么需要裁剪"（防止更新幅度过大）

如果一步更新太猛，模型可能直接崩溃（概率论中的"方差爆炸"）。

PPO 的解决方案：计算新旧策略的概率比 r = π_新/π_旧，用 clip 限制在 [1-ε, 1+ε]：

```
clip(r, 1-ε, 1+ε)  →  比如 ε=0.2，则概率比被限制在 [0.8, 1.2]
```

**一句话**：裁剪 = 步长保护，不让模型一步走太远，类似优化中的学习率衰减。

### 4.4 第四步：理解"为什么需要 KL 惩罚"（防止遗忘）

训练后模型可能学到了"歪门邪道"——比如为了得分只输出特定格式，丧失了语言能力。

KL 散度（你在概率论学过）衡量两个分布的差异：

```
KL(π_θ || π_ref) = Σ π_θ(x) * log(π_θ(x) / π_ref(x))
```

在损失函数中减去 β * KL，迫使新模型不要偏离参考模型太远。

**一句话**：KL 惩罚 = "别学歪了"，用你学过的 KL 散度量化"歪了多少"。

### 4.5 完整 GRPO 目标函数（现在你能看懂了）

```
J_GRPO(θ) = E[ 1/G * Σ min( π_θ/π_θold * A_i, clip(π_θ/π_θold, 1-ε, 1+ε) * A_i ) - β * KL(π_θ || π_ref) ]
```

逐项拆解：

| 公式部分 | 含义 | 你对应的数学知识 |
|---------|------|-----------------|
| E[...] | 期望（对所有 prompt 取平均） | 概率论的期望 E[X] |
| 1/G * Σ | 对 G 个输出取平均 | 求和再除以个数 |
| π_θ/π_θold | 新旧策略的概率比 | 两个概率的比值 |
| A_i | 第 i 个输出的优势 | 标准化分数 z-score |
| min(..., clip(...)) | 裁剪后的策略梯度 | 取较小值 = 更保守的更新 |
| β * KL(...) | KL 散度惩罚 | 概率论的 KL 散度 D(P‖Q) |

### 4.6 三个模型角色

| 模型 | 符号 | 作用 | 生活类比 |
|------|------|------|---------|
| Policy Model | π_θ | 正在训练的模型，参数被更新 | 正在学习的学生 |
| Old Policy | π_θold | 冻结参数用于计算优势，定期同步 | 上一轮考试时的学生水平 |
| Reference Model | π_ref | 参考基线，KL 惩罚防止漂移 | 学生的"初始性格"，不能学歪 |

### 4.7 GRPO vs PPO：为什么去掉价值网络

| 维度 | PPO | GRPO | 为什么这样改 |
|------|-----|------|-------------|
| 价值网络 | 需要 critic 模型 | 不需要 | 组内均值已经估计了"平均能得多少分"，不需要额外模型 |
| 显存占用 | 2x（policy + critic） | 1x（仅 policy） | 少一个模型 = 省一半显存 |
| 优势估计 | GAE + value function | 组内 mean/std 归一化 | 直接用样本统计量估计，更简单 |
| KL 惩罚 | 在 reward 中 | 在 loss 中 | 简化计算，优势不受 KL 影响 |

**核心洞察**：如果你已经采样了 G 个输出并打了分，它们的均值就是"平均能得多少分"的天然估计——不需要再训练一个价值网络来预测这个值。这就是 GRPO 的全部精髓。

---

## 五、分阶段复现路线（每阶段标注你需要的知识）

### 阶段 0：环境准备（已完成 ✅）

- [x] 模型下载到 `/data/models`
- [x] 本文档创建

### 阶段 1：玩具算法理解（本机 CPU，~1小时）

> **你需要**：Python 基础、numpy 基本操作、概率论的均值/标准差概念
> **不需要**：GPU、深度学习框架、任何 RL 知识

- **目标**：用一个数字排序任务，在 CPU 上亲手实现 GRPO 的每一步
- **工具**：Conda + Jupyter Notebook
- **素材**：[djemec/rl_grpo_explainer.ipynb](https://github.com/djemec/descriptive_notebooks/blob/main/rl_grpo_explainer.ipynb)
- **你会学到**：
  - 如何用 numpy 实现一个最简单的"策略"（输出概率分布）
  - 如何采样多个输出（组采样）
  - 如何计算奖励和优势（就是均值和标准差的标准化）
  - 如何用梯度更新策略参数
- **理解检查点**：
  - [ ] 能解释"为什么优势要除以标准差"
  - [ ] 能解释"为什么不用原始奖励而用优势"
  - [ ] 能手动算出一个简单例子的优势值

### 阶段 2：最小 LLM GRPO（本机 5070Ti 16GB，~2小时）

> **你需要**：阶段 1 的概念理解、Python 基础
> **不需要**：深入了解 Transformer 内部结构（把它当黑盒用就行）

- **目标**：用真实 LLM 跑通 GRPO 训练，观察 TensorBoard 曲线
- **模型**：`/data/models/Qwen2.5-0.5B-Instruct`
- **数据**：`trl-lib/DeepMath-103K` 或 `openai/gsm8k`
- **框架**：TRL GRPOTrainer（封装好的库，几行代码就能跑）
- **关键配置**：`num_generations=4, max_completion_length=64, bf16=True`
- **你会学到**：
  - LLM 的输入输出是什么（prompt → token → 概率 → 文字）
  - TRL 库如何封装 GRPO 算法
  - TensorBoard 如何可视化训练过程
- **理解检查点**：
  - [ ] 能解释 `num_generations` 参数的含义（就是组大小 G）
  - [ ] 能在 TensorBoard 中找到 reward 曲线并解释趋势
  - [ ] 能解释训练前后模型输出的变化

### 阶段 3：阅读源码理解细节（本机，~2小时）

> **你需要**：阶段 1-2 的概念理解、能读 Python 代码
> **不需要**：能自己从零写出全部代码

- **目标**：逐行理解 GRPO 实现，看清楚每一步的代码长什么样
- **素材**：[transparent-grpo](https://github.com/siyuan-harry/transparent-grpo)（~400行单文件）
- **流程**：`Generate → Reward → Advantage → Update` 线性代码，从上到下读
- **你会学到**：
  - 概率比 π_θ/π_θold 在代码中怎么算（取 log 相减）
  - KL 散度在代码中怎么实现（Schulman 估计器）
  - 裁剪函数在 PyTorch 中就是 `torch.clamp`
- **理解检查点**：
  - [ ] 能在代码中指出哪一行计算优势
  - [ ] 能在代码中指出哪一行计算 KL 散度
  - [ ] 能解释为什么取 log 概率而不是直接用概率

### 阶段 4：正式训练（无头 4090，Docker，~3-5小时）

> **你需要**：阶段 2-3 的理解、基本的 Docker 命令
> **不需要**：深入 Docker 知识（我会提供完整 Dockerfile 和命令）

- **目标**：更大模型 + 更大组 + vLLM 加速，观察更明显的训练效果
- **模型**：`/data/models/Qwen2.5-1.5B-Instruct` 或 `3B`
- **环境**：Docker 容器（nvidia/cuda 基础镜像）
- **关键配置**：`num_generations=8, max_completion_length=256, use_vllm=True`
- **你会学到**：
  - 更大组（G=8）如何让优势估计更稳定（统计学：样本越多估计越准）
  - vLLM 如何加速生成（批处理推理引擎）
  - Docker 如何隔离环境
- **理解检查点**：
  - [ ] 能解释为什么 G=8 比 G=4 更稳定（样本量与估计方差的关系）
  - [ ] 能对比 0.5B 和 1.5B 模型的训练效果差异

### 阶段 5：自定义奖励函数 + 对比实验（4090，~3小时）

> **你需要**：阶段 4 的训练经验、Python 函数编写
> **不需要**：奖励模型训练知识（我们用规则奖励）

- **目标**：实现论文中的 format_reward + accuracy_reward，对比不同奖励组合
- **实验设计**：
  - 实验 A：仅 correctness reward → 观察模型是否学会答题
  - 实验 B：correctness + format reward → 观察格式是否改善
  - 实验 C：correctness + format + length reward → 观察回答长度变化
- **你会学到**：
  - 奖励函数设计如何影响模型行为
  - 多目标奖励如何加权组合
  - "奖励黑客"（reward hacking）现象——模型钻空子
- **理解检查点**：
  - [ ] 能解释为什么仅用 correctness reward 可能效果不好
  - [ ] 能观察到至少一种奖励黑客现象
  - [ ] 能设计一个自己的奖励函数

### 阶段 6：评估与可视化（~1小时）

> **你需要**：前面所有阶段的理解
> **不需要**：额外知识

- **目标**：训练前后模型输出对比 + TensorBoard 指标分析
- **工具**：TensorBoard、[UNIPO 交互可视化](https://poloclub.github.io/unipo/)
- **你会学到**：
  - 如何科学地评估模型改进（对比测试）
  - 如何解读训练曲线中的异常信号
- **理解检查点**：
  - [ ] 能用训练前后的模型分别回答 10 道题并对比正确率
  - [ ] 能解释 TensorBoard 中 reward、kl、loss 三条曲线的关系

---

## 六、依赖环境

### 阶段 1（玩具学习）

```bash
conda create -n grpo-learn python=3.10 -y
conda activate grpo-learn
pip install numpy matplotlib jupyter
```

### 阶段 2-3（TRL 训练 + 源码阅读）

```bash
conda create -n grpo-tutorial python=3.10 -y
conda activate grpo-tutorial
pip install torch transformers trl datasets accelerate peft tensorboard
```

### 阶段 4-6（Docker 正式训练）

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

## 七、关键代码模板

### 最小 GRPO 训练脚本

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

### 自定义奖励函数

```python
import re

def format_reward(completions, **kwargs):
    """奖励 <think>...</think><answer>...</answer> 格式"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in completions]

def accuracy_reward(completions, answers, **kwargs):
    """奖励正确答案"""
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

### TensorBoard 可视化

```bash
tensorboard --logdir output/runs --port 6006
```

关键指标：
- `reward`：平均奖励（应上升）
- `reward_std`：组内奖励标准差（反映探索性）
- `kl`：KL 散度（应保持在合理范围）
- `loss`：总损失

---

## 七点五、GRPO 训练调参实战经验

> 以下经验来自在 RTX 5070 Ti (16GB) 上用 TRL 训练 Qwen2.5-0.5B 做加法任务的多轮实验。

### 实验环境

- **GPU**: RTX 5070 Ti (16GB)
- **模型**: Qwen2.5-0.5B-Instruct (494M 参数)
- **框架**: TRL GRPOTrainer
- **任务**: 整数加法（Answer with just the number）

### 三轮训练对比

| 轮次 | 任务 | G | temp | lr | beta | max_steps | batch | accum | 结果 |
|------|------|---|------|-----|------|-----------|-------|-------|------|
| 第1轮 | 2位数 | 4 | 1.0 | 1e-5 | 0.04 | 50 | 4 | 2 | 90%→65%（**退步**）|
| 第2轮 | 2位数 | 4 | 0.7 | 5e-6 | 0.1 | 300 | 4 | 2 | 几乎无学习信号 |
| 第3轮 | 3位数 | 6 | 0.8 | 5e-6 | 0.1 | 300 | 2 | 3 | 50%→80%（**成功**）|

### 核心教训

#### 1. `frac_reward_zero_std` 是最重要的指标

> **如果组内 G 个回答全对或全错，优势全是 0，模型学不到任何东西。**

- 第2轮：`frac_reward_zero_std` ≈ 100%，300 步几乎白跑
- 第3轮：`frac_reward_zero_std` = 63%，37% 的步有学习信号，最终准确率提升 30%

**调参的首要目标是让 `frac_reward_zero_std` 降下来**，否则其他参数再怎么调都没用。

#### 2. 任务难度要匹配模型水平

| 任务 | 基座准确率 | 组内多样性 | 效果 |
|------|-----------|-----------|------|
| 两位数加法 | ~90% | 极低（几乎全对） | 学不到东西 |
| 三位数加法 | ~50% | 较高（有对有错） | 有效学习 |

**经验法则**：基座模型准确率在 30%-70% 时，GRPO 最容易生效。太高（全对）或太低（全错）都不行。

#### 3. 温度的选择是平衡艺术

| 温度 | 效果 |
|------|------|
| 0.7 | 太确定，组内答案几乎相同，`zero_std` 高 |
| 0.8 | 平衡，组内有一定多样性 ✅ |
| 1.0 | 太随机，模型乱说长篇废话，entropy 飙升 |

**经验**：0.8 是 0.5B 模型做加法任务的一个较好起点，具体任务需要实验调整。

#### 4. 梯度裁剪是必须的

- 不加 `max_grad_norm`：梯度飙到 388~524，训练不稳定
- 加 `max_grad_norm=0.5`：梯度被截断，但有时裁剪太狠（原始梯度 500+ → 裁到 0.5，信息丢失）

**经验**：0.5 偏保守，1.0 可能更好。但如果不确定，先设 0.5 保底。

#### 5. KL 惩罚需要动态调整

| beta | KL 表现 |
|------|---------|
| 0.04 | KL 失控到 4.3+ |
| 0.1 | KL 平均 1.6，最高 7.8（仍偏高） |

**经验**：如果 KL 持续 >2.0，加大 `beta`。0.1 对 0.5B 模型仍不够，可能需要 0.2。但太大也会限制学习。

#### 6. 显存管理

**关键公式**：每步序列数 = `per_device_train_batch_size × num_generations`

| 配置 | 序列数 | 显存 |
|------|--------|------|
| batch=4, G=4 | 16 | ~13 GB |
| batch=2, G=6 | 12 | ~10 GB |

**技巧**：增大 G 时同步降低 batch，保持序列数不超显存。用 `gradient_accumulation_steps` 补偿等效 batch_size。

#### 7. TRL 的整除约束

> `generation_batch_size`（= `batch_size × gradient_accumulation_steps`）必须能被 `num_generations` 整除。

| batch | accum | 乘积 | G=6 能否整除 |
|-------|-------|------|-------------|
| 2 | 4 | 8 | ❌ 报错 |
| 2 | 3 | 6 | ✅ |

**经验**：先确定 G，再反推 batch × accum = G 的倍数。

### 调参决策流程

```
1. 选任务难度 → 让基座准确率在 30%-70%
2. 选 G 和 temp → 让 frac_reward_zero_std < 70%
3. 调 batch 和 accum → 满足显存 + 整除约束
4. 设 max_grad_norm → 防梯度爆炸
5. 设 beta → 控制 KL 在 1.0 以下
6. 设 lr 和 warmup → 配合以上参数
7. 跑 50 步观察日志 → 确认有学习信号
8. 跑完整训练 → 对比训练前后准确率
```

### 常见问题速查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `frac_reward_zero_std=1.0` | 任务太简单或温度太低 | 换更难的题 / 升温 / 增大 G |
| KL 持续上升 | beta 太小 | 加大 beta |
| 梯度爆炸 | 没加梯度裁剪 | 设 max_grad_norm=0.5~1.0 |
| 显存不足 | 序列数太多 | 降 batch 或 G，用 accum 补偿 |
| reward 不涨 | 学习信号不足 | 检查 zero_std，调整任务难度 |
| 回答太长乱说 | 温度太高 | 降温 + 限制 max_completion_length |
| `ValueError: generation_batch_size` | 整除约束 | 调 accum 使 batch×accum 能被 G 整除 |

---

## 八、参考资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Towards AI 文章 | [链接](https://towardsai.com/p/l/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation) | 论文原文 |
| DeepSeekMath 论文 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | GRPO 原始论文 |
| TRL GRPO 文档 | [HF Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) | 官方实现文档 |
| HF Course Ch12 | [链接](https://huggingface.co/docs/course/main/en/chapter12/4) | GRPO 教程 |
| HF Cookbook | [链接](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) | 完整 Notebook |
| transparent-grpo | [GitHub](https://github.com/siyuan-harry/transparent-grpo) | ~400行单文件实现 |
| mini-grpo | [GitHub](https://github.com/JFan5/mini-grpo) | ~500行纯 PyTorch |
| GRPO Explainer | [GitHub](https://github.com/djemec/descriptive_notebooks/blob/main/rl_grpo_explainer.ipynb) | CPU 教学 Notebook |
| UNIPO 可视化 | [poloclub.github.io/unipo](https://poloclub.github.io/unipo/) | 交互式算法可视化 |
| HF Blog: GRPO = PPO without critic | [链接](https://huggingface.co/blog/garg-aayush/derive-grpo-loss) | 逐步推导 GRPO 目标 |

---

## 九、进度追踪

- [x] 阶段 0：环境准备（模型下载 + 文档创建）
- [x] 阶段 1：玩具算法理解（`stage1_toy_grpo.ipynb`）
- [x] 阶段 2：最小 LLM GRPO 训练（`stage2_trl_grpo.ipynb`）
- [x] 阶段 3：阅读源码（`stage3_source_code.ipynb`）
- [x] 阶段 4：4090 正式训练（`stage4_notebook.ipynb` + `stage4_train.py` + `stage4_eval.py` + `stage4_deploy.sh` + `Dockerfile`）
- [x] 阶段 4：训练记录与评估结果（`output/stage4_runs/` + `output/stage4_trainer_state.json` + `output/stage4_eval_results.md`）
- [x] 全过程训练总结（`stage4_summary.md`）
- [ ] 阶段 5：自定义奖励函数实验
- [ ] 阶段 6：评估与可视化

---

## 十、仓库结构

```
illustrated-grpo/
├── README.md                      # 项目主文档：GRPO 原理 + 硬件 + 进度
├── stage4_summary.md              # 全过程训练总结（每次训练的参数/结果/改动原因）
├── 4090_agent_上手指南.md          # 4090 服务器环境说明
├── requirements.txt               # 依赖说明
├── Dockerfile                     # 阶段 4 Docker 镜像定义
├── .gitignore                     # 排除模型权重/checkpoint
│
├── stage1_toy_grpo.ipynb          # 阶段 1：numpy 手写 GRPO
├── stage1_training_visualization.png  # 阶段 1 训练可视化图
├── stage2_trl_grpo.ipynb          # 阶段 2：TRL + 0.5B 训练
├── stage3_source_code.ipynb       # 阶段 3：TRL 源码解读
├── stage3_experiments.py          # 阶段 3：K1/K3、epochs、clip 对比实验
├── stage4_notebook.ipynb          # 阶段 4：教学 notebook
├── stage4_train.py                # 阶段 4：1.5B 训练脚本
├── stage4_eval.py                 # 阶段 4：评估脚本
├── stage4_deploy.sh               # 阶段 4：Docker 部署脚本
│
└── output/                        # 训练记录（不含模型权重）
    ├── README.md                  # 训练输出说明
    ├── stage4_eval_results.md     # 阶段 4 评估结果汇总
    ├── stage4_trainer_state.json  # 阶段 4 完整 500 步训练历史
    ├── stage4_runs/               # 阶段 4 TensorBoard 日志
    ├── runs/                      # 阶段 2 TensorBoard 日志
    └── completions/               # 阶段 2 训练中模型生成的回答记录
```

---

## 十一、学习建议

1. **不要跳阶段**：每个阶段建立在前一个的理解上，跳过会导致后面看不懂
2. **先直觉后公式**：每个概念先用生活类比理解，再看数学公式，最后看代码
3. **动手改参数**：跑通后改一改参数（如 G=4 改成 G=8），观察变化，加深理解
4. **遇到不懂的术语**：回到第 0.3 节术语表查
5. **遇到不懂的公式**：回到第 4 节看对应的生活类比和数学知识映射
6. **每个阶段的"理解检查点"都要过**：确认自己能回答那些问题再进入下一阶段
7. **想了解完整训练历程**：看 `stage4_summary.md`，记录了每次训练的参数、结果、改动原因和效果改进
