# GRPO 训练全过程总结：从第一次实验到最终结果

> 从 numpy 玩具模型到 4090 服务器 1.5B 正式训练的完整记录

---

## 一、训练全记录（按时间线）

以下是每一次训练/实验的完整记录：参数是什么、结果如何、为什么改参数、改了之后效果如何。

---

### 训练 1：阶段 1 — numpy 玩具 GRPO（CPU）

**时间**：学习阶段，最早做

**目的**：不碰 GPU，只用 numpy 在 CPU 上从零实现 GRPO 的每一步，理解算法原理

**任务**：数字排序（给 5 个数字，输出降序排列）

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| G（组大小） | 4 | 每个 prompt 生成 4 个回答 |
| lr（学习率） | 0.05 | 玩具模型用大学习率 |
| epsilon（PPO 裁剪） | 0.2 | 标准值 |
| beta（KL 惩罚） | 0.005 | 很小的 KL 权重 |
| temperature | 1.0 | 高温度保证组内多样性 |
| num_steps | 300 | 300 步训练 |
| sync_every | 5 | 每 5 步同步旧策略 |

**结果**：

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| 平均奖励 | 0.475 | 0.912 | **+0.438** |

**结论**：GRPO 算法在玩具任务上有效，奖励从 ~0.5 上升到 ~0.9，模型学会了排序。

**意义**：验证了对 GRPO 6 个步骤（组采样、奖励、优势、策略更新、KL 惩罚、旧策略同步）的理解是正确的，为后续用 TRL 训练真正的 LLM 打下基础。

---

### 训练 2：阶段 2 第一轮 — TRL + 0.5B + 两位数加法（5070Ti）

**时间**：阶段 2 第一次尝试

**目的**：用 TRL 的 GRPOTrainer 训练 Qwen2.5-0.5B-Instruct 学会加法

**任务**：两位数加法（如 25+37=?）

**参数**：

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B-Instruct |
| GPU | RTX 5070 Ti (16GB) |
| G | 4 |
| temperature | 0.7 |
| learning_rate | 5e-6 |
| epsilon | 0.2 |
| beta | 0.1 |
| max_steps | 300 |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 2 |
| max_completion_length | 16 |
| 奖励函数 | correctness_reward（仅正确性） |

**结果**：训练跑完了，但暴露了**核心问题**：

- `frac_reward_zero_std` **几乎一直是 1.0** — 组内 4 个回答要么全对要么全错
- 模型学不到东西，因为优势 = (r - mean) / std，当 std=0 时没有学习信号
- 两位数加法对 0.5B 模型太简单，组内缺乏对错混合

**为什么改参数**：

| 问题 | 原因 | 改进方向 |
|------|------|---------|
| 组内缺乏多样性 | G=4 太小，组内全对/全错概率高 | 增大 G |
| 温度太低 | temp=0.7 太确定，生成相似回答 | 升高温度 |
| 题目太简单 | 两位数加法 0.5B 基本都会 | 换更难的题 |

---

### 训练 3：阶段 2 第二轮 — TRL + 0.5B + 三位数加法（5070Ti）

**时间**：阶段 2 第二次尝试（优化后）

**目的**：解决第一轮 `frac_reward_zero_std=1.0` 的问题

**参数变更**：

| 参数 | 第一轮 | 第二轮 | 改动原因 |
|------|--------|--------|---------|
| G | 4 | **6** | 更大的组，降低全对/全错概率（12.5%→3.1%） |
| temperature | 0.7 | **0.8** | 略升温，增加组内多样性 |
| 题目难度 | 两位数加法 | **三位数加法** | 更难的题让模型更容易答错，增加组内对错混合 |
| batch_size | 4 | **2** | G 增大后降低 batch 省显存（2×6=12序列 vs 4×4=16序列） |
| grad_accum | 2 | **3** | 等效 batch=6，必须能被 G=6 整除 |

**其他参数不变**：lr=5e-6, epsilon=0.2, beta=0.1, max_steps=300

**结果**：

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| 准确率（20道三位数加法） | 50.0% | 80.0% | **+30.0%** |

具体例子（部分）：

```
题目        | 正确答案 | 训练前  | 训练后
900+795    | 1695    | 1795   | 1695  ✓
681+687    | 1368    | 1374   | 1368  ✓
646+602    | 1248    | 1258   | 1248  ✓
762+425    | 1187    | 1187   | 1187  ✓
```

**效果评价**：
- `frac_reward_zero_std` 明显改善，不再一直是 1.0
- 准确率提升 30%，说明 GRPO 训练有效
- 但训练后仍有 20% 答错，模型还有提升空间
- 评估完成后 Jupyter kernel 因 OOM 崩溃（16GB 显存不够同时加载两个模型）

**引出的问题**：第二轮虽然有效，但我们不知道当前参数（K1 KL、3次内更新、clip=0.2）是否最优。于是进入阶段 3 做参数对比实验。

---

### 训练 4-9：阶段 3 参数调优实验（0.5B, 5070Ti, 30步/实验）

**时间**：阶段 3

**目的**：用 transparent-grpo 的手写实现做 3 组对比实验，为阶段 4 的参数选择提供依据

**公共参数**（所有实验相同）：

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B-Instruct |
| GPU | RTX 5070 Ti (16GB) |
| group_size | 4 |
| learning_rate | 5e-6 |
| beta | 0.1 |
| temperature | 0.8 |
| num_steps | 30 |

#### 实验 1（训练 4 vs 训练 5）：K1 vs K3 — KL 放在哪里？

**背景**：GRPO 论文中有两种 KL 放置方式：
- K1：KL 放在 reward 里（`reward = r - beta * KL`）
- K3：KL 放在 loss 里（`loss = pg_loss + beta * KL`）

**参数差异**：

| | 训练 4 (K1) | 训练 5 (K3) |
|--|------------|------------|
| kl_mode | K1（KL in reward） | K3（KL in loss） |
| 其他参数 | 完全相同 | 完全相同 |

**结果**：

| 指标 | K1（训练 4） | K3（训练 5） |
|------|-------------|-------------|
| Reward 变化 | 0.72→0.72 | 0.68→**0.57**（退步！） |
| MaxKL | **9.33** | 22.39 |
| MeanLoss | 0.0161 | 0.0545 |
| MeanGrad | 0.60 | 0.48 |

**结论**：**选 K1**。K1 的 KL 控制力更强（MaxKL 9 vs 22），K3 的 reward 反而退步了。

**对阶段 4 的影响**：TRL 默认使用 K1（KL in reward），与实验结论一致，无需额外配置。

---

#### 实验 2（训练 6 vs 7 vs 8）：inner_update_epochs = 1 vs 3 vs 10

**背景**：每个 batch 的回答生成后，可以用同一批数据更新模型几次。更新太少学得慢，更新太多可能过拟合导致 KL 失控。

**参数差异**：

| | 训练 6 | 训练 7 | 训练 8 |
|--|--------|--------|--------|
| inner_update_epochs | 1 | 3 | 10 |

**结果**：

| 指标 | epochs=1 | epochs=3 | epochs=10 |
|------|----------|----------|-----------|
| Reward 变化 | 0.62→0.80 | 0.70→0.80 | 0.82→0.80（退步） |
| MaxKL | 9.45 | **5.61** | **24.04**（爆炸！） |
| MeanGrad | 0.88 | 0.95 | 0.52 |

**结论**：**选 3 次**。3 次的 MaxKL 最低（5.61），训练最稳定。10 次导致 KL 失控（24.04），模型偏离参考模型太远。

**对阶段 4 的影响**：TRL 默认 `num_iterations=3`，与实验结论一致，无需额外配置。

---

#### 实验 3（训练 9a vs 9b vs 9c）：clip_epsilon = 0.1 vs 0.2 vs 0.3

**背景**：PPO 裁剪范围控制每次更新的幅度。太小（0.1）学得慢但稳定，太大（0.3）学得快但可能失控。

**参数差异**：

| | 训练 9a | 训练 9b | 训练 9c |
|--|---------|---------|---------|
| clip_epsilon | 0.1（保守） | 0.2（默认） | 0.3（激进） |

**结果**：

| 指标 | clip=0.1 | clip=0.2 | clip=0.3 |
|------|----------|----------|----------|
| Reward 变化 | 0.68→0.78 | 0.60→0.60 | 0.65→0.78 |
| MaxKL | **3.84**（最稳定） | 14.21 | **24.47**（爆炸！） |
| MeanGrad | 1.63 | 0.76 | 1.06 |

**结论**：0.1 的 KL 控制最好（3.84），但 reward 提升一般。0.2 是效率与稳定的平衡点。0.3 导致 KL 爆炸。

**对阶段 4 的影响**：选择 **epsilon=0.2**（TRL 默认值），在 30 步短实验中表现一般，但 500 步长训练中更可靠。

---

#### 阶段 3 实验中的 OOM 问题

5070Ti (16GB) 连续跑 8 个实验时 OOM，原因是 Jupyter kernel 占用 9GB 显存未释放。

**修复**：
1. `kill` 占用显存的进程
2. 实验间添加 `gc.collect()` + `torch.cuda.empty_cache()` + `time.sleep(2)`
3. `group_size` 固定为 4（不使用更大的组）

---

### 训练 10：阶段 4 — 1.5B + G=8 + 4090 服务器（最终训练）

**时间**：阶段 4，最终正式训练

**目的**：在 4090 (48GB) 上用更大的模型和更大的组做正式 GRPO 训练

**基于阶段 3 实验的参数选择**：

| 参数 | 阶段 2 第二轮 | 阶段 4 | 改动原因 |
|------|--------------|--------|---------|
| 模型 | 0.5B | **1.5B** | 4090 显存够，更大模型能力更强 |
| G | 6 | **8** | G 越大 zero_std 概率越低（3.1%→0.8%），优势估计更准 |
| max_steps | 300 | **500** | 更大模型需要更多步数收敛 |
| warmup_steps | 20 | **30** | 配合 500 步，稍长 warmup |
| batch_size | 2 | **4** | 4090 显存大，可以增大 batch |
| grad_accum | 3 | **2** | 4×2=8，能被 G=8 整除 |
| max_completion_length | 16 | **32** | 给更长生成空间 |
| 奖励函数 | 仅正确性 | **正确性 + 格式** | 更细粒度信号，增加组内奖励多样性 |
| vLLM | 不用 | **计划用→关闭** | TRL 1.9.2 + vLLM 版本不兼容 |
| 环境 | Conda | **Docker** | 服务器环境隔离，可复现 |

**不变参数**（来自阶段 3 实验结论）：

| 参数 | 值 | 依据 |
|------|-----|------|
| learning_rate | 5e-6 | 一直保持，配合 warmup 稳定 |
| epsilon | 0.2 | 阶段 3 实验 3 结论：0.2 是平衡点 |
| beta | 0.1 | 一直保持 |
| temperature | 0.8 | 阶段 2 第二轮已验证 |
| KL 模式 | K1（TRL 默认） | 阶段 3 实验 1 结论：K1 优于 K3 |
| inner_update_epochs | 3（TRL 默认） | 阶段 3 实验 2 结论：3 次最稳定 |

**部署过程中的 5 轮调试**：

| 轮次 | 报错 | 修复 |
|------|------|------|
| 1 | `GRPOConfig` 不接受 `model_name_or_path` | TRL 1.9.2 移除了该参数，改为传给 `GRPOTrainer` |
| 2 | vLLM 0.26 与 TRL 1.9.2 不兼容 | 关闭 vLLM（`use_vllm=False`），从 Dockerfile 移除 vLLM |
| 3 | `clip_epsilon` 参数不存在 | TRL 1.9.2 改名为 `epsilon`，`temperature`/`do_sample` 移入 `generation_kwargs` |
| 4 | `GRPOTrainer` 不接受 `config=` | TRL 1.9.2 改为 `args=config` |
| 5 | 奖励函数收到非字符串 completion | 添加 `extract_text()` 兼容 str 和 list[dict] 格式 |

**最终训练参数**（`stage4_train.py`）：

```python
GRPOConfig(
    num_generations=8,
    max_completion_length=32,
    generation_kwargs={"temperature": 0.8, "do_sample": True},
    use_vllm=False,
    learning_rate=5e-6,
    max_steps=500,
    warmup_steps=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    beta=0.1,
    epsilon=0.2,
    max_grad_norm=0.5,
    bf16=True,
    gradient_checkpointing=True,
)
```

**奖励函数**：

| 奖励函数 | 最高分 | 条件 |
|---------|-------|------|
| correctness_reward | 1.0 | 答案正确 |
| format_reward | 0.2 | 回答是 3-4 位纯数字 |
| **总计** | **1.2** | 正确且格式好 |

**训练过程指标变化**：

| 阶段 | Step | Reward | KL | Entropy | frac_reward_zero_std |
|------|------|--------|-----|---------|---------------------|
| 初期 | 0 | 0.7 | 0 | 0.77 | 0 |
| 中期 | 164 | 0.7-1.2 | 1.67 | 0.77 | 0-1 |
| 中后期 | 295 | 0.95 | 0.20 | 0.03 | 0 |
| 后期 | 395 | 1.2 | 0.01 | 0.001 | 1 |
| 末期 | 500 | 1.2 | 0 | 0.002 | 1 |

> 注：后期 `frac_reward_zero_std=1` 是因为模型全答对了（reward=1.2 满分），不是没有学习信号。

**训练统计**：

| 指标 | 值 |
|------|-----|
| 总步数 | 500 |
| 总耗时 | 5 分 45 秒 |
| 每步耗时 | ~0.25 秒 |
| 训练速度 | 2.28 it/s |
| 最终 train_loss | 0.0925 |
| 最终 reward | 1.2（满分） |
| 最终 KL | ~0（完全收敛） |

**评估结果**（20 道三位数加法，seed=999，greedy decoding）：

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| 准确率 | 90.0% | 100.0% | **+10.0%** |

---

### 所有训练一览表

| # | 阶段 | 模型 | GPU | G | temp | lr | epsilon | beta | steps | 奖励 | 结果 |
|---|------|------|-----|---|------|-----|---------|------|-------|------|------|
| 1 | 1 | ToyPolicy | CPU | 4 | 1.0 | 0.05 | 0.2 | 0.005 | 300 | 正确率 | 0.475→0.912 |
| 2 | 2-1 | 0.5B | 5070Ti | 4 | 0.7 | 5e-6 | 0.2 | 0.1 | 300 | 正确性 | zero_std≈1.0，学不到东西 |
| 3 | 2-2 | 0.5B | 5070Ti | 6 | 0.8 | 5e-6 | 0.2 | 0.1 | 300 | 正确性 | 50%→80% (+30%) |
| 4 | 3-1a | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | K1: MaxKL=9.33 ✓ |
| 5 | 3-1b | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | K3: MaxKL=22.39, reward 退步 |
| 6 | 3-2a | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | epochs=1: MaxKL=9.45 |
| 7 | 3-2b | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | epochs=3: MaxKL=5.61 ✓ |
| 8 | 3-2c | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | epochs=10: MaxKL=24.04 爆炸 |
| 9a | 3-3a | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.1 | 0.1 | 30 | 正确性 | clip=0.1: MaxKL=3.84 最稳 |
| 9b | 3-3b | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.2 | 0.1 | 30 | 正确性 | clip=0.2: MaxKL=14.21 平衡 |
| 9c | 3-3c | 0.5B | 5070Ti | 4 | 0.8 | 5e-6 | 0.3 | 0.1 | 30 | 正确性 | clip=0.3: MaxKL=24.47 爆炸 |
| 10 | 4 | 1.5B | 4090 | 8 | 0.8 | 5e-6 | 0.2 | 0.1 | 500 | 正确性+格式 | 90%→100% (+10%) |

---

### 参数演进逻辑链

```
阶段1（玩具）
  └─ G=4, lr=0.05, epsilon=0.2, beta=0.005 → reward 0.475→0.912 ✓
     │
     ▼
阶段2第一轮（0.5B, 两位数加法）
  └─ G=4, temp=0.7 → zero_std≈1.0，学不到东西 ✗
     │  问题：组内缺乏多样性
     ▼
阶段2第二轮（0.5B, 三位数加法）
  └─ G=4→6, temp=0.7→0.8, 换三位数 → 50%→80% ✓
     │  但不知道 K1/K3、epochs、clip 哪个最优
     ▼
阶段3实验（0.5B, 30步/实验）
  ├─ K1 vs K3 → K1 胜（MaxKL 9 vs 22）
  ├─ epochs 1/3/10 → 3 胜（MaxKL 5.61）
  └─ clip 0.1/0.2/0.3 → 0.2 选为平衡点
     │  实验结论指导阶段4参数
     ▼
阶段4（1.5B, 4090, G=8, 500步）
  └─ K1 + epochs=3 + clip=0.2 + G=8 + 双奖励 → 90%→100% ✓
```

---

## 二、环境与部署

### 硬件

| 角色 | 机器 | GPU | 显存 |
|------|------|-----|------|
| 主控机 | 5070Ti | RTX 5070 Ti | 16GB |
| 训练机 | 4090 服务器 (192.168.1.103) | RTX 4090 魔改 | 48GB |

### 4090 服务器环境

| 项目 | 值 |
|------|-----|
| OS | Ubuntu Server 24.04.3 LTS |
| GPU 驱动 | 580.173.02 |
| Docker | 29.1.3 + NVIDIA Container Toolkit |
| SSH | `ssh 4090`（免密已配） |
| CUDA（Docker 内） | 12.4.1 |
| 存储 | NVMe 891G（热数据）+ HDD 3.4T（冷数据） |

---

## 二、阶段 4 部署过程

### 2.1 文件准备（5070Ti 本机）

创建了 4 个核心文件：

| 文件 | 用途 |
|------|------|
| `stage4_train.py` | 训练脚本（1.5B + G=8 + 双奖励函数） |
| `stage4_deploy.sh` | 部署脚本（检查环境 + 构建 Docker + 运行） |
| `Dockerfile` | Docker 镜像定义（CUDA 12.4 + PyTorch + TRL） |
| `stage4_eval.py` | 评估脚本（对比训练前后准确率） |

### 2.2 模型下载

4090 上没有 1.5B 模型，通过 modelscope 下载：

```bash
ssh 4090 "source /home/jzd/miniconda3/etc/profile.d/conda.sh && conda activate base && \
  pip install modelscope && \
  python -c \"from modelscope import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/mnt/nvme_gm9_1tb/models')\""
```

modelscope 下载到 `/mnt/nvme_gm9_1tb/models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master/`，创建符号链接简化路径：

```bash
ln -sf .../snapshots/master /mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct
```

### 2.3 文件传输

```bash
rsync -avz --progress "/data/RL/The Illustrated GRPO/" \
  4090:"/mnt/nvme_gm9_1tb/grpo-tutorial/"
```

### 2.4 Docker 镜像构建

```bash
ssh 4090 "cd /mnt/nvme_gm9_1tb/grpo-tutorial && sudo docker build -t grpo-stage4 ."
```

**Dockerfile 内容**：

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# 安装 Python 3.10 + pip
# 安装 PyTorch 2.6.0 (CUDA 12.4)
# 安装 TRL 1.9.2 + Transformers 5.14.1 + Datasets + Accelerate + TensorBoard
```

构建耗时约 15 分钟（首次需下载 PyTorch ~2.5GB + CUDA 库 ~2GB）。

最终镜像大小：28.7GB。

### 2.5 路径适配

根据 `4090_agent_上手指南.md` 调整了所有路径：

| 项目 | 原路径（假想） | 实际路径 |
|------|---------------|---------|
| 模型 | `/data/models/Qwen2.5-1.5B-Instruct` | `/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct` |
| 输出 | `output/grpo_1.5b_addition` | `/mnt/hdd_wd_4tb/grpo_output/grpo_1.5b_addition` |
| 日志 | `logs/stage4` | `/mnt/hdd_wd_4tb/grpo_output/logs/stage4` |
| Docker | 无需 sudo | 需要 `sudo docker` |

---

---

## 三、阶段 4 部署踩坑与修复（共 5 轮）

> 这 5 轮调试都发生在阶段 4 的部署过程中，详见上方「训练 10」的描述。

---

## 四、4090 上的文件布局

```
/mnt/nvme_gm9_1tb/                              ← NVMe SSD (891G)
├── models/
│   ├── Qwen2.5-1.5B-Instruct/                  ← 原始模型 (2.9G，符号链接)
│   └── models/                                 ← modelscope 缓存
│       └── Qwen--Qwen2.5-1.5B-Instruct/
│           └── snapshots/master/               ← 实际文件位置
│
└── grpo-tutorial/                              ← 项目代码
    ├── Dockerfile
    ├── stage4_train.py
    ├── stage4_eval.py
    ├── stage4_deploy.sh
    ├── stage4_notebook.ipynb
    └── ...（阶段 1-3 的文件）

/mnt/hdd_wd_4tb/                                ← HDD (3.4T)
└── grpo_output/
    └── grpo_1.5b_addition/                     ← 训练输出
        ├── model.safetensors                   (5.8G，训练后模型)
        ├── config.json
        ├── tokenizer.json
        ├── training_args.bin
        ├── checkpoint-300/                     (18G，含 optimizer state)
        ├── checkpoint-400/                     (18G)
        ├── checkpoint-500/                     (18G)
        └── runs/                               ← TensorBoard 日志
            └── Aug03_06-26-13_.../
                └── events.out.tfevents         (841K)
```

---

## 六、经验总结

### 做对了的事

1. **先在本机 (5070Ti) 做参数实验**：用 0.5B 小模型跑 K1/K3、epochs、clip 对比，为 4090 上的参数选择提供依据
2. **先读 `4090_agent_上手指南.md`**：避免了路径错误、Docker 权限问题
3. **逐步调试**：每次报错只改一个地方，确认后再继续

### 踩的坑

1. **vLLM 版本兼容**：TRL 1.9.2 + vLLM 0.26 不兼容，vLLM 0.25 需要 CUDA 13。最终放弃 vLLM，影响不大（加法任务生成短）
2. **TRL API 变更**：1.9.2 大量参数名变了（`clip_epsilon`→`epsilon`，`config`→`args`，`model_name_or_path` 移到 Trainer）
3. **Transformers 5.x API 变更**：`apply_chat_template` 返回格式变了，`torch_dtype` 废弃
4. **modelscope 下载路径**：不是直接的模型名目录，而是 `models/Qwen--模型名/snapshots/master/`，需要符号链接
5. **5070Ti OOM**：Jupyter kernel 占显存不释放，需要手动 kill + gc.collect

### 下一步建议

1. **用更难的题**：1.5B 基座 90% 太强，三位数加法不够有挑战性。阶段 5 可以用四位数加法或两位数乘法
2. **清理 checkpoints**：3 个 checkpoint 共 54G，只保留 checkpoint-500 即可省 36G
3. **尝试 vLLM**：如果后续需要更长生成（如推理任务），需要解决 vLLM 版本问题（可能需要降级 TRL 或升级 CUDA）
