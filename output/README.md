# output/ 目录说明

本目录保存 GRPO 训练过程中产生的记录文件，**不含模型权重**（已被 .gitignore 排除）。

## 目录结构

```
output/
├── README.md                      # 本文件
├── stage4_eval_results.md         # 阶段 4 评估结果汇总（90%→100%）
├── stage4_trainer_state.json      # 阶段 4 完整 500 步训练历史（每步所有指标）
├── stage4_runs/                   # 阶段 4 TensorBoard 日志（4090 训练）
│   ├── Aug03_06-25-49_*/          #    tfevents 文件（7.5KB，初始日志）
│   └── Aug03_06-26-13_*/          #    tfevents 文件（840KB，主训练日志）
├── runs/                          # 阶段 2 TensorBoard 日志（5070Ti 训练）
│   ├── Aug02_21-26-14_*/          #    第一轮训练（G=4，两位数加法）
│   ├── Aug02_23-25-32_*/          #    第二轮训练（G=6，三位数加法）
│   ├── Aug02_23-35-16_*/
│   └── Aug03_00-13-34_*/
└── completions/                   # 阶段 2 训练中模型生成的回答记录
    └── completions_00001~00300.parquet  # 300 步，每步保存模型生成的回答
```

## 如何查看 TensorBoard 日志

```bash
# 安装 tensorboard
pip install tensorboard

# 启动
tensorboard --logdir output/

# 浏览器打开 http://localhost:6006
```

## 如何查看训练历史

```python
import json

with open("output/stage4_trainer_state.json") as f:
    state = json.load(f)

# 查看每步的指标
for entry in state["log_history"][:5]:
    print(f"Step {entry['step']}: reward={entry['reward']:.2f}, kl={entry['kl']:.4f}")
```

## 框架版本（阶段 4 Docker 环境）

- TRL: 1.9.2
- Transformers: 5.14.1
- PyTorch: 2.11.0+cu128
- Datasets: 5.0.1
- Tokenizers: 0.22.2