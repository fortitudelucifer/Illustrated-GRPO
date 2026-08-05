#!/bin/bash
# ============================================================
# 阶段 4：部署脚本
# 在 4090 服务器上构建 Docker 镜像并运行训练
# 适配 4090 实际环境（见 4090_agent_上手指南.md）
# ============================================================

set -e

# 配置
IMAGE_NAME="grpo-stage4"
CONTAINER_NAME="grpo-train"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "阶段 4：部署到 4090 服务器"
echo "============================================================"

# 1. 检查 Docker + GPU（4090 已装 Docker + NVIDIA Container Toolkit）
echo ">>> 检查 Docker 和 GPU 环境..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1
echo "Docker + GPU 环境正常"

# 2. 检查模型是否存在
MODEL_PATH="/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：模型不存在: $MODEL_PATH"
    echo "请先下载模型："
    echo "  ssh 4090 \"source /home/jzd/miniconda3/etc/profile.d/conda.sh && conda activate base && python -c \\\"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/mnt/nvme_gm9_1tb/models')\\\"\""
    exit 1
fi
echo "模型存在: $MODEL_PATH"

# 3. 创建输出目录
sudo mkdir -p /mnt/hdd_wd_4tb/grpo_output/grpo_1.5b_addition
sudo mkdir -p /mnt/hdd_wd_4tb/grpo_output/logs/stage4
sudo chown -R $(whoami) /mnt/hdd_wd_4tb/grpo_output

# 4. 构建镜像
echo ""
echo ">>> 构建 Docker 镜像（首次约 10-15 分钟，走代理）..."
sudo docker build -t $IMAGE_NAME "$PROJECT_DIR"

# 5. 运行训练
echo ""
echo ">>> 启动训练容器..."
echo "    模型: /mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct"
echo "    项目: $PROJECT_DIR"
echo "    输出: /mnt/hdd_wd_4tb/grpo_output/"
echo ""

sudo docker run --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size=16g \
    -v /mnt/nvme_gm9_1tb/models:/mnt/nvme_gm9_1tb/models:ro \
    -v /mnt/hdd_wd_4tb/grpo_output:/mnt/hdd_wd_4tb/grpo_output \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    $IMAGE_NAME \
    python stage4_train.py

echo ""
echo "============================================================"
echo "训练完成！"
echo "  模型输出: /mnt/hdd_wd_4tb/grpo_output/grpo_1.5b_addition/"
echo "  TensorBoard 日志: /mnt/hdd_wd_4tb/grpo_output/logs/stage4/"
echo ""
echo "查看 TensorBoard："
echo "  ssh 4090 -L 6006:localhost:6006"
echo "  sudo docker run --rm --gpus all -p 6006:6006 -v /mnt/hdd_wd_4tb/grpo_output:/data grpo-stage4 tensorboard --logdir=/data/logs/stage4 --host=0.0.0.0 --port=6006"
echo "============================================================"
