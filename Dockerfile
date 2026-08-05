# ============================================================
# 阶段 4：Docker 镜像
# 基于 NVIDIA CUDA 12.4 镜像，安装 PyTorch + TRL + vLLM
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 别名
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 安装 PyTorch（CUDA 12.4）
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# 安装 TRL 和相关库
RUN pip install \
    transformers \
    trl \
    datasets \
    accelerate \
    tensorboard \
    modelscope

# 设置工作目录
WORKDIR /workspace

# 默认命令
CMD ["/bin/bash"]
