# 4090算力节点：零记忆Agent上手指南

> 本文件专为"没有对话记忆的新agent"设计。新agent只需读本文件即可开始操作4090。
> 完整装机记录见同目录下 `4090_ubuntu_server_装机记录.md`。

## 环境概况

| 项目 | 值 |
|---|---|
| 4090主机IP | `192.168.1.103` |
| SSH用户 | `jzd` |
| SSH别名 | `ssh 4090`（已在5070Ti的 `~/.ssh/config` 配好） |
| 免密SSH | ✅ 已配置（5070Ti → 4090） |
| 免密sudo | ✅ 已配置（`jzd` 用户无需密码执行sudo） |
| 操作系统 | Ubuntu Server 24.04.3 LTS |
| 内核 | 6.8.0-71-generic |
| GPU | RTX 4090 48GB（魔改），驱动 580.173.02，CUDA ≤ 13.0 |
| Docker | 29.1.3 + NVIDIA Container Toolkit（`--gpus all` 已可用） |
| Conda | Miniconda 26.5.3（`/home/jzd/miniconda3`，base环境Python 3.14.6） |
| 代理 | Docker通过 `192.168.1.100:7897`（5070Ti Clash Verge）访问外网 |
| WoL | `boot4090` 命令可远程开机（MAC: `9c:6b:00:97:3b:4b`） |

## 快速验证（新agent第一步）

执行以下命令确认4090在线且环境正常：
```bash
ssh 4090 "hostname && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && docker --version && df -h /mnt/nvme_gm9_1tb /mnt/hdd_wd_4tb | tail -2"
```

预期输出：
```
4090
NVIDIA GeForce RTX 4090, 49140 MiB, 580.173.02
Docker version 29.1.3, build 29.1.3-0ubuntu3~24.04.2
/dev/nvme1n1p1  938G  ...  891G   1% /mnt/nvme_gm9_1tb
/dev/sda1       3.6T  ...  3.4T   1% /mnt/hdd_wd_4tb
```

如果SSH连不上，4090可能关机了，在5070Ti终端执行 `boot4090` 远程开机，等待30秒后重试。

## 存储路径

| 路径 | 容量 | 用途 |
|---|---|---|
| `/var/lib/docker`（系统盘） | 1.8T空闲 | Docker默认存储（镜像/容器/卷） |
| `/mnt/nvme_gm9_1tb/models` | 891G | 热数据：当前训练用的模型权重 |
| `/mnt/nvme_gm9_1tb/datasets` | | 热数据：当前使用的数据集 |
| `/mnt/nvme_gm9_1tb/docker` | | 预留：Docker数据目录迁移用 |
| `/mnt/hdd_wd_4tb/models_archive` | 3.4T | 冷数据：归档模型 |
| `/mnt/hdd_wd_4tb/datasets_archive` | | 冷数据：归档数据集 |
| `/mnt/hdd_wd_4tb/logs` | | 训练日志 |
| `/mnt/hdd_wd_4tb/downloads` | | 下载的原始文件 |

## 常用操作速查

**跑GPU容器**：
```bash
ssh 4090 "sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi"
```

**拉取Docker镜像**（已配代理，可直接pull）：
```bash
ssh 4090 "sudo docker pull <image>"
```

**挂载数据到容器**：
```bash
ssh 4090 "sudo docker run --gpus all -v /mnt/nvme_gm9_1tb/models:/models -v /mnt/nvme_gm9_1tb/datasets:/datasets <image>"
```

**执行sudo命令**（免密，无需 `-t`）：
```bash
ssh 4090 "sudo <command>"
```

**查看GPU状态**：
```bash
ssh 4090 "nvidia-smi"
```

**查看磁盘空间**：
```bash
ssh 4090 "df -h /mnt/nvme_gm9_1tb /mnt/hdd_wd_4tb"
```

**使用conda**（SSH非交互式需先source）：
```bash
ssh 4090 "source /home/jzd/miniconda3/etc/profile.d/conda.sh && conda activate base && python --version"
```

**创建新conda环境**：
```bash
ssh 4090 "source /home/jzd/miniconda3/etc/profile.d/conda.sh && conda create -n myenv python=3.11 -y"
```

## 监控工具

三个工具均已安装，不占GPU显存，只读GPU状态：

| 工具 | 类型 | 启动方式 | 访问方式 | 说明 |
|---|---|---|---|---|
| **btop** | 终端TUI | `ssh 4090 -t "btop"` | SSH终端 | 按需运行，退出即消失，~5MB内存 |
| **glances** | 终端+Web | `ssh 4090 -t "glances"`（终端）<br>`ssh 4090 -t "glances -w"`（Web） | 终端 或 `http://192.168.1.103:61208` | 按需运行，Web模式~50MB内存 |
| **netdata** | Web常驻 | systemd自启 | `http://192.168.1.103:19999` | 开机自动运行，~200-500MB内存，70+GPU指标 |

**从5070Ti浏览器直接访问**：
- Netdata仪表盘：`http://192.168.1.103:19999`（实时，含GPU利用率/温度/功耗/显存/编码器等历史图表）
- Glances Web：`http://192.168.1.103:61208`（需先在4090上启动 `glances -w`）

**SSH终端快速查看**：
```bash
ssh 4090 -t "btop"          # 终端实时监控（CPU/内存/GPU/磁盘/网络）
ssh 4090 -t "glances"       # 另一个终端监控（含Docker容器）
```

**停止/启动netdata**：
```bash
ssh 4090 "sudo systemctl stop netdata"   # 停止（释放内存）
ssh 4090 "sudo systemctl start netdata"  # 启动
```

## 注意事项

- **不要在4090上直接装CUDA toolkit**：采用混合方案（驱动在系统层，CUDA由Docker/conda管理），不同容器可用不同CUDA版本
- **Docker镜像拉取走代理**：已通过systemd drop-in配置 `HTTP_PROXY=http://192.168.1.100:7897`，`docker pull` 自动走代理，无需额外设置
- **数据盘命名规则**：`/mnt/硬盘类型_品牌_容量`，未来加盘延续此规则
- **5070Ti是主控机**：所有操作从5070Ti发起，5070Ti上的agent通过SSH控制4090
- **装机记录全文**：`/data/test/4090 setup/4090_ubuntu_server_装机记录.md`，包含完整装机过程和排错记录
