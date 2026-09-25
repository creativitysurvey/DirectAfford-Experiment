# DirectAfford-Experiment

代码与实验整理仓库。该仓库当前为私有仓库，包含 DirectAfford、LASO/OpenAD/PointBERT 相关代码和实验脚本。

> **复现状态说明（请先读）**：仓库目前没有附带数据集或模型权重。现有训练代码、数据格式和启动脚本之间存在不一致，因此下面给出的是环境准备、数据来源和代码当前实际输入要求；不能据此认为所有论文实验已经在干净环境中端到端复现。请按本文核对数据后再启动训练。

## 目录概览

- `src/`：模型、数据集、训练和评估代码。
- `src/baselines/`：基线实现。
- `environment.yml`：Conda 环境依赖。
- `download_clip.py`：下载 CLIP 文本编码器。
- `download_openshape.py`：从 Hugging Face 下载 OpenShape Point-BERT 权重。
- `run_laso_training.sh`、`start_exp2_training.sh`：实验启动脚本；目前参数和所引用脚本与仓库代码并不完全匹配，见“已知复现限制”。

## 环境安装

建议使用 Linux、Python 3.10 和支持 CUDA 的 NVIDIA GPU。仓库环境文件指定 PyTorch 2.0.1 / CUDA 11.8：

```bash
git clone git@github.com:creativitysurvey/DirectAfford-Experiment.git
cd DirectAfford-Experiment

conda env create -f environment.yml
conda activate directafford_env

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

若只做 CPU 测试，数据预处理和模型运行仍可能很慢；正式训练建议使用 CUDA GPU。

## 数据获取与目录

### LASO 数据集

LASO 官方代码仓库提供数据集下载说明和 Google Drive 链接：

- 官方代码与说明：[yl3800/LASO](https://github.com/yl3800/LASO)
- 官方数据集下载：[Google Drive](https://drive.google.com/file/d/1P4CwQeSALtUOgzhg1ILCiKovE-tcXlSu/view)

请遵循数据集发布方的许可条款。下载后解压到本地，不要提交到 Git。当前仓库的 LASO 数据扫描脚本读取：

```text
data/laso_dataset/LASO_dataset/anno_val.pkl
```

训练与评估代码对 LASO 的目录/文件名要求并不统一，运行前应先检查对应脚本的参数和数据读取逻辑。

### DirectAfford full-shape 数据

仓库中的 `src/dataset.py` 和 `src/train.py` 实际读取以下 pickle 文件：

```text
data/full_shape_train_data.pkl
data/full_shape_val_data.pkl
```

每个样本至少需要包含 `full_shape`（含 `coordinate` 与按 affordance 标注的 `label`）和 `affordance` 字段。当前仓库没有给出这两个文件的公开下载链接、生成脚本或完整数据转换流程。若你没有从项目原始数据来源获得这两个文件，仅下载 LASO 并不能直接满足该训练入口的输入要求。不要把这些数据文件上传到 GitHub；请在取得授权后通过合适的数据托管服务分享，并在此处补充固定版本和校验和。

## 预训练权重获取

### CLIP 文本编码器

模型默认使用 Hugging Face 上的 `openai/clip-vit-base-patch32`。可按需下载到仓库根目录的 `clip-vit-base-patch32/`：

```bash
python download_clip.py
```

也可以让 Transformers 在首次运行时从 Hugging Face 下载。离线环境需先在可联网机器下载模型，再按 Transformers 缓存或本地目录方式传入。

### OpenShape Point-BERT 权重（可选）

`download_openshape.py` 会下载 Hugging Face 仓库 `OpenShape/openshape-pointbert-shapenet` 的 `model.pt`，并保存为 `data/pointbert.pt`：

```bash
python -m pip install huggingface_hub
python download_openshape.py
```

此文件是 OpenShape 预训练权重，不是本仓库训练得到的 DirectAfford checkpoint。使用前请确认对应评估/基线脚本确实读取该路径。

### DirectAfford 训练权重

训练入口将模型权重写到 `data/directafford_weights.pth`。仓库未提供该权重的下载地址；需使用兼容的数据运行训练获得。训练输出和 checkpoint 已被 `.gitignore` 排除。

## 当前代码的 DirectAfford 训练入口

在仓库根目录先准备好上述两个 `full_shape_*.pkl` 文件以及 CLIP 权重，再执行：

```bash
python src/train.py
```

当前 `src/train.py` 不解析命令行参数；它默认读取 `./data/full_shape_train_data.pkl` 和 `./data/full_shape_val_data.pkl` 所在的 `data/` 目录，使用 fold 0、2048 点、batch size 16，并训练 3 个 epoch。权重保存到 `data/directafford_weights.pth`。这些参数是代码当前默认值，不代表论文配置已经完整核验。

LASO 相关的评估脚本位于 `src/eval*.py`。每个脚本使用的数据路径、checkpoint 路径和参数可能不同，运行前请查看其顶部配置和命令行参数，并将数据与权重路径设置为本机实际位置。

## 已知复现限制

- `run_laso_training.sh` 调用 `src/train_laso.py` 和 `src/train_pointbert.py`，但这两个入口并未出现在仓库当前文件清单中；该脚本也把 LASO 路径设为 `./data/laso`。不能直接视为可运行的完整训练流程。
- `start_exp2_training.sh` 向训练脚本传递 `--data_dir` 和 `--epochs`，而仓库当前 `src/train.py` 不解析这些参数；它不能按脚本说明可靠地启动训练。
- LASO 原始数据与 `src/dataset.py` 需要的 `full_shape_*.pkl` 是不同的数据格式。仓库尚未包含从官方 LASO 数据到此格式的完整转换说明。
- 首次运行前，请先在小规模数据上核验数据字段、模型前向和保存路径，再启动长时间训练；记录实际 commit、配置、随机种子、数据版本和权重哈希，便于复核结果。

## 引用

若使用 LASO 数据集或其基线，请引用 LASO 原论文，并遵循官方仓库中列出的引用与许可要求：

```bibtex
@InProceedings{Li_2024_CVPR,
  author    = {Li, Yicong and Zhao, Na and Xiao, Junbin and Feng, Chun and Wang, Xiang and Chua, Tat-Seng},
  title     = {LASO: Language-guided Affordance Segmentation on 3D Object},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
  pages     = {14251--14260}
}
```
