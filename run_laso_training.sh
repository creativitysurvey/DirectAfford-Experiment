#!/bin/bash
echo "======================================================="
echo "🚀 启动 LASO 真实数据集 [几天几夜] 从零重训马拉松"
echo "======================================================="

# 设置 LASO 数据集路径环境变量，让之前的 Python 脚本读取
export DATASET_PATH="./data/laso"
export WEIGHTS_OUTPUT_DIR="./data/baselines_laso"
mkdir -p $WEIGHTS_OUTPUT_DIR

echo "▶️ [1/4] 开始重训 OpenAD (InfoNCE)..."
python src/train_openad.py --dataset_path $DATASET_PATH --output_dir $WEIGHTS_OUTPUT_DIR

echo "▶️ [2/4] 开始重训 Aff3DFunc (SupCon)..."
python src/train_aff3dfunc.py --dataset_path $DATASET_PATH --output_dir $WEIGHTS_OUTPUT_DIR

echo "▶️ [3/4] 开始重训 LASO Proxy (Contrastive with Dense Attention)..."
# 注意：您需要使用类似于 train_kdtpc.py 的逻辑为 LASO 写一个训练脚本
python src/train_laso.py --dataset_path $DATASET_PATH --output_dir $WEIGHTS_OUTPUT_DIR

echo "▶️ [4/4] 开始重训 Point-BERT (Pure Hierarchical)..."
python src/train_pointbert.py --dataset_path $DATASET_PATH --output_dir $WEIGHTS_OUTPUT_DIR

echo "▶️ [5/5] 开始重训 DirectAfford (Ours)..."
python src/train.py --dataset_path $DATASET_PATH --output_dir $WEIGHTS_OUTPUT_DIR

echo "✅ 史诗级重训完成！所有真实物理权重已就绪。"
