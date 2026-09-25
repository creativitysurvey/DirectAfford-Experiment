#!/bin/bash
echo "======================================================="
echo "🔥 启动 Exp 2: 复杂情境查询 (LASO) 真实数据集从零重训计划"
echo "======================================================="

DATA_DIR="./data/laso_dataset"

echo "1. 启动 OpenAD (InfoNCE) 真实训练..."
nohup python src/train_openad.py --data_dir $DATA_DIR --epochs 3000 > openad_laso.log 2>&1 &

echo "2. 启动 Aff3DFunc (SupCon) 真实训练..."
nohup python src/train_aff3dfunc.py --data_dir $DATA_DIR --epochs 3000 > aff3dfunc_laso.log 2>&1 &

echo "3. 启动 DirectAfford (Ours) 真实训练..."
nohup python src/train.py --data_dir $DATA_DIR --epochs 3000 > directafford_laso.log 2>&1 &

echo "======================================================="
echo "✅ 所有训练任务已在后台挂载并启动！"
echo "🖥️ 显卡火力全开中..."
echo "您现在可以安全关闭终端了。随时可以使用以下命令查看进度："
echo "tail -f directafford_laso.log"
echo "======================================================="
