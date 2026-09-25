import sys
import os
import torch
import inspect

# 强行注入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'baselines/PointBERT')))
import models.Point_BERT as Point_BERT_module

print("\n🔍 启动 Point-BERT 内部架构 X光透视...")

# ==========================================
# 1. 暴力解包官方权重
# ==========================================
weight_path = "./data/baselines/pointbert_official.pth"
print(f"🔄 正在强行解包权重文件: {weight_path}")
checkpoint = torch.load(weight_path, map_location='cpu')

if isinstance(checkpoint, dict):
    print(f"[*] Checkpoint 顶层 Keys: {list(checkpoint.keys())}")
    # 智能寻找权重字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

print(f"[*] 提取到的有效权重层数: {len(state_dict)}")
print("[*] 窥探真实物理权重的部分键名 (前 10 个):")
for k in list(state_dict.keys())[:10]:
    print(f"  - {k}")

# ==========================================
# 2. 反向扫描源码定义的类
# ==========================================
print("\n⚙️ 逆向工程: 扫描源码中的模型类与参数...")
for name, obj in inspect.getmembers(Point_BERT_module, inspect.isclass):
    # 只看他们自己写的类，过滤掉引入的库
    if obj.__module__ == 'models.Point_BERT':
        try:
            sig = inspect.signature(obj.__init__)
            print(f"  👉 发现核心网络类: {name}")
            print(f"     需要传入参数: {sig}")
        except ValueError:
            pass

print("-" * 50)
print("✨ 透视完成！")
