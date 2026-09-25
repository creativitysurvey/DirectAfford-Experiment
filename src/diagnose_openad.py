import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'baselines/OpenAD')))
from models.openad_pn2 import OpenAD_PN2

class DummyArgs:
    def __init__(self):
        self.model = 'openad_pn2'
        self.emb_dims = 512
        self.k = 20
args = DummyArgs()

print("\n🔍 启动 OpenAD 内部架构 X光透视...")
# 核心修复：把模型推到 GPU
model = OpenAD_PN2(args, num_classes=1, normal_channel=False).cuda()
weight_path = "./data/baselines/openad_official.pth"
checkpoint = torch.load(weight_path, map_location='cuda')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ 权重装载基本正常。")
print("\n⚙️ 注入随机点云，测试真实 Forward 结构...")

# 核心修复：把点云也推到 GPU
dummy_pc = torch.randn(1, 3, 2048).cuda()

with torch.no_grad():
    out = model(dummy_pc, ["sit"])

print("-" * 50)
print(f"[*] 模型真实返回 shape: {out.shape}")
print(f"[*] 预测值的最大值: {out.max().item():.6f}")
print(f"[*] 预测值的最小值: {out.min().item():.6f}")
print(f"[*] 预测值的均值: {out.mean().item():.6f}")
print("[*] 前 10 个原始 Logits 值:")
print(out.flatten()[:10].cpu().numpy())
print("-" * 50)
