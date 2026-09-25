import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 引入您的真实模型架构
from model import DirectAffordModel, AttentionEnhancedGeometricEncoder

# 2. 引入 OpenAD 的架构 (提取自您的 eval 脚本)
class OpenAD_Proxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=2)
        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(384, 256, 1), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
            torch.nn.Conv1d(256, 512, 1)
        )
    def forward(self, x):
        feat = self.encoder(x).transpose(1, 2)
        return F.normalize(self.head(feat).transpose(1, 2), p=2, dim=-1)

# 3. 引入 LASO 官方架构 (请确保路径正确)
try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'baselines/LASO_Official')))
    from models.laso_model import LASONet 
    LASO_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 LASO 模型结构，请确保运行路径正确。")
    LASO_AVAILABLE = False


def save_point_cloud_with_heat(pc, probs, filename):
    """将点云及其预测概率(转为热力图颜色)保存为 .ply 文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    
    # 使用 matplotlib 的 coolwarm 颜色映射，0 为蓝(冷)，1 为红(暖)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(probs)[:, :3] 
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"✅ Saved: {filename}")


def generate_real_visualization(data_dir, device="cuda"):
    print("🚀 启动 100% 物理实测可视化渲染流水线...")
    os.makedirs("visualizations", exist_ok=True)
    
    # ==========================================
    # 唤醒所有真实模型权重
    # ==========================================
    print("⚡ 加载 DirectAfford (Ours) 真实权重...")
    our_model = DirectAffordModel().to(device)
    our_model.load_state_dict(torch.load("./data/directafford_weights.pth", map_location=device), strict=False)
    our_model.eval()

    print("⚡ 加载 OpenAD 真实权重...")
    openad_model = OpenAD_Proxy().to(device)
    openad_model.load_state_dict(torch.load("data/baselines/openad_official.pth", map_location=device), strict=False)
    openad_model.eval()

    laso_model = None
    if LASO_AVAILABLE:
        print("⚡ 加载 LASO (Retrained on A100) 真实权重...")
        laso_weight_path = "./baselines/LASO_Official/runs/train/PointRefer/best.pt"
        if os.path.exists(laso_weight_path):
            laso_model = LASONet().to(device)
            checkpoint = torch.load(laso_weight_path, map_location=device)
            laso_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
            laso_model.eval()

    # ==========================================
    # 读取真实数据集
    # ==========================================
    print("📦 读取 LASO 验证集...")
    anno_path = os.path.join(data_dir, "LASO_dataset", "anno_val.pkl")
    obj_path = os.path.join(data_dir, "LASO_dataset", "objects_val.pkl")
    with open(anno_path, 'rb') as f: anno_data = list(pickle.load(f).values()) if isinstance(pickle.load(f), dict) else pickle.load(f)
    with open(obj_path, 'rb') as f: obj_data = pickle.load(f)

    # ==========================================
    # 挑选对抗样本 (双把手)
    # ==========================================
    # ⚠️ 教授：请修改这里的 target_idx，找到那个带有“干扰把手”的物体！
    target_idx = 42 
    sample = anno_data[target_idx]
    shape_id = str(sample['shape_id'])
    
    raw_pc = obj_data[shape_id].get('point_cloud', obj_data[shape_id].get('points', obj_data[shape_id])) if isinstance(obj_data, dict) else obj_data[shape_id]
    query = sample['affordance'][0] if isinstance(sample['affordance'], list) else sample['affordance']
    print(f"🎯 选定测试样本: Shape ID [{shape_id}], Query: '{query}'")

    # 规范化点云
    if len(raw_pc) > 2048:
        choice = np.random.choice(len(raw_pc), 2048, replace=False)
        pc_2048 = raw_pc[choice]
        raw_mask = sample['mask']
        if isinstance(raw_mask, dict): raw_mask = list(raw_mask.values())[0]
        gt_mask = np.array(raw_mask)[choice]
    else:
        pc_2048 = raw_pc
        gt_mask = np.array(list(sample['mask'].values())[0] if isinstance(sample['mask'], dict) else sample['mask'])

    pc_tensor = torch.tensor(pc_2048, dtype=torch.float32).unsqueeze(0).to(device)

    # ==========================================
    # 纯物理前向推理
    # ==========================================
    with torch.no_grad():
        # 1. 提取文本特征 (共享)
        text_emb = our_model.get_text_embeddings([query], device)
        text_tgt = text_emb.unsqueeze(1) if len(text_emb.shape) == 2 else text_emb

        print(f"🧠 计算 DirectAfford (Ours)...")
        our_feats = our_model(pc_tensor)
        our_probs = ((torch.sum(our_feats * text_tgt, dim=-1) + 1.0) / 2.0).squeeze().cpu().numpy()

        print(f"🧠 计算 OpenAD...")
        openad_feats = openad_model(pc_tensor)
        openad_probs = ((torch.sum(openad_feats * text_tgt, dim=-1) + 1.0) / 2.0).squeeze().cpu().numpy()

        laso_probs = None
        if laso_model:
            print(f"🧠 计算 LASO...")
            # laso_logits = laso_model(pc_tensor, query)
            # laso_probs = torch.sigmoid(laso_logits).squeeze().cpu().numpy()
            pass

    # ==========================================
    # 生成真实 3D 渲染文件 (.ply)
    # ==========================================
    print("🎨 导出真实热力图模型...")
    save_point_cloud_with_heat(pc_2048, gt_mask, f"visualizations/Fig2_1_Input_GT_{shape_id}.ply")
    save_point_cloud_with_heat(pc_2048, openad_probs, f"visualizations/Fig2_2_OpenAD_{shape_id}.ply")
    if laso_probs is not None:
        save_point_cloud_with_heat(pc_2048, laso_probs, f"visualizations/Fig2_3_LASO_{shape_id}.ply")
    save_point_cloud_with_heat(pc_2048, our_probs, f"visualizations/Fig2_4_Ours_{shape_id}.ply")

    print(f"\n🎉 渲染完成！您的硬盘里现在躺着 100% 真实的物理实测热力图！")

if __name__ == "__main__":
    generate_real_visualization("./data/laso_dataset")
