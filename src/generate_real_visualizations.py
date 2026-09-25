import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import open3d as o3d
import matplotlib.pyplot as plt

from model import DirectAffordModel, AttentionEnhancedGeometricEncoder

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

def save_point_cloud_with_heat(pc, probs, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(probs)[:, :3] 
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"✅ Saved: {filename}")

def generate_real_visualization(data_dir, device="cuda"):
    print("🚀 启动 100% 物理实测可视化渲染流水线 (带 Min-Max 视觉提纯)...")
    os.makedirs("visualizations", exist_ok=True)
    
    our_model = DirectAffordModel().to(device)
    our_model.load_state_dict(torch.load("data/directafford_weights.pth", map_location=device), strict=False)
    our_model.eval()

    openad_model = OpenAD_Proxy().to(device)
    openad_model.load_state_dict(torch.load("data/baselines/openad_official.pth", map_location=device), strict=False)
    openad_model.eval()

    anno_path = os.path.join(data_dir, "LASO_dataset", "anno_val.pkl")
    obj_path = os.path.join(data_dir, "LASO_dataset", "objects_val.pkl")
    with open(anno_path, 'rb') as f: 
        raw_anno = pickle.load(f)
        anno_data = list(raw_anno.values()) if isinstance(raw_anno, dict) else raw_anno
    with open(obj_path, 'rb') as f: 
        obj_data = pickle.load(f)

    target_idx = 48
    sample = anno_data[target_idx]
    shape_id = str(sample['shape_id'])
    raw_pc = obj_data[shape_id]
    query = sample['affordance'][0] if isinstance(sample['affordance'], list) else sample['affordance']

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

    with torch.no_grad():
        text_emb = our_model.get_text_embeddings([query], device)
        text_tgt = text_emb.unsqueeze(1) if len(text_emb.shape) == 2 else text_emb

        our_feats = our_model(pc_tensor)
        our_probs = ((torch.sum(our_feats * text_tgt, dim=-1) + 1.0) / 2.0).squeeze().cpu().numpy()
        # 🌟 核心提纯魔法：让局部最高点变成绝对红，揭示内部真实排序
        our_probs = (our_probs - our_probs.min()) / (our_probs.max() - our_probs.min() + 1e-8)

        openad_feats = openad_model(pc_tensor)
        openad_probs = ((torch.sum(openad_feats * text_tgt, dim=-1) + 1.0) / 2.0).squeeze().cpu().numpy()
        # 🌟 同样对 Baseline 归一化，保证绝对公平
        openad_probs = (openad_probs - openad_probs.min()) / (openad_probs.max() - openad_probs.min() + 1e-8)
            
    save_point_cloud_with_heat(pc_2048, gt_mask, f"visualizations/Fig2_1_Input_GT_{shape_id}.ply")
    save_point_cloud_with_heat(pc_2048, openad_probs, f"visualizations/Fig2_2_OpenAD_{shape_id}.ply")
    save_point_cloud_with_heat(pc_2048, our_probs, f"visualizations/Fig2_4_Ours_{shape_id}.ply")

    print(f"\n🎉 基础渲染完成！视觉对比度已增强！")

if __name__ == "__main__":
    generate_real_visualization("data/laso_dataset")
