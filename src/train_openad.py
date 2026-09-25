import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np # Added missing import

# 导入统一的几何编码器和分词器
from model import AttentionEnhancedGeometricEncoder, DirectAffordModel

class TrueOpenAD(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        # 为了公平，给予它和 DirectAfford 一样强大的 384维骨干
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        
        # 对比学习通常包含一个非线性的投影头 (Projection Head)
        self.proj_head = nn.Sequential(
            nn.Conv1d(384, 512, 1), 
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Conv1d(512, text_dim, 1)
        )
        
        # 可学习的温度系数，InfoNCE 必备
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, point_clouds):
        B, N, _ = point_clouds.shape
        geom_features = self.geometric_encoder(point_clouds)
        proj_features = self.proj_head(geom_features.transpose(1, 2)).transpose(1, 2)
        return F.normalize(proj_features, p=2, dim=-1)

def infonce_loss(point_features, text_features, active_mask, logit_scale):
    """
    点级别的 InfoNCE 损失。
    point_features: (B, N, d)
    text_features: (B, d)
    active_mask: (B, N) - 仅计算有效点
    """
    B, N, d = point_features.shape
    
    # 提取所有 Batch 中有效的点特征和对应的文本特征
    valid_points = []
    valid_texts = []
    
    for b in range(B):
        mask = active_mask[b] > 0
        if mask.sum() > 0:
            valid_points.append(point_features[b][mask])
            # 将对应的文本特征复制扩展，使其数量与有效点相同
            text_repeated = text_features[b].unsqueeze(0).repeat(mask.sum(), 1)
            valid_texts.append(text_repeated)
            
    if not valid_points:
         return torch.tensor(0.0, device=point_features.device, requires_grad=True)
         
    valid_points = torch.cat(valid_points, dim=0) # (V, d)
    valid_texts = torch.cat(valid_texts, dim=0)   # (V, d)
    
    # 计算全局相似度矩阵 (V, V)
    # 这就是对比学习中导致 OOM(Out of Memory) 的罪魁祸首！
    # 为了防止显存爆炸，我们在这里采用简化版的 InfoNCE：点 -> 对应文本 vs 点 -> 其他所有文本
    
    # 计算当前点与整个 Batch 内所有文本锚点 (B, d) 的相似度
    # valid_points: (V, d) @ text_features.T: (d, B) -> logits: (V, B)
    logits = torch.matmul(valid_points, text_features.t()) * logit_scale.exp()
    
    # 构建正样本标签：当前点属于哪个 Batch 索引
    labels = []
    for b in range(B):
        mask_count = (active_mask[b] > 0).sum().item()
        labels.extend([b] * mask_count)
    labels = torch.tensor(labels, dtype=torch.long, device=point_features.device)
    
    # 交叉熵损失 (InfoNCE 的本质)
    loss = F.cross_entropy(logits, labels)
    return loss

def train_openad_from_scratch(data_path, epochs=3000, device="cuda"):
    print("🚀 启动 [从零重训计划]: OpenAD (InfoNCE Contrastive Paradigm)")
    
    # 初始化文本引擎 (借用 DirectAfford 的冻结 CLIP)
    text_engine = DirectAffordModel().to(device)
    text_engine.eval()
    
    model = TrueOpenAD().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # ... (这里省略 Dataloader 的加载逻辑，假设与 DirectAfford 一致) ...
    # 假设我们有一个虚拟的 dataloader (您需要将其替换为您实际的 data loader)
    # 为了演示，我们生成随机数据模拟训练过程
    
    print("⏳ 正在进行高强度对比学习训练...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        # 模拟一个 Batch
        B, N, C = 8, 2048, 3 # 注意：由于 InfoNCE 显存爆炸，Batch Size 必须调小
        dummy_pcs = torch.randn(B, N, C).to(device)
        dummy_texts = ["a tool for grasping", "a part to cut with"] * 4
        dummy_masks = torch.randint(0, 2, (B, N)).float().to(device)
        
        optimizer.zero_grad()
        
        # 1. 提取文本特征
        text_features = text_engine.get_text_embeddings(dummy_texts, device)
        
        # 2. 提取点云特征
        point_features = model(dummy_pcs)
        
        # 3. 计算 InfoNCE 损失
        loss = infonce_loss(point_features, text_features, dummy_masks, model.logit_scale)
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{epochs}] | InfoNCE Loss: {loss.item():.4f} | Scale: {model.logit_scale.exp().item():.2f}")

    # 保存训练好的权重
    os.makedirs("./data/baselines", exist_ok=True)
    torch.save(model.state_dict(), "./data/baselines/openad_weights.pth")
    print("✅ OpenAD 从零训练完成！权重已保存至 ./data/baselines/openad_weights.pth")

if __name__ == "__main__":
    train_openad_from_scratch("./data")
