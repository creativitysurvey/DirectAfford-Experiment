import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Import the geometric encoder
from model import AttentionEnhancedGeometricEncoder, DirectAffordModel

class TrueAff3DFunc(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        # Ensure parity with a 384-dim backbone
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        
        # Aff3DFunc uses local and global projections for multilevel alignment
        self.local_proj = nn.Sequential(
            nn.Conv1d(384, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, text_dim, 1)
        )
        # We simplify to focus on local part-level affordances for this benchmark
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, point_clouds):
        geom_features = self.geometric_encoder(point_clouds)
        local_features = self.local_proj(geom_features.transpose(1, 2)).transpose(1, 2)
        return F.normalize(local_features, p=2, dim=-1)

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Simplified Supervised Contrastive Loss.
    features: (V, d) - valid point features
    labels: (V) - corresponding affordance class indices
    """
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create mask for positive pairs (same label)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    
    # Exclude self-similarity
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(features.size(0)).view(-1, 1).to(features.device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log-probabilities
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
    
    # Loss
    loss = - mean_log_prob_pos.mean()
    return loss

def train_aff3dfunc_from_scratch(data_path, epochs=3000, device="cuda"):
    print("🚀 启动 [从零重训计划]: Aff3DFunc (Supervised Contrastive Paradigm)")
    
    model = TrueAff3DFunc().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print("⏳ 正在进行监督对比学习训练 (SupCon)...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        # 模拟一个 Batch
        B, N, C = 8, 2048, 3 
        dummy_pcs = torch.randn(B, N, C).to(device)
        
        # 模拟监督标签 (0, 1, 2) 代表不同的 affordance
        dummy_labels = torch.randint(0, 3, (B, N)).to(device)
        dummy_masks = torch.randint(0, 2, (B, N)).to(device) # Active points
        
        optimizer.zero_grad()
        
        # 提取点云特征
        point_features = model(dummy_pcs)
        
        # 展平并提取有效点
        valid_features = point_features[dummy_masks > 0]
        valid_labels = dummy_labels[dummy_masks > 0]
        
        # 计算 SupCon 损失
        loss = supervised_contrastive_loss(valid_features, valid_labels, model.temperature)
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{epochs}] | SupCon Loss: {loss.item():.4f}")

    os.makedirs("./data/baselines", exist_ok=True)
    torch.save(model.state_dict(), "./data/baselines/aff3dfunc_weights.pth")
    print("✅ Aff3DFunc 从零训练完成！权重已保存至 ./data/baselines/aff3dfunc_weights.pth")

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
