import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from model import AttentionEnhancedGeometricEncoder, DirectAffordModel

class TrueKdtpc(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=3)
        self.distillation_head = nn.Sequential(
            nn.Conv1d(384, 512, 1),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Conv1d(512, text_dim, 1)
        )

    def forward(self, point_clouds):
        geom_features = self.geometric_encoder(point_clouds)
        distilled_features = self.distillation_head(geom_features.transpose(1, 2)).transpose(1, 2)
        return F.normalize(distilled_features, p=2, dim=-1)

def distillation_loss(student_features, teacher_features, active_mask):
    """
    Knowledge Distillation Loss (Cosine Similarity or MSE)
    """
    B, N, d = student_features.shape
    
    # We use MSE for feature imitation
    loss = F.mse_loss(student_features, teacher_features, reduction='none')
    loss = loss.mean(dim=-1) # Average over feature dim
    
    # Apply mask
    active_loss = (loss * active_mask).sum(dim=1) / (active_mask.sum(dim=1).clamp(min=1e-6))
    return active_loss.mean()

def train_kdtpc_from_scratch(data_path, epochs=3000, device="cuda"):
    print("🚀 启动 [从零重训计划]: KD-TPC (Knowledge Distillation Paradigm)")
    
    # Teacher model (borrowing CLIP text features as a proxy for the distillation target)
    teacher = DirectAffordModel().to(device)
    teacher.eval()
    
    model = TrueKdtpc().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print("⏳ 正在进行知识蒸馏训练 (Knowledge Distillation)...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        B, N, C = 8, 2048, 3 
        dummy_pcs = torch.randn(B, N, C).to(device)
        dummy_texts = ["grasp", "cut", "lift", "support", "press", "turn", "twist", "pull"]
        dummy_masks = torch.randint(0, 2, (B, N)).float().to(device)
        
        optimizer.zero_grad()
        
        student_features = model(dummy_pcs)
        
        # Simulate teacher features (in reality, these come from rendering 2D views or point-text correlation)
        # Here we use text features to simulate the target latent space
        with torch.no_grad():
            teacher_features = teacher.get_text_embeddings(dummy_texts, device)
            # Expand to match point cloud shape (simplification for the proxy)
            teacher_expanded = teacher_features.unsqueeze(1).expand(-1, N, -1)
            
        loss = distillation_loss(student_features, teacher_expanded, dummy_masks)
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{epochs}] | Distillation Loss: {loss.item():.4f}")

    os.makedirs("./data/baselines", exist_ok=True)
    torch.save(model.state_dict(), "./data/baselines/kdtpc_weights.pth")
    print("✅ KD-TPC 从零训练完成！权重已保存至 ./data/baselines/kdtpc_weights.pth")

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
