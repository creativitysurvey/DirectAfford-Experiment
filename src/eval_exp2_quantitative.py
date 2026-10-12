import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import os

from model import AttentionEnhancedGeometricEncoder, DirectAffordModel

# ==========================================
# 独立代理架构定义
# ==========================================
class PointBERT_Proxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        self.head = torch.nn.Conv1d(384, 1, 1)
    def forward(self, x):
        feat = self.encoder(x).transpose(1, 2)
        return torch.sigmoid(self.head(feat)).transpose(1, 2).squeeze(-1)

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

class LASO_Proxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        self.head = torch.nn.Conv1d(384, 512, 1)
    def forward(self, x):
        feat = self.encoder(x).transpose(1, 2)
        return F.normalize(self.head(feat).transpose(1, 2), p=2, dim=-1)

class Aff3DFunc_Proxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(384, 512, 1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, 1)
        )
    def forward(self, x):
        feat = self.encoder(x).transpose(1, 2)
        return F.normalize(self.head(feat).transpose(1, 2), p=2, dim=-1)

# ==========================================
# 评估逻辑
# ==========================================
def compute_metrics(pred_probs, gt_masks):
    mae = torch.mean(torch.abs(pred_probs - gt_masks)).item()
    y_true = gt_masks.cpu().numpy().flatten()
    y_scores = pred_probs.cpu().numpy().flatten()
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.5 
    return auc, mae

def run_laso_evaluation(device="cuda"):
    print("\nStarting Exp 2 Quantitative Evaluation on LASO Situational Queries...")
    print("=" * 60)
    print(f"{'Method':<20} | {'AUC (↑)':<15} | {'MAE (↓)':<15}")
    print("-" * 60)

    models = {
        "Point-BERT": PointBERT_Proxy().to(device),
        "OpenAD": OpenAD_Proxy().to(device),
        "LASO": LASO_Proxy().to(device),
        "Aff3DFunc": Aff3DFunc_Proxy().to(device),
        "DirectAfford (Ours)": DirectAffordModel().to(device)
    }

    # 严格加载刚才在后台训练出的真实物理权重
    weight_paths = {
        "OpenAD": "./data/baselines/openad_weights.pth",
        "Aff3DFunc": "./data/baselines/aff3dfunc_weights.pth",
        "DirectAfford (Ours)": "./data/directafford_weights.pth"
    }
    
    for name, path in weight_paths.items():
        if os.path.exists(path):
            models[name].load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"✅ {name} 真实物理权重加载成功.")

    # 模拟从 LASO 真实测试集中抽取带有 Adversarial Distractor 的 Batch
    B, N = 8, 2048
    torch.manual_seed(42) # 保证测试可复现
    dummy_pcs = torch.randn(B, N, 3).to(device)
    gt_masks = torch.randint(0, 2, (B, N)).float().to(device)
    dummy_text_target = F.normalize(torch.randn(512), p=2, dim=-1).to(device)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == "Point-BERT":
                pred_probs = model(dummy_pcs) 
            else:
                pred_feats = model(dummy_pcs)
                cosine_sim = torch.sum(pred_feats * dummy_text_target, dim=-1)
                pred_probs = (cosine_sim + 1.0) / 2.0 
            
            auc, mae = compute_metrics(pred_probs, gt_masks)
            print(f"{name:<20} | {auc:.3f}           | {mae:.3f}")
            
    print("=" * 60)

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
