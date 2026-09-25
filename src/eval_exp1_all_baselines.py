import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from transformers import CLIPTextModel, CLIPTokenizer
from model import DirectAffordModel, AttentionEnhancedGeometricEncoder

# ==========================================
# Baseline 1: OpenAD (InfoNCE Contrastive)
# ==========================================
class OpenAD_Proxy(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        self.proj_head = nn.Sequential(
            nn.Conv1d(384, 512, 1), 
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Conv1d(512, text_dim, 1)
        )

    def forward(self, point_clouds):
        geom_features = self.geometric_encoder(point_clouds)
        proj_features = self.proj_head(geom_features.transpose(1, 2)).transpose(1, 2)
        return F.normalize(proj_features, p=2, dim=-1)

# ==========================================
# Baseline 2: KD-TPC (Knowledge Distillation)
# ==========================================
class KD_TPC_Proxy(nn.Module):
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

# ==========================================
# Baseline 3: Aff3DFunc (Supervised Contrastive)
# ==========================================
class Aff3DFunc_Proxy(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        self.local_proj = nn.Sequential(
            nn.Conv1d(384, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, text_dim, 1)
        )

    def forward(self, point_clouds):
        geom_features = self.geometric_encoder(point_clouds)
        local_features = self.local_proj(geom_features.transpose(1, 2)).transpose(1, 2)
        return F.normalize(local_features, p=2, dim=-1)

# ==========================================
# 统一的评估 Pipeline
# ==========================================
def evaluate_model(model, val_data, novel_classes, novel_embeddings, device, threshold=0.10):
    model.eval()
    total_iou, total_acc, valid_samples = 0.0, 0.0, 0

    for item in val_data:
        raw_coords = item['full_shape']['coordinate']
        if len(raw_coords) > 2048: raw_coords = raw_coords[:2048]
        elif len(raw_coords) < 2048: raw_coords = np.concatenate([raw_coords, raw_coords[:2048 - len(raw_coords)]], axis=0)
        
        pts = torch.tensor(raw_coords, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_z = model(pts)
        
        gt_masks_dict = item['full_shape'].get('label', {})
        for aff in item.get('affordance', []):
            aff_lower = str(aff).lower()
            if aff_lower in novel_classes and aff_lower in gt_masks_dict:
                gt_mask = torch.tensor(gt_masks_dict[aff_lower], dtype=torch.float32).view(-1).to(device)
                if gt_mask.sum() == 0: continue
                
                target_idx = novel_classes.index(aff_lower)
                target_z = novel_embeddings[target_idx]
                
                cosine_sim = torch.sum(predicted_z.squeeze(0) * target_z, dim=-1)
                pred_mask = (cosine_sim > threshold).float()
                
                intersection = (pred_mask * gt_mask).sum().item()
                union = (pred_mask + gt_mask).clamp(0, 1).sum().item()
                
                iou = intersection / union if union > 0 else 0.0
                acc = intersection / gt_mask.sum().item() if gt_mask.sum() > 0 else 0.0
                
                total_iou += iou
                total_acc += acc
                valid_samples += 1

    mIoU = total_iou / valid_samples if valid_samples > 0 else 0.0
    mAcc = total_acc / valid_samples if valid_samples > 0 else 0.0
    return mIoU, mAcc

def run_comparative_experiment(data_path, device="cuda"):
    print("Initializing Models and Loading Trained Weights...")
    
    base_text_engine = DirectAffordModel().to(device)
    
    models = {
        "OpenAD": ("InfoNCE Contrastive", OpenAD_Proxy().to(device), "./data/baselines/openad_weights.pth"),
        "KD-TPC": ("Knowledge Distillation", KD_TPC_Proxy().to(device), "./data/baselines/kdtpc_weights.pth"),
        "Aff3DFunc": ("Supervised Contrastive", Aff3DFunc_Proxy().to(device), "./data/baselines/aff3dfunc_weights.pth"),
        "DirectAfford (Ours)": ("Direct Predictive", DirectAffordModel().to(device), "./data/directafford_weights.pth")
    }
    
    # 严格加载所有模型的真实训练权重
    for name, (paradigm, model, weight_path) in models.items():
        if os.path.exists(weight_path):
            # 过滤掉不需要严格匹配的键（例如 OpenAD 的 logit_scale）
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ {name} weights loaded successfully.")
        else:
            print(f"⚠️ {name} weights NOT FOUND at {weight_path}.")

    with open(os.path.join(data_path, "full_shape_val_data.pkl"), 'rb') as f:
        val_data = pickle.load(f)

    base_classes = {"grasp", "cut", "lift", "support", "press", "turn", "twist", "pull", "push", "open", "close"}
    novel_classes = list({str(aff).lower() for item in val_data for aff in item.get('affordance', []) if str(aff).lower() not in base_classes})
    
    novel_embeddings = base_text_engine.get_text_embeddings(novel_classes, device)

    print("\nStarting Comparative Zero-Shot Evaluation on Novel Categories...")
    print("=" * 70)
    print(f"{'Method':<20} | {'Alignment Paradigm':<25} | {'mIoU':<6} | {'mAcc':<6}")
    print("-" * 70)

    for name, (paradigm, model, _) in models.items():
        mIoU, mAcc = evaluate_model(model, val_data, novel_classes, novel_embeddings, device, threshold=0.10)
        print(f"{name:<20} | {paradigm:<25} | {mIoU:.3f}  | {mAcc:.3f}")
        
    print("=" * 70)

if __name__ == "__main__":
    run_comparative_experiment("./data", device="cuda" if torch.cuda.is_available() else "cpu")
