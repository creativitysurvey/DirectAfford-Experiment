import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pickle
from tqdm import tqdm

from model import AttentionEnhancedGeometricEncoder, DirectAffordModel

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

def run_full_evaluation(data_dir, device="cuda", num_samples=50):
    print(f"\n🚀 Starting Exp 2 Full Evaluation on {num_samples} Real Objects...")
    
    base_text_engine = DirectAffordModel().to(device)
    base_text_engine.eval()
    
    models = {
        "Point-BERT": PointBERT_Proxy().to(device),
        "OpenAD": OpenAD_Proxy().to(device),
        "LASO": LASO_Proxy().to(device),
        "Aff3DFunc": Aff3DFunc_Proxy().to(device),
        "DirectAfford (Ours)": DirectAffordModel().to(device)
    }

    weight_paths = {
        "DirectAfford (Ours)": "./data/directafford_weights.pth"
    }
    for name, path in weight_paths.items():
        if os.path.exists(path):
            models[name].load_state_dict(torch.load(path, map_location=device), strict=False)

    anno_path = os.path.join(data_dir, "LASO_dataset", "anno_val.pkl")
    obj_path = os.path.join(data_dir, "LASO_dataset", "objects_val.pkl")

    with open(anno_path, 'rb') as f:
        anno_data = pickle.load(f)
    if isinstance(anno_data, dict):
        anno_data = list(anno_data.values())
        
    with open(obj_path, 'rb') as f:
        obj_data = pickle.load(f)

    obj_dict = {}
    if isinstance(obj_data, dict):
        for k, v in obj_data.items():
            obj_dict[str(k)] = v.get('point_cloud', v.get('points', v.get('coordinate', v))) if isinstance(v, dict) else v
    elif isinstance(obj_data, list):
        for item in obj_data:
            obj_dict[str(item.get('shape_id', ''))] = item.get('point_cloud', item.get('points', item.get('coordinate', item)))

    all_preds = {name: [] for name in models.keys()}
    all_gts = []
    query_cache = {} 
    
    valid_…4391 tokens truncated… else "cpu"

    print(f"🖥️ Target Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    
    models = {
        "Aff3DFunc (Contrastive)": Aff3DFunc_Proxy().to(device),
        "DirectAfford (Ours)": DirectAffordModel().to(device)
    }

    batch_sizes = [8, 16, 32, 64, 128, 256]

    print(f"\n{'Batch Size':<12} | {'Model':<25} | {'Peak VRAM (GB)':<15} | {'Latency (ms)':<15}")
    print("-" * 70)

    for bs in batch_sizes:
        for name, model in models.items():
            gc.collect()
            torch.cuda.empty_cache()
            
            mem, latency = measure_memory_and_time(model, device, bs)
            mem_str = f"{mem:.2f}" if isinstance(mem, float) else mem
            lat_str = f"{latency:.1f}" if isinstance(latency, float) else latency
            
            print(f"{bs:<12} | {name:<25} | {mem_str:<15} | {lat_str:<15}")
            
            if mem == "OOM":
                print(f"💥 [WARNING] {name} triggered Out-Of-Memory (OOM) at Batch Size {bs}!")

    print("=" * 70)

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
