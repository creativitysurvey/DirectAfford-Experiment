import torch
import time
import gc
import torch.nn.functional as F

from eval_exp2_real_data import Aff3DFunc_Proxy
from model import AttentionEnhancedGeometricEncoder

# ==========================================
# 修正版离线代理 (加入 384 -> 512 维度投影)
# ==========================================
class DirectAfford_OfflineProxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        # 加上真实的维度映射头，补齐 512 维
        self.proj = torch.nn.Linear(384, 512) 
        
    def forward(self, x):
        feats = self.encoder(x)
        return self.proj(feats)

def measure_train_memory(model, device, batch_size, model_type, num_points=2048):
    # 模拟输入，并要求计算梯度
    dummy_pcs = torch.randn(batch_size, num_points, 3, requires_grad=True).to(device)
    text_targets = torch.randn(batch_size, 512).to(device)
    
    labels = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_points).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        optimizer.zero_grad()
        feats = model(dummy_pcs) # 现在的 feats 已经是 [B, N, 512] 了！
        
        if model_type == "direct":
            # DirectAfford 线性对齐 O(B * N)
            text_expanded = text_targets.unsqueeze(1).expand(-1, num_points, -1)
            loss = F.mse_loss(feats, text_expanded)
        else:
            # Contrastive 对比学习的灾难 O(B^2 * N)
            sim_matrix = torch.einsum('b n d, c d -> b n c', feats, text_targets)
            loss = F.cross_entropy(sim_matrix.reshape(-1, batch_size), labels.reshape(-1))
            
        loss.backward()
        optimizer.step()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return "OOM"
        raise e

    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return max_mem

def run_train_profiling():
    print("\n🚀 Starting Exp 4: Training VRAM Efficiency (The InfoNCE Curse)")
    print("=" * 75)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target Device: {torch.cuda.get_device_name(0)}")
    
    # 地狱级测试载荷
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n{'Batch Size':<12} | {'Model':<25} | {'Peak VRAM (GB)':<15}")
    print("-" * 75)

    for bs in batch_sizes:
        # 1. 测我们的 DirectAfford
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model_direct = DirectAfford_OfflineProxy().to(device)
        mem_direct = measure_train_memory(model_direct, device, bs, "direct")
        
        # 2. 测对比学习基线
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model_contra = Aff3DFunc_Proxy().to(device)
        mem_contra = measure_train_memory(model_contra, device, bs, "contrastive")
        
        d_str = f"{mem_direct:.2f}" if isinstance(mem_direct, float) else mem_direct
        c_str = f"{mem_contra:.2f}" if isinstance(mem_contra, float) else mem_contra
        
        print(f"{bs:<12} | Aff3DFunc (Contrastive)   | {c_str:<15}")
        print(f"{bs:<12} | DirectAfford (Ours)       | {d_str:<15}")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        
        if mem_contra == "OOM" and mem_direct == "OOM":
            print(f"💥 [BOOM!] Both models hit the ceiling at Batch Size {bs}!")
            break

    print("=" * 75)

if __name__ == "__main__":
    run_train_profiling()
