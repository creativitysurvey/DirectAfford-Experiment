import torch
import time
import gc

from eval_exp2_real_data import Aff3DFunc_Proxy
from model import AttentionEnhancedGeometricEncoder

# ==========================================
# 离线脱机代理 (彻底剥离 CLIP，纯测 3D 骨干与对齐头的 VRAM)
# ==========================================
class DirectAfford_OfflineProxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentionEnhancedGeometricEncoder(embed_dim=384, num_layers=4)
        
    def forward(self, x):
        return self.encoder(x)

def measure_memory_and_time(model, device, batch_size, num_points=2048):
    dummy_pcs = torch.randn(batch_size, num_points, 3).to(device)

    model.eval()
    with torch.no_grad():
        try:
            model(dummy_pcs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                return "OOM", "OOM"
            raise e

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5): 
            model(dummy_pcs)
                
    torch.cuda.synchronize()
    end_time = time.time()

    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_time = ((end_time - start_time) / 5) * 1000 

    return max_mem, avg_time

def run_efficiency_profiling():
    print("\n🚀 Starting Exp 4: Asymptotic Efficiency Profiling (Hellfire Offline Test)")
    print("=" * 75)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    
    models = {
        "Aff3DFunc (Contrastive)": Aff3DFunc_Proxy().to(device),
        "DirectAfford (Ours)": DirectAfford_OfflineProxy().to(device)
    }

    # 针对 80GB A100 的地狱级载荷
    batch_sizes = [64, 128, 256, 512, 1024, 2048]

    print(f"\n{'Batch Size':<12} | {'Model':<25} | {'Peak VRAM (GB)':<15} | {'Latency (ms)':<15}")
    print("-" * 75)

    for bs in batch_sizes:
        for name, model in models.items():
            gc.collect()
            torch.cuda.empty_cache()
            
            mem, latency = measure_memory_and_time(model, device, bs)
            mem_str = f"{mem:.2f}" if isinstance(mem, float) else mem
            lat_str = f"{latency:.1f}" if isinstance(latency, float) else latency
            
            print(f"{bs:<12} | {name:<25} | {mem_str:<15} | {lat_str:<15}")
            
            if mem == "OOM":
                print(f"💥 [BOOM!] {name} triggered Out-Of-Memory (OOM) at Batch Size {bs}!")

    print("=" * 75)

if __name__ == "__main__":
    run_efficiency_profiling()
