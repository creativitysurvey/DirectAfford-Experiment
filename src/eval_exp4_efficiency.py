import torch
import time
import gc
from model import DirectAffordModel
from eval_exp2_real_data import Aff3DFunc_Proxy

def measure_memory_and_time(model, device, batch_size, num_points=2048):
    # 构造极限测试数据
    dummy_pcs = torch.randn(batch_size, num_points, 3).to(device)
    dummy_text = torch.randn(batch_size, 512).to(device) if isinstance(model, DirectAffordModel) else None

    # 预热 (Warmup)
    model.eval()
    with torch.no_grad():
        try:
            if isinstance(model, DirectAffordModel):
                model(dummy_pcs)
            else:
                model(dummy_pcs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                return "OOM", "OOM"
            raise e

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 极速压测
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5): 
            if isinstance(model, DirectAffordModel):
                feats = model(dummy_pcs)
            else:
                feats = model(dummy_pcs)
                
    torch.cuda.synchronize()
    end_time = time.time()

    # 提取峰值显存 (转化为 GB) 和 耗时 (ms)
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_time = ((end_time - start_time) / 5) * 1000 

    return max_mem, avg_time

def run_efficiency_profiling():
    print("\n🚀 Starting Exp 4: Asymptotic Efficiency Profiling (Hardware Limit Test)")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    run_efficiency_profiling()
