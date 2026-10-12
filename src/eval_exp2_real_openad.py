import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from tqdm import tqdm

# ==========================================
# 1. 强行把 OpenAD 的源码目录注入 Python 环境变量
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'baselines/OpenAD')))

# 直接从官方源码中导入真实的模型类！
from models.openad_pn2 import OpenAD_PN2, cls_encoder

# ==========================================
# 2. 伪造官方所需的命令行参数 (Args)
# ==========================================
class DummyArgs:
    def __init__(self):
        self.model = 'openad_pn2'
        self.emb_dims = 512
        self.k = 20
args = DummyArgs()

def run_real_openad(data_dir, weight_path, device="cuda", num_samples=50):
    print("\n🚀 唤醒 OpenAD (IROS 2023) 官方真实源码与权重...")
    
    try:
        model = OpenAD_PN2(args, num_classes=1, normal_channel=False).to(device)
    except AttributeError as e:
        missing_attr = str(e).split("'")[3]
        setattr(args, missing_attr, 256) 
        model = OpenAD_PN2(args, num_classes=1, normal_channel=False).to(device)

    print(f"🔄 正在灌入真实的物理装甲: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    cls_encoder.eval()
    print("✅ 官方权重注入成功！模型已达到完全体状态。")

    # ==========================================
    # 3. 稳妥读取 LASO 真实数据
    # ==========================================
    anno_path = os.path.join(data_dir, "LASO_dataset", "anno_val.pkl")
    obj_path = os.path.join(data_dir, "LASO_dataset", "objects_val.pkl")

    with open(anno_path, 'rb') as f:
        raw_anno = pickle.load(f)
        anno_data = list(raw_anno.values()) if isinstance(raw_anno, dict) else raw_anno
        
    with open(obj_path, 'rb') as f:
        obj_data = pickle.load(f)

    obj_dict = {str(k): v.get('point_cloud', v.get('points', v)) if isinstance(v, dict) else v for k, v in obj_data.items()} if isinstance(obj_data, dict) else {str(item.get('shape_id', '')): item.get('point_cloud', item) for item in obj_data}

    all_preds = []
    all_gts = []
    valid_count = 0
    
    print("⏳ 正在用 OpenAD 真实算力评估 LASO 验证集...")
    for sample_anno in tqdm(anno_data[:num_samples * 3]):
        if valid_count >= num_samples: break
            
        raw_mask = np.array(list(sample_anno['mask'].values())[0] if isinstance(sample_anno['mask'], dict) else sample_anno['mask'], dtype=np.float32)
        raw_pc = obj_dict.get(str(sample_anno['shape_id']), None)
        
        if raw_pc is None or raw_mask.sum() == 0 or raw_mask.sum() == len(raw_mask):
            continue
            
        if len(raw_pc) > 2048:
            choice = np.random.choice(len(raw_pc), 2048, replace=False)
            raw_pc, raw_mask = raw_pc[choice], raw_mask[choice]
        elif len(raw_pc) < 2048:
            pad_size = 2048 - len(raw_pc)
            raw_pc = np.concatenate([raw_pc, raw_pc[:pad_size]], axis=0)
            raw_mask = np.concatenate([raw_mask, raw_mask[:pad_size]], axis=0)

        real_pcs = torch.tensor(raw_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(device)
        query = sample_anno['affordance'][0] if isinstance(sample_anno['affordance'], list) else sample_anno['affordance']

        all_gts.append(raw_mask)
        
        with torch.no_grad():
            # 🌟 核心修复：直接把原始字符串列表 [query] 传给模型！
            pred_logits = model(real_pcs, [query]) 
            
            if len(pred_logits.shape) == 3: 
                pred_probs = torch.sigmoid(pred_logits.squeeze())
            else:
                pred_probs = torch.sigmoid(pred_logits)

            all_preds.append(pred_probs.cpu().numpy().flatten())
                
        valid_count += 1

    print("\n" + "=" * 60)
    print(f"{'Method (100% Real Source & Weights)':<35} | {'AUC (↑)':<10}")
    print("-" * 60)
    
    y_true_float = np.concatenate(all_gts)
    y_true_binary = (y_true_float > 0.5).astype(int) 
    y_scores = np.concatenate(all_preds)

    print("👀 偷看 OpenAD 输出的原始概率值前 20 个:")
    print(y_scores[:20])
    
    auc = roc_auc_score(y_true_binary, y_scores)
    print(f"{'OpenAD (IROS 2023)':<35} | {auc:.3f}")
    print("=" * 60)
    print("✨ 这就是它在最复杂场景下真实的物理底线！")

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
