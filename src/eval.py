import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from model import DirectAffordModel

def evaluate_zero_shot_with_suppression(model, data_path, device="cuda"):
    print("Starting Zero-Shot Evaluation (Normalized Spatial Sweep with Background Suppression)...")
    model.eval()
    
    with open(os.path.join(data_path, "full_shape_val_data.pkl"), 'rb') as f:
        val_data = pickle.load(f)

    base_classes = {"grasp", "cut", "lift", "support", "press", "turn", "twist", "pull", "push", "open", "close"}
    novel_classes = set()
    for item in val_data:
        affordance_labels = item.get('affordance', [])
        for aff in affordance_labels:
            if isinstance(aff, str) and aff.lower() not in base_classes:
                novel_classes.add(aff.lower())
    
    novel_classes = list(novel_classes)

    # 背景锚点
    bg_prompts = ["background", "unrelated shape", "other parts", "nothing"]
    novel_text_embeds = model.get_text_embeddings(novel_classes, device)
    bg_text_embeds = model.get_text_embeddings(bg_prompts, device)

    thresholds = np.arange(0.10, 0.95, 0.05)
    best_thresh = 0.10
    best_miou = 0.0
    best_macc = 0.0

    print(f"\n=============================================")
    print(f"   Zero-Shot Grounding (Background Suppression)")
    print(f"=============================================")

    for thresh in thresholds:
        total_iou = 0.0
        total_acc = 0.0
        valid_samples = 0

        for idx, item in enumerate(val_data):
            if idx > 0 and idx % 40 == 0 and thresh == thresholds[0]:
                print(f"Processed {idx} batches...")

            raw_coords = item['full_shape']['coordinate']
            if len(raw_coords) > 2048:
                raw_coords = raw_coords[:2048]
            elif len(raw_coords) < 2048:
                pad_size = 2048 - len(raw_coords)
                raw_coords = np.concatenate([raw_coords, raw_coords[:pad_size]], axis=0)
                
            pts = torch.tensor(raw_coords, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_z = model(pts)
            
            affordance_labels = item.get('affordance', [])
            gt_masks_dict = item['full_shape'].get('label', {})
            
            for aff in affordance_labels:
                aff_lower = str(aff).lower()
                if aff_lower in novel_classes and aff_lower in gt_masks_dict:
                    
                    # 【核心修复】：强行拍扁成一维 (2048,)，彻底杜绝广播！
                    gt_mask = torch.tensor(gt_masks_dict[aff_lower], dtype=torch.float32).view(-1).to(device)
                    if gt_mask.sum() == 0:
                        continue
                    
                    target_idx = novel_classes.index(aff_lower)
                    target_embed = novel_text_embeds[target_idx].unsqueeze(0)
                    combined_embeds = torch.cat([target_embed, bg_text_embeds], dim=0)
                    
                    sim_matrix = torch.matmul(pred_z.squeeze(0), combined_embeds.t())
                    
                    target_scores = sim_matrix[:, 0]
                    bg_max_scores = sim_matrix[:, 1:].max(dim=1)[0]
                    
                    target_scores_norm = (target_scores - target_scores.min()) / (target_scores.max() - target_scores.min() + 1e-8)
                    
                    # 【核心修复】：强行拍扁成一维 (2048,)
                    pred_mask = ((target_scores_norm > thresh) & (target_scores > bg_max_scores)).float().view(-1)

                    intersection = (pred_mask * gt_mask).sum().item()
                    union = (pred_mask + gt_mask).clamp(0, 1).sum().item()
                    
                    iou = intersection / union if union > 0 else 0.0
                    acc = intersection / gt_mask.sum().item() if gt_mask.sum() > 0 else 0.0

                    total_iou += iou
                    total_acc += acc
                    valid_samples += 1

        if valid_samples > 0:
            miou = total_iou / valid_samples
            macc = total_acc / valid_samples
            print(f" Threshold +{thresh:.2f} | mIoU: {miou:.3f} | mAcc: {macc:.3f}")
            if miou > best_miou:
                best_miou = miou
                best_macc = macc
                best_thresh = thresh

    print(f"=============================================")
    print(f" [Optimal Zero-Shot Peak] Threshold: +{best_thresh:.2f}")
    print(f" Novel Classes mIoU : {best_miou:.3f}")
    print(f" Novel Classes mAcc : {best_macc:.3f}")
    print(f"=============================================")

if __name__ == "__main__":
    raise SystemExit("Legacy entry point disabled: proxy/random-data or unverified evaluation. Use src/train.py or src/evaluate.py. Official baselines require separate verified adapters.")
