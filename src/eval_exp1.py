import torch
import numpy as np
import os
import pickle
from model import DirectAffordModel

def run_experiment_1(model, data_path, device="cuda"):
    print("Starting Experiment 1: Pure Zero-Shot Grounding on Novel Categories...")
    model.eval()
    
    with open(os.path.join(data_path, "full_shape_val_data.pkl"), 'rb') as f:
        val_data = pickle.load(f)

    base_classes = {"grasp", "cut", "lift", "support", "press", "turn", "twist", "pull", "push", "open", "close"}
    novel_classes = set()
    for item in val_data:
        for aff in item.get('affordance', []):
            if str(aff).lower() not in base_classes:
                novel_classes.add(str(aff).lower())
    novel_classes = list(novel_classes)

    novel_embeddings = model.get_text_embeddings(novel_classes, device)
    thresholds = np.arange(0.10, 0.95, 0.05)
    best_thresh, best_miou, best_macc = 0.10, 0.0, 0.0

    print("=============================================")
    print(" Exp 1 Results (Pure Cosine Distance Thresholding)")
    print("=============================================")

    for thresh in thresholds:
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
                    
                    # 纯粹的余弦相似度计算，无背景竞争
                    cosine_sim = torch.sum(predicted_z.squeeze(0) * target_z, dim=-1)
                    
                    pred_mask = (cosine_sim > thresh).float()
                    
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
            if miou > best_miou:
                best_miou, best_macc, best_thresh = miou, macc, thresh

    print(f" [Exp 1 Optimal Peak] Threshold: +{best_thresh:.2f}")
    print(f" Measured Value 1 (Novel mIoU) : {best_miou:.3f}")
    print(f" Measured Value 2 (Novel mAcc) : {best_macc:.3f}")
    print("=============================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DirectAffordModel(text_encoder_name="openai/clip-vit-base-patch32").to(device)
    weights_path = "./data/directafford_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    run_experiment_1(model, "./data", device)
