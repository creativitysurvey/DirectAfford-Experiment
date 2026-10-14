"""Evaluate a compatible DirectAfford checkpoint on a complete real-data split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import DirectAffordModel
from real_data import file_hash
from metrics import compute_metrics
from train import build_dataset, probabilities, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--classes", choices=["seen", "novel"], default="novel")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit-queries",action="store_true")
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.audit_queries and args.split != "val":
        parser.error("Query audit is development-only; use --split val")
    # Only load trusted checkpoints. Fail on missing/incompatible weights.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["dataset"] == "full_shape" and args.split == "test":
        parser.error("full_shape currently provides val only; use --split val --classes novel")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    config_args = argparse.Namespace(**{**config, "data_dir": args.data_dir})
    seed_all(config["seed"])
    dataset = build_dataset(config_args, args.split, args.classes)
    if config["dataset"] == "laso" and args.split == "test":
        test_ids = {sample[3] for sample in dataset.samples}
        for split, ids in checkpoint["split_ids"].items():
            if test_ids.intersection(ids):
                raise ValueError(f"Test shapes overlap with {split}")
    model = DirectAffordModel(config["text_encoder"],architecture=config.get("architecture","legacy")).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions, targets, queries, per_sample = [], [], [], []
    audit_differences=[]
    alternate_pool=list(dict.fromkeys(dataset[i][2] for i in range(min(128,len(dataset))))) if args.audit_queries else []
    if args.audit_queries and len(alternate_pool)<2:
        raise ValueError("Query audit needs at least two distinct queries")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for pcs, masks, text in loader:
            p = probabilities(model, pcs.to(args.device), text, config["loss"], config["logit_scale"]).cpu().numpy()
            if args.audit_queries and len(audit_differences)<128:
                wrong_queries=[next(q for q in alternate_pool if q!=correct) for correct in text]
                wrong=probabilities(model,pcs.to(args.device),wrong_queries,config["loss"],config["logit_scale"]).cpu().numpy()
                audit_differences.extend(np.abs(p-wrong).mean(axis=1).tolist())
            y = masks.numpy()
            predictions.extend(p)
            targets.extend(y)
            queries.extend(text)
            per_sample.extend(compute_metrics(a,b) for a,b in zip(p,y))
    predictions, targets = np.stack(predictions), np.stack(targets)
    raw_predictions = predictions.copy()
    calibration_hash = None
    if args.calibration:
        from calibration import apply_calibration
        parameters = json.loads(args.calibration.read_text())
        if parameters["checkpoint_sha256"] != file_hash(args.checkpoint):
            raise ValueError("Calibration belongs to another checkpoint")
        if parameters["source_split"] != "val":
            raise ValueError("Calibration must be fitted on validation only")
        predictions = apply_calibration(predictions,parameters["slope"],parameters["bias"])
        calibration_hash = file_hash(args.calibration)
        per_sample = [compute_metrics(a,b) for a,b in zip(predictions,targets)]
    defined_auc = [s["auc"] for s in per_sample if s["auc"] is not None]
    report = {"method": "DirectAfford", "dataset":config["dataset"], "architecture":config.get("architecture","legacy"),
              "query_audit_mean_absolute_change":float(np.mean(audit_differences[:128])) if audit_differences else None,
              "calibration_sha256":calibration_hash, "raw_pooled":compute_metrics(raw_predictions,targets), "query_mode": config["laso_query_mode"], "loss": config["loss"], "split": args.split,
              "classes": args.classes if config["dataset"] == "full_shape" else "LASO queries",
              "checkpoint_sha256": file_hash(args.checkpoint), "data_hashes": dataset.hashes,
              "seed": config["seed"], "samples": len(dataset), "num_points": config["num_points"],
              "pooled": compute_metrics(predictions,targets),
              "macro_auc": float(np.mean(defined_auc)) if defined_auc else None,
              "macro_mae": float(np.mean([s["mae"] for s in per_sample])),
              "undefined_auc_samples": len(dataset)-len(defined_auc),
              "sampling": "seeded fixed per sample; all annotations retained", "torch": torch.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/"predictions.npz", probabilities=predictions, masks=targets, queries=np.asarray(queries),raw_probabilities=raw_predictions,
                        shape_ids=np.asarray([s[3] for s in dataset.samples]) if config["dataset"] == "laso" else np.asarray([str(i) for i in range(len(dataset))]))
    (args.output_dir/"metrics.json").write_text(json.dumps(report, indent=2))
    (args.output_dir/"per_sample.json").write_text(json.dumps(per_sample, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
