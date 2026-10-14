"""Real-data trainer with explicit objectives, validation and resumable checkpoints."""
import argparse
import fcntl
import json
import time
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import DirectAffordModel
from loss import loss_components
from selection import CheckpointSelection
from metrics import compute_metrics
from real_data import LASORealDataset, file_hash


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(args, split, class_mode="seen"):
    if args.dataset == "laso":
        return LASORealDataset(args.data_dir, split, args.num_points, args.seed, args.laso_query_mode)
    from dataset import AffordanceNetKFoldDataset
    dataset = AffordanceNetKFoldDataset(args.data_dir, args.fold, split, args.num_points, class_mode)
    dataset.hashes = {str(p.resolve()): file_hash(p) for p in [Path(args.data_dir)/f"full_shape_{split}_data.pkl"]}
    return dataset


def probabilities(model, pcs, queries, mode, scale):
    text = model.get_text_embeddings(queries, pcs.device)
    features,logits = model.forward_with_logits(pcs,text,scale)
    cosine = (features * text.unsqueeze(1)).sum(-1)
    # Preserve original scoring for the alignment-only ablation.
    return ((cosine + 1) / 2).clamp(0, 1) if mode == "alignment" else torch.sigmoid(logits)


def validate(model, loader, device, mode, scale):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for pcs, masks, queries in loader:
            predictions.append(probabilities(model, pcs.to(device), queries, mode, scale).cpu().numpy())
            targets.append(masks.numpy())
    p, y = np.concatenate(predictions), np.concatenate(targets)
    result = compute_metrics(p, y)
    individual = [compute_metrics(a,b) for a,b in zip(p,y)]
    aucs = [r["auc"] for r in individual if r["auc"] is not None]
    result.update(macro_auc=float(np.mean(aucs)) if aucs else None,
                  undefined_auc_samples=len(individual)-len(aucs),
                  target_mean=float(y.mean()), probability_mean=float(p.mean()),
                  positive_probability_mean=float(p[y > 0.5].mean()) if (y > 0.5).any() else None,
                  negative_probability_mean=float(p[y <= 0.5].mean()) if (y <= 0.5).any() else None)
    return result


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["laso", "full_shape"], default="laso")
    parser.add_argument("--data-dir", "--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--laso-query-mode", choices=["official", "affordance"], default="official")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--architecture",choices=["legacy","local","query_local"],default="legacy")
    parser.add_argument("--loss", choices=["alignment", "bce", "alignment_bce"], default="alignment")
    parser.add_argument("--logit-scale", type=float, default=10.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--selection", choices=["mae", "auc", "joint"], default="mae")
    parser.add_argument("--target-auc", type=float, default=0.871)
    parser.add_argument("--target-mae", type=float, default=0.089)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", default="outputs/directafford")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_points) <= 0 or args.lr <= 0 or args.logit_scale <= 0:
        parser.error("epochs, batch size, points, learning rate and scale must be positive")
    if (args.alignment_weight < 0 or args.bce_weight < 0 or
        not 0 < args.target_auc < 1 or not 0 < args.target_mae < 1 or
        args.lr_patience < 0 or args.early_stop_patience < 0 or args.log_every < 0 or
        not 0 < args.min_lr <= args.lr):
        parser.error("Invalid weights, targets, patience, log interval or minimum LR")
    if args.architecture == "query_local" and (args.loss == "alignment" or args.bce_weight <= 0):
        parser.error("query_local requires a nonzero BCE objective")
    active_weight = (args.alignment_weight if args.loss == "alignment" else args.bce_weight
                     if args.loss == "bce" else args.alignment_weight + args.bce_weight)
    if active_weight <= 0:
        parser.error("Selected objective must have a positive loss weight")
    if not all(np.isfinite(v) for v in (args.lr,args.logit_scale,args.alignment_weight,args.bce_weight,args.target_auc,args.target_mae,args.min_lr)):
        parser.error("Numeric options must be finite")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output/"run.lock", "a") as run_lock:
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another updated trainer owns {output}; do not start duplicates")
        seed_all(args.seed)
        train_dataset, val_dataset = build_dataset(args, "train"), build_dataset(args, "val")
        if args.dataset == "laso":
            train_ids = {s[3] for s in train_dataset.samples}
            val_ids = {s[3] for s in val_dataset.samples}
            if train_ids & val_ids:
                raise ValueError("Train/validation shape overlap: supply disjoint splits")
        split_ids = ({"train": sorted(train_ids), "val": sorted(val_ids)} if args.dataset == "laso" else None)
        hashes = {"train": train_dataset.hashes, "val": val_dataset.hashes}
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output/"last.pth").exists() and not args.resume:
            raise FileExistsError("Existing run: choose a new output directory or --resume")
        model = DirectAffordModel(args.text_encoder,architecture=args.architecture).to(args.device)
        # Includes geometric encoder AND regression head. Frozen text parameters excluded.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        config = vars(args).copy()
        config["resume"] = str(args.resume) if args.resume else None
        selector = CheckpointSelection(args.target_auc,args.target_mae)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                     patience=args.lr_patience, min_lr=args.min_lr) if args.scheduler == "plateau" else None)
        start, history, stale = 1, [], 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            for key in ("dataset", "laso_query_mode", "data_dir", "batch_size", "num_points", "fold", "seed", "lr", "loss", "logit_scale", "text_encoder"):
                if checkpoint["config"][key] != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            if checkpoint["data_hashes"] != hashes:
                raise ValueError("Dataset changed since checkpoint")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Old checkpoints may resume only with legacy defaults; historical AUC
            # peaks cannot be reconstructed without their saved model weights.
            defaults = dict(architecture="legacy",alignment_weight=1.0,bce_weight=1.0,selection="mae",target_auc=0.871,
                            target_mae=0.089,scheduler="none",lr_patience=5,min_lr=1e-6,early_stop_patience=0)
            for key, default in defaults.items():
                if checkpoint["config"].get(key,default) != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}; start a separate run")
            start, history = checkpoint["epoch"]+1, checkpoint["history"]
            if "selection_state" in checkpoint:
                selector.load_state_dict(checkpoint["selection_state"])
                stale = checkpoint.get("stale_epochs",0)
            else:
                selector.best_mae = checkpoint["best_mae"]
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if args.epochs < start:
                raise ValueError("Requested epochs already completed; increase --epochs to resume")
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.set_rng_state(checkpoint["rng_torch"])
            if args.device.startswith("cuda") and checkpoint["rng_cuda"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        metadata = {"config": config, "data_hashes": hashes, "torch": torch.__version__, "numpy": np.__version__,
                    "train_samples": len(train_dataset), "val_samples": len(val_dataset),
                    "source_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                    "selection": f"validation {args.selection}; dual and joint checkpoints; no test selection"}
        (output/"run.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"event":"ready","start_epoch":start,"end_epoch":args.epochs,
                          "train_samples":len(train_dataset),"val_samples":len(val_dataset),
                          "output_dir":str(output),"selection":args.selection}),flush=True)
        for epoch in range(start, args.epochs+1):
            epoch_start = time.monotonic()
            lr_used = optimizer.param_groups[0]["lr"]
            model.train()
            total, align_total, bce_total, count = 0.0, 0.0, 0.0, 0
            for batch_index, (pcs, masks, queries) in enumerate(train_loader,1):
                pcs, masks = pcs.to(args.device), masks.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                text = model.get_text_embeddings(queries, args.device)
                prediction,logits = model.forward_with_logits(pcs,text,args.logit_scale)
                alignment, bce = loss_components(prediction,text,masks,args.logit_scale,logits=logits)
                loss = (args.alignment_weight*alignment if args.loss == "alignment" else
                        args.bce_weight*bce if args.loss == "bce" else
                        args.alignment_weight*alignment + args.bce_weight*bce)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                optimizer.step()
                total += loss.item()*len(pcs)
                align_total += alignment.item()*len(pcs)
                bce_total += bce.item()*len(pcs)
                count += len(pcs)
                if args.log_every and batch_index % args.log_every == 0:
                    print(json.dumps({"event":"batch","epoch":epoch,"batch":batch_index,
                          "batches":len(train_loader),"train_loss":total/count}),flush=True)
            metrics = validate(model, val_loader, args.device, args.loss, args.logit_scale)
            row = {"epoch": epoch, "train_loss": total/count,"alignment_loss":align_total/count,
                   "bce_loss":bce_total/count,"lr":lr_used,"seconds":time.monotonic()-epoch_start,
                   "validation_targets_met":metrics["auc"] is not None and metrics["auc"] > args.target_auc and metrics["mae"] < args.target_mae,**metrics}
            history.append(row)
            improved = selector.update(metrics)
            stale = 0 if improved[args.selection] else stale + 1
            if scheduler is not None:
                if metrics["auc"] is None:
                    raise ValueError("AUC scheduler requires two validation target classes")
                scheduler.step(metrics["auc"])
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                       "config": config, "data_hashes": hashes, "split_ids": split_ids,
                       "best_mae": selector.best_mae, "selection_state":selector.state_dict(),
                       "scheduler":scheduler.state_dict() if scheduler else None,"stale_epochs":stale,"history": history,
                       "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                       "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            atomic_save(payload, output/"last.pth")
            for criterion in ("mae","auc","joint","feasible_auc"):
                if improved[criterion]:
                    atomic_save(payload,output/f"best_{criterion}.pth")
            if improved[args.selection]:
                atomic_save(payload,output/"best.pth")
            temporary = output/"history.json.tmp"
            temporary.write_text(json.dumps(history, indent=2))
            os.replace(temporary,output/"history.json")
            print(json.dumps(row), flush=True)
            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(json.dumps({"event":"early_stop","epoch":epoch,"criterion":args.selection}),flush=True)
                break


if __name__ == "__main__":
    main()
