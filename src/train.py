"""Real-data trainer with explicit objectives, validation and resumable checkpoints."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import DirectAffordModel
from loss import dense_alignment_loss
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
    features = model(pcs)
    text = model.get_text_embeddings(queries, pcs.device)
    cosine = (features * text.unsqueeze(1)).sum(-1)
    # Preserve original scoring for the alignment-only ablation.
    return ((cosine + 1) / 2).clamp(0, 1) if mode == "alignment" else torch.sigmoid(cosine * scale)


def validate(model, loader, device, mode, scale):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for pcs, masks, queries in loader:
            predictions.append(probabilities(model, pcs.to(device), queries, mode, scale).cpu().numpy())
            targets.append(masks.numpy())
    return compute_metrics(np.concatenate(predictions), np.concatenate(targets))


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
    parser.add_argument("--loss", choices=["alignment", "bce", "alignment_bce"], default="alignment")
    parser.add_argument("--logit-scale", type=float, default=10.0)
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", default="outputs/directafford")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_points) <= 0 or args.lr <= 0 or args.logit_scale <= 0:
        parser.error("epochs, batch size, points, learning rate and scale must be positive")
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
    model = DirectAffordModel(args.text_encoder).to(args.device)
    # Includes geometric encoder AND regression head. Frozen text parameters excluded.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    config = vars(args).copy()
    config["resume"] = str(args.resume) if args.resume else None
    start, best, history = 1, float("inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("dataset", "laso_query_mode", "data_dir", "batch_size", "num_points", "fold", "seed", "lr", "loss", "logit_scale", "text_encoder"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if checkpoint["data_hashes"] != hashes:
            raise ValueError("Dataset changed since checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, history = checkpoint["epoch"]+1, checkpoint["best_mae"], checkpoint["history"]
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
                "selection": "validation MAE (seen classes for full_shape); no test selection"}
    (output/"run.json").write_text(json.dumps(metadata, indent=2))
    for epoch in range(start, args.epochs+1):
        model.train()
        total, count = 0.0, 0
        for pcs, masks, queries in train_loader:
            pcs, masks = pcs.to(args.device), masks.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(pcs)
            text = model.get_text_embeddings(queries, args.device)
            loss = dense_alignment_loss(prediction, text, masks, args.loss, args.logit_scale)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            total += loss.item()*len(pcs)
            count += len(pcs)
        metrics = validate(model, val_loader, args.device, args.loss, args.logit_scale)
        row = {"epoch": epoch, "train_loss": total/count, **metrics}
        history.append(row)
        improved = metrics["mae"] < best
        best = min(best, metrics["mae"])
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                   "config": config, "data_hashes": hashes, "split_ids": split_ids, "best_mae": best, "history": history,
                   "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                   "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_save(payload, output/"last.pth")
        if improved:
            atomic_save(payload, output/"best.pth")
        (output/"history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
