"""Train-subset capacity probe or same-object validation query diagnostic.

All pairs use identical point indices for both queries. No test data are read.
These restricted-set metrics are diagnostics, not full-split model accuracy.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from model import DirectAffordModel
from real_data import LASORealDataset, file_hash, sample_points
from train import probabilities, seed_all
from metrics import compute_metrics


def paired_inputs(dataset, max_shapes=8, seed=42):
    """One maximally different mask pair per shape; sample independently of query."""
    if max_shapes <= 0:
        raise ValueError("max_shapes must be positive")
    groups = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        groups[sample[3]].append(i)
    rng = np.random.default_rng(seed)
    pcs, masks, queries, records = [], [], [], []
    eligible = 0
    for shape in rng.permutation(sorted(groups)):
        candidates = groups[shape]
        best, difference = None, 0.0
        for left, i in enumerate(candidates):
            for j in candidates[left+1:]:
                a, b = dataset.samples[i], dataset.samples[j]
                if a[2] == b[2]:
                    continue
                if a[0].shape != b[0].shape or not np.allclose(a[0],b[0]):
                    raise ValueError("Same-shape annotations have inconsistent coordinates")
                delta = float(np.abs(np.asarray(a[1])-np.asarray(b[1])).mean())
                if delta > difference:
                    best, difference = (i,j), delta
        if best is None or difference <= 0.01:
            continue
        eligible += 1
        if len(records) >= max_shapes:
            continue
        shape_seed = seed + int(hashlib.sha256(str(shape).encode()).hexdigest()[:16],16)
        pair_masks, pair_queries = [], []
        for index in best:
            points, mask, affordance, _, questions = dataset.samples[index]
            query = questions[0] if questions else affordance
            points, mask = sample_points(points,mask,dataset.num_points,np.random.default_rng(shape_seed))
            pcs.append(points); masks.append(mask); queries.append(query)
            pair_masks.append(mask); pair_queries.append(query)
        if pair_queries[0] == pair_queries[1]:
            raise ValueError("Different affordances have identical questions")
        records.append({"shape_id":str(shape),"annotation_indices":list(best),
                        "queries":pair_queries,"full_mask_difference":difference,
                        "sampled_mask_difference":float(np.abs(pair_masks[0]-pair_masks[1]).mean())})
    if not records:
        raise ValueError("No same-object distinct-function pairs with different masks found")
    return np.stack(pcs),np.stack(masks),queries,records,eligible


def query_diagnostic(predictions, masks):
    p, y = np.asarray(predictions),np.asarray(masks)
    compute_metrics(p,y)  # validates probability and target ranges
    if p.ndim != 2 or len(p)%2:
        raise ValueError("Expected adjacent two-query pairs")
    p, y = p.reshape(-1,2,p.shape[-1]),y.reshape(-1,2,y.shape[-1])
    matched = np.abs(p-y).mean(axis=(1,2))
    swapped = np.abs(p[:,::-1,:]-y).mean(axis=(1,2))
    delta = y[:,0]-y[:,1]
    mass = np.abs(delta).sum()
    return {"pairs":len(p),"matched_mae":float(matched.mean()),"swapped_mae":float(swapped.mean()),
            "swapped_minus_matched_mae":float((swapped-matched).mean()),
            "fraction_pairs_correct_query_lower_mae":float((matched<swapped-1e-8).mean()),
            "mean_prediction_change":float(np.abs(p[:,0]-p[:,1]).mean()),
            "target_weighted_direction_margin":float(((p[:,0]-p[:,1])*delta).sum()/mass) if mass else None,
            "interpretation":"Positive swapped-minus-matched MAE and direction margin support query discrimination"
                             " on these selected pairs only; they do not establish full validation/test quality."}


def predict(model, pcs, queries, mode, scale, device, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0,len(pcs),batch_size):
            chunks.append(probabilities(model,torch.as_tensor(pcs[i:i+batch_size],device=device),
                                        queries[i:i+batch_size],mode,scale).cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["overfit","queries"],required=True)
    parser.add_argument("--data-dir",required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--architecture",choices=["query_local","query_tokens"],default="query_tokens")
    parser.add_argument("--text-encoder",default="openai/clip-vit-base-patch32")
    parser.add_argument("--num-points",type=int,default=2048)
    parser.add_argument("--max-shapes",type=int,default=8)
    parser.add_argument("--steps",type=int,default=300)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if min(args.num_points,args.max_shapes,args.steps,args.batch_size)<=0 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error("Sizes, steps and LR must be positive")
    if args.mode == "queries" and args.checkpoint is None:
        parser.error("Query diagnostic requires --checkpoint")
    if args.mode == "overfit" and args.checkpoint is not None:
        parser.error("Capacity probe starts fresh; omit --checkpoint")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty diagnostic directory")
    seed_all(args.seed)
    checkpoint_hash,checkpoint_config = None,None
    if args.mode == "queries":
        checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        config = checkpoint["config"]
        if config["dataset"] != "laso":
            raise ValueError("Requires a LASO checkpoint")
        checkpoint_hash,checkpoint_config = file_hash(args.checkpoint),config
        args.architecture = config.get("architecture","legacy")
        args.num_points,args.text_encoder = config["num_points"],config["text_encoder"]
        mode,scale = config["loss"],config["logit_scale"]
        query_mode = config["laso_query_mode"]
    else:
        mode,scale,query_mode = "bce",10.0,"official"
    split = "train" if args.mode == "overfit" else "val"
    ds = LASORealDataset(args.data_dir,split,args.num_points,args.seed,query_mode)
    if args.mode == "queries":
        # Bind diagnostics to the validation data used by this checkpoint.
        if ds.hashes != checkpoint["data_hashes"]["val"]:
            raise ValueError("Validation data/path hashes differ from checkpoint")
    pcs,y,queries,pairs,eligible = paired_inputs(ds,args.max_shapes,args.seed)
    model = DirectAffordModel(args.text_encoder,architecture=args.architecture).to(args.device)
    if args.mode == "queries":
        model.load_state_dict(checkpoint["model"],strict=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    history = []
    start = time.monotonic()
    def record(step):
        p = predict(model,pcs,queries,mode,scale,args.device,args.batch_size)
        row = {"step":step,"subset_metrics":compute_metrics(p,y),"query_diagnostic":query_diagnostic(p,y)}
        history.append(row)
        (args.output_dir/"history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        return p
    p = record(0)
    if args.mode == "overfit":
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        for step in range(1,args.steps+1):
            indices = rng.choice(len(pcs),min(args.batch_size,len(pcs)),replace=False)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _,_,logits = model.score_queries(torch.as_tensor(pcs[indices],device=args.device),
                                             [queries[i] for i in indices],scale)
            loss = F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y[indices],device=args.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite probe loss")
            loss.backward();optimizer.step()
            if step%25 == 0 or step == args.steps:
                p = record(step)
        torch.save({"model":model.state_dict(),"architecture":args.architecture,
                    "purpose":"training subset capacity diagnostic only"},args.output_dir/"overfit_probe.pth")
    report = {"mode":args.mode,"split":split,"architecture":args.architecture,
              "torch":torch.__version__,"numpy":np.__version__,
              "checkpoint_sha256":checkpoint_hash,"checkpoint_config":checkpoint_config,
              "seed":args.seed,"num_points":args.num_points,"data_hashes":ds.hashes,
              "eligible_shapes":eligible,"selected_pairs":pairs,"history":history,
              "zero_prediction_mae":float(np.abs(y).mean()),"seconds":time.monotonic()-start,
              "probe_config":{"steps":args.steps if args.mode=='overfit' else 0,
                              "lr":args.lr if args.mode=='overfit' else None,
                              "batch_size":args.batch_size,"text_encoder":args.text_encoder},
              "source_hashes":{str(f.relative_to(Path(__file__).resolve().parents[1])):file_hash(f)
                               for f in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/model.py']},
              "limitations":"Restricted pairs chosen for mask difference. Capacity probe evaluates its own training"
                             " subset with fixed questions. No independent generalization or target attainment claim."}
    np.savez_compressed(args.output_dir/"predictions.npz",probabilities=p,masks=y,queries=np.asarray(queries))
    (args.output_dir/"diagnostic.json").write_text(json.dumps(report,indent=2))
    print(f"Diagnostic: {args.output_dir/'diagnostic.json'}",flush=True)


if __name__ == "__main__":
    main()
