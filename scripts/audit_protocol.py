"""Read-only validation protocol audit; optionally export existing structure weights.

Does not train, fit calibration, read test labels, or change checkpoint selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

OFFICIAL_REF = "c412ec376310fd9d2de0d3ea06516dea50b9d4e8"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_metrics(probabilities, targets, inclusive=False):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if p.ndim != 2 or p.shape != y.shape or not p.size:
        raise ValueError("Expected matching nonempty [samples, points] arrays")
    if any(not np.isfinite(a).all() or ((a < 0) | (a > 1)).any() for a in (p, y)):
        raise ValueError("Predictions and masks must be finite and in [0,1]")
    binary = (y >= 0.5) if inclusive else (y > 0.5)

    def auc(a, b):
        return float(roc_auc_score(b, a)) if np.unique(b).size == 2 else None

    scores = [auc(a, b) for a, b in zip(p, binary)]
    valid = [a for a in scores if a is not None]
    return {"pooled_auc": auc(p.reshape(-1), binary.reshape(-1)),
            "macro_auc_defined_samples": float(np.mean(valid)) if valid else None,
            "mae": float(np.abs(p - y).mean()), "samples": len(p),
            "points": int(p.size), "undefined_auc_samples": len(p) - len(valid),
            "all_negative_samples": int((~binary.any(axis=1)).sum()),
            "all_positive_samples": int(binary.all(axis=1).sum()),
            "exact_half_label_points": int((y == 0.5).sum()),
            "threshold": ">=0.5" if inclusive else ">0.5",
            "official_overall_auc_comparable": len(valid) == len(p)}


def audit_export(directory):
    metadata_path = directory / "metrics.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split") != "val" or metadata.get("dataset") != "laso":
        raise ValueError("Protocol development audit accepts LASO val exports only")
    if metadata.get("calibration_sha256"):
        raise ValueError("Use raw exports, not calibrated exports")
    path = directory / "predictions.npz"
    with np.load(path, allow_pickle=False) as arrays:
        p, y = arrays["probabilities"], arrays["masks"]
        ids, queries = arrays["shape_ids"], arrays["queries"]
    if ids.ndim != 1 or queries.ndim != 1 or len(ids) != len(p) or len(queries) != len(p):
        raise ValueError("Export identifiers do not align with predictions")
    if metadata["samples"] != len(p) or metadata["num_points"] != p.shape[1]:
        raise ValueError("Export dimensions disagree with metadata")
    strict = protocol_metrics(p, y)
    inclusive = protocol_metrics(p, y, inclusive=True)
    if not np.isclose(strict["mae"], metadata["pooled"]["mae"], atol=1e-7, rtol=1e-6):
        raise ValueError("Export MAE disagrees with metadata")
    recorded = metadata["pooled"]["auc"]
    actual = strict["pooled_auc"]
    if (recorded is None) != (actual is None) or (actual is not None and not np.isclose(recorded, actual)):
        raise ValueError("Export AUC disagrees with metadata")
    return {"directory": str(directory.resolve()),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "metrics_sha256": digest(metadata_path), "predictions_sha256": digest(path),
            "data_hashes": metadata["data_hashes"], "architecture": metadata.get("architecture"),
            "query_mode": metadata["query_mode"], "seed": metadata["seed"],
            "unique_shapes": len(np.unique(ids)), "unique_queries": len(np.unique(queries)),
            "query_audit_mean_absolute_change": metadata.get("query_audit_mean_absolute_change"),
            "current_strict": strict, "laso_inclusive": inclusive,
            "scope": "Same existing predictions; aggregation/threshold comparison only."
                      " Does not reproduce upstream voting, input point order, or baseline scores."}


def audit_histories(root):
    results = []
    for name in ("legacy_bce", "local_bce", "query_local_bce"):
        path = root / "structure" / name / "history.json"
        rows = json.loads(path.read_text())
        if not rows:
            raise ValueError(f"Empty history: {path}")
        def summary(row):
            return {key: row.get(key) for key in ("epoch", "auc", "macro_auc", "mae")}
        pooled = [r for r in rows if r.get("auc") is not None]
        macro = [r for r in rows if r.get("macro_auc") is not None]
        results.append({"candidate": name, "history_sha256": digest(path), "epochs": len(rows),
                        "highest_pooled_auc": summary(max(pooled, key=lambda r: r["auc"])) if pooled else None,
                        "highest_strict_macro_auc": summary(max(macro, key=lambda r: r["macro_auc"])) if macro else None,
                        "lowest_mae": summary(min(rows, key=lambda r: r["mae"])),
                        "note": "History macro uses >0.5; its best epoch may lack a saved checkpoint."})
    return results


def run(args):
    root, out = args.stage2_root.resolve(), args.output_dir.resolve()
    # Keep audit outputs outside the completed experiment tree.
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Choose an audit output directory separate from the stage2 tree")
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Wait until stage2 is complete before auditing")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty audit output directory")
    histories = audit_histories(root)
    exports = sorted((root / "calibration_diagnostics").glob("*/validation_export"))
    if not exports:
        raise ValueError("No existing validation exports found")
    out.mkdir(parents=True, exist_ok=True)
    reports = [audit_export(p) for p in exports]
    links = []
    if args.export_structures:
        repo = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": args.gpu}
        cache = {}
        for name in ("legacy_bce", "local_bce", "query_local_bce"):
            for criterion in ("joint", "auc"):
                checkpoint = root / "structure" / name / f"best_{criterion}.pth"
                fingerprint = digest(checkpoint)
                if fingerprint not in cache:
                    export = out / "exports" / f"{name}_{criterion}"
                    log = out / f"{name}_{criterion}.log"
                    command = [sys.executable, "-u", str(repo / "src/evaluate.py"),
                               "--checkpoint", str(checkpoint), "--data-dir", str(Path(args.data_dir).resolve()),
                               "--split", "val", "--audit-queries", "--device", "cuda", "--output-dir", str(export)]
                    print(json.dumps({"event": "export_validation", "candidate": name,
                                      "criterion": criterion, "log": str(log)}), flush=True)
                    with open(log, "w") as handle:
                        subprocess.run(command, cwd=repo, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, check=True)
                    audited = audit_export(export)
                    if audited["checkpoint_sha256"] != fingerprint:
                        raise ValueError("Checkpoint hash changed during evaluation")
                    reports.append(audited)
                    cache[fingerprint] = str(export)
                links.append({"candidate": name, "criterion": criterion,
                              "checkpoint_sha256": fingerprint, "export": cache[fingerprint]})
    report = {"purpose": "Validation protocol exploration; no independent test claim or target attainment claim",
              "upstream_reference": f"https://github.com/yl3800/LASO/blob/{OFFICIAL_REF}/eval_lyc.py",
              "upstream_rules": {"auc": "per-sample ROC AUC, then sample-weighted average",
                                 "threshold": ">=0.5", "mae": "unthresholded soft masks",
                                 "votes": 3, "loader": "all object points, no loader resampling",
                                 "undefined": "Upstream sums propagate NaN for empty-positive samples;"
                                              " all-positive samples can raise. Here both are counted and excluded"
                                              " from defined-sample macro; equivalence requires none."},
              "stage2_root": str(root), "stage2_plan_sha256": digest(root / "stage2_plan.json"),
              "histories": histories, "exports": reports, "checkpoint_exports": links,
              "comparability": "Require matching validation data hashes, sample order, query mode, point sampling,"
                               " voting, weights and split. Screenshot baseline provenance remains unverified."}
    (out / "protocol_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"histories": histories, "export_metrics": [
        {"directory": r["directory"], "strict": r["current_strict"], "inclusive": r["laso_inclusive"]}
        for r in reports]}, indent=2, allow_nan=False), flush=True)
    print(f"Report: {out / 'protocol_audit.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-structures", action="store_true")
    parser.add_argument("--data-dir", default="data/laso_dataset")
    parser.add_argument("--gpu", default="1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
