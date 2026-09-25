#!/usr/bin/env python3
"""Audit full-shape pickle schema and affordance-mask coverage before training."""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_CLASSES = [
    "grasp", "lift", "contain", "open", "sit", "support",
    "wrap-grasp", "pour", "move", "display", "push", "pull",
    "listen", "wear", "press", "cut", "stab", "hold",
]


def as_names(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(item) for item in value]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    root = Path(args.data_dir)

    for split, filename in (
        ("train", "full_shape_train_data.pkl"),
        ("val", "full_shape_val_data.pkl"),
    ):
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

        with path.open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, (list, tuple)):
            raise SystemExit(f"{path}: expected list/tuple, got {type(rows).__name__}")

        annotated = Counter()
        exact = Counter()
        positive_masks = Counter()
        mask_points = Counter()
        affordance_vocab = Counter()
        label_vocab = Counter()
        missing_candidates = defaultdict(Counter)
        malformed = 0

        for row in rows:
            if not isinstance(row, dict):
                malformed += 1
                continue
            names = as_names(row.get("affordance"))
            shape = row.get("full_shape")
            labels = shape.get("label", {}) if isinstance(shape, dict) else {}
            if not isinstance(labels, dict):
                malformed += 1
                labels = {}

            affordance_vocab.update(names)
            label_vocab.update(str(key) for key in labels)
            for name in names:
                if name not in EXPECTED_CLASSES:
                    continue
                annotated[name] += 1
                if name not in labels:
                    missing_candidates[name].update(str(key) for key in labels)
                    continue
                exact[name] += 1
                try:
                    import numpy as np
                    mask = np.asarray(labels[name]).reshape(-1)
                    mask_points[name] += int(mask.size)
                    positive_masks[name] += int((mask > 0.5).sum())
                except Exception as exc:
                    print(f"WARNING: cannot inspect mask {name!r} in {path}: {exc}")

        print(f"\n[{split}] {path} samples={len(rows)} malformed_records={malformed}")
        print("affordance vocabulary:", sorted(affordance_vocab))
        print("label-key vocabulary:", sorted(label_vocab))
        print("class              annotated exact_mask positive_points/total_points")
        for name in EXPECTED_CLASSES:
            print(
                f"{name:18s} {annotated[name]:9d} {exact[name]:10d} "
                f"{positive_masks[name]:14d}/{mask_points[name]}"
            )
        absent = [name for name in EXPECTED_CLASSES if not annotated[name]]
        if absent:
            print("expected classes absent from affordance field:", absent)
        missing = [name for name in EXPECTED_CLASSES if annotated[name] and not exact[name]]
        if missing:
            print("candidate keys for missing exact matches:")
            for name in missing:
                print(f"  {name}: {missing_candidates[name].most_common(10)}")


if __name__ == "__main__":
    main()
