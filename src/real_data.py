"""Strict adapter for the repo's documented LASO pickle schema.
Only load trusted pickle files. No missing-object or random-label fallback.
"""
import csv
import hashlib
import pickle
from pathlib import Path
import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_points(points, mask, num_points, rng):
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3 or not len(points):
        raise ValueError("Expected nonempty (N, >=3) coordinates")
    points = points[:, :3]
    if len(mask) != len(points) or not np.isfinite(points).all() or not np.isfinite(mask).all():
        raise ValueError("Nonfinite values or point/mask length mismatch")
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError("Masks must lie in [0, 1]")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) == num_points:
        indices = np.arange(num_points)
    elif len(points) > num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = np.concatenate([np.arange(len(points)), rng.choice(len(points), num_points-len(points), replace=True)])
    return points[indices], mask[indices]


class LASORealDataset:
    def __init__(self, data_dir, split="train", num_points=2048, seed=42, query_mode="official",
                 sampling="fixed", keep_ratio=1.0):
        if sampling not in {"fixed", "epoch"} or not np.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
            raise ValueError("Invalid training sampling policy")
        if (split != "train" or sampling == "fixed") and (sampling != "fixed" or keep_ratio != 1):
            raise ValueError("Epoch sampling/point thinning are training-only; fixed requires keep_ratio=1")
        self.split, self.query_mode = split, query_mode
        self.num_points, self.seed = num_points, seed
        self.sampling, self.keep_ratio, self.epoch = sampling, keep_ratio, 0
        base = Path(data_dir)
        if (base / "LASO_dataset").is_dir():
            base = base / "LASO_dataset"
        self.paths = [base / f"anno_{split}.pkl", base / f"objects_{split}.pkl"]
        self.hashes = {str(p.resolve()): file_hash(p) for p in self.paths}
        with open(self.paths[0], "rb") as f:
            annotations = pickle.load(f)
        with open(self.paths[1], "rb") as f:
            objects = pickle.load(f)
        annotations = list(annotations.values()) if isinstance(annotations, dict) else annotations
        if isinstance(objects, list):
            keys = [str(o["shape_id"]) for o in objects]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate shape IDs")
            objects = dict(zip(keys, objects))
        if not isinstance(objects, dict) or not isinstance(annotations, list):
            raise ValueError("Unsupported LASO schema; convert explicitly before training")
        objects = {str(k): v for k, v in objects.items()}
        self.questions = {}
        if query_mode == "official":
            csv_path = base / "Affordance-Question.csv"
            self.hashes[str(csv_path.resolve())] = file_hash(csv_path)
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    key = (row["Object"], row["Affordance"])
                    if key in self.questions:
                        raise ValueError("Duplicate object/affordance question row")
                    self.questions[key] = row
        elif query_mode != "affordance":
            raise ValueError("Unknown LASO query mode")
        self.samples = []
        for annotation in annotations:
            shape_id = str(annotation["shape_id"])
            query = annotation["affordance"]
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ambiguous multi-query annotation; expand explicitly")
                query = query[0]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Expected a nonempty query string")
            mask = annotation["mask"]
            if isinstance(mask, dict):
                mask = mask[query]  # never pick an unrelated mask
            points = objects[shape_id]  # missing shapes are errors
            if isinstance(points, dict):
                fields = [k for k in ("point_cloud", "points", "coordinate") if k in points]
                if len(fields) != 1:
                    raise ValueError("Missing or ambiguous coordinate field")
                points = points[fields[0]]
            points = np.asarray(points, dtype=np.float32)
            sample_points(points, mask, num_points, np.random.default_rng(0))
            points = points[:, :3] - points[:, :3].mean(axis=0)
            radius = np.linalg.norm(points, axis=1).max()
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("Degenerate point cloud cannot be normalized")
            points = points / radius
            questions = None
            if query_mode == "official":
                row = self.questions[(annotation["class"], query)]
                columns = [f"Question{i}" for i in (range(1,15) if split == "train" else [0])]
                questions = [row[c] for c in columns]
                if any(not isinstance(q, str) or not q.strip() for q in questions):
                    raise ValueError("Missing or empty official question")
            self.samples.append((points, mask, query, shape_id, questions))
        if not self.samples:
            raise ValueError("Empty dataset")

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        points, mask, query, _, questions = self.samples[index]
        if questions:
            query = questions[np.random.randint(len(questions))] if self.split == "train" else questions[0]
        if self.sampling == "epoch":
            # Independent of call order/global RNG; restored from epoch on resume.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,index,self.epoch]))
            if self.keep_ratio < 1:
                keep = max(1,int(len(points)*self.keep_ratio))
                indices = rng.choice(len(points),keep,replace=False)
                points, mask = points[indices], np.asarray(mask)[indices]
            points, mask = sample_points(points,mask,self.num_points,rng)
        else:
            # Legacy path stays byte-for-byte reproducible for validation and old runs.
            points, mask = sample_points(points, mask, self.num_points, np.random.default_rng(self.seed + index))
        return points, mask, query
