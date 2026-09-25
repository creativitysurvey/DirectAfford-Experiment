import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# Canonical 3D AffordanceNet classes. The local pickle files use label-style
# names such as "openable" and "wrap_grasp"; normalize those to query names.
ALL_CLASSES = [
    "grasp", "lift", "contain", "open", "lay", "sit", "support",
    "wrap-grasp", "pour", "display", "push", "pull", "listen",
    "wear", "press", "cut", "stab", "move",
]

ALIASES = {
    "openable": "open",
    "layable": "lay",
    "sittable": "sit",
    "wrap_grasp": "wrap-grasp",
    "pourable": "pour",
    "pushable": "push",
}


def normalize_affordance(name):
    """Normalize dataset label names to canonical query names."""
    value = str(name).strip().casefold()
    value = ALIASES.get(value, value)
    return value.replace("_", "-")


def farthest_point_sample(xyz, npoint):
    N, _ = xyz.shape
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids.astype(np.int32)


class AffordanceNetKFoldDataset(Dataset):
    def __init__(self, data_dir, fold_id=0, split="train", num_points=2048):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0 <= fold_id < 3:
            raise ValueError("fold_id must be 0, 1, or 2")

        self.num_points = num_points
        self.split = split
        file_name = "full_shape_train_data.pkl" if split == "train" else "full_shape_val_data.pkl"
        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "rb") as f:
            self.raw_data = pickle.load(f)

        fold_size = len(ALL_CLASSES) // 3
        novel_classes = ALL_CLASSES[fold_id * fold_size : (fold_id + 1) * fold_size]
        self.target_classes = (
            [c for c in ALL_CLASSES if c not in novel_classes]
            if split == "train"
            else novel_classes
        )

        self.valid_data = []
        self.skipped_missing_mask = 0
        for item in self.raw_data:
            affordances = item.get("affordance", [])
            if isinstance(affordances, str):
                affordances = [affordances]
            active_affs = {normalize_affordance(name) for name in affordances}

            points = item.get("full_shape", {})
            labels = points.get("label", {}) if isinstance(points, dict) else {}
            if not isinstance(labels, dict):
                self.skipped_missing_mask += len(active_affs.intersection(self.target_classes))
                continue

            # Resolve stored label keys through the same normalization so the
            # human-readable query class and mask remain paired correctly.
            label_by_class = {}
            for key, mask in labels.items():
                canonical = normalize_affordance(key)
                if canonical in label_by_class:
                    raise ValueError(
                        f"Multiple label keys map to {canonical!r} in shape "
                        f"{item.get('shape_id', '<unknown>')}"
                    )
                label_by_class[canonical] = mask

            for class_name in self.target_classes:
                if class_name not in active_affs:
                    continue
                if class_name not in label_by_class:
                    self.skipped_missing_mask += 1
                    continue
                self.valid_data.append(
                    {
                        "points": points,
                        "query_class": class_name,
                        "mask": label_by_class[class_name],
                    }
                )

        if not self.valid_data:
            raise ValueError(
                f"No usable {split} samples found in {file_path} for fold {fold_id}; "
                "check affordance names and label masks."
            )

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        cls_name = item["query_class"]
        text_query = f"Where can I {cls_name} this object?"

        raw_points = np.asarray(item["points"]["coordinate"], dtype=np.float32)
        raw_mask = np.asarray(item["mask"], dtype=np.float32).reshape(-1)
        if raw_points.ndim != 2 or raw_points.shape[1] < 3:
            raise ValueError(f"Expected point coordinates with shape (N, >=3), got {raw_points.shape}")
        if len(raw_mask) != len(raw_points):
            raise ValueError(
                f"Point/mask length mismatch for {cls_name}: "
                f"{len(raw_points)} points versus {len(raw_mask)} mask values"
            )
        raw_points = raw_points[:, :3]

        if len(raw_points) > self.num_points:
            fps_idx = farthest_point_sample(raw_points, self.num_points)
            sampled_points = raw_points[fps_idx, :]
            sampled_mask = raw_mask[fps_idx]
        elif len(raw_points) < self.num_points:
            pad_idx = np.random.choice(len(raw_points), self.num_points, replace=True)
            sampled_points = raw_points[pad_idx, :]
            sampled_mask = raw_mask[pad_idx]
        else:
            sampled_points = raw_points
            sampled_mask = raw_mask

        return torch.tensor(sampled_points), torch.tensor(sampled_mask), text_query
