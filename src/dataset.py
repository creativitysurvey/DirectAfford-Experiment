import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

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
    def __init__(self, data_dir, fold_id=0, split='train', num_points=2048):
        self.num_points = num_points
        self.split = split
        
        file_name = 'full_shape_train_data.pkl' if split == 'train' else 'full_shape_val_data.pkl'
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, 'rb') as f:
            self.raw_data = pickle.load(f)
            
        all_classes = [
            "grasp", "lift", "contain", "open", "sit", "support", 
            "wrap-grasp", "pour", "move", "display", "push", "pull",
            "listen", "wear", "press", "cut", "stab", "hold"
        ]
        
        fold_size = len(all_classes) // 3
        novel_classes = all_classes[fold_id*fold_size : (fold_id+1)*fold_size]
        self.target_classes = [c for c in all_classes if c not in novel_classes] if split == 'train' else novel_classes
        
        self.valid_data = []
        for item in self.raw_data:
            aff = item['affordance']
            active_affs = []
            if isinstance(aff, list):
                active_affs = [a for a in aff if a in self.target_classes]
            elif isinstance(aff, dict):
                active_affs = [k for k in aff.keys() if k in self.target_classes]
            elif isinstance(aff, str) and aff in self.target_classes:
                active_affs = [aff]
                
            for a in active_affs:
                self.valid_data.append({
                    'points': item['full_shape'], 
                    'query_class': a
                })

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        cls_name = item['query_class']
        text_query = f"Where can I {cls_name} this object?"
        
        raw_points = np.array(item['points']['coordinate'], dtype=np.float32)
        # 精准提取当前功能的 0/1 Mask
        raw_mask = np.array(item['points']['label'][cls_name], dtype=np.float32)
        
        if len(raw_points.shape) > 1 and raw_points.shape[1] > 3:
            raw_points = raw_points[:, :3]
            
        # 物理空间同步降采样：点云与 Mask 必须共享同一个索引
        if raw_points.shape[0] > self.num_points:
            fps_idx = farthest_point_sample(raw_points, self.num_points)
            sampled_points = raw_points[fps_idx, :]
            sampled_mask = raw_mask[fps_idx]
        elif raw_points.shape[0] < self.num_points:
            pad_idx = np.random.choice(raw_points.shape[0], self.num_points, replace=True)
            sampled_points = raw_points[pad_idx, :]
            sampled_mask = raw_mask[pad_idx]
        else:
            sampled_points = raw_points
            sampled_mask = raw_mask
            
        return torch.tensor(sampled_points), torch.tensor(sampled_mask), text_query
