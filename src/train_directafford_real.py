import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import time

from model import DirectAffordModel

# ==========================================
# 1. 极其硬核的 LASO 真实数据加载器
# ==========================================
class LASORealDataset(Dataset):
    def __init__(self, data_dir, split="train", num_points=2048):
        self.num_points = num_points
        anno_path = os.path.join(data_dir, "LASO_dataset", f"anno_{split}.pkl")
        obj_path = os.path.join(data_dir, "LASO_dataset", f"objects_{split}.pkl")
        
        print(f"📦 正在将 {split} 集载入物理内存...")
        with open(anno_path, 'rb') as f:
            self.anno_data = pickle.load(f)
        with open(obj_path, 'rb') as f:
            obj_data = pickle.load(f)
            
        # 统一 anno 格式为 List
        if isinstance(self.anno_data, dict):
            self.anno_data = list(self.anno_data.values())
            
        # 构建 O(1) 的 3D 点云哈希表，极速提效
        print("🔨 正在构建 3D 点云哈希索引...")
        self.obj_dict = {}
        if isinstance(obj_data, dict):
            for k, v in obj_data.items():
                self.obj_dict[str(k)] = v.get('point_cloud', v.get('points', v.get('coordinate', v))) if isinstance(v, dict) else v
        elif isinstance(obj_data, list):
            for item in obj_data:
                shape_id = str(item.get('shape_id', ''))
                self.obj_dict[shape_id] = item.get('point_cloud', item.get('points', item.get('coordinate', item)))
        
        print(f"✅ {split} 集载入完成！共发现 {len(self.anno_data)} 条有效情境指令。")

    def __len__(self):
        return len(self.anno_data)

    def __getitem__(self, idx):
        sample = self.anno_data[idx]
        shape_id = str(sample['shape_id'])
        
        # 提取 Query
        query = sample['affordance']
        if isinstance(query, list): query = query[0]
        
        # 提取 Mask
        raw_mask = sample['mask']
        if isinstance(raw_mask, dict):
            raw_mask = raw_mask.get(query, list(raw_mask.values())[0])
        raw_mask = np.array(raw_mask, dtype=np.float32)
        
        # 提取 点云
        raw_pc = self.obj_dict.get(shape_id, np.zeros((len(raw_mask), 3), dtype=np.float32))
        if isinstance(raw_pc, dict): # 极强容错
             raw_pc = raw_pc.get('point_cloud', raw_pc.get('points', raw_pc.get('coordinate', np.zeros((len(raw_mask), 3)))))
        
        # 空间归一化与采样 (截断或 Padding 到 2048)
        if len(raw_pc) > self.num_points:
            # 随机采样
            choice = np.random.choice(len(raw_pc), self.num_points, replace=False)
            pc = raw_pc[choice]
            mask = raw_mask[choice]
        elif len(raw_pc) < self.num_points:
            pad_size = self.num_points - len(raw_pc)
            pc = np.concatenate([raw_pc, raw_pc[:pad_size]], axis=0)
            mask = np.concatenate([raw_mask, raw_mask[:pad_size]], axis=0)
        else:
            pc = raw_pc
            mask = raw_mask

        return torch.tensor(pc, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), query

# ==========================================
# 2. 真实物理训练引擎
# ==========================================
def train_directafford_real(data_dir, epochs=300, batch_size=32, device="cuda"):
    print("🚀 启动 [真实物理重训计划]: DirectAfford (Ours)")
    
    # 初始化模型
    model = DirectAffordModel().to(device)
    optimizer = optim.AdamW(model.geometric_encoder.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none') # 使用 BCE 更好的处理密集预测
    
    # 挂载真实的 DataLoader
    train_dataset = LASORealDataset(data_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    print("=======================================================")
    print("🖥️ A6000 显卡火力全开中 (Real LASO Data)...")
    print("=======================================================")
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (pcs, masks, queries) in enumerate(train_loader):
            pcs, masks = pcs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 1. 实时提取真实 Query 的文本向量 (由于文本引擎冻结，可以用 torch.no_grad 提速，但模型内已处理)
            with torch.no_grad():
                text_targets = model.get_text_embeddings(queries, device) # (B, 512)
                
            # 2. 提取 3D 几何特征
            point_features = model(pcs) # (B, N, 512)
            
            # 3. Direct Predictive Alignment (直接流形对齐)
            # 计算点与文本的余弦相似度，并放缩到 logits 空间
            text_targets = text_targets.unsqueeze(1) # (B, 1, 512)
            logits = torch.sum(point_features * text_targets, dim=-1) * 10.0 # Temperature scaling
            
            # 4. 计算 Loss (只关注有有效点的 batch)
            loss = criterion(logits, masks)
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Real Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # 覆盖保存最新真实的物理权重
        torch.save(model.state_dict(), "./data/directafford_weights.pth")

if __name__ == "__main__":
    train_directafford_real("./data/laso_dataset", epochs=300, batch_size=32)
