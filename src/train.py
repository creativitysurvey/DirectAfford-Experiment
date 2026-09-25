import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DirectAffordModel
from dataset import AffordanceNetKFoldDataset

def train_one_epoch(model, dataloader, optimizer, device, save_path):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (point_clouds, point_masks, text_queries) in enumerate(dataloader):
        point_clouds = point_clouds.to(device)
        point_masks = point_masks.to(device)
        
        optimizer.zero_grad()
        
        target_z = model.get_text_embeddings(text_queries, device)
        pred_z = model(point_clouds)
        
        from loss import direct_predictive_alignment_loss
        loss = direct_predictive_alignment_loss(pred_z, target_z, point_masks)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}: Physical Alignment Loss = {loss.item():.4f} | Mask Active Points = {int(point_masks.sum().item())}")
            
        # [核心补丁]：每 400 个 Batch 自动固化一次物理权重
        if batch_idx > 0 and batch_idx % 400 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"--> [Auto-Checkpoint] 物理流形权重已安全更新至 Batch {batch_idx}")
            
    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing physically on device: {device}")
    
    model = DirectAffordModel(text_encoder_name="./clip-vit-base-patch32").to(device)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4, 
        weight_decay=1e-4
    )
    
    data_directory = "../data" if os.path.exists("../data/full_shape_train_data.pkl") else "./data"
    save_path = os.path.join(data_directory, "directafford_weights.pth")
    print(f"Initializing dense physical dataset from: {data_directory}")
    
    train_dataset = AffordanceNetKFoldDataset(data_dir=data_directory, fold_id=0, split='train', num_points=2048)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    
    print("Starting Direct Predictive Training Loop on Real Physical Data (Dense Supervision)...")
    
    try:
        for epoch in range(1, 4):
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, save_path)
            print(f"=== Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ===")
            torch.save(model.state_dict(), save_path)
            print(f"=== Epoch 权重已保存至 {save_path} ===")
            
    except KeyboardInterrupt:
        # [安全补丁]：捕获前台的 Ctrl+C 中断，保证权重不丢失
        print("\n[中断预警] 捕获到用户中断操作！正在紧急固化当前物理权重...")
        torch.save(model.state_dict(), save_path)
        print(f"--> [紧急固化成功] 最新权重已保存至 {save_path}，可以安全退出。")
