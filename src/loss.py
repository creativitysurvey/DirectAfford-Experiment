import torch
import torch.nn.functional as F

def direct_predictive_alignment_loss(predicted_point_embeddings, target_text_embeddings, point_masks):
    """
    严格按照论文 Equation (VI.1) 和 Algorithm 2 实现的 Direct Predictive Loss。
    修正了 Mask 维度广播陷阱。
    """
    B, N, D = predicted_point_embeddings.shape
    
    # 扩展 Text Target 以匹配点云形状 (B, N, D)
    target_expanded = target_text_embeddings.unsqueeze(1).expand(-1, N, -1)
    
    # 计算公式 (VI.1) 中的分子部分（余弦相似度）
    cosine_sim = F.cosine_similarity(predicted_point_embeddings, target_expanded, dim=-1)
    
    # Equation (VI.1): L = 1 - (z_p · z_T) / (||z_p|| * ||z_T||)
    point_loss = 1.0 - cosine_sim
    
    # 【修复点】强行拍扁 point_masks，确保其形状严格为 (B, N)
    active_mask = (point_masks.view(B, N) > 0.5).float()
    
    # 避免分母为 0
    num_active = active_mask.sum(dim=1).clamp(min=1e-6)
    
    # 计算 Batch 内每个 Instance 的平均距离损失
    loss_batch = (point_loss * active_mask).sum(dim=1) / num_active
    
    # 防御机制：如果整个 Batch 没有激活点，返回 0 梯度
    if active_mask.sum() == 0:
        return predicted_point_embeddings.sum() * 0.0
        
    return loss_batch.mean()



def loss_components(predicted, target, masks, scale=10.0, logits=None):
    """Expose competing positive-alignment and dense segmentation terms."""
    alignment = direct_predictive_alignment_loss(predicted, target, masks)
    if logits is None:
        logits = (predicted * target.unsqueeze(1)).sum(-1) * scale
    bce = F.binary_cross_entropy_with_logits(logits, masks.reshape_as(logits))
    return alignment, bce


def dense_alignment_loss(predicted, target, masks, mode="alignment_bce", scale=10.0,
                         alignment_weight=1.0, bce_weight=1.0):
    """Optional background-supervised variants; original alignment stays available."""
    if mode not in {"alignment", "bce", "alignment_bce"}:
        raise ValueError("Unknown loss mode")
    if alignment_weight < 0 or bce_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    alignment, bce = loss_components(predicted, target, masks, scale)
    if mode == "alignment":
        return alignment_weight * alignment
    if mode == "bce":
        return bce_weight * bce
    return alignment_weight * alignment + bce_weight * bce
