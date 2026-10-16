import torch
import torch.nn.functional as F


def point_pair_ranking_loss(logits, masks, points_per_class=64, generator=None):
    """Per-query logistic ranking, positives >0.5, negatives <=0.5.

    Uniformly sample up to K points per class without replacement, average all
    sampled positive/negative pairs, then average queries with both classes.
    Single-class queries contribute no ranking gradient; BCE still trains them.
    A private generator keeps this sampling separate from model/dropout RNG.
    """
    if logits.ndim != 2 or logits.shape != masks.shape or not logits.numel():
        raise ValueError("Ranking expects matching nonempty [batch, points] tensors")
    if not isinstance(points_per_class,int) or not 1 <= points_per_class <= 1024:
        raise ValueError("Ranking sample budget must be an integer in [1,1024]")
    if not torch.isfinite(logits).all() or not torch.isfinite(masks).all() or ((masks<0)|(masks>1)).any():
        raise ValueError("Invalid logits or soft targets")
    terms = []
    for scores,target in zip(logits,masks):
        positive = torch.nonzero(target>0.5,as_tuple=True)[0]
        negative = torch.nonzero(target<=0.5,as_tuple=True)[0]
        if not positive.numel() or not negative.numel():
            continue
        if positive.numel()>points_per_class:
            positive = positive[torch.randperm(positive.numel(),device=logits.device,generator=generator)[:points_per_class]]
        if negative.numel()>points_per_class:
            negative = negative[torch.randperm(negative.numel(),device=logits.device,generator=generator)[:points_per_class]]
        differences = scores[positive].unsqueeze(1)-scores[negative].unsqueeze(0)
        terms.append(F.softplus(-differences).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0.0

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
