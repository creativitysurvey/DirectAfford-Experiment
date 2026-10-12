import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def fps(xyz, npoint, random_start=True):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = (torch.randint(0, N, (B,), dtype=torch.long, device=device) if random_start
                else torch.zeros(B, dtype=torch.long, device=device))
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

class AttentionEnhancedGeometricEncoder(nn.Module):
    def __init__(self, num_patches=256, k_neighbors=32, embed_dim=384, num_layers=4):
        super().__init__()
        self.num_patches = num_patches
        self.k_neighbors = k_neighbors
        self.phi_local = nn.Sequential(
            nn.Conv2d(3, 64, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU()
        )
        self.e_pos = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=1024, 
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        if N < max(self.num_patches, self.k_neighbors):
            raise ValueError("Too few points for configured patches/neighbors")
        center_idx = fps(xyz, self.num_patches, random_start=self.training)
        centers = index_points(xyz, center_idx)
        sq_dist = square_distance(centers, xyz)
        knn_idx = sq_dist.topk(self.k_neighbors, dim=-1, largest=False)[1]
        neighbor_xyz = index_points(xyz, knn_idx) 
        neighbor_xyz = neighbor_xyz - centers.view(B, self.num_patches, 1, 3)
        
        patch_features = self.phi_local(neighbor_xyz.permute(0, 3, 1, 2)) 
        patch_tokens = torch.max(patch_features, dim=-1)[0].transpose(1, 2) 
        
        pos_embeddings = self.e_pos(centers)
        global_tokens = self.transformer(patch_tokens + pos_embeddings)
        
        dist = square_distance(xyz, centers) 
        idx = dist.topk(3, dim=-1, largest=False)[1]
        dist = dist.topk(3, dim=-1, largest=False)[0]
        weight = 1.0 / (dist + 1e-8)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        
        dense_features = torch.sum(index_points(global_tokens, idx) * weight.view(B, N, 3, 1), dim=2) 
        return dense_features

class DirectAffordModel(nn.Module):
    def __init__(self, text_encoder_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_dim = self.text_encoder.config.projection_dim if hasattr(self.text_encoder.config, 'projection_dim') else 512
        self.geom_dim = 384
        
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=self.geom_dim)
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.geom_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.text_dim)
        )

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def get_text_embeddings(self, text_queries, device):
        with torch.no_grad():
            inputs = self.tokenizer(text_queries, padding=True, return_tensors="pt", truncation=True).to(device)
            text_features = self.text_encoder(**inputs).pooler_output
            return F.normalize(text_features, p=2, dim=-1)

    def forward(self, point_clouds):
        geom_features = self.geometric_encoder(point_clouds)
        predicted_z = self.regression_head(geom_features)
        return F.normalize(predicted_z, p=2, dim=-1)
