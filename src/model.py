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
    def __init__(self, text_encoder_name="openai/clip-vit-base-patch32", architecture="legacy"):
        if architecture not in {"legacy", "local", "query_local", "query_tokens"}:
            raise ValueError("Unknown architecture")
        super().__init__()
        self.architecture = architecture
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_dim = getattr(self.text_encoder.config, "hidden_size", getattr(self.text_encoder.config, "projection_dim", 512))
        self.geom_dim = 384
        
        self.geometric_encoder = AttentionEnhancedGeometricEncoder(embed_dim=self.geom_dim)
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.geom_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.text_dim)
        )

        if architecture != "legacy":
            # Point coordinates + neighbor-offset statistics retain details
            # that are otherwise available only through patch interpolation.
            self.point_local = nn.Sequential(nn.Linear(12,64),nn.GELU(),nn.Linear(64,64))
            self.local_fusion = nn.Sequential(nn.Linear(self.geom_dim+64,self.geom_dim),
                                              nn.LayerNorm(self.geom_dim),nn.GELU())
        if architecture in {"query_local", "query_tokens"}:
            self.query_film = nn.Linear(self.text_dim,2*self.geom_dim)
            self.query_head = nn.Sequential(nn.Linear(self.geom_dim,128),nn.GELU(),
                                             nn.Linear(128,1))
        if architecture == "query_tokens":
            self.token_projection = nn.Linear(self.text_dim,self.geom_dim)
            self.token_attention = nn.MultiheadAttention(self.geom_dim,6,dropout=0,batch_first=True)
            self.token_norm = nn.LayerNorm(self.geom_dim)
            self.token_ffn = nn.Sequential(nn.Linear(self.geom_dim,512),nn.GELU(),
                                            nn.Linear(512,self.geom_dim))
            self.token_output_norm = nn.LayerNorm(self.geom_dim)

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def get_text_embeddings(self, text_queries, device):
        with torch.no_grad():
            inputs = self.tokenizer(text_queries, padding=True, return_tensors="pt", truncation=True).to(device)
            text_features = self.text_encoder(**inputs).pooler_output
            return F.normalize(text_features, p=2, dim=-1)

    def score_queries(self, point_clouds, queries, scale=10.0):
        """One text encoding; token context is only consumed by the new variant."""
        if self.architecture != "query_tokens":
            text = self.get_text_embeddings(queries,point_clouds.device)
            prediction,logits = self.forward_with_logits(point_clouds,text,scale)
        else:
            with torch.no_grad():
                inputs = self.tokenizer(queries,padding=True,return_tensors="pt",truncation=True).to(point_clouds.device)
                encoded = self.text_encoder(**inputs)
                text = F.normalize(encoded.pooler_output,p=2,dim=-1)
                tokens = encoded.last_hidden_state
                padding = ~inputs["attention_mask"].bool()
            prediction,logits = self.forward_with_logits(point_clouds,text,scale,tokens,padding)
        return prediction,text,logits

    def encode_points(self, point_clouds):
        geom = self.geometric_encoder(point_clouds)
        if self.architecture == "legacy":
            return geom
        # Deterministic KNN descriptors; includes self, shared for all variants.
        neighbors = min(16,point_clouds.shape[1])
        distances = square_distance(point_clouds,point_clouds).clamp_min(0)
        indices = distances.topk(neighbors,dim=-1,largest=False).indices
        relative = index_points(point_clouds,indices)-point_clouds.unsqueeze(2)
        descriptors = torch.cat([point_clouds,relative.mean(2),relative.amax(2),relative.amin(2)],dim=-1)
        local = self.point_local(descriptors)
        return self.local_fusion(torch.cat([geom,local],dim=-1))

    def forward(self, point_clouds):
        geom = self.encode_points(point_clouds)
        return F.normalize(self.regression_head(geom),p=2,dim=-1)

    def forward_with_logits(self, point_clouds, text_features, scale=10.0, token_features=None, token_padding=None):
        geom = self.encode_points(point_clouds)
        prediction = F.normalize(self.regression_head(geom),p=2,dim=-1)
        logits = (prediction*text_features.unsqueeze(1)).sum(-1)*scale
        if self.architecture in {"query_local", "query_tokens"}:
            gamma,beta = self.query_film(text_features).chunk(2,dim=-1)
            conditioned = geom*(1+torch.tanh(gamma).unsqueeze(1))+beta.unsqueeze(1)
            if self.architecture == "query_tokens":
                if token_features is None or token_padding is None:
                    raise ValueError("query_tokens needs token context; use score_queries")
                if token_padding.all(dim=1).any():
                    raise ValueError("Text context must contain at least one valid token")
                context = self.token_projection(token_features)
                attention,_ = self.token_attention(conditioned,context,context,
                                                    key_padding_mask=token_padding,need_weights=False)
                conditioned = self.token_norm(conditioned+attention)
                conditioned = self.token_output_norm(conditioned+self.token_ffn(conditioned))
            logits = logits+self.query_head(conditioned).squeeze(-1)
        return prediction,logits
