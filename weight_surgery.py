import torch
import os
from src.model import DirectAffordModel

def perform_surgery(official_weights_path, output_weights_path):
    print("🔪 启动终极吻合手术...")
    
    state_dict = torch.load(official_weights_path, map_location="cpu")
    if 'model' in state_dict: state_dict = state_dict['model']
    elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']

    target_model = DirectAffordModel(text_encoder_name="openai/clip-vit-base-patch32")
    target_state_dict = target_model.state_dict()

    new_state_dict = {}
    matched = 0

    print("🔍 进行 6 层 Transformer 精准血管映射...")
    for target_key, target_tensor in target_state_dict.items():
        if target_key.startswith("text_encoder.") or target_key.startswith("tokenizer.") or target_key.startswith("patch_embed.") or target_key.startswith("fp_mlp.") or target_key.startswith("regression_head."):
             new_state_dict[target_key] = target_tensor
             continue

        source_key = None
        if target_key.startswith("transformer.layers."):
            parts = target_key.split(".")
            layer_idx = parts[2]
            suffix = ".".join(parts[3:])

            if suffix == "norm1.weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.norm.weight"
            elif suffix == "norm1.bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.norm.bias"
            elif suffix == "self_attn.in_proj_weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.fn.to_qkv.weight"
            elif suffix == "self_attn.in_proj_bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.fn.to_qkv.bias"
            elif suffix == "self_attn.out_proj.weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.fn.to_out.0.weight"
            elif suffix == "self_attn.out_proj.bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.0.fn.to_out.0.bias"
            elif suffix == "norm2.weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.norm.weight"
            elif suffix == "norm2.bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.norm.bias"
            elif suffix == "linear1.weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.fn.net.0.weight"
            elif suffix == "linear1.bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.fn.net.0.bias"
            elif suffix == "linear2.weight": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.fn.net.3.weight"
            elif suffix == "linear2.bias": source_key = f"module.ppat.transformer.layers.{layer_idx}.1.fn.net.3.bias"

        if source_key and source_key in state_dict:
            source_tensor = state_dict[source_key]
            if source_tensor.shape == target_tensor.shape:
                new_state_dict[target_key] = source_tensor
                matched += 1
            else:
                print(f"⚠️ 形状错配: {target_key}")
                new_state_dict[target_key] = target_tensor
        else:
            new_state_dict[target_key] = target_tensor

    print(f"✅ 成功植入 Transformer 先验矩阵数量: {matched} / 72")
    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
    torch.save(new_state_dict, output_weights_path)
    print(f"✅ 权重已固化至: {output_weights_path}")

if __name__ == "__main__":
    perform_surgery("./data/pointbert.pt", "./data/directafford_weights.pth")
