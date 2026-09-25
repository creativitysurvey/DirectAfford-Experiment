import torch

def probe(file_path):
    print(f"正在探测文件: {file_path}")
    state_dict = torch.load(file_path, map_location="cpu")
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    print("\n--- 官方权重的真实前 30 个键名 (Key Names) ---")
    keys = list(state_dict.keys())
    for k in keys[:30]:
        print(f"'{k}': {state_dict[k].shape}")
        
    print("\n--- 官方权重的真实后 30 个键名 (Key Names) ---")
    for k in keys[-30:]:
        print(f"'{k}': {state_dict[k].shape}")

if __name__ == "__main__":
    probe("./data/pointbert.pt")
