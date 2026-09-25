import pickle
import os

anno_path = "data/laso_dataset/LASO_dataset/anno_val.pkl"
with open(anno_path, 'rb') as f: 
    raw_anno = pickle.load(f)
    anno_data = list(raw_anno.values()) if isinstance(raw_anno, dict) else raw_anno

print("\n🔍 正在扫描验证集，为您寻找完美的实验对象...\n" + "="*80)
print(f"{'Index (target_idx)':<20} | {'Query (功能指令)':<30} | {'Shape ID'}")
print("="*80)

for i in range(100):
    query = anno_data[i]['affordance'][0] if isinstance(anno_data[i]['affordance'], list) else anno_data[i]['affordance']
    shape_id = anno_data[i]['shape_id']
    # 标红那些可能符合 "把手/握持/倾倒" 的高价值 Query
    if any(keyword in query.lower() for keyword in ['hold', 'pour', 'grasp', 'handle']):
        print(f"\033[91m{i:<20} | {query:<30} | {shape_id}\033[0m  <-- 🌟 高价值目标！")
    else:
        print(f"{i:<20} | {query:<30} | {shape_id}")
