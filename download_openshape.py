import os
import shutil
from huggingface_hub import hf_hub_download

# 强行设置国内高速镜像环境变量，打通网络壁垒
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 正在通过高速镜像源下载 OpenShape 官方权重 (OpenShape/openshape-pointbert-shapenet, 约 61MB)...")

try:
    # 从官方仓库精准定位 model.pt 文件
    file_path = hf_hub_download(
        repo_id="OpenShape/openshape-pointbert-shapenet", 
        filename="model.pt"
    )
    # 将下载好的底层缓存文件直接复制并重命名到我们的工程目录
    os.makedirs("./data", exist_ok=True)
    shutil.copy(file_path, "./data/pointbert.pt")
    print("✅ 下载并部署成功！权重已放置于: ./data/pointbert.pt")
except Exception as e:
    print(f"❌ 下载失败: {e}")
