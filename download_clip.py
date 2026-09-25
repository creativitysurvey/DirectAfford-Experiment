# run this on a machine WITH internet access
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "openai/clip-vit-base-patch32"
save_path = "./clip-vit-base-patch32"

print(f"Downloading {model_id}...")
tokenizer = CLIPTokenizer.from_pretrained(model_id)
model = CLIPTextModel.from_pretrained(model_id)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Saved locally to {save_path}")
