from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

# Load pretrained ViT model
model_name = "google/vit-base-patch16-224"
imageprocessor = AutoImageProcessor.from_pretrained(model_name)
vitmodel = AutoModel.from_pretrained(model_name)
vitmodel.eval()

# Put model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vitmodel.to(device)

def get_vit_embedding(image_np, model_name="google/vit-base-patch16-224", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    inputs = imageprocessor(images=image_np, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vitmodel(**inputs)
    
    return outputs.last_hidden_state[:, 0]  # CLS token
