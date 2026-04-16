# load_model.py
import torch
from model import CompleteViTModel  # or your class name
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_vit_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    print("🤖 Loading trained model...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # If model keys don't have 'model.' prefix
    if not next(iter(state_dict.keys())).startswith("model."):
        print("Adding 'model.' prefix to checkpoint keys...")
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    model = CompleteViTModel()
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded and ready!")
    return model
