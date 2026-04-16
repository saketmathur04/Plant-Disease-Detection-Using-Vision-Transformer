import torch

ckpt = torch.load("best_vit_model.pth", map_location="cpu")
print("\nKeys in checkpoint:", list(ckpt.keys())[:10])
if "model_state_dict" in ckpt:
    print("✅ Found nested model_state_dict:", len(ckpt["model_state_dict"]))
elif "state_dict" in ckpt:
    print("✅ Found nested state_dict:", len(ckpt["state_dict"]))
else:
    print("✅ Appears to be a raw state_dict:", len(ckpt))
