import torch
import json

# Load the checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')

print("Keys in checkpoint:", checkpoint.keys())

# Check if it contains model_state_dict
if 'model_state_dict' in checkpoint:
    print("✅ Found model_state_dict - extracting model weights...")
    model_weights = checkpoint['model_state_dict']
    
    # Save just the model weights
    torch.save(model_weights, 'model_weights.pth')
    print("✅ Saved model weights as 'model_weights.pth'")
    
elif 'state_dict' in checkpoint:
    print("✅ Found state_dict - extracting model weights...")
    model_weights = checkpoint['state_dict']
    torch.save(model_weights, 'model_weights.pth')
    print("✅ Saved model weights as 'model_weights.pth'")
    
elif 'model' in checkpoint:
    print("✅ Found model - extracting model weights...")
    model_weights = checkpoint['model']
    torch.save(model_weights, 'model_weights.pth')
    print("✅ Saved model weights as 'model_weights.pth'")
    
else:
    print("❌ No model weights found in checkpoint")
    print("Available keys:", list(checkpoint.keys()))