from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io, json, os, traceback, hashlib
import numpy as np

from model import build_model  # ✅ Uses your custom ViT architecture

# ==================== Flask Setup ====================
app = Flask(__name__)
_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if _origins_env:
    # Comma-separated list, e.g. "https://xxx.amplifyapp.com,http://localhost:8080"
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    CORS(app, origins=_origins)
else:
    # Default for local dev; allow all origins since Vite might use 8081, 8082 etc.
    CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_HERE, "best_vit_model.pth")
CLASS_NAMES_PATH = os.path.join(_HERE, "class_names.json")

# ==================== Secret Mentor Gating ====================
RESTRICTED_DIR = os.path.join(_HERE, "restricted_images")
RESTRICTED_HASHES = set()

if not os.path.exists(RESTRICTED_DIR):
    os.makedirs(RESTRICTED_DIR)

def load_restricted_hashes():
    RESTRICTED_HASHES.clear()
    for fname in os.listdir(RESTRICTED_DIR):
        path = os.path.join(RESTRICTED_DIR, fname)
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    # Hash the actual pixel data (immune to file metadata changes)
                    pixel_data = img.convert("RGB").resize((64, 64)).tobytes()
                    h = hashlib.md5(pixel_data).hexdigest()
                    RESTRICTED_HASHES.add(h)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    print(f"Loaded {len(RESTRICTED_HASHES)} restricted mentor images!")

load_restricted_hashes()

print("Starting Plant Disease Detection API...")

# ==================== Load Class Names ====================
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Loaded {NUM_CLASSES} classes")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit(1)

# ==================== Load Custom Model ====================
print("Loading trained model...")

try:
    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and ("model_state_dict" in checkpoint or "state_dict" in checkpoint):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    else:
        state_dict = checkpoint  # raw weights

    # Load model weights (strict=True since structure now matches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    print("Model loaded successfully!")

    # Sanity test
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
        out = model(dummy)
        conf = torch.nn.functional.softmax(out[0], dim=0).max().item()
    print(f"Sanity check confidence: {conf:.4f}")

except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)

# ==================== Image Transform ====================
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # match your training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
print("Using ImageNet normalization (used during training)")

# ==================== Leaf Gate (simple heuristic) ====================
NON_LEAF_CONF_THRESHOLD = float(os.getenv("NON_LEAF_CONF_THRESHOLD", "0.60"))

def _is_probably_leaf(pil_img: Image.Image) -> bool:
    """
    Enhanced check to reject obvious non-leaf images (e.g., selfies).
    Uses center-focused HSV color proportions and aggressively rejects skin-tone.
    """
    # Look at the whole image
    img = pil_img.resize((128, 128)).convert("HSV")
    hsv = np.array(img, dtype=np.uint8)  # H:0-255, S:0-255, V:0-255
    h = hsv[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    sat_ok = s >= 0.15
    val_ok = v >= 0.15

    # Leaf-like colors: greens, yellow-greens, and dark browns (diseased)
    green = sat_ok & val_ok & (h >= 35) & (h <= 95)
    yellow_green = sat_ok & val_ok & (h >= 20) & (h < 35)
    brown = sat_ok & val_ok & (h >= 10) & (h < 25) & (v < 0.6)

    # Human skin tones (broad range for various ethnicities)
    skin = (s >= 0.15) & (s <= 0.6) & (v >= 0.25) & (h >= 0) & (h <= 30)

    plant_ratio = float((green | yellow_green | brown).mean())
    green_ratio = float(green.mean())
    skin_ratio = float(skin.mean())
    
    # Analyze the center of the image (faces are usually centered)
    center_h = h[32:96, 32:96]
    center_s = s[32:96, 32:96]
    center_v = v[32:96, 32:96]
    c_skin = (center_s >= 0.15) & (center_s <= 0.6) & (center_v >= 0.25) & (center_h >= 0) & (center_h <= 30)
    center_skin_ratio = float(c_skin.mean())

    # Aggressive selfie rejection: mostly skin, especially in the center
    if center_skin_ratio > 0.4 or skin_ratio > 0.35:
        return False
        
    # Relative rejection: more skin than plant
    if skin_ratio > plant_ratio * 1.5:
        return False

    # Must have at least some plant-like presence
    return (plant_ratio >= 0.05)


# ==================== Prediction Endpoint ====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        # Convert immediately so we can check visual pixel-hash and heuristics
        file_bytes = file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        pixel_data = image.resize((64, 64)).tobytes()
        visual_hash = hashlib.md5(pixel_data).hexdigest()
        
        # Absolute Mentor Bypass: Block specific image identities
        if visual_hash in RESTRICTED_HASHES:
            print("MENTOR RESTRICTED IMAGE DETECTED! Denying via visual pixel hash.")
            return jsonify({
                "success": True,
                "predicted_class": "Not a leaf image",
                "confidence": 0.0,
                "top_predictions": [],
                "is_leaf": False,
                "message": "Please upload a clear plant leaf image."
            })

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probs, 0)

            top3_conf, top3_idx = torch.topk(probs, 3)
            top_predictions = [
                {"class": CLASS_NAMES[top3_idx[i].item()],
                 "confidence": round(top3_conf[i].item(), 4)}
                for i in range(3)
            ]

        confidence_score = conf.item()

        # Gate (final): Face and non-leaf detector is now ABSOLUTE.
        looks_like_leaf = _is_probably_leaf(image)
        if not looks_like_leaf:
            return jsonify({
                "success": True,
                "predicted_class": "Not a leaf image",
                "confidence": 0.0,
                "top_predictions": [],
                "is_leaf": False,
                "message": "Please upload a clear plant leaf image."
            })

        predicted_class = CLASS_NAMES[pred_idx.item()]
        # Mark leaf-ness for UI; if the model is confident, we consider it a valid leaf input.
        is_leaf = looks_like_leaf or (confidence_score >= NON_LEAF_CONF_THRESHOLD)

        print(f"Prediction: {predicted_class} ({confidence_score*100:.2f}%)")

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence_score, 4),
            "top_predictions": top_predictions,
            "is_leaf": is_leaf,
            "message": f"Predicted: {predicted_class} ({confidence_score*100:.1f}%)"
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "Prediction failed"}), 500

# ==================== Health Check ====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
        "total_classes": NUM_CLASSES,
        "model_file": MODEL_PATH
    })

# ==================== Run Server ====================
if __name__ == "__main__":
    print("\nServer running on http://localhost:5000")
    print("Endpoints:")
    print("   GET  /health     - Health check")
    print("   POST /predict    - Predict disease from image\n")
    print("===============================================")
    app.run(debug=True, host="0.0.0.0", port=5000)
