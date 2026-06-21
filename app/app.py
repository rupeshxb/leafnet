"""
LeafNet — Flask backend for potato leaf disease classification.
Loads the trained MobileNetV2 model and serves predictions via a
simple upload-and-result web interface, per the deployment design
described in Chapter 3 §3.5 and Chapter 2 §2.6 of the LeafNet thesis.
"""

import os
# Suppress TensorFlow GPU warnings on CPU-only deployment environments
# Set CUDA_VISIBLE_DEVICES=-1 to explicitly disable GPU even if CUDA is present
# Remove or comment this line if deploying to a GPU-enabled server in future
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import base64
import io
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))

# ── Load the trained model once at startup, not per-request ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "MobileNetV2_final.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Class index mapping — MUST match the training generator's class_indices ──
# Confirmed from Colab Cell 7 output: {'Early_Blight': 0, 'Healthy': 1, 'Late_Blight': 2}
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]

# ── Plain-language labels and recommended actions, per §2.6 usability design ──
DISPLAY_INFO = {
    "Early_Blight": {
        "label": "Early Blight",
        "action": (
            "This is consistent with Early Blight (Alternaria solani). "
            "Remove and discard affected leaves, avoid overhead watering, "
            "and consult your local agricultural extension officer about "
            "an appropriate fungicide for your area."
        ),
    },
    "Late_Blight": {
        "label": "Late Blight",
        "action": (
            "This is consistent with Late Blight (Phytophthora infestans), "
            "a fast-spreading and serious disease. Isolate or remove affected "
            "plants promptly and contact your local agricultural extension "
            "officer as soon as possible, since this disease can spread "
            "rapidly under cool, humid conditions."
        ),
    },
    "Healthy": {
        "label": "Healthy",
        "action": (
            "No signs of Early Blight or Late Blight were detected in this "
            "leaf. Continue routine monitoring of your crop."
        ),
    },
}

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 65.0  # below this the image is likely not a clear potato leaf


def preprocess_image(image_file):
    """
    Preprocess an uploaded image to match the exact pipeline used during
    training (Colab Cell 7): resize to 224x224, convert to RGB, rescale
    pixel values to [0, 1]. Returns a (1, 224, 224, 3) array ready for
    model.predict().
    """
    img = Image.open(image_file)
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def image_to_base64(image_file):
    image_file.seek(0)
    data = image_file.read()
    encoded = base64.b64encode(data).decode("utf-8")
    mime = "image/jpeg"
    name = getattr(image_file, "filename", "")
    if name.lower().endswith(".png"):
        mime = "image/png"
    elif name.lower().endswith(".webp"):
        mime = "image/webp"
    return f"data:{mime};base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "leaf_image" not in request.files or request.files["leaf_image"].filename == "":
        return render_template("index.html", result=None, error="Please choose an image to upload.")

    file = request.files["leaf_image"]

    try:
        img_b64 = image_to_base64(file)
        img_array = preprocess_image(file)
    except Exception:
        return render_template("index.html", result=None,
                                error="Could not read that file. Please upload a JPG or PNG image.")

    predictions = model.predict(img_array, verbose=0)[0]   # shape (3,)
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    all_probs = [
        {"class_key": CLASS_NAMES[i], "label": DISPLAY_INFO[CLASS_NAMES[i]]["label"],
         "pct": round(float(predictions[i]) * 100, 1)}
        for i in range(len(CLASS_NAMES))
    ]

    if confidence < CONFIDENCE_THRESHOLD:
        result = {
            "class_key": "unclear",
            "label": "Cannot identify",
            "confidence": f"{confidence:.1f}",
            "action": (
                "The image could not be confidently identified as a potato leaf. "
                "Tips for a better result: hold the leaf steady and close to the camera, "
                "ensure it is in sharp focus, use good natural lighting, and make sure "
                "the leaf fills most of the frame with a plain background if possible."
            ),
            "all_probs": all_probs,
            "image_b64": img_b64,
        }
        return render_template("index.html", result=result, error=None)

    info = DISPLAY_INFO[predicted_class]
    result = {
        "class_key": predicted_class,
        "label": info["label"],
        "confidence": f"{confidence:.1f}",
        "action": info["action"],
        "all_probs": all_probs,
        "image_b64": img_b64,
    }

    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)