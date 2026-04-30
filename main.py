import os

# Optional: reduces TensorFlow oneDNN warning messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Configuration
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "intel_cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

IMG_SIZE = (150, 150)

# =========================
# Load Model and Classes
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Intel Image Classification API",
    description="CNN image classifier for glacier, sea, forest, and street images.",
    version="1.0.0",
)

# For development, this allows all origins.
# Later, replace "*" with your deployed Vercel frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helper Function
# =========================


def preprocess_image(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        image_array = np.array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid JPG, JPEG, or PNG image.",
        )


# =========================
# Routes
# =========================


@app.get("/")
def root():
    return {
        "message": "Intel Image Classification API is running.",
        "classes": class_names,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are supported.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    predicted_class = class_names[predicted_index]

    all_predictions = {
        class_names[i]: round(float(predictions[0][i]), 4)
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_predictions": all_predictions,
    }
