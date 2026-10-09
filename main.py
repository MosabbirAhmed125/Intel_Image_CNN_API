import asyncio
import gc
import json
import os
from io import BytesIO

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "intel_cnn_model.onnx")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

IMG_SIZE = (150, 150)
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 10_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")

# --- ONNX Runtime session (single-thread, CPU only) ---
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

prediction_lock = asyncio.Lock()

app = FastAPI(
    title="Intel Image Classification API",
    description="CNN image classifier for buildings, forest, glacier, mountain, sea, and street images.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://intel-image-cnn.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes):
    """Validate, resize to 150×150, normalise to [0, 1] float32, return (1, 150, 150, 3)."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail="Image dimensions are too large. Please upload a smaller image.",
            )

        image = image.resize(IMG_SIZE)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid JPG, JPEG, or PNG image.",
        )


@app.get("/")
def root():
    return {
        "status": "running",
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

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 5 MB.",
        )

    processed_image = preprocess_image(image_bytes)

    try:
        async with prediction_lock:
            raw_output = session.run([output_name], {input_name: processed_image})

        raw = np.asarray(raw_output[0][0], dtype=np.float64)

        # The model's final Dense layer already uses softmax, so `raw` is
        # normally a probability vector. Only apply softmax if the output
        # does not sum to ~1.0 (i.e. unnormalized logits).
        if 0.98 <= float(np.sum(raw)) <= 1.02:
            preds = raw
        else:
            exp = np.exp(raw - np.max(raw))
            preds = exp / exp.sum()

        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index])
        predicted_class = class_names[predicted_index]

        all_predictions = {
            class_names[i]: round(float(preds[i]), 4)
            for i in range(len(class_names))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_predictions": all_predictions,
        }

    finally:
        del image_bytes
        del processed_image
        if "raw_output" in locals():
            del raw_output
        gc.collect()
