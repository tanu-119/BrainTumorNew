from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import io
import base64
from PIL import Image
import requests
import shutil
import gdown

app = Flask(__name__)

IMG_SIZE = (128, 128)
label_mapping = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor",
    3: "No Tumor"
}

# Model configuration
#MODEL_URL = "https://drive.google.com/uc?export=download&id=1-3KZAIoDLV98_5f9KH84tL07QyQBawxT"
#MODEL_DIR = "model"
#MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.h5")

MODEL_PATH = "brain_tumor_model.h5"
def download_model():
    """Download the model file using gdown if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model with gdown...")
        try:
            file_id = "1XIV4QM0Wj74DYyayfEwjth4WJM2pbhtH"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download model with gdown: {e}")
            raise
    else:
        print("Model already exists. Skipping download.")


def load_tumor_model():
    """Load the model with error handling"""
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully with standard method")
        return model
    except Exception as e:
        print(f"Standard load failed: {e}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully with tf.keras method")
            return model
        except Exception as e:
            print(f"All model loading methods failed: {e}")
            raise


# Ensure the model is downloaded and then loaded
download_model()
model = load_tumor_model()


def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


def get_mask_image(mask_array):
    mask = (mask_array.squeeze() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).convert("L")
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/')
def home():
    return render_template('braintumor.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    try:
        img = preprocess_image(file)
        seg_pred, clf_pred = model.predict(img)
        tumor_type = label_mapping[np.argmax(clf_pred)]
        mask_b64 = get_mask_image(seg_pred[0])

        return jsonify({
            'tumor_type': tumor_type,
            'mask_image': f"data:image/png;base64,{mask_b64}",
            'confidence': float(np.max(clf_pred))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)