from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import io
import base64
from PIL import Image
import zipfile
import gdown

app = Flask(__name__)

IMG_SIZE = (128, 128)

label_mapping = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor",
    3: "No Tumor"
}

# Model paths
MODEL_ZIP_PATH = "brain_tumor_model.zip"
MODEL_PATH = "brain_tumor_model.h5"

def download_and_extract_model():
    """Download and unzip the model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading model ZIP from Google Drive...")
            file_id = "1XIV4QM0Wj74DYyayfEwjth4WJM2pbhtH"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_ZIP_PATH, quiet=False, fuzzy=True)
            print("Model ZIP downloaded successfully!")

            print("Extracting model from ZIP...")
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall()
            print("Model extracted.")
        except Exception as e:
            print(f"Error during model download/extraction: {e}")
            raise
    else:
        print("Model already exists. Skipping download.")

def load_tumor_model():
    """Load the model with error handling."""
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully with standard method.")
        return model
    except Exception as e:
        print(f"Standard load failed: {e}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded with tf.keras fallback.")
            return model
        except Exception as e:
            print(f"All model loading methods failed: {e}")
            raise

# Prepare model
download_and_extract_model()
model = load_tumor_model()

def preprocess_image(file):
    """Resize and normalize image."""
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def get_mask_image(mask_array):
    """Convert segmentation mask to base64 PNG."""
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
