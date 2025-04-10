import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import zipfile
import gdown
import io
import base64

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
            st.write("Downloading model ZIP from Google Drive...")
            file_id = "1XIV4QM0Wj74DYyayfEwjth4WJM2pbhtH"  # Replace with your actual file ID
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_ZIP_PATH, quiet=False, fuzzy=True)
            st.write("Model ZIP downloaded successfully!")

            st.write("Extracting model from ZIP...")
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall()  # Extract the model to the current directory
            st.write("Model extracted.")
        except Exception as e:
            st.error(f"Error during model download/extraction: {e}")
            raise
    else:
        st.write("Model already exists. Skipping download.")

def load_tumor_model():
    """Load the model with error handling."""
    try:
        model = load_model(MODEL_PATH, compile=False)
        st.write("Model loaded successfully with standard method.")
        return model
    except Exception as e:
        st.error(f"Standard load failed: {e}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.write("Model loaded with tf.keras fallback.")
            return model
        except Exception as e:
            st.error(f"All model loading methods failed: {e}")
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

# Streamlit app interface
st.title("Brain Tumor Detection")
st.markdown("Upload an MRI image to predict the tumor type and segmentation.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    try:
        img = preprocess_image(uploaded_file)
        seg_pred, clf_pred = model.predict(img)
        tumor_type = label_mapping[np.argmax(clf_pred)]
        mask_b64 = get_mask_image(seg_pred[0])

        st.write(f"**Predicted Tumor Type**: {tumor_type}")
        st.write(f"**Confidence**: {float(np.max(clf_pred)):.2f}")
        
        # Display segmentation mask
        st.image(f"data:image/png;base64,{mask_b64}", caption="Segmentation Mask", use_column_width=True)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
