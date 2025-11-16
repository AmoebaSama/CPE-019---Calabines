import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

# -------------------
# Streamlit App Title
# -------------------
st.title("VTuber vs Human Classifier (Pre-trained)")

# -------------------
# Paths
# -------------------
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (224, 224)

# -------------------
# Load Pre-trained Model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Loaded pre-trained model successfully!")
else:
    st.error("Pre-trained model not found! Upload `vtuber_model.h5` in the repo root.")
    st.stop()  # Stop further execution

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = image.img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    if confidence >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-confidence:.3f})")
