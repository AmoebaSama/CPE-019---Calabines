import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

st.title("VTuber vs Human Classifier")

# Paths
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (224, 224)

# Load pre-trained model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Loaded pre-trained model.")
else:
    st.error("Pre-trained model not found! Upload 'vtuber_model.h5' in the repo root.")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = image.img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    # Reverse predictions if needed
    if confidence >= 0.5:
        st.success(f"Predicted: Human (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: VTuber (Confidence: {1-confidence:.3f})")
