import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# -------------------
# Streamlit title
# -------------------
st.title("VTuber vs Human Classifier")

# -------------------
# Paths
# -------------------
MODEL_PATH = "vtuber_model.h5"  # Must be in repo
IMAGE_SIZE = (224, 224)

# -------------------
# Load model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Loaded trained model!")
else:
    st.error(f"Model file not found at {MODEL_PATH}. Please upload vtuber_model.h5 to the repo.")

# -------------------
# Image Upload & Prediction
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "png", "jpeg"])

if uploaded_file and os.path.exists(MODEL_PATH):
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)

    if confidence >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-confidence:.3f})")
