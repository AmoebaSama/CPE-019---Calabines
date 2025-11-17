import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------
# Streamlit App Title
# -------------------
st.title("VTuber vs Human Classifier")

st.write("""
### How to use:
1. Upload an image of either a **Human** or a **VTuber**.
2. You may use sample images here:  
   🔗 https://drive.google.com/drive/folders/1Yi9iDOr0HRudf-E_EXMmkgGg85RrvV8C?usp=sharing  
3. See if the image that you have uploaded is a **Human** or a **VTuber**.
P.S. The uploaded image will be grayscaled.
Created by: **Calabines, Ayden Jarrick J.**
""")

# -------------------
# Dataset and Model Paths
# -------------------
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)  # smaller size for lighter model

# -------------------
# Load Model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("Pre-trained model not found! Upload 'vtuber_model.h5' in the repo root.")
    st.stop()

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    
    pred = model.predict(img_array)[0][0]
    
    # VTuber=1, Human=0
    if pred >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {pred:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-pred:.3f})")

