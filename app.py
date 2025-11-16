import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load lightweight classifier model
@st.cache_resource
def load_model():
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    x = tf.keras.layers.Dense(2, activation="softmax")(base.output)
    return tf.keras.Model(base.input, x)

model = load_model()

CLASSES = ["Human", "VTuber"]

def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def classify(img):
    arr = preprocess(img)
    pred = model.predict(arr)[0]
    idx = np.argmax(pred)
    return CLASSES[idx], float(pred[idx])

# ===================
# Streamlit UI
# ===================
st.set_page_config(page_title="VTuber Detector", layout="centered")
st.title("VTuber vs Human Detector")
st.write("Upload an image below to classify!")

uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=300)

    label, conf = classify(img)
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: `{conf:.3f}`")
