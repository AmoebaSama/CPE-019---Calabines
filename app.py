import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale handwritten digit image.")

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

# Preprocess function
def preprocess(img):
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img)/255.0
    arr = arr.reshape(1,28,28,1)
    return arr

# File uploader
uploaded = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=150)
    arr = preprocess(img)
    prediction = model.predict(arr)
    label = np.argmax(prediction)
    st.write(f"Predicted Digit: **{label}**")