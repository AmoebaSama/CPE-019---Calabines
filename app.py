# =============================
# 2️⃣ Import libraries
# =============================
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# =============================
# 3️⃣ Load pretrained model
# (MobileNetV2 transfer learning)
# =============================
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    x = tf.keras.layers.Dense(2, activation="softmax")(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

model = load_model()

# =============================
# 4️⃣ Preprocess image
# =============================
def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# =============================
# 5️⃣ Prediction function
# =============================
CLASSES = ["Human", "VTuber"]

def predict(img):
    arr = preprocess(img)
    pred = model.predict(arr)[0]
    idx = np.argmax(pred)
    return CLASSES[idx], pred[idx]

# =============================
# 6️⃣ STREAMLIT UI
# =============================
%%writefile app.py
import streamlit as st
from PIL import Image
from vtuber_classifier import predict

st.title("VTuber vs Human Classifier")
st.write("Upload an image. The model will determine if it's a VTuber or a real human.")

uploaded = st.file_uploader("Upload a file", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=250)
    
    label, confidence = predict(img)
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: `{confidence:.3f}`")




