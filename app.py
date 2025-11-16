import streamlit as st
import numpy as np
from PIL import Image
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

# -------------------
# Streamlit title
# -------------------
st.title("VTuber vs Human Classifier")

# -------------------
# Paths
# -------------------
DATASET_DIR = "dataset"            # folder inside repo
MODEL_PATH = "vtuber_model.h5"    # model file in repo
IMAGE_SIZE = (128, 128)           # smaller image to reduce model size
BATCH_SIZE = 16

# -------------------
# Data generators
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.15
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# -------------------
# Load or create model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
else:
    # small MobileNetV2
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1],3), alpha=0.35)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.info("No saved model found. Ready to train a new model.")

# -------------------
# Train model button
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a few minutes!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5  # keep small for Streamlit Cloud
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

# -------------------
# Image Upload and Prediction
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # preprocess
    img_array = image.img_to_array(img.resize(IMAGE_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # prediction
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    if confidence >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-confidence:.3f})")
