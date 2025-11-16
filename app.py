# app.py
import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model

# -------------------
# App title
# -------------------
st.title("VTuber vs Human Classifier (Grayscale)")

# -------------------
# Dataset and model paths
# -------------------
DATASET_DIR = "dataset"  # Folder inside repo root
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)  # smaller size for light model
BATCH_SIZE = 8  # small batch to reduce memory

# -------------------
# Dataset check
# -------------------
if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset folder not found at {DATASET_DIR}! Upload it to the repo root.")
    st.stop()

# -------------------
# Data generators (grayscale)
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# -------------------
# Load or build model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
else:
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.info("No saved model found. Ready to train a new model.")

# -------------------
# Train model
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a few minutes!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5  # keep small to reduce file size
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

# -------------------
# Image upload and prediction
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)[0][0]

    # reverse prediction if needed
    if pred >= 0.5:
        st.success(f"Predicted: Human (Confidence: {pred:.3f})")
    else:
        st.success(f"Predicted: VTuber (Confidence: {1-pred:.3f})")
