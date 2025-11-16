import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# -------------------
# Streamlit App Title
# -------------------
st.title("VTuber vs Human Classifier (Grayscale Lightweight CNN)")

# -------------------
# Dataset and Model Paths
# -------------------
DATASET_DIR = "dataset"  # Folder in your repo
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 5  # small for lightweight model

# -------------------
# Data Generator (Grayscale)
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20
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
# Build Lightweight CNN
# -------------------
if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
else:
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.info("No saved model found. Ready to train a new model.")

# -------------------
# Train Model Button
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a while!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # Grayscale
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]

    # -------------------
    # Reverse prediction if necessary
    # -------------------
    # VTuber = cartoonish = 1
    # Human = realistic = 0
    label = "VTuber" if pred >= 0.5 else "Human"
    confidence = pred if pred >= 0.5 else 1 - pred
    
    st.success(f"Predicted: {label} (Confidence: {confidence:.3f})")
