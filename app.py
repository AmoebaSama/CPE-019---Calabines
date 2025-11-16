import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# -------------------
# Streamlit App Title
# -------------------
st.title("VTuber vs Human Classifier")

# -------------------
# Dataset and Model Paths
# -------------------
DATASET_DIR = "dataset"  # folder in repo root
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)  # smaller size for lighter model
BATCH_SIZE = 16

# -------------------
# Data Generators (Grayscale)
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# -------------------
# Load or Build Model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------
# Train Model Button
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a while!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5  # small epoch number for demo
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

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
    
    # Reversing the prediction if VTuber=1, Human=0
    if pred >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {pred:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-pred:.3f})")
