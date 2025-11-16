import streamlit as st
import numpy as np
from PIL import Image
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------
# App Title
# -------------------
st.title("VTuber vs Human Classifier (Small CNN)")

# -------------------
# Dataset and model path
# -------------------
DATASET_DIR = "dataset"  # folder in repo
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)  # small size for tiny CNN
BATCH_SIZE = 16

# -------------------
# Data Generators
# -------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# -------------------
# Build small CNN
# -------------------
if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
else:
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
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
# Train Model Button
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a while!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    # Reverse labels if VTuber=0, Human=1
    if confidence >= 0.5:
        st.success(f"Predicted: Human (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: VTuber (Confidence: {1-confidence:.3f})")
