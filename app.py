# app.py
import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -------------------
# Streamlit title
# -------------------
st.title("VTuber vs Human Classifier")

# -------------------
# Paths and settings
# -------------------
DATASET_DIR = "dataset"  # make sure this exists in repo
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
COLOR_MODE = "grayscale"  # easier, smaller model

# -------------------
# Data Generators
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    color_mode=COLOR_MODE,
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode=COLOR_MODE,
    shuffle=True
)

# -------------------
# Load or create model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
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
    st.info("No pre-trained model found. Ready to train a new model.")

# -------------------
# Train button
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
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    
    # Map prediction using class_indices
    class_indices = train_gen.class_indices
    if pred >= 0.5:
        label = [k for k,v in class_indices.items() if v==1][0]
        confidence = pred
    else:
        label = [k for k,v in class_indices.items() if v==0][0]
        confidence = 1 - pred
        
    st.success(f"Predicted: {label} (Confidence: {confidence:.3f})")
