import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# -------------------
# Streamlit Title
# -------------------
st.title("VTuber vs Human Classifier (Grayscale, Small Model)")

# -------------------
# Dataset path
# -------------------
# Change this path if your dataset is elsewhere in Drive
DATASET_DIR = "/content/drive/MyDrive/EMTECH 2 Finals/dataset"
IMAGE_SIZE = (128, 128)  # smaller to reduce model size
BATCH_SIZE = 8           # smaller batch for lightweight model

# -------------------
# Data Generators (Grayscale)
# -------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',  # use grayscale to reduce complexity
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
# Build small MobileNetV2-based model
# -------------------
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(128,128,1))  # grayscale channel
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

st.info("Model ready. You can train or upload an image to predict.")

# -------------------
# Train Button
# -------------------
if st.button("Train Model"):
    st.write("Training... This may take a few minutes.")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3  # small number for testing
    )
    st.success("Training complete!")

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    # Reverse prediction if necessary
    if confidence >= 0.5:
        st.success(f"Predicted: Human (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: VTuber (Confidence: {1-confidence:.3f})")
