import streamlit as st
import os
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

# -------------------
# Streamlit App Title
# -------------------
st.title("VTuber vs Human Classifier")

# -------------------
# Dataset and Model Paths
# -------------------
DATASET_DIR = "dataset"  # folder in repo
MODEL_PATH = "vtuber_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# -------------------
# Data Generators
# -------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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
# Load or Build Model
# -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.info("Loaded existing trained model.")
else:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # freeze for small size

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
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
        epochs=5  # keep small to limit file size
    )
    model.save(MODEL_PATH)
    st.success("Training complete and model saved!")

# -------------------
# Upload and Predict
# -------------------
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")  # ensures 3 channels
    img = img.resize(IMAGE_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array, verbose=0)[0][0]
    confidence = float(pred)

    # Map prediction according to class indices
    if 'vtuber' in train_gen.class_indices and train_gen.class_indices['vtuber'] == 1:
        if confidence >= 0.5:
            st.success(f"Predicted: VTuber (Confidence: {confidence:.3f})")
        else:
            st.success(f"Predicted: Human (Confidence: {1-confidence:.3f})")
    else:
        if confidence >= 0.5:
            st.success(f"Predicted: Human (Confidence: {confidence:.3f})")
        else:
            st.success(f"Predicted: VTuber (Confidence: {1-confidence:.3f})")
