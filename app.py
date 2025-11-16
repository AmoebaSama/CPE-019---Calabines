import streamlit as st
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model, load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# -------------------
# Streamlit title
# -------------------
st.title("VTuber vs Human Classifier (MobileNetV2)")

# -------------------
# Paths
# -------------------
DATASET_DIR = "dataset"  # This folder must be inside your repo
MODEL_PATH = "vtuber_model.h5"  # The model will be saved/loaded here
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -------------------
# Data generators
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
    st.write("Training... This may take some time!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
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
    
    # Preprocess
    img_array = image.img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = float(pred)
    
    if confidence >= 0.5:
        st.success(f"Predicted: VTuber (Confidence: {confidence:.3f})")
    else:
        st.success(f"Predicted: Human (Confidence: {1-confidence:.3f})")
