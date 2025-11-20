import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import os

# ------------------------------
# Load Trained Model
# ------------------------------
MODEL_PATH = "garbage_classification_model.h5"      # or .h5
model = load_model(MODEL_PATH)

# Class names (same order as train_data.class_indices)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("♻ Garbage Classification ML App")
st.write("Upload an image of waste material to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_img = preprocess_image(img)

    # Predict
    prediction = model.predict(input_img)[0]
    class_idx = np.argmax(prediction)
    class_name = CLASS_NAMES[class_idx]

    st.subheader(f"🟢 Predicted Class: **{class_name.upper()}**")

    # Show probabilities
    st.write("### Prediction Confidence:")
    for i, prob in enumerate(prediction):
        st.write(f"{CLASS_NAMES[i]}: **{prob*100:.2f}%**")
