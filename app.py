import streamlit as st
st.write("App started...")   # Debug line

import numpy as np
from tensorflow.keras.models import load_model

# Load model safely
try:
    model = load_model("cnn_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

from PIL import Image

class_names = ['labels', 'images']

st.title("Image Classification using CNN")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    img = Image.open(uploaded_file).resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: {predicted_class}")