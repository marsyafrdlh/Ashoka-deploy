import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_cnn_model():
    model = load_model('CNN_Model.h5', custom_objects={"F1Score": tf.keras.metrics.Metric})
    return model

model = load_cnn_model()

# Define class names
class_names = ["Normal", "Abnormal"]

# App title
st.title("Image Classification with CNN model")
st.write("Upload an image to classify whether it is Normal or Abnormal.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((256, 256))  # Resize to match model input
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[int(prediction[0] > 0.5)]  # Binary classification

    # Display prediction
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
