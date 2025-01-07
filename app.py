import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Streamlit app title and description
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained PyTorch model.")

@st.cache_resource
def load_model():
    model_path = r"CNN_Model_Complete.pth"  # Adjust the file extension if needed
    st.write(f"Loading Keras model from: {model_path}")
    try:
        model = load_model(model_path)  # Load the Keras model
        return model
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# Fungsi untuk preprocessing gambar

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = img_to_array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_class(image, model):
    classnames = ['normal', 'abnormal']  # Adjust labels as needed
    processed_image = preprocess_image(image)
    processed_image = processed_image.numpy()  # Convert PyTorch tensor to NumPy array if needed
    processed_image = np.transpose(processed_image, (0, 2, 3, 1))  # Rearrange dimensions for Keras
    outputs = model.predict(processed_image)  # Predict with Keras
    class_idx = np.argmax(outputs, axis=1)[0]
    confidence = outputs[0][class_idx] * 100
    return classnames[class_idx], confidence


# Fungsi utama aplikasi Streamlit
def main():
    # Upload gambar
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Buka gambar menggunakan PIL
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image: {e}")
            return

        # Tampilkan gambar yang diunggah
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Muat model
        st.write("Loading model...")
        model = load_model()
        if model is None:
            return

        # Prediksi kelas
        st.write("Classifying image...")
        try:
            result, confidence = predict_class(image, model)
            st.write(f"Prediction: {result}, Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error during classification: {e}")

# Jalankan aplikasi
if __name__ == '__main__':
    main()
