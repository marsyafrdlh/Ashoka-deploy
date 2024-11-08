import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

import ssl
from urllib.request import urlopen

# Fungsi untuk memuat model klasifikasi
@st.cache_resource
def load_classification_model():
    # Misal menggunakan ResNet18 yang sudah dilatih pada ImageNet atau model kustom
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 output untuk klasifikasi biner
    model.load_state_dict(torch.load('model_normal_abnormal.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Muat model
model = load_classification_model()

# Define preprocessing sesuai dengan model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Aplikasi Streamlit
st.title("Image Classification")
st.write("Upload an image to classify as Normal or Abnormal.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocess gambar
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Prediksi
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        normal_prob, abnormal_prob = probabilities

    # Menampilkan hasil
    st.write(f"Normal Probability: {normal_prob.item():.2f}")
    st.write(f"Abnormal Probability: {abnormal_prob.item():.2f}")
    if abnormal_prob > normal_prob:
        st.write("Prediction: **Abnormal**")
    else:
        st.write("Prediction: **Normal**")