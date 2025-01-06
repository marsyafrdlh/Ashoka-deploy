import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np


# Streamlit app
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained model.")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = "CNN_SisiAtas.h5"  # Ganti dengan path model PyTorch Anda
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set model ke evaluasi mode
    return model

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Sesuaikan dengan preprocessing saat melatih model
    ])
    image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    return image

# Fungsi untuk prediksi kelas
def predict_class(image, model):
    classnames = ['normal', 'abnormal']  # Sesuaikan dengan label kelas Anda
    processed_image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(processed_image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = torch.softmax(outputs, dim=1)[0][class_idx].item() * 100
    return f"Class: {classnames[class_idx]}, Confidence: {confidence:.2f}%"

# Fungsi utama aplikasi Streamlit
def main():
    # Upload gambar
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Buka gambar menggunakan PIL
        image = Image.open(uploaded_file).convert("RGB")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        
        # Muat model
        st.write("Loading model...")
        model = load_model()

        # Prediksi kelas
        st.write("Classifying image...")
        result = predict_class(image, model)
        st.write(result)
        st.pyplot(figure)

if __name__ == '__main__':
    main()
