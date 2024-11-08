import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

import ssl
from urllib.request import urlopen

# Definisikan kembali model yang sama dengan model yang telah dilatih
# Misalnya, kita menggunakan ResNet18 untuk klasifikasi
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Misalnya output 1 neuron untuk klasifikasi biner

# Muat state dictionary ke model
model.load_state_dict(torch.load("model_normal_abnormal.pth", map_location=torch.device("cpu")))

# Atur model ke mode evaluasi
model.eval()

# Define preprocessing sesuai dengan yang dilakukan saat melatih
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Ukuran input sesuai dengan model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app untuk klasifikasi gambar
st.title("Deteksi Kondisi Normal dan Abnormal")
st.write("Unggah gambar untuk mendeteksi apakah kondisinya normal atau abnormal.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Pindahkan input dan model ke GPU jika tersedia
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    # Prediksi
    with torch.no_grad():
        output = model(input_batch)

    # Menggunakan sigmoid untuk mengubah keluaran ke probabilitas (jika model binary)
    probability = torch.sigmoid(output).item()

    # Menampilkan hasil
    if probability > 0.5:
        st.write("Prediksi: **Abnormal** dengan probabilitas {:.2f}%".format(probability * 100))
    else:
        st.write("Prediksi: **Normal** dengan probabilitas {:.2f}%".format((1 - probability) * 100))

