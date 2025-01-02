import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image

# Streamlit app
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained model.")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    # Muat model dari file .h5
    model_path = r'/content/drive/Mydrive/Ashoka dataset/model.h5'  # Ganti dengan path model .h5 Anda
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    test_image = image.resize((128, 128))  # Sesuaikan dengan ukuran input model
    test_image = keras_image.img_to_array(test_image)  # Mengubah gambar menjadi array
    test_image = test_image / 255.0  # Normalisasi pixel gambar (0-1)
    test_image = np.expand_dims(test_image, axis=0)  # Tambahkan dimensi batch
    return test_image

# Fungsi untuk prediksi kelas
def predict_class(image, model):
    classnames = ['normal', 'abnormal']  # Sesuaikan dengan label kelas Anda
    processed_image = preprocess_image(image)
    
    # Prediksi menggunakan model
    predictions = model.predict(processed_image)
    
    # Softmax untuk mendapatkan probabilitas kelas
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()  # Convert ke numpy array untuk manipulasi lebih lanjut
    
    # Menentukan kelas dengan probabilitas tertinggi
    image_class = classnames[np.argmax(scores)]
    confidence = scores[np.argmax(scores)] * 100  # Confidence dalam persen
    
    return f"Class: {image_class}, Confidence: {confidence:.2f}%"

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
