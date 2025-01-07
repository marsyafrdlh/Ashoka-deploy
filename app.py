import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Streamlit app title and description
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained PyTorch model.")

# Definisikan ulang arsitektur model (sesuaikan dengan model Anda)
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 256 * 256, 2)  # Output sesuai jumlah kelas (2: 'normal' dan 'abnormal')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = r"C:/Users/MARSHA\Documents/GitHub/Ashoka-deploy/CNN_Model_Complete.pth"
    st.write(f"Loading model from: {model_path}")  # Debugging path
    try:
        # Jika model disimpan dengan state_dict
        model = YourModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Jika model disimpan sebagai objek lengkap, gunakan:
        # model = torch.load(model_path, map_location=torch.device('cpu'))

        model.eval()
        return model
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize gambar ke ukuran 256x256
        transforms.ToTensor(),         # Convert gambar ke tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisasi
    ])
    image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    return image

# Fungsi untuk prediksi kelas
def predict_class(image, model):
    classnames = ['normal', 'abnormal']  # Sesuaikan label kelas
    processed_image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(processed_image)  # Prediksi
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = torch.softmax(outputs, dim=1)[0][class_idx].item() * 100
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
