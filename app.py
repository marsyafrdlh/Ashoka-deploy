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
    model_path = r"C:\Users\MARSHA\Documents\GitHub\Ashoka-deploy\CNN_Model_Complete.pth"
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


# Jalankan aplikasi
if __name__ == '__main__':
    main()
