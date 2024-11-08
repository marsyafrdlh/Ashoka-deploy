import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

import ssl
from urllib.request import urlopen


# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
@st.cache
def load_labels():
    import requests
    import json
    response = requests.get(LABELS_URL)
    return json.loads(response.text)

labels = load_labels()

# Streamlit app
st.title("Image Classification with Streamlit")
st.write("Upload an image to classify using a pretrained model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Check if CUDA is available and move the input and model to GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Show results
    st.write("Top 5 Predictions:")
    for i in range(top5_prob.size(0)):
        st.write(f"{labels[top5_catid[i]]}: {top5_prob[i].item():.2f}")
    st.image(detected_img, caption="Detected Image", use_column_width=True)
