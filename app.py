import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

import ssl
from urllib.request import urlopen

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

if uploaded_file is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Processing...")
    
    # Load model
    model = load_model()
    
    # Perform object detection
    detected_img = detect_objects(image, model)
    
    # Convert BGR to RGB for displaying with Streamlit
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    
    # Display detected image
    st.image(detected_img, caption="Detected Image", use_column_width=True)

