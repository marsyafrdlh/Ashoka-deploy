import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_hub as hub
import torch
import numpy as np
import cv2
import os
import ssl
from urllib.request import urlopen


# Streamlit app
st.title("Image Classification with Streamlit")
st.write("Upload an image to classify using a pretrained model.")

def main():
# Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open image using PIL
        image = Image.open(uploaded_file)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

# Object Detection function
def detect_objects(image, model):
    # Convert image to numpy array
    img_array = np.array(image)
    # Convert RGB to BGR format (OpenCV standard)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = model(img_array)
    # Get detection results
    results_img = np.squeeze(results.render())  # Render the detected results on the image
    
    return results_img

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

