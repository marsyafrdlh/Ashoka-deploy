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