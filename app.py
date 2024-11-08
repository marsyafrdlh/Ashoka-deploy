import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import h5py
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

def predict_class(image):
    tf.keras.models.load_model(r'/content/drive/Mydrive/Ashoka dataset')
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model)])    
    test_image = image.resize((128, 128))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    classnames = ['normal', 
                 'abnormal']    
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = classnames[np.argmax(scores)]
    return result

if __name__ == '__main__':
    main()               