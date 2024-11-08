import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub


def main():
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open image using PIL
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        result = predict_class(img)
        st.write(result)


def predict_class(image):
    # Load your trained model
    model = tf.keras.models.load_model(r'/content/drive/MyDrive/Ashoka_dataset')
    
    # Preprocess the image
    test_image = image.resize((128, 128))  # Resize image to match model input
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize image
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Class names
    classnames = ['normal', 'abnormal']
    
    # Make prediction
    predictions = model.predict(test_image)
    
    # Convert predictions to softmax probabilities
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    # Get the predicted class
    image_class = classnames[np.argmax(scores)]
    
    return f'Prediction: {image_class}'


if __name__ == '__main__':
    main()
