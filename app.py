import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Streamlit app
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained model.")

def main():
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open image using PIL
        image = Image.open(uploaded_file)

        # Display image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict and display result
        result = predict_class(image)
        st.write(f"Classification Result: {result}")

def predict_class(image):
    # Load pre-trained model from TensorFlow Hub
    model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"
    model = tf.keras.Sequential([hub.KerasLayer(model_url)])

    # Preprocess image
    test_image = image.resize((224, 224))  # Resize to match model input size
    test_image = np.array(test_image) / 255.0  # Normalize image
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(test_image)
    predicted_class = np.argmax(predictions[0])

    # Map class index to class name (example classes)
    classnames = ["Normal", "Abnormal"]  # Replace with actual class names
    return classnames[predicted_class] if predicted_class < len(classnames) else "Unknown"

if __name__ == '__main__':
    main()
