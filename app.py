import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load private model
@st.cache_resource
def load_private_model():
    return tf.keras.models.load_model('CNN_SisiAtas.h5')

# Preprocess image for the private model
def preprocess_private_dataset_image(image, target_size):
    img = image.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# Private model classification
def private_model_classification():
    st.title("Private Model Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        try:
            model = load_private_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        
        # Resize and preprocess image for model
        target_size = (224, 224)  # Update this if your model expects a different size
        img_array = preprocess_private_dataset_image(image, target_size)
        
        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # Define class names
        class_names = ['normal', 'abnormal']
        
        # Display prediction
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

# Main function for Streamlit app
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("Private Model",))
    
    if choice == "Private Model":
        private_model_classification()

if __name__ == "__main__":
    main()
