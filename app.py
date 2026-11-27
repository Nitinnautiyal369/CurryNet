import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="CurryNet", page_icon="🥘")

st.title("🥘 CurryNet")
st.write("Upload a food image to identify the dish.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_model.keras")

model = load_model()

# Load class names
with open("classes.txt", "r") as file:
    class_names = [line.strip() for line in file]

# Upload image
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare image for prediction
    image = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display result
    st.success(f"Prediction: {predicted_class.title()}")
    st.metric("Confidence", f"{confidence:.2f}%")
