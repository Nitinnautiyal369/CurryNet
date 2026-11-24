import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="CurryNet", page_icon="🥘")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_model.keras")

@st.cache_data
def load_classes():
    with open("classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_classes()

st.title("🥘CurryNet")
st.write("Upload a photo of a dish to see what the model thinks it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image to match training size
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_batch)
        scores = predictions[0]
        top_idx = np.argmax(scores)
        confidence = scores[top_idx] * 100

    st.success(f"Prediction: **{class_names[top_idx].title()}** ({confidence:.1f}% confidence)")

    # Show probability breakdown for all classes
    st.subheader("Class Probabilities")
    for name, score in zip(class_names, scores):
        st.progress(float(score), text=f"{name.title()}: {score * 100:.1f}%")
