# CurryNet

An end-to-end computer vision web app built with TensorFlow and Streamlit that identifies regional Indian dishes from uploaded photos.

**Live Demo:** https://currynet-l8hkag49adqvekj4qljzm6.streamlit.app/


---

### Why I Built This
Most introductory vision projects stop at standard datasets like CIFAR-10 or Cats vs. Dogs in a local notebook. I wanted to build an end-to-end pipeline on a custom dataset of Indian dishes—where visually similar colors and textures (like golden-brown gravies vs. roasted flatbreads) make classification trickier—and deploy it so anyone can test it directly in a browser.

---

### How It Works
- **Backbone:** MobileNetV2 (pre-trained on ImageNet). I chose MobileNetV2 over heavier architectures like VGG16 or ResNet50 so the saved model stays compact (~14 MB) and runs fast on CPU-only cloud servers.
- **Training Strategy (Two-Stage Fine-Tuning):**
  1. **Head Training:** Froze the MobileNetV2 base and trained a custom classification head (`GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense`) for 10 epochs at `lr=0.001`.
  2. **Fine-Tuning:** Unfroze the top 30 layers of MobileNetV2 and trained for 6 additional epochs at a lower learning rate (`lr=0.0001`) with horizontal flip, rotation, zoom, and contrast augmentations so the network adapts to specific food textures without overfitting.
- **Web Interface:** Built with Streamlit (`app.py`), caching the model in memory for quick predictions and displaying class-wise confidence scores.

---

### Project Structure
```text
├── app.py               # Streamlit web application
├── train2.py            # Two-stage model training and fine-tuning script
├── food_model.keras     # Trained TensorFlow/Keras model
├── classes.txt          # Class labels mapped alphabetically from training folders
├── requirements.txt     # Python dependencies for deployment
└── sample_images/       # Sample dish photos for quick testing
```

---

### Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd simple-indian-dish-classifier-by-nitinnautiyal369
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

*(Optional) To retrain the model from scratch, place your class folders inside a `dataset/` directory and run `python train2.py`.*

---

### Known Limitations & Future Improvements
- **Out-of-distribution inputs:** Since the final layer uses `softmax`, uploading a non-food photo will still force a prediction among the trained dishes. Adding a confidence threshold warning (e.g., flagging predictions below 65%) is next on my list.
- **Grad-CAM visualization:** Planning to add activation heatmaps in the UI to show which regions of the plate drove the model's prediction.
