import streamlit as st
from PIL import Image
import torch

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

st.title("Object Detection App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    results = model(image)

    # Convert to PIL image with detections rendered
    detected_image = Image.fromarray(results.render()[0])  # Render adds bounding boxes to the image
    st.image(detected_image, caption="Detected Image", use_column_width=True)


