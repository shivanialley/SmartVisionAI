import streamlit as st
from PIL import Image
import numpy as np
from src.inference import detect

st.header("Object Detection")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    results = detect(np.array(img))
    st.image(results[0].plot(), caption="Detections")