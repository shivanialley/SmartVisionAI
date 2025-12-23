import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.inference import classify


st.header('Image Classification')
file = st.file_uploader('Upload Image', type=['jpg','png'])
if file:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300)
    label, conf = classify(np.array(img))
    st.success(f'Prediction: {label} ({conf:.2f})')