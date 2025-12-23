import cv2
import numpy as np


def preprocess_image(img, size=224):
    img = cv2.resize(img, (size, size))
    img = img.astype('float32') / 255.0
    return img