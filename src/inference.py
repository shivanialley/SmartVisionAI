import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from src.preprocessing import preprocess_image
from src.confiq import CLASSES


clf_model = load_model('models/classification/efficientnet.h5')
yolo_model = YOLO('yolov8n.pt')


def classify(img):
    img = preprocess_image(img)
    pred = clf_model.predict(img.reshape(1,224,224,3))[0]
    idx = np.argmax(pred)
    return CLASSES[idx], float(pred[idx])


def detect(image_path):
    return yolo_model(image_path)   