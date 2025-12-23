from ultralytics import YOLO


def load_yolo(weights_path):
    return YOLO(weights_path)