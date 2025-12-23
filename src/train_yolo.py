from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

model.export(format="pt")