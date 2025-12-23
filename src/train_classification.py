import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.classification_models import build_model

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "data/classification/train"
MODEL_DIR = "models/classification"
MODEL_PATH = f"{MODEL_DIR}/efficientnet.h5"

train_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

model = build_model(num_classes=train_data.num_classes)

model.fit(train_data, epochs=EPOCHS)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

print("✅ Classification model saved")