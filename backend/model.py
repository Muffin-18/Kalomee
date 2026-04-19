from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("test images/test_image_1.jpg")

import cv2

image = cv2.imread("test images/test_image_1.jpg")

crops = []
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    crops.append(crop)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

model = ResNet50(weights='imagenet')
def classify_food(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds
final_results = []

for crop in crops:
    pred = classify_food(crop)
    label = decode_prediction(pred)  # your mapping
    final_results.append(label)

print(final_results)