from ultralytics import YOLO
import cv2
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# -------------------------------
# 1. Load YOLO model
# -------------------------------
yolo_model = YOLO("yolov8n.pt")

# -------------------------------
# 2. Load image
# -------------------------------
img_path = "test images/test_image_5.jpg"
image = cv2.imread(img_path)

# -------------------------------
# 3. Run YOLO detection
# -------------------------------
results = yolo_model(img_path)

# -------------------------------
# 4. Extract crops
# -------------------------------
crops = []
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    crops.append(crop)

# -------------------------------
# 5. Load ResNet50 model
# -------------------------------
resnet_model = ResNet50(weights='imagenet')

# -------------------------------
# 6. Classification function
# -------------------------------
def classify_food(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = resnet_model.predict(img)

    # Convert to human-readable labels
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]       # class name
    confidence = decoded[2]  # probability

    return label, confidence

# -------------------------------
# 7. Final pipeline
# -------------------------------
final_results = []

for crop in crops:
    label, confidence = classify_food(crop)
    final_results.append({
        "label": label,
        "confidence": float(confidence)
    })

# -------------------------------
# 8. Output
# -------------------------------
print(final_results)