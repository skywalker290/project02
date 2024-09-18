from paddleocr import PaddleOCR
import re
from super_gradients.training import models  
import requests
import cv2
import os

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_name = os.path.join('Images', image_url.split("/")[-1])
        with open(image_name, 'wb') as f:
            f.write(response.content)
        return image_name
    return None

indian_number_plate_initials = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", 
    "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TR", 
    "TS", "UK", "UP", "WB", "AN", "CH", "DN", "DL", "LA", "PY"
]

def recognize_vehicle_number(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    img = cv2.imread(image_path)
    result = ocr.ocr(img)
    if line :
        for line in result:
            if word :
                for word in line:
                    detected_text = word[1][0]
                    confidence = word[1][1]
                    print(detected_text, confidence)
                    
                    if len(detected_text) >= 2 and detected_text[:2] in indian_number_plate_initials:
                        return detected_text
    return None

def detect_vehicle_type(image_path):
    model = models.get("yolo_nas_s", pretrained_weights="coco")
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

    image = cv2.imread(image_path)
    if image is None:
        return None

    results = model.predict(image)

    class_ids = results.prediction.labels  
    for class_id in class_ids:
        class_name = results.class_names[int(class_id)]
        if class_name in vehicle_classes:
            return class_name  
    return None  
