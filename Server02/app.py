import os
import cv2
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import re
from super_gradients.training import models  # Import the YOLO-NAS model
import cv2

app = Flask(__name__)

# Ensure the "Images" folder exists
if not os.path.exists('Images'):
    os.makedirs('Images')

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        # Create a unique filename from the URL
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
    
    for line in result:
        for word in line:
            detected_text = word[1][0]
            confidence = word[1][1]
            print(detected_text, confidence)
            
            if len(detected_text) >= 2 and detected_text[:2] in indian_number_plate_initials:
                return detected_text
    return None

def detect_vehicle_type(image_path):
    # Load YOLO-NAS model
    model = models.get("yolo_nas_s", pretrained_weights="coco")
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Run YOLO-NAS inference
    results = model.predict(image)

    # `results.prediction` contains bounding boxes, confidences, and class IDs as NumPy arrays
    class_ids = results.prediction.labels  # Class IDs are stored here

    # Iterate through detected objects
    for class_id in class_ids:
        class_name = results.class_names[int(class_id)]
        if class_name in vehicle_classes:
            return class_name  # Return the first detected vehicle type

    return None  # Return None if no vehicle is detected


@app.route('/upload', methods=['POST'])
def upload_image():
    image_url = request.json.get('image_url')
    if image_url:
        # Download the image
        image_path = download_image(image_url)
        if image_path:
            # Perform OCR and recognize the vehicle number
            vehicle_number = recognize_vehicle_number(image_path)
            vehicle_type = detect_vehicle_type(image_path)
            if vehicle_number:
                return jsonify({"vehicle_number": vehicle_number,"vehicle_type":vehicle_type}), 200
            else:
                return jsonify({"error": "Vehicle number not found"}), 404
        else:
            return jsonify({"error": "Failed to download image"}), 400
    return jsonify({"error": "Image URL not provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
