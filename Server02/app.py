import os
import cv2
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import re

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

def recognize_vehicle_number(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    img = cv2.imread(image_path)
    result = ocr.ocr(img)
    pattern = r'^[A-Za-z]{2}\d{2}[A-Za-z]{2}\d{4}$'
    
    for line in result:
        for word in line:
            detected_text = word[1][0]
            confidence = word[1][1]
            if re.match(pattern, detected_text):
                return detected_text
    return None

@app.route('/upload', methods=['POST'])
def upload_image():
    image_url = request.json.get('image_url')
    if image_url:
        # Download the image
        image_path = download_image(image_url)
        if image_path:
            # Perform OCR and recognize the vehicle number
            vehicle_number = recognize_vehicle_number(image_path)
            if vehicle_number:
                return jsonify({"vehicle_number": vehicle_number}), 200
            else:
                return jsonify({"error": "Vehicle number not found"}), 404
        else:
            return jsonify({"error": "Failed to download image"}), 400
    return jsonify({"error": "Image URL not provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
