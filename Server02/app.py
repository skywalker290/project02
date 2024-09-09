import os
from functions import *
from flask import Flask, request, jsonify

import cv2

app = Flask(__name__)

if not os.path.exists('Images'):
    os.makedirs('Images')


@app.route('/upload', methods=['POST'])
def upload_image():
    image_url = request.json.get('image_url')
    if image_url:
        image_path = download_image(image_url)
        if not image_path:
            return jsonify({"error": "Failed to download image"}), 400
        return jsonify({"output": "Image uploaded successfully"}), 200
    return jsonify({"error": "Image URL not provided"}), 400

@app.route('/scan',methods = ['POST'])
def inference():
    image_name = request.json.get('image_name')
    image_path = os.path.join('Images', image_name)
    
    if image_path:
        vehicle_number = recognize_vehicle_number(image_path)
        vehicle_type = detect_vehicle_type(image_path)
        if vehicle_number:
            return jsonify({"vehicle_number": vehicle_number,"vehicle_type":vehicle_type}), 200
        else:
            return jsonify({"error": "Vehicle number not found"}), 404
    else:
        return jsonify({"error": "Failed to find image"}), 400

if __name__ == '__main__':
    app.run(debug=True)
