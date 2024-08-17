from super_gradients.training import models
import cv2

import logging
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Set logging level to WARNING or ERROR to suppress INFO logs
logging.getLogger().setLevel(logging.WARNING)

os.environ['CRASH_HANDLER'] = 'FALSE'

# Load the YOLO-NAS model
model = models.get("yolo_nas_s", pretrained_weights="coco")

# Get all the class names that the model can detect
class_names = model._class_names

# Write class names to a file
with open('yolo-nas-classes.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Classes of interest for vehicle detection
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

import os

# Path to the folder
folder_path = 'test_images'

# Get all file paths in the folder
file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]

# Print the file paths
# for file_path in file_paths:
#     print(file_path)



# # Load input image
# image_path = 'bike2.png'  # Path to your image

for image_path in file_paths:
    image = cv2.imread(f"{image_path}")
    if image is None:
        print("Failed to load image. Check the file path.")
        exit()

    # Run YOLO-NAS inference
    results = model.predict(image)

    # Get class names
    class_names = results.class_names  # These will be COCO class names

    # Filter predictions by vehicle classes and print probabilities
    detected_vehicles = []

    # `results.prediction` contains bounding boxes, confidences, and class IDs as NumPy arrays
    bboxes = results.prediction.bboxes_xyxy
    confidences = results.prediction.confidence
    class_ids = results.prediction.labels  # Class IDs are stored here

    # Iterate through all detected objects
    for i in range(len(bboxes)):
        class_id = int(class_ids[i])
        conf = confidences[i]
        class_name = class_names[class_id]
        print(class_name)

        if class_name in vehicle_classes:
            detected_vehicles.append((class_name, conf))

    # Print detected vehicle categories with probabilities
    # Print detected vehicle categories with probabilities
    if detected_vehicles:
        # Print only the first detected vehicle
        vehicle, confidence = detected_vehicles[0]
        print(f"Detected vehicle: {vehicle} with confidence:{image_path} {confidence:.2f}")
    else:
        print("No vehicles detected.")
