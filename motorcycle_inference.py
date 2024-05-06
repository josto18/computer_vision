# This file handles the motorcycle inference process

from ultralytics import YOLO
from PIL import Image
import uuid

# Load the model
model = YOLO('yolov8l')

# Crop the image to the bounding box of the detected motorcycle
def crop_image(image_path, box, save_path):
    image = Image.open(image_path)
    xyxy_list = box.xyxy[0].tolist()
    cropped_image = image.crop((xyxy_list[0], xyxy_list[1], xyxy_list[2], xyxy_list[3]))
    return cropped_image


# Inference with YoloV8
def inference(image_path):
    # Perform inference
    results = model.predict(image_path, save=True)

    # Iterate through all the detected bouding boxes
    for box in results[0].boxes:
        # Check if the detected object is a motorcycle - ID = 3
        if box.cls.item() == 3:

            xyxy_list = box.xyxy[0].tolist()
            height = xyxy_list[3] - xyxy_list[1]
            width = xyxy_list[2] - xyxy_list[0]

            # Check if the box is larger than 300x300 to weed out shit images
            if width > 300 and height > 300:
                # Crop the image to the bounding box of the detected motorcycle
                filename = str(uuid.uuid4()) + '.jpg'

                # # Crop the image to the bounding box of the detected motorcycle
                cropped_image_path = f'/Volumes/Stojcevski/Vision/processed/{filename}'

                cropped_image = crop_image(image_path, box, cropped_image_path)
                if cropped_image and filename:
                    return cropped_image, filename