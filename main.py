import motorcycle_inference
import os
import boto3
from PIL import Image
import io

# Data path
raw_data_path = '/Volumes/Stojcevski/Vision/raw'
output_path = '/Volumes/Stojcevski/Vision/processed'

# Connect to S3 
s3 = boto3.client('s3')
bucket_name = "cv-logo-detection"
save_folder = "processed/"
raw_folder = "raw/"

# Iterate through all the images in the raw data folder
i = 0
for image in os.listdir(raw_data_path):
    if (image.endswith('.jpg') or image.endswith('.png')) and image != '.DS_Store' and image[0] != ".":
        # Only perform on the first 20 images in test env
        if os.environ.get('env') == 'test' and i == 20:
            break
        image_path = os.path.join(raw_data_path, image)
        cropped_image, filename = motorcycle_inference.inference(image_path)

        if cropped_image and filename:
            # Save the image to the buffer
            buffer = io.BytesIO()
            cropped_image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            try:
                # Insert into the S3 bucket
                s3.put_object(Body=buffer, Bucket=bucket_name, Key=save_folder+filename)

                # Close the buffer to free memory
                buffer.close()

                print(f"Saved cropped image to S3: {filename}")
            except Exception as e:
                # Close the buffer to free memory
                buffer.close()

                print(f"Error saving cropped image to S3: {e}")

        i += 1

# Use transfer learning to train the model on the new data
# Load in annotations from S3
# Train the model on the new data
# Save the model to S3