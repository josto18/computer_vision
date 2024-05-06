## Computer vision - motorcycle logo detector

1. This project uses YoloV8 from Ultralytics to detect motorcycles in an image. To install `pip install ultralytics` and then you will be able to run the script
2. The entrypoint is `main.py` which firstly performs inference on an image to extract all the motorcycles. It uses the `motorcycle_inference.py` package.
3. Inside of the `motorcycle_inference.py` package is the YoloV8 detector which is only looking for the ID 3 on classified bounding boxes. This corresponds to the class of motorcycle
4. To run the program use command `env=test python3 main.py` - Make sure all the packages are installed and make sure your aws credentials are configured. To test you will also have to have some images already stored in a folder

---------------------------------------
4. TODO - This will be pulling dynamically from S3 which will need to be setup. A trigger will exist in cloud functions that tells SageMaker to start inference on the new image.
5. TODO - Connec the S3 bucket to CVAT so we can add annotations directly in the cloud and the images will never have to touch anyones machines
6. TODO - Annotations will save in the cloud and will be used to create the logo detector/classifier. There will be 6 classes
7. TODO - Save the trained model weights so we can use in an inference pipeline, the goal is to return the logo class that has been detected
8. TODO - With the returned 

---------------------------------------
### S3 Setup
- Contact jordy.stoj@gmail.com to configure your aws credentials on your machine. You will need to do this by installing the AWS CLI
- Find it here Visit the AWS CLI official page and download the appropriate installer - https://aws.amazon.com/cli/
- Add in your credentials into ~/.aws/config and ~/.aws/credentials
