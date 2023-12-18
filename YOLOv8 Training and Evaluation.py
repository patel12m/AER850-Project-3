!pip install ultralytics

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='/content/drive/MyDrive/Project 3 Data/data/data.yaml', epochs=100, batch=5, imgsz=1216, name='trained_yolov8_model')

# Paths to the images
image_paths = [
    '/content/drive/MyDrive/Project 3 Data/data/evaluation/ardmega.jpg',
    '/content/drive/MyDrive/Project 3 Data/data/evaluation/arduno.jpg',
    '/content/drive/MyDrive/Project 3 Data/data/evaluation/rasppi.jpg'
]

for image_path in image_paths:
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Perform prediction using the YOLO model
    results = model.predict(image)
    annotated_image = results[0].plot()

    # Display the annotated image
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Annotated Image")
    plt.axis('off')
    plt.show()
