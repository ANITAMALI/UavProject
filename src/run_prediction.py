from ultralytics import YOLO
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

model_path = "C:\\Users\Boaz\Downloads\\best (3).pt"

def bgr2rgb(image):
    """Convert OpenCV BGR image to RGB for correct display in Matplotlib."""
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def run_yolo_on_images(images:list, model_path="C:\\Users\Boaz\Downloads\\best (3).pt", conf=0.2, classes=[1], imgsz=640):
    """
    Run YOLO inference on a list of images and return result images with bounding boxes.
    Args:
        images: list of np.ndarray (BGR images)
        model_path: path to YOLO weights
        conf: confidence threshold
        classes: list of class indices to detect
        imgsz: inference image size
    Returns:
        List of images with bounding boxes drawn.
    """
    model = YOLO(model_path)
    all_boxes = []
    for img in images:
        results = model.predict(source=img, conf=conf, classes=classes, imgsz=imgsz)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        all_boxes.append(boxes)
    return all_boxes

