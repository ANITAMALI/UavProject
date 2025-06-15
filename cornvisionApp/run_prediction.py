from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os

### Define paths to the model weights files ###
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "assets", "crop_detector.pt")
model2_path = os.path.join(current_dir, "assets", "row_detector.pt")


def bgr2rgb(image):
    """Convert OpenCV BGR image to RGB for correct display in Matplotlib."""
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def update_progress_bar(self, value):
    self.progress_bar.progress['value'] = int(float(self.progress_bar.progress['value']) + value)
    self.root.update_idletasks()  # Process events to update UI

def find_crops(images:list, self, path=model_path, conf=0.2, classes=1, imgsz=640):
    """
    Run YOLO inference on a list of images and return result images with bounding boxes.
    Args:
        images: list of np.ndarray (BGR images)
        path: path to YOLO weights
        conf: confidence threshold
        classes: list of class indices to detect
        imgsz: inference image size
    Returns:
        List of images with bounding boxes drawn.
    """
    model = YOLO(path)
    annotated_images = []
    all_boxes = []
    for img in images:
        results = model.predict(source=img, conf=conf, classes=classes, imgsz=imgsz)
        update_progress_bar(self, int(25/len(images)))
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        all_boxes.append(boxes)
        annotated_images.append(result.plot())  # add annotated image
    del results, model  # Free memory
    return all_boxes, annotated_images

def count_rows(images, path=None, conf=0.2, classes=0, imgsz=640):
    model2 = YOLO(model2_path)
    annotated_images = []
    all_boxes = []
    for img in images:
        results = model2.predict(source=img, conf=conf, classes=classes, imgsz=imgsz)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        all_boxes.append(boxes)
        annotated_images.append(result.plot())  # add annotated image
    del results, model2
    return all_boxes, annotated_images
