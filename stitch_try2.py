from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


model_path = "C:\\Users\Boaz\Downloads\\best (3).pt"

def bgr2rgb(image):
    """Convert OpenCV BGR image to RGB for correct display in Matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_yolo_on_images(images, model_path="C:\\Users\Boaz\Downloads\\best (3).pt", conf=0.2, classes=[1], imgsz=640):
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
    results_imgs = []
    for img in images:
        results = model.predict(source=img, conf=conf, classes=classes, imgsz=imgsz)
        result = results[0]
        # Draw bounding boxes on the image
        img_with_boxes = result.plot(labels=False)
        results_imgs.append(img_with_boxes)

    for idx, img in enumerate(results_imgs):
        plt.figure()
        plt.imshow(bgr2rgb(img))
        plt.axis("off")
        plt.title(f"Result {idx + 1}")
        plt.show()
    return results_imgs

