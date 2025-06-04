from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
def generate_circle_mask(image, boxes, radius=160):
    mask = np.zeros_like(image, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv.circle(mask, (cx, cy), radius, (0, 255, 0), -1)  # green circle
    cv.imwrite("C:\\Users\Boaz\Desktop\\University Courses\D\Project\Viktor Images\Stitch\\verbose\mask1.jpg", mask)
    return mask