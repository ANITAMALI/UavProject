import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def draw_yolo_boxes(image_path, label_path, class_names=None):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            color = (0, 255, 0) if int(cls) == 0 else (255, 0, 0)
            label = class_names[int(cls)] if class_names else str(int(cls))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 8)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(image_path.name)
    plt.axis('off')
    plt.show()

# 🔁 Iterate only over images that have corresponding .txt annotations
folder = Path(r"C:\Users\Boaz\Desktop\University Courses\D\Project\Viktor Images\61-106")

for img_path in folder.glob("*.JPG"):
    label_path = img_path.with_suffix('.txt')
    if label_path.exists():
        draw_yolo_boxes(img_path, label_path)
