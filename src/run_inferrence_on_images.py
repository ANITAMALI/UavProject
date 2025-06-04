from ultralytics import YOLO

def run_model_on_all_images():
    if not loaded_images:
        print("No images loaded.")
        return

    model = YOLO("weights/best.pt")  # Adjust path as needed

    for img_data in loaded_images:
        pil_image = img_data["original_pil"]
        result = model.predict(source=pil_image, save=False, conf=0.4, classes=[1])

        # Show or process results as needed
        boxes = result[0].boxes  # YOLOv8+ structure
        print(f"{os.path.basename(img_data['file_path'])} → {len(boxes)} detections")
