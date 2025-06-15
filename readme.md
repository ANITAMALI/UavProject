# MaizeFieldAnalyzer

A Python application that detects maize sprouts in UAV images using a custom‑trained YOLOv10 model, stitches the shots into a single orthomosaic, and overlays a plant‑density heat‑map for quick agronomic insights.

---

## ✨ Features

- **Object detection** – YOLOv10 model fine‑tuned on aerial maize imagery.
- **Automatic stitching** – perspective‑correct panorama generation for flight lines.
- **Heat‑map overlay** – visualizes stand density and highlights gaps.
- **GUI** – built with **ttkbootstrap + TkinterDnD2**, drag‑and‑drop image loading, zoom, and batch processing.
- **Portable build** – optional PyInstaller one‑file EXE (CPU‑only) for Windows.

## 📸 Demo

> *(Add a short GIF / screenshot here)*

---

## 🚀 Quick start

```bash
# 1. Clone the repo
$ git clone https://github.com/ANITAMALI/UavProject.git
$ cd UavProject

# 2. Create & activate a virtual environment
$ python -m venv venv
$ source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the app
$ python application.py
```

### Inference only

If all you need is prediction on a folder of JPG/PNG images:

```bash
$ python run_inference_on_images.py --input /path/to/images --weights weights/last.pt --save
```

---

## 🛠️ Project structure

```text
📦UavProject
 ┣ 📂assets          # icons, sample images, model weights
 ┣ 📂gui              # ttkbootstrap GUI modules
 ┣ 📂stitching        # image‑stitch & panorama utilities
 ┣ 📂yolo_utils       # training / inference helpers
 ┣ 📜application.py   # main GUI entry point
 ┣ 📜requirements.txt # pinned runtime deps (generated via pip‑compile / pipreqs)
 ┣ 📜README.md        # you are here
 ┗ 📜LICENSE
```

---

## 🏋️ Training (optional)

1. Label new images in *Roboflow* or *Label Studio* using the single class **crop**.
2. Export YOLO‑format TXT + images ➜ `datasets/maize/`.
3. Update `data.yaml` with class list & paths.
4. Train:
   ```bash
   yolo train model=yolov10n.pt data=data.yaml epochs=100 imgsz=640 batch=16 device=0
   ```
5. Evaluate and log metrics (Precision, Recall, mAP\@50) as per your **Train/Val** table.

---

## 🧾 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv10
- [OpenCV](https://opencv.org/) for computer‑vision utilities
- Inspiration and field data courtesy of the University of XYZ Agronomy Lab

