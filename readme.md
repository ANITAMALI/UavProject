# MaizeFieldAnalyzer

A Python application that detects maize sprouts in UAV images using a custom‑trained YOLOv10 model, stitches the shots into a single orthomosaic, and outputs a plant‑density heat‑map for quick agronomic insights.

** Cannot install on Python version 3.13.3; only versions >=3.9,<3.13 are supported. **
---

## ✨ Features

- **Object detection** – YOLOv10 model fine‑tuned on aerial maize imagery.
- **Automatic stitching** – perspective‑correct panorama generation for flight lines.
- **Heat‑map** – visualizes stand density and highlights gaps.
- **GUI** – built with **ttkbootstrap**.

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

# 3. Install dependencies (located at the repo root)
$ pip install -r requirements.txt

# 4. Run the app (all source lives in the sub‑folder)
$ cd cornvisionApp
$ python main.py
```
---

## 🛠️ Project structure

```text
📦UavProject
 ┣ 📂sampleImages          # Samples for testing
 ┣ 📂cornvisionApp
 ┃ ┣ 📂assets              # icons, model weights
 ┃ ┣ components.py
 ┃ ┣ generate_heatmap.py
 ┃ ┣ gui.py
 ┃ ┣ gui_events.py
 ┃ ┣ main.py                # GUI entry point
 ┃ ┣ my_styles.py
 ┃ ┣ perform_analysis.py
 ┃ ┗ run_prediction.py
 ┣ 📜requirements.txt       # pinned runtime deps at repo root
 ┣ 📜README.md              # you are here
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

