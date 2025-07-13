# Player Tracking using YOLO + DeepSORT + TorchReID

This repository provides two pipelines for real-time **player detection and tracking** in sports footage using:
- **YOLOv11 for detection**
- **DeepSORT for object tracking**
- **TorchReID (OSNet) for appearance-based re-identification**

---

## 📂 Project Structure

```
.
├── detection_deepsort/
│   ├── yolo_detector.py
│   ├── tracker.py
│   └── yolo_detection_tracking.py
├── detection_torchreid/
│   ├── main.py
│   ├── deep_sort/
│   └── tracker/
├── requirements.txt
└── README.md
```

---

## ✅ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/player-tracking.git
cd player-tracking
```

---

### 2. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate         # On Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipelines

### ▶️ A. Detection + DeepSORT (Fast & Lightweight)

#### 🔗 Model Download
- **YOLOv11 model**: [Download YOLO model](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

#### 🧠 To Run:
```bash
cd detection_deepsort
python yolo_detection_tracking.py
```

---

### ▶️ B. Detection + DeepSORT + TorchReID (Appearance-Aware ReID)

#### 🔗 Models Required
- **YOLOv11 model**: [Download YOLO model](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
- **OSNet (TorchReID) model**: [Download OSNet model](https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view)

Ensure the `.pth` file is placed in a `models/` directory or update the path in code accordingly.

#### 🧠 To Run:
```bash
cd detection_torchreid
python main.py
```

---

## 🧠 Notes
- The YOLO model is fine-tuned on sports datasets to detect **players only**.
- DeepSORT assigns consistent IDs using motion and appearance embeddings.
- TorchReID further enhances identity consistency across occlusions and re-entries.

---

## 📎 References

- [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
- [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime)
- [TorchReID by Kaiyang Zhou](https://github.com/KaiyangZhou/deep-person-reid)

---

