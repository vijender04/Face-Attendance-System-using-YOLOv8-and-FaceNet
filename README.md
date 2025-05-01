# Face-Attendance-System-using-YOLOv8-and-FaceNet


A Python-based real-time face attendance system leveraging YOLOv8 for face detection and FaceNet for face recognition. This system detects faces through webcam feed and matches them against a preloaded dataset to mark attendance in a CSV file.

## 🔍 Features

- Real-time face detection using YOLOv8 (Ultralytics)
- Face recognition using `keras-facenet` (FaceNet)
- Attendance logging with timestamps
- Automatic CSV generation per session
- Works offline (once model and dataset are loaded)

---

## 📂 Dataset Structure

Your image dataset should be placed in a folder called `photos/` with filenames as the person's name. Example:


Each image should clearly show the face of the person for accurate embedding generation.

---

## 🧪 Requirements

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
opencv-python
numpy
keras-facenet
torch
ultralytics
scikit-learn
