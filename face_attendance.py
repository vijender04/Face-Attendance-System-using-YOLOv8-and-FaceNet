from keras_facenet import FaceNet
import cv2  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from ultralytics import YOLO
from datetime import datetime
import csv


embedder = FaceNet()


users_face = os.listdir('photos')
all_users_face = []
all_users_face_name = []
for user in users_face:
    person_image = cv2.imread(f'photos/{user}')
    person_rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    person_resize_image = cv2.resize(person_rgb_image, (160, 160)).astype(np.float32)
    all_users_face.append(person_resize_image)
    all_users_face_name.append(user.split('.')[0])
all_users_face = np.array(all_users_face)


face_embeddings = embedder.embeddings(all_users_face)


cap = cv2.VideoCapture(0)
model = YOLO("yolov8n-face[1].pt", task="detect")


session_attendance = {}
today = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{today}.csv"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame, device='cpu') 
    all_faces = []
    face_cordinates = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                croped_frame = frame[y1:y2, x1:x2]
                croped_frame = cv2.cvtColor(croped_frame, cv2.COLOR_BGR2RGB)
                croped_frame = cv2.resize(croped_frame, (160, 160)).astype(np.float32)
                all_faces.append(croped_frame)
                face_cordinates.append((x1, y1))

    if len(all_faces) > 0:
        cctv_face_embeddings = embedder.embeddings(np.array(all_faces))
        similarities = cosine_similarity(face_embeddings, cctv_face_embeddings)
        similarities_labels = similarities.argmax(axis=1)

        for i, label in enumerate(similarities_labels):
            if similarities[i, label] > 0.65:
                name = all_users_face_name[i]
                current_time = datetime.now().strftime("%H:%M:%S")

                
                if name not in session_attendance:
                    session_attendance[name] = {"in": current_time, }
                

                cv2.putText(frame, name, face_cordinates[label], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time"])
    for name, times in session_attendance.items():
        writer.writerow([name, times["in"] ])
