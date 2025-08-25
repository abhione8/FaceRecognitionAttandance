
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Path to known face images
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance.csv'

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"[WARNING] No face found in {filename}")

# Setup attendance file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'r+') as f:
        entries = f.readlines()
        recorded_names = [line.split(',')[0] for line in entries]
        if name not in recorded_names:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{now}\n")
            print(f"[INFO] Attendance marked for {name} at {now}")

# Start webcam
video_capture = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

        # Scale back to original size
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)

        # Draw landmarks
        for feature_points in landmarks.values():
            for point in feature_points:
                x, y = point[0] * 4, point[1] * 4
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Face Recognition + Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
