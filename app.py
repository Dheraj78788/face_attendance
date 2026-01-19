from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
import pickle
import csv
from datetime import datetime
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages

# -----------------------------
# Paths & Data
# -----------------------------
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.pkl"
USERS_CSV = "users.csv"
ATTENDANCE_CSV = "attendance.csv"
IMAGE_PATH = "images"

# Make images folder if not exists
os.makedirs(IMAGE_PATH, exist_ok=True)

# -----------------------------
# Load LBPH Face Recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(MODEL_PATH):
    recognizer.read(MODEL_PATH)

# Load label map
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)
else:
    label_map = {}

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Register Page
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_id = request.form["user_id"].strip()
        name = request.form["name"].strip()

        # Check duplicate
        if os.path.exists(USERS_CSV):
            with open(USERS_CSV, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == user_id:
                        flash(f"User {name} already registered!", "error")
                        return redirect(url_for("register"))

        # Create user folder
        user_folder = os.path.join(IMAGE_PATH, user_id)
        os.makedirs(user_folder, exist_ok=True)

        # Open webcam to capture face
        cap = cv2.VideoCapture(0)
        count = 0
        flash("Capturing face images. Press Q to quit.", "info")

        while count < 20:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                count += 1
                cv2.imwrite(f"{user_folder}/{count}.jpg", face)

            cv2.imshow("Register Face - Press Q to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save user data
        with open(USERS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([user_id, name])

        # Update LBPH model
        update_model()

        flash(f"User {name} registered successfully!", "success")
        return redirect(url_for("home"))

    return render_template("register.html")

# -----------------------------
# Update LBPH Model
# -----------------------------
def update_model():
    faces = []
    labels = []
    current_label = 0
    label_map.clear()

    for person in os.listdir(IMAGE_PATH):
        person_path = os.path.join(IMAGE_PATH, person)
        if not os.path.isdir(person_path):
            continue
        label_map[current_label] = person
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

# -----------------------------
# Attendance Page
# -----------------------------
@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

# -----------------------------
# Face Recognition API
# -----------------------------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as BGR

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    result = {"name": None, "confidence": None}

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)
        name = label_map.get(label, "Unknown")

        # LBPH: lower confidence = better match
        if confidence < 90:  # threshold
            mark_attendance(name)
            result["name"] = name
            result["confidence"] = confidence
        else:
            result["name"] = None
            result["confidence"] = confidence

        break  # only process first face

    return result

    data = request.get_json()
    img_data = data["image"]

    # Decode base64
    img_str = re.sub('^data:image/.+;base64,', '', img_data)
    img_bytes = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_bytes)).convert("L")  # GRAYSCALE!
    img = img.resize((200, 200))  # EXACT size
    img_np = np.array(img)

    try:
        label, confidence = recognizer.predict(img_np)
        name = label_map.get(label)
        print(f"Detected: {name}, Confidence: {confidence}")
        if confidence < 90:  # Relax threshold
            mark_attendance(name)
            return {"name": name}
        else:
            return {"name": None}
    except Exception as e:
        print("Recognition error:", e)
        return {"name": None}

# -----------------------------
# Mark Attendance
# -----------------------------
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])
    print(f"✅ Attendance marked: {name}")

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
