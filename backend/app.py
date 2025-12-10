from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow your mobile app to talk to this server

# --- DATABASE SIMULATION ---
known_face_encodings = []
known_face_names = []

def load_known_faces():
    print("Loading database...")
    try:
        # Load Rudransh (MAKE SURE rudransh.jpg EXISTS)
        img_rudransh = face_recognition.load_image_file("rudransh.jpg")
        enc_rudransh = face_recognition.face_encodings(img_rudransh)[0]
        known_face_encodings.append(enc_rudransh)
        known_face_names.append("Rudransh Gupta")
        print("Database loaded.")
    except Exception as e:
        print(f"Error loading face database: {e}")
        print("Make sure 'rudransh.jpg' is in the backend folder!")

# Load faces when app starts
load_known_faces()

@app.route('/', methods=['GET'])
def home():
    return "AttendAI Backend is Running!"

@app.route('/scan', methods=['POST'])
def scan_attendance():
    # 1. Check if image is in the request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No image sent"}), 400

    file = request.files['file']
    
    # 2. Convert uploaded image to something OpenCV can read
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 3. Convert BGR (OpenCV) to RGB (Face Recognition)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. Find faces in the uploaded photo
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    detected_name = "Unknown"
    status = "Absent"

    # 5. Compare found faces with known faces
    if len(face_encodings) > 0:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                detected_name = known_face_names[first_match_index]
                status = "Present"
                break
    else:
        return jsonify({"status": "error", "message": "No face detected in photo"})

    print(f"Result: {detected_name} is {status}")
    
    return jsonify({
        "status": "success",
        "name": detected_name,
        "attendance": status
    })

if __name__ == '__main__':
    # '0.0.0.0' allows access from other devices (like your phone) on the same network
    app.run(host='0.0.0.0', port=5000, debug=True)