from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow your mobile app to talk to this server

# --- CONFIGURATION ---
# Lower this number to make it STRICTER. 
# 0.6 is default. 0.50 is good. 0.45 is very strict.
TOLERANCE = 0.50 

# --- DATABASE SIMULATION ---
known_face_encodings = []
known_face_names = []

def load_known_faces():
    print("Loading database...")
    
    # List of students to load
    students_to_load = [
        ("rudransh.jpg", "Rudransh Gupta"),
        ("omkar.jpg", "Omkar Jadhav"),
        ("pushkar.jpg", "Pushkar Jaju"),
        ("Devesh.jpg", "Devesh Mahajan")
    ]

    for filename, name in students_to_load:
        try:
            # Load image file
            img = face_recognition.load_image_file(filename)
            # Encode face
            encodings = face_recognition.face_encodings(img)
            
            if len(encodings) > 0:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"✅ Loaded: {name}")
            else:
                print(f"⚠️ Warning: No face found in '{filename}'. Skipping.")
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find file '{filename}'. Skipping {name}.")
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")

    print(f"Database loaded with {len(known_face_names)} students.")

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
    debug_distance = 1.0 # Default high distance

    # 5. Compare found faces with known faces
    if len(face_encodings) > 0:
        # We only check the first face found in the image
        face_encoding = face_encodings[0]
        
        # Calculate the distance to every known face
        # Lower distance = More similar (0.0 is exact match)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the index of the face with the smallest distance
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        debug_distance = best_match_distance

        # STRICT CHECK: Only accept if distance is less than our TOLERANCE (0.50)
        if best_match_distance < TOLERANCE:
            detected_name = known_face_names[best_match_index]
            status = "Present"
        else:
            detected_name = "Unknown"
            status = "Absent"
            
    else:
        return jsonify({"status": "error", "message": "No face detected in photo"})

    print(f"Scan Result: {detected_name} | Distance Score: {debug_distance:.4f} (Threshold: {TOLERANCE})")
    
    return jsonify({
        "status": "success",
        "name": detected_name,
        "attendance": status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)