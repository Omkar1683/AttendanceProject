from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import face_recognition
import numpy as np
import cv2
from datetime import datetime
from bson import ObjectId
import os
import io
from pymongo import MongoClient
from urllib.parse import quote_plus

# Internal modules
from auth import hash_password, verify_password, generate_token, token_required, role_required
from models import get_models, setup_indexes
import analytics

app = Flask(__name__)
CORS(app)

# Database Connection Configuration
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

# Connect to MongoDB
def get_db_connection():
    """Get database connection"""
    try:
        client = MongoClient(MONGO_URI)
        db = client['attendai_db']
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        return db
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return None

db = get_db_connection()

# Initialize Models
models = get_models(db) if db is not None else None

# --- GLOBAL CACHE ---
# We load encodings into memory for speed
known_face_encodings = []
known_face_names = []
known_student_ids = []

def load_faces_from_db():
    global known_face_encodings, known_face_names, known_student_ids
    
    print("🔄 Loading faces from database...")
    if db is None or models is None:
        print("❌ No database connection!")
        return

    try:
        # Use StudentModel to get all encodings
        students = models['students'].get_all_encodings()
        
        known_face_encodings = []
        known_face_names = []
        known_student_ids = []

        count = 0
        for student in students:
            name = student.get('name', 'Unknown')
            encoding_list = student.get('encoding', [])
            student_id = str(student['_id'])
            
            if len(encoding_list) > 0:
                encoding = np.array(encoding_list)
                known_face_names.append(name)
                known_face_encodings.append(encoding)
                known_student_ids.append(student_id)
                count += 1
            
        print(f"✅ Loaded {count} student faces.")
    except Exception as e:
        print(f"❌ Error loading faces: {e}")

# Load faces on startup
load_faces_from_db()

# --- AUTHENTICATION ENDPOINTS ---

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'status': 'error', 'message': 'Missing credentials'}), 400
    
    email = data.get('email')
    password = data.get('password')
    
    # Use UserModel
    user = models['users'].find_by_email(email)
    
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    
    if verify_password(password, user['password']):
        token = generate_token(str(user['_id']), user['email'], user['role'])
        return jsonify({
            'status': 'success',
            'token': token,
            'user': {
                'name': user['name'],
                'email': user['email'],
                'role': user['role'],
                'id': str(user['_id'])
            }
        })
    
    return jsonify({'status': 'error', 'message': 'Invalid password'}), 401

@app.route('/signup', methods=['POST'])
def signup():
    """Register a new user (teacher or student)"""
    data = request.json
    
    # Validate required fields
    required_fields = ['email', 'password', 'name', 'role']
    if not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    
    # Check if user already exists
    existing_user = models['users'].find_by_email(data['email'])
    if existing_user:
        return jsonify({'status': 'error', 'message': 'Email already registered'}), 409
    
    # Create user using UserModel
    user_id = models['users'].create_user(
        email=data['email'],
        password_hash=hash_password(data['password']),
        role=data['role'],
        name=data['name'],
        department=data.get('department'),
        roll_no=data.get('roll_no') if data['role'] == 'student' else None
    )
    
    if user_id:
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user_id': user_id
        }), 201
    else:
        return jsonify({'status': 'error', 'message': 'Registration failed'}), 500

@app.route('/students/register', methods=['POST'])
@token_required
@role_required('teacher')
def register_student():
    """Register a student with face encoding"""
    data = request.json
    
    # Validate required fields
    required_fields = ['name', 'roll_no', 'encoding']
    if not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    
    # Validate encoding length
    if len(data['encoding']) != 128:
        return jsonify({'status': 'error', 'message': 'Face encoding must be 128 dimensions'}), 400
    
    # Check if student already exists
    existing_student = models['students'].find_by_roll_no(data['roll_no'])
    if existing_student:
        return jsonify({'status': 'error', 'message': 'Student with this roll number already exists'}), 409
    
    # Create student using StudentModel
    student_id = models['students'].create_student(
        name=data['name'],
        roll_no=data['roll_no'],
        encoding=data['encoding'],
        email=data.get('email'),
        phone=data.get('phone'),
        department=data.get('department'),
        batch=data.get('batch'),
        user_id=data.get('user_id')
    )
    
    if student_id:
        # Reload face encodings
        load_faces_from_db()
        
        return jsonify({
            'status': 'success',
            'message': 'Student registered successfully',
            'student_id': student_id
        }), 201
    else:
        return jsonify({'status': 'error', 'message': 'Registration failed'}), 500

# --- DATA FETCHING ENDPOINTS ---

@app.route('/classes', methods=['GET'])
@token_required
def get_classes():
    teacher_id = request.args.get('teacher_id')
    print(f"🔍 GET /classes called with teacher_id: {teacher_id}")
    
    # Use ClassModel
    if teacher_id:
        print(f"🔍 Querying classes for teacher: {teacher_id}")
        classes = models['classes'].find_by_teacher(teacher_id)
        print(f"🔍 Found {len(classes)} classes")
    else:
        print(f"🔍 No teacher_id provided, fetching all classes")
        classes = models['classes'].find_many({})
        print(f"🔍 Found {len(classes)} total classes")
    
    result = []
    for cls in classes:
        result.append({
            'id': str(cls['_id']),
            'name': cls['name'],
            'code': cls.get('code', ''),
            'batch': cls.get('batch', ''),
            'total_students': cls.get('total_students', 0)
        })
    
    print(f"🔍 Returning {len(result)} classes to frontend")
    return jsonify({'status': 'success', 'data': result})

# --- SESSION MANAGEMENT ---

@app.route('/sessions/create', methods=['POST'])
@token_required
@role_required('teacher')
def create_session():
    data = request.json
    class_id = data.get('class_id')
    location = data.get('location', 'Classroom')
    
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
    
    # Use SessionModel to create session
    session_id = models['sessions'].create_session(
        class_id=class_id,
        teacher_id=request.user['user_id'],
        location=location
    )
    
    return jsonify({
        'status': 'success',
        'session_id': session_id
    })

@app.route('/sessions/stop', methods=['POST'])
@token_required
@role_required('teacher')
def stop_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
    
    # Use SessionModel to end session
    success = models['sessions'].end_session(session_id)
    
    if success:
        return jsonify({'status': 'success', 'message': 'Session stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop session'}), 400

# --- ATTENDANCE SCANNING ---

@app.route('/scan', methods=['POST'])
# @token_required # Commented out for easier testing with raw requests if needed, but recommended
def scan_attendance():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No image sent"}), 400

    file = request.files['file']
    session_id = request.form.get('session_id')
    
    # Process image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    detected_people = []
    
    if len(face_encodings) > 0:
        # Check if we should skip matching (for student registration when DB is empty)
        skip_matching = len(known_face_encodings) == 0
        
        for idx, face_encoding in enumerate(face_encodings):
            name = "Unknown"
            status = "Absent"
            student_id = None
            
            # If database is not empty, try to match faces
            if not skip_matching:
                # Compare faces
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < 0.50:
                    name = known_face_names[best_match_index]
                    student_id = known_student_ids[best_match_index]
                    status = "Present"
                    
                    # Log attendance if session_id provided
                    if session_id and student_id:
                        # Use AttendanceLogModel to mark attendance
                        attendance_id = models['attendance_logs'].mark_attendance(
                            session_id=session_id,
                            student_id=student_id,
                            student_name=name,
                            status='Present',
                            marked_by='AI',
                            confidence=1.0 - face_distances[best_match_index]
                        )
                        
                        # Only increment if new attendance was marked
                        if attendance_id:
                            # Use SessionModel to increment scanned count
                            models['sessions'].increment_scanned(session_id)
            
            detected_people.append({
                "name": name,
                "status": status,
                "student_id": student_id,
                "encoding": face_encoding.tolist()  # Always include encoding
            })

        return jsonify({
            "status": "success",
            "people": detected_people,
            "count": len(detected_people)
        })
        
    else:
        return jsonify({"status": "error", "message": "No faces detected"})

# --- REPORTING & ANALYTICS ---

@app.route('/analytics/today', methods=['GET'])
@token_required
def get_today_analytics():
    class_id = request.args.get('class_id')
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
        
    summary = analytics.get_today_summary(db, class_id)
    return jsonify({'status': 'success', 'data': summary})

@app.route('/analytics/defaulters', methods=['GET'])
@token_required
def get_defaulters():
    class_id = request.args.get('class_id')
    threshold = int(request.args.get('threshold', 75))
    
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
        
    defaulters = analytics.get_defaulters_list(db, class_id, threshold)
    return jsonify({'status': 'success', 'data': defaulters})

@app.route('/reports/class', methods=['GET'])
@token_required
def get_class_report():
    class_id = request.args.get('class_id')
    month = int(request.args.get('month', datetime.now().month))
    year = int(request.args.get('year', datetime.now().year))
    
    if not class_id:
        return jsonify({'status': 'error', 'message': 'Class ID required'}), 400
        
    report = analytics.get_monthly_report(db, class_id, month, year)
    return jsonify({'status': 'success', 'data': report})

@app.route('/reports/student/<student_id>', methods=['GET'])
@token_required
def get_student_stats(student_id):
    # In a real app, user `request.user['user_id']` mapping to `student_id`
    # For now we assume student_id passed is the "face recognition db id" or we map it
    
    # We need to map the User ID (from auth) to the Student ID (from faces collection)
    # For simplicity in this demo, we'll assume they are linked or same, 
    # or we handle this mapping in the frontend/db
    
    report = analytics.get_student_report(db, student_id)
    return jsonify({'status': 'success', 'data': report})

@app.route('/reports/export-csv', methods=['GET'])
@token_required
def export_report_csv():
    class_id = request.args.get('class_id')
    month = int(request.args.get('month', datetime.now().month))
    year = int(request.args.get('year', datetime.now().year))
    
    report = analytics.get_monthly_report(db, class_id, month, year)
    csv_data = analytics.export_to_csv(report, "Class Report")
    
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_report_{month}_{year}.csv'
    )

@app.route('/attendance/manual', methods=['POST'])
@token_required
@role_required('teacher')
def manual_attendance():
    data = request.json
    student_id = data.get('student_id')
    session_id = data.get('session_id')
    status = data.get('status', 'Present') # Present or Absent
    
    # Use StudentModel and AttendanceLogModel
    if status == 'Present':
        # Get student name using StudentModel
        student = models['students'].find_by_id(student_id)
        student_name = student['name'] if student else 'Unknown'
        
        # Mark attendance using AttendanceLogModel
        models['attendance_logs'].mark_attendance(
            session_id=session_id,
            student_id=student_id,
            student_name=student_name,
            status='Present',
            marked_by='Manual'
        )
    else:
        # Remove attendance using AttendanceLogModel
        models['attendance_logs'].delete_attendance(session_id, student_id)
        
    return jsonify({'status': 'success', 'message': 'Updated'})

@app.route('/notifications/send', methods=['POST'])
@token_required
@role_required('teacher')
def send_notification():
    data = request.json
    target = data.get('target') # 'defaulters', 'all', 'critical'
    message = data.get('message')
    class_id = data.get('class_id')
    
    # In a real app, this would integrate with FCM or similar
    # Use NotificationModel to create notification
    notification_id = models['notifications'].create_notification(
        class_id=class_id,
        target=target,
        message=message,
        sent_by=request.user['user_id']
    )
    
    return jsonify({'status': 'success', 'message': 'Notifications queued'})


@app.route('/', methods=['GET'])
def home():
    return "AttendAI Complete Backend API is Running!"

if __name__ == '__main__':
    # Ensure DB is ready
    if db is None:
        print("⚠️ Warning: Running without Database Connection")
    
    app.run(host='0.0.0.0', port=5000, debug=True)