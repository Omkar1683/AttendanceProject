"""
Example Usage of AttendAI MongoDB Models

This file demonstrates how to use the models.py module for common operations.
"""

from models import get_models, setup_indexes
from auth import hash_password
from datetime import datetime
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus

# Database Connection Configuration
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

def get_db_connection():
    """Get database connection"""
    try:
        client = MongoClient(MONGO_URI)
        db = client['attendai_db']
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        return db
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return None


def example_user_operations(models):
    """Example user operations"""
    print("\n=== USER OPERATIONS ===\n")
    
    # Create a teacher
    teacher_id = models['users'].create_user(
        email="teacher@ves.ac.in",
        password_hash=hash_password("password123"),
        role="teacher",
        name="Prof. Kumar",
        department="MCA"
    )
    print(f"✅ Created teacher: {teacher_id}")
    
    # Create a student user
    student_id = models['users'].create_user(
        email="student@ves.ac.in",
        password_hash=hash_password("student123"),
        role="student",
        name="Raj Sharma",
        department="MCA",
        roll_no="MCA001"
    )
    print(f"✅ Created student: {student_id}")
    
    # Find user by email
    user = models['users'].find_by_email("teacher@ves.ac.in")
    print(f"✅ Found user: {user['name']}")
    
    # Find all teachers
    teachers = models['users'].find_by_role("teacher")
    print(f"✅ Found {len(teachers)} teachers")
    
    return teacher_id, student_id


def example_student_operations(models, user_id):
    """Example student operations"""
    print("\n=== STUDENT OPERATIONS ===\n")
    
    # Create face encoding (mock 128-dim vector)
    face_encoding = np.random.rand(128).tolist()
    
    # Register student with face
    student_id = models['students'].create_student(
        name="Raj Sharma",
        roll_no="MCA001",
        encoding=face_encoding,
        email="student@ves.ac.in",
        phone="+919876543210",
        department="MCA",
        batch="MCA 2A",
        user_id=user_id
    )
    print(f"✅ Registered student with face: {student_id}")
    
    # Find by roll number
    student = models['students'].find_by_roll_no("MCA001")
    print(f"✅ Found student: {student['name']}")
    
    # Get all encodings for face recognition
    encodings = models['students'].get_all_encodings()
    print(f"✅ Loaded {len(encodings)} face encodings")
    
    # Update face encoding
    new_encoding = np.random.rand(128).tolist()
    models['students'].update_encoding(student_id, new_encoding)
    print(f"✅ Updated face encoding")
    
    return student_id


def example_class_operations(models, teacher_id):
    """Example class operations"""
    print("\n=== CLASS OPERATIONS ===\n")
    
    # Create a class
    class_id = models['classes'].create_class(
        name="Artificial Intelligence",
        code="AI-501",
        teacher_id=teacher_id,
        total_students=65,
        batch="MCA 2A",
        department="MCA",
        schedule="Mon 10-12, Thu 2-4"
    )
    print(f"✅ Created class: {class_id}")
    
    # Find by code
    cls = models['classes'].find_by_code("AI-501")
    print(f"✅ Found class: {cls['name']}")
    
    # Find by teacher
    classes = models['classes'].find_by_teacher(teacher_id)
    print(f"✅ Teacher has {len(classes)} classes")
    
    # Update enrollment
    models['classes'].update_enrollment(class_id, 70)
    print(f"✅ Updated enrollment to 70")
    
    return class_id


def example_session_operations(models, class_id, teacher_id):
    """Example session operations"""
    print("\n=== SESSION OPERATIONS ===\n")
    
    # Create session
    session_id = models['sessions'].create_session(
        class_id=class_id,
        teacher_id=teacher_id,
        location="Room 504"
    )
    print(f"✅ Created session: {session_id}")
    
    # Find active session
    active = models['sessions'].find_active_session(class_id)
    print(f"✅ Active session: {active['_id']}")
    
    # Increment scanned count
    models['sessions'].increment_scanned(session_id)
    print(f"✅ Incremented scanned count")
    
    # End session
    models['sessions'].end_session(session_id)
    print(f"✅ Ended session")
    
    # Find completed sessions
    completed = models['sessions'].find_by_class(class_id, status="completed")
    print(f"✅ Found {len(completed)} completed sessions")
    
    return session_id


def example_attendance_operations(models, session_id, student_id):
    """Example attendance operations"""
    print("\n=== ATTENDANCE OPERATIONS ===\n")
    
    # Mark attendance (AI)
    log_id = models['attendance_logs'].mark_attendance(
        session_id=session_id,
        student_id=student_id,
        student_name="Raj Sharma",
        status="Present",
        marked_by="AI",
        confidence=0.95
    )
    print(f"✅ Marked attendance (AI): {log_id}")
    
    # Get session attendance
    logs = models['attendance_logs'].find_by_session(session_id)
    print(f"✅ Found {len(logs)} attendance records")
    
    # Get student attendance history
    history = models['attendance_logs'].find_by_student(student_id)
    print(f"✅ Student has {len(history)} attendance records")
    
    # Count present
    present_count = models['attendance_logs'].count_present_in_session(session_id)
    print(f"✅ Present in session: {present_count}")
    
    # Manual attendance update
    models['attendance_logs'].mark_attendance(
        session_id=session_id,
        student_id=student_id,
        student_name="Raj Sharma",
        status="Absent",
        marked_by="Manual"
    )
    print(f"✅ Updated to Absent (Manual)")


def example_notification_operations(models, class_id, teacher_id):
    """Example notification operations"""
    print("\n=== NOTIFICATION OPERATIONS ===\n")
    
    # Create notification
    notif_id = models['notifications'].create_notification(
        class_id=class_id,
        target="defaulters",
        message="Your attendance is below 75%",
        sent_by=teacher_id
    )
    print(f"✅ Created notification: {notif_id}")
    
    # Get class notifications
    notifs = models['notifications'].find_by_class(class_id)
    print(f"✅ Found {len(notifs)} notifications")
    
    # Mark as read
    models['notifications'].mark_as_read(notif_id)
    print(f"✅ Marked notification as read")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("AttendAI Models - Usage Examples")
    print("=" * 60)
    
    # Connect to database
    db = get_db_connection()
    if db is None:
        print("❌ Failed to connect to database")
        return
    
    # Get model instances
    models = get_models(db)
    print("\n✅ Connected to database and loaded models")
    
    # Run examples
    teacher_id, student_user_id = example_user_operations(models)
    student_id = example_student_operations(models, student_user_id)
    class_id = example_class_operations(models, teacher_id)
    session_id = example_session_operations(models, class_id, teacher_id)
    example_attendance_operations(models, session_id, student_id)
    example_notification_operations(models, class_id, teacher_id)
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print("\nNote: These are example operations. In production,")
    print("clean up the test data or use a separate test database.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
