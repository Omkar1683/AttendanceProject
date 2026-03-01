"""
Database utilities for initialization, seeding, and management
"""
from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
from bson import ObjectId
import sys
sys.path.append('.')
from auth import hash_password

# Configuration
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"


username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

def get_db_connection():
    """Get database connection"""
    try:
        client = MongoClient(MONGO_URI)
        print(f"DEBUG: Client type: {type(client)}")
        db = client['attendai_db']
        print(f"DEBUG: DB type: {type(db)}")
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        return db
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return None

def initialize_database():
    """Initialize database with required collections and indexes"""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        # Create collections if they don't exist (implicit creation is sufficient)
        # existing_collections = db.list_collection_names()
        # required_collections = ['users', 'students', 'classes', 'sessions', 'attendance_logs', 'notifications']
        # for collection in required_collections: ...
        
        # Create indexes
        print("\n📊 Creating indexes...")
        
        # Users collection indexes
        db.users.create_index("email", unique=True)
        print("  - Created unique index on users.email")
        
        # Sessions collection indexes
        db.sessions.create_index([("class_id", 1), ("date", -1)])
        print("  - Created index on sessions.class_id and date")
        
        # Attendance logs indexes
        db.attendance_logs.create_index([("session_id", 1), ("student_id", 1)])
        print("  - Created index on attendance_logs.session_id and student_id")
        
        # Classes collection indexes
        db.classes.create_index("teacher_id")
        print("  - Created index on classes.teacher_id")
        
        print("\n✅ Database initialization complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

def seed_sample_data():
    """Seed database with sample data for testing"""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        print("\n🌱 Seeding sample data...")
        
        # 1. Create sample users (teachers and students)
        users_col = db['users']
        
        # Clear existing users (optional - comment out if you want to preserve data)
        # users_col.delete_many({})
        
        # Sample Teacher
        teacher_password = hash_password("teacher123")
        teacher = {
            'email': 'prof.XYZ@ves.ac.in',
            'password': teacher_password,
            'role': 'teacher',
            'name': 'Prof. Xyz',
            'department': 'MCA',
            'created_at': datetime.now()
        }
        
        # Check if teacher already exists
        existing_teacher = users_col.find_one({'email': teacher['email']})
        if not existing_teacher:
            teacher_result = users_col.insert_one(teacher)
            teacher_id = teacher_result.inserted_id
            print(f"✅ Created teacher: {teacher['email']}")
        else:
            teacher_id = existing_teacher['_id']
            print(f"  Teacher already exists: {teacher['email']}")
        
        # Sample Students (these are separate from the students collection for face recognition)
        # In a real system, you'd link these with the face recognition students
        sample_students = [
            {
                'email': 'rudransh@ves.ac.in',
                'password': hash_password('student123'),
                'role': 'student',
                'name': 'Rudransh Gupta',
                'roll_no': '13',
                'department': 'MCA',
                'created_at': datetime.now()
            },
            {
                'email': 'omkar@ves.ac.in',
                'password': hash_password('student123'),
                'role': 'student',
                'name': 'Omkar Student',
                'roll_no': '14',
                'department': 'MCA',
                'created_at': datetime.now()
            }
        ]
        
        for student in sample_students:
            existing = users_col.find_one({'email': student['email']})
            if not existing:
                users_col.insert_one(student)
                print(f"✅ Created student user: {student['email']}")
            else:
                print(f"  Student user already exists: {student['email']}")
        
        # 2. Create sample classes/subjects
        classes_col = db['classes']
        
        sample_classes = [
            {
                'name': 'Artificial Intelligence (Theory)',
                'code': 'AI-TH-501',
                'teacher_id': teacher_id,
                'batch': 'MCA 2A',
                'total_students': 65,
                'created_at': datetime.now()
            },
            {
                'name': 'Python Programming (Lab)',
                'code': 'PY-LAB-502',
                'teacher_id': teacher_id,
                'batch': 'MCA 2A',
                'total_students': 65,
                'created_at': datetime.now()
            },
            {
                'name': 'Deep Learning (Theory)',
                'code': 'DL-TH-503',
                'teacher_id': teacher_id,
                'batch': 'MCA 2A',
                'total_students': 65,
                'created_at': datetime.now()
            }
        ]
        
        class_ids = []
        for cls in sample_classes:
            existing = classes_col.find_one({'code': cls['code']})
            if not existing:
                result = classes_col.insert_one(cls)
                class_ids.append(result.inserted_id)
                print(f"✅ Created class: {cls['name']}")
            else:
                class_ids.append(existing['_id'])
                print(f"  Class already exists: {cls['name']}")
        
        # 3. Create sample sessions (past sessions for testing reports)
        sessions_col = db['sessions']
        
        if class_ids:
            # Create a few completed sessions for the first class
            ai_class_id = class_ids[0]
            
            sample_sessions = [
                {
                    'class_id': ai_class_id,
                    'teacher_id': teacher_id,
                    'date': datetime(2025, 11, 1, 10, 0),
                    'started_at': datetime(2025, 11, 1, 10, 0),
                    'ended_at': datetime(2025, 11, 1, 11, 30),
                    'status': 'completed',
                    'total_scanned': 58,
                    'location': 'Room 504'
                },
                {
                    'class_id': ai_class_id,
                    'teacher_id': teacher_id,
                    'date': datetime(2025, 11, 5, 10, 0),
                    'started_at': datetime(2025, 11, 5, 10, 0),
                    'ended_at': datetime(2025, 11, 5, 11, 30),
                    'status': 'completed',
                    'total_scanned': 60,
                    'location': 'Room 504'
                },
                {
                    'class_id': ai_class_id,
                    'teacher_id': teacher_id,
                    'date': datetime(2025, 11, 10, 10, 0),
                    'started_at': datetime(2025, 11, 10, 10, 0),
                    'ended_at': datetime(2025, 11, 10, 11, 30),
                    'status': 'completed',
                    'total_scanned': 55,
                    'location': 'Room 504'
                }
            ]
            
            for session in sample_sessions:
                existing = sessions_col.find_one({
                    'class_id': session['class_id'],
                    'date': session['date']
                })
                if not existing:
                    sessions_col.insert_one(session)
                    print(f"✅ Created sample session for {session['date'].strftime('%Y-%m-%d')}")
        
        print("\n✅ Sample data seeding complete!")
        print("\n📝 Sample Credentials:")
        print("   Teacher: prof.omkar@vesit.edu / teacher123")
        print("   Student: rudransh@student.edu / student123")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        return False

def add_student_to_students_collection(name, roll_no, encoding):
    """Helper function to add a new student with face encoding"""
    db = get_db_connection()
    if db is None:
        return None
    
    try:
        students_col = db['students']
        
        student_doc = {
            'name': name,
            'roll_no': roll_no,
            'encoding': encoding.tolist() if hasattr(encoding, 'tolist') else encoding,
            'created_at': datetime.now()
        }
        
        result = students_col.insert_one(student_doc)
        print(f"✅ Added student: {name} (Roll: {roll_no})")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Error adding student: {e}")
        return None

def get_all_classes():
    """Get all classes from database"""
    db = get_db_connection()
    if db is None:
        return []
    
    try:
        classes_col = db['classes']
        classes = list(classes_col.find({}))
        
        # Convert ObjectId to string for JSON serialization
        for cls in classes:
            cls['_id'] = str(cls['_id'])
            cls['teacher_id'] = str(cls['teacher_id'])
        
        return classes
        
    except Exception as e:
        print(f"❌ Error fetching classes: {e}")
        return []

def cleanup_old_sessions(days=30):
    """Clean up sessions older than specified days"""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        sessions_col = db['sessions']
        result = sessions_col.delete_many({
            'date': {'$lt': cutoff_date},
            'status': 'completed'
        })
        
        print(f"✅ Deleted {result.deleted_count} old sessions")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False

if __name__ == '__main__':
    """Run database initialization and seeding"""
    print("=" * 60)
    print("AttendAI Database Setup")
    print("=" * 60)
    
    # Initialize database
    if initialize_database():
        # Seed sample data
        seed_sample_data()
        
        print("\n" + "=" * 60)
        print("✅ Database setup complete!")
        print("=" * 60)
        print("\nYou can now start the Flask server with: python app.py")
    else:
        print("\n❌ Database setup failed!")
