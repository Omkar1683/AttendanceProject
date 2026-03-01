"""
Script to manually add classes to MongoDB
Run this script to populate the classes collection with sample data
"""

from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
from bson import ObjectId

# MongoDB connection (same as in app.py)
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['attendai_db']

def add_sample_classes():
    """Add sample classes to the database"""
    
    # First, let's get a teacher ID
    print("Fetching teachers from database...")
    teachers = list(db.users.find({"role": "teacher"}))
    
    if not teachers:
        print("❌ No teachers found in database. Please create a teacher account first.")
        return
    
    print(f"✅ Found {len(teachers)} teacher(s)")
    for i, teacher in enumerate(teachers):
        print(f"{i+1}. {teacher.get('name', 'Unknown')} ({teacher.get('email', 'No email')}) - ID: {teacher['_id']}")
    
    # Use the first teacher by default
    teacher_id = teachers[0]['_id']
    teacher_name = teachers[0].get('name', 'Unknown')
    print(f"\n📝 Using teacher: {teacher_name} (ID: {teacher_id})")
    
    # Define sample classes
    sample_classes = [
        {
            'name': 'Computer Science 101',
            'code': 'CS101',
            'teacher_id': teacher_id,
            'students': [],  # No students enrolled yet
            'total_students': 30,
            'batch': '2024',
            'department': 'Computer Science',
            'schedule': 'Mon, Wed 10:00-11:30',
            'createdAt': datetime.now()
        },
        {
            'name': 'Data Structures',
            'code': 'CS201',
            'teacher_id': teacher_id,
            'students': [],
            'total_students': 35,
            'batch': '2024',
            'department': 'Computer Science',
            'schedule': 'Tue, Thu 14:00-15:30',
            'createdAt': datetime.now()
        },
        {
            'name': 'Database Systems',
            'code': 'CS301',
            'teacher_id': teacher_id,
            'students': [],
            'total_students': 40,
            'batch': '2024',
            'department': 'Computer Science',
            'schedule': 'Mon, Wed 14:00-15:30',
            'createdAt': datetime.now()
        },
        {
            'name': 'Web Development',
            'code': 'CS401',
            'teacher_id': teacher_id,
            'students': [],
            'total_students': 25,
            'batch': '2024',
            'department': 'Computer Science',
            'schedule': 'Fri 10:00-13:00',
            'createdAt': datetime.now()
        }
    ]
    
    print("\n📚 Adding sample classes...")
    for cls in sample_classes:
        try:
            # Check if class with same code already exists
            existing = db.classes.find_one({'code': cls['code']})
            if existing:
                print(f"⚠️  Class {cls['code']} already exists, skipping...")
                continue
            
            result = db.classes.insert_one(cls)
            print(f"✅ Added: {cls['name']} ({cls['code']}) - ID: {result.inserted_id}")
        except Exception as e:
            print(f"❌ Error adding {cls['code']}: {e}")
    
    print("\n✨ Done! Checking total classes in database...")
    total_classes = db.classes.count_documents({})
    print(f"📊 Total classes in database: {total_classes}")
    
    # List all classes
    print("\n📋 All classes:")
    all_classes = db.classes.find({})
    for cls in all_classes:
        print(f"  - {cls['name']} ({cls['code']}) - {cls.get('total_students', 0)} students")

if __name__ == '__main__':
    print("=" * 60)
    print("  Adding Sample Classes to MongoDB")
    print("=" * 60)
    add_sample_classes()
    print("\n" + "=" * 60)
    print("  Script completed!")
    print("=" * 60)
