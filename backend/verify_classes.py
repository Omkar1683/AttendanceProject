"""
Simple script to verify classes in MongoDB
"""

from pymongo import MongoClient
from urllib.parse import quote_plus

# MongoDB connection (same as in app.py)
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['attendai_db']

print("Checking classes in database...")
classes = list(db.classes.find({}))
print(f"\nTotal classes: {len(classes)}")

if classes:
    print("\nClasses found:")
    for cls in classes:
        print(f"  - {cls['name']} ({cls['code']}) - {cls.get('totalStudents', 0)} students")
else:
    print("\nNo classes found in database.")
    
print("\nChecking teachers in database...")
teachers = list(db.users.find({"role": "teacher"}))
print(f"Total teachers: {len(teachers)}")
for teacher in teachers:
    print(f"  - {teacher.get('name', 'Unknown')} ({teacher.get('email', 'No email')}) - ID: {teacher['_id']}")
