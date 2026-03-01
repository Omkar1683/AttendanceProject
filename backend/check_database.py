"""
Script to check classes in the database
"""

from pymongo import MongoClient
from urllib.parse import quote_plus
from bson import ObjectId
import json

# MongoDB connection
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['attendai_db']

# Check all teachers
teachers = list(db.users.find({"role": "teacher"}))
print(f"Total Teachers: {len(teachers)}")
for teacher in teachers:
    print(f"  - {teacher.get('name')}: ID={str(teacher['_id'])}")

# Check all classes  
classes = list(db.classes.find({}))
print(f"\nTotal Classes: {len(classes)}")

for cls in classes:
    teacher_id = cls.get('teacher_id') or cls.get('teacher')
    print(f"  - {cls['name']}: teacher_id={str(teacher_id) if teacher_id else 'None'}")

