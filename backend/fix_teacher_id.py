"""
Script to update classes to match the logged-in teacher
"""

from pymongo import MongoClient
from urllib.parse import quote_plus
from bson import ObjectId

# MongoDB connection
RAW_USERNAME = "devesh"
RAW_PASSWORD = "Devesh_1234"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['attendai_db']

# The teacher ID from your login (from the backend logs)
LOGGED_IN_TEACHER_ID = "698ddd6daf11a72903a31b58"

print(f"Updating all classes to teacher ID: {LOGGED_IN_TEACHER_ID}")

# Update all classes
result = db.classes.update_many(
    {},  # Match all classes
    {"$set": {"teacher_id": ObjectId(LOGGED_IN_TEACHER_ID)}}
)

print(f"✅ Updated {result.modified_count} classes")

# Verify the update
classes = list(db.classes.find({}))
print(f"\nVerification - Total classes: {len(classes)}")
for cls in classes:
    print(f"  - {cls['name']}: teacher_id={str(cls['teacher_id'])}")
