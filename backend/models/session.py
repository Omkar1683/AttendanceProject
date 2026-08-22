"""
models/session.py
-----------------
SessionModel        — attendance session records in `sessions` collection.
AttendanceLogModel  — per-student attendance records in `attendance_logs`.
setup_indexes()     — creates all required DB indexes.
"""
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from .base import BaseModel


class SessionModel(BaseModel):
    """Attendance session model."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["class_id", "teacher_id", "date", "started_at", "status", "total_scanned"],
            "properties": {
                "class_id":      {"bsonType": "objectId"},
                "teacher_id":    {"bsonType": "objectId"},
                "date":          {"bsonType": "date"},
                "started_at":    {"bsonType": "date"},
                "ended_at":      {"bsonType": "date"},
                "status":        {"enum": ["active", "completed"]},
                "total_scanned": {"bsonType": "int", "minimum": 0},
                "location":      {"bsonType": "string"},
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'sessions')

    def create_session(
        self,
        class_id: str,
        teacher_id: str,
        location: str = "Classroom",
    ) -> Optional[str]:
        """Create (or resume) an active session for a class."""
        # Return existing active session if one already exists
        active = self.find_one({'class_id': ObjectId(class_id), 'status': 'active'})
        if active:
            return str(active['_id'])

        now = datetime.now()
        doc = {
            'class_id':      ObjectId(class_id),
            'teacher_id':    ObjectId(teacher_id),
            'date':          now,
            'started_at':    now,
            'status':        'active',
            'total_scanned': 0,
            'location':      location,
        }
        return self.insert_one(doc)

    def end_session(self, session_id: str) -> bool:
        return self.update_one(
            {'_id': ObjectId(session_id)},
            {'$set': {'status': 'completed', 'ended_at': datetime.now()}},
        )

    def increment_scanned(self, session_id: str) -> bool:
        result = self.collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$inc': {'total_scanned': 1}},
        )
        return result.modified_count > 0

    def find_active_session(self, class_id: str) -> Optional[Dict]:
        return self.find_one({'class_id': ObjectId(class_id), 'status': 'active'})

    def find_by_class(self, class_id: str, status: str = None) -> List[Dict]:
        query = {'class_id': ObjectId(class_id)}
        if status:
            query['status'] = status
        return self.find_many(query)

    def find_by_date_range(
        self, class_id: str, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        return self.find_many({
            'class_id': ObjectId(class_id),
            'date': {'$gte': start_date, '$lt': end_date},
        })

    def find_sessions_for_date(self, class_id: str, date_str: str) -> List[Dict]:
        """Return all sessions for a class on a specific calendar day (YYYY-MM-DD)."""
        try:
            day = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return []
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day.replace(hour=23, minute=59, second=59, microsecond=999999)
        return self.find_many({
            'class_id': ObjectId(class_id),
            'date':     {'$gte': day_start, '$lte': day_end},
        })


class AttendanceLogModel(BaseModel):
    """Per-student attendance records."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "student_id", "student_name", "timestamp", "status", "marked_by"],
            "properties": {
                "session_id":   {"bsonType": "objectId"},
                "student_id":   {"bsonType": "string"},
                "student_name": {"bsonType": "string"},
                "timestamp":    {"bsonType": "date"},
                "status":       {"enum": ["Present", "Absent"]},
                "marked_by":    {"enum": ["AI", "Manual"]},
                "confidence":   {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'attendance_logs')

    def mark_attendance(
        self,
        session_id: str,
        student_id: str,
        student_name: str,
        status: str = "Present",
        marked_by: str = "AI",
        confidence: float = None,
    ) -> Optional[str]:
        """Upsert an attendance record. Returns the document ID if new, else existing ID."""
        existing = self.find_one({
            'session_id': ObjectId(session_id),
            'student_id': student_id,
        })

        if existing:
            # Existing record — update in place (do NOT increment session scanned count again)
            update_data = {'status': status, 'timestamp': datetime.now(), 'marked_by': marked_by}
            if confidence is not None:
                update_data['confidence'] = confidence
            self.update_one({'_id': existing['_id']}, {'$set': update_data})
            return None  # Return None → caller knows it's a duplicate (no increment)

        doc = {
            'session_id':   ObjectId(session_id),
            'student_id':   student_id,
            'student_name': student_name,
            'timestamp':    datetime.now(),
            'status':       status,
            'marked_by':    marked_by,
        }
        if confidence is not None:
            doc['confidence'] = confidence

        return self.insert_one(doc)

    def find_by_session(self, session_id: str) -> List[Dict]:
        return self.find_many({'session_id': ObjectId(session_id)})

    def find_by_student(self, student_id: str) -> List[Dict]:
        return self.find_many({'student_id': student_id})

    def count_present_in_session(self, session_id: str) -> int:
        return self.count({'session_id': ObjectId(session_id), 'status': 'Present'})

    def delete_attendance(self, session_id: str, student_id: str) -> bool:
        return self.delete_one({
            'session_id': ObjectId(session_id),
            'student_id': student_id,
        })


# ── Index setup ───────────────────────────────────────────────────────────────

def setup_indexes(db) -> None:
    """Create all required MongoDB indexes. Safe to call multiple times."""
    print("🔧 Creating database indexes…")

    db.users.create_index([("email", ASCENDING)], unique=True)
    db.users.create_index([("role", ASCENDING)])

    db.students.create_index([("roll_no", ASCENDING)])
    db.students.create_index([("user_id", ASCENDING)])

    db.classes.create_index([("code", ASCENDING)], unique=True)
    db.classes.create_index([("teacher_id", ASCENDING)])
    db.classes.create_index([("batch", ASCENDING)])

    db.sessions.create_index([("class_id", ASCENDING), ("date", DESCENDING)])
    db.sessions.create_index([("status", ASCENDING)])
    db.sessions.create_index([("teacher_id", ASCENDING)])

    db.attendance_logs.create_index(
        [("session_id", ASCENDING), ("student_id", ASCENDING)], unique=True
    )
    db.attendance_logs.create_index([("student_id", ASCENDING)])
    db.attendance_logs.create_index([("timestamp", DESCENDING)])

    db.notifications.create_index([("class_id", ASCENDING), ("created_at", DESCENDING)])
    db.notifications.create_index([("target", ASCENDING)])

    print("✅ All indexes created.")
