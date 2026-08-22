"""
models/student.py
-----------------
StudentModel — face-encoded student records in the `students` collection.
"""
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId

from .base import BaseModel


class StudentModel(BaseModel):
    """Student model for face recognition and registration."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "roll_no", "encoding", "created_at"],
            "properties": {
                "name":       {"bsonType": "string", "minLength": 2},
                "roll_no":    {"bsonType": "string"},
                "encoding":   {"bsonType": "array", "minItems": 128, "maxItems": 128,
                               "items": {"bsonType": "double"}},
                "email":      {"bsonType": "string"},
                "phone":      {"bsonType": "string"},
                "department": {"bsonType": "string"},
                "batch":      {"bsonType": "string"},
                "user_id":    {"bsonType": "objectId"},
                "created_at": {"bsonType": "date"},
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'students')

    # ── Create ────────────────────────────────────────────────────────────────

    def create_student(
        self,
        name: str,
        roll_no: str,
        encoding: List[float],
        email: str = None,
        phone: str = None,
        department: str = None,
        batch: str = None,
        user_id: str = None,
    ) -> Optional[str]:
        """Register a new student; returns string ID or None."""
        if len(encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")

        doc = {
            'name': name,
            'roll_no': roll_no,
            'encoding': encoding,
            'created_at': datetime.now(),
        }
        if email:
            doc['email'] = email.lower()
        if phone:
            doc['phone'] = phone
        if department:
            doc['department'] = department
        if batch:
            doc['batch'] = batch
        if user_id:
            doc['user_id'] = ObjectId(user_id)

        return self.insert_one(doc)

    # ── Queries ───────────────────────────────────────────────────────────────

    def find_by_roll_no(self, roll_no: str) -> Optional[Dict]:
        return self.find_one({'roll_no': roll_no})

    def find_by_batch(self, batch: str) -> List[Dict]:
        return self.find_many({'batch': batch})

    def get_all_encodings(self) -> List[Dict]:
        """Return name + encoding for every student (used by face cache)."""
        return self.find_many({}, projection={'name': 1, 'encoding': 1})

    # ── Updates ───────────────────────────────────────────────────────────────

    def update_encoding(self, student_id: str, new_encoding: List[float]) -> bool:
        if len(new_encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")
        return self.update_one(
            {'_id': ObjectId(student_id)},
            {'$set': {'encoding': new_encoding}},
        )
