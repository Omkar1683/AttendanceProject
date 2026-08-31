"""
models/class_model.py
---------------------
ClassModel — subject/class records in the `classes` collection.
"""
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId

from .base import BaseModel


class ClassModel(BaseModel):
    """Class / Subject model."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "code", "teacher_id", "total_students", "created_at"],
            "properties": {
                "name":           {"bsonType": "string"},
                "code":           {"bsonType": "string"},
                "teacher_id":     {"bsonType": "objectId"},
                "batch":          {"bsonType": "string"},
                "total_students": {"bsonType": "int", "minimum": 1},
                "department":     {"bsonType": "string"},
                "schedule":       {"bsonType": "string"},
                "created_at":     {"bsonType": "date"},
                "students": {
                    "bsonType": "array",
                    "items":    {"bsonType": "objectId"},
                    "description": "List of enrolled student ObjectIds",
                },
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'classes')

    # ── Create ────────────────────────────────────────────────────────────────

    def create_class(
        self,
        name: str,
        code: str,
        teacher_id: str,
        total_students: int,
        student_ids: List[str] = None,
        batch: str = None,
        department: str = None,
        schedule: str = None,
    ) -> Optional[str]:
        doc = {
            'name': name,
            'code': code.upper(),
            'teacher_id': ObjectId(teacher_id),
            'total_students': total_students,
            'created_at': datetime.now(),
            'students': [ObjectId(sid) for sid in (student_ids or []) if sid],
        }
        if batch:
            doc['batch'] = batch
        if department:
            doc['department'] = department
        if schedule:
            doc['schedule'] = schedule
        return self.insert_one(doc)

    # ── Queries ───────────────────────────────────────────────────────────────

    def find_by_code(self, code: str) -> Optional[Dict]:
        return self.find_one({'code': code.upper()})

    def find_by_teacher(self, teacher_id: str) -> List[Dict]:
        return self.find_many({'teacher_id': ObjectId(teacher_id)})

    def find_by_batch(self, batch: str) -> List[Dict]:
        return self.find_many({'batch': batch})

    # ── Update ────────────────────────────────────────────────────────────────

    def update_enrollment(self, class_id: str, total_students: int) -> bool:
        return self.update_one(
            {'_id': ObjectId(class_id)},
            {'$set': {'total_students': total_students}},
        )

    def add_student(self, class_id: str, student_id: str) -> bool:
        """Add a single student to the class students array (idempotent)."""
        return self.update_one(
            {'_id': ObjectId(class_id)},
            {'$addToSet': {'students': ObjectId(student_id)}},
        )

    def update_class_details(
        self,
        class_id: str,
        student_ids: List[str],
        batch: str = None,
    ) -> bool:
        """
        Update the enrolled students list and optionally the batch.
        Keeps total_students in sync with the actual students array length.
        Does NOT modify the schema — uses existing fields only.
        """
        update = {
            'students': [ObjectId(sid) for sid in student_ids],
            'total_students': len(student_ids) if student_ids else 0,
        }
        if batch is not None:
            update['batch'] = batch
        return self.update_one(
            {'_id': ObjectId(class_id)},
            {'$set': update},
        )
