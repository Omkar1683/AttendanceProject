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
