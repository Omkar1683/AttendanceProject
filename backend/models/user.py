"""
models/user.py
--------------
UserModel — authentication and profile documents in the `users` collection.
"""
import re
from datetime import datetime
from typing import Optional, List, Dict

from .base import BaseModel


class UserModel(BaseModel):
    """User model for authentication and profiles."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["email", "password", "role", "name", "created_at"],
            "properties": {
                "email":      {"bsonType": "string", "pattern": r"^[^\s@]+@[^\s@]+\.[^\s@]+$"},
                "password":   {"bsonType": "string", "minLength": 8},
                "role":       {"enum": ["teacher", "student"]},
                "name":       {"bsonType": "string", "minLength": 2},
                "department": {"bsonType": "string"},
                "roll_no":    {"bsonType": "string"},
                "created_at": {"bsonType": "date"},
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'users')

    # ── Create ────────────────────────────────────────────────────────────────

    def create_user(
        self,
        email: str,
        password_hash: bytes,
        role: str,
        name: str,
        department: str = None,
        roll_no: str = None,
    ) -> Optional[str]:
        """Insert a new user document; return its string ID."""
        doc = {
            'email': email.lower(),
            'password': password_hash,
            'role': role,
            'name': name,
            'created_at': datetime.now(),
        }
        if department:
            doc['department'] = department
        if roll_no and role == 'student':
            doc['roll_no'] = roll_no
        return self.insert_one(doc)

    # ── Queries ───────────────────────────────────────────────────────────────

    def find_by_email(self, email: str) -> Optional[Dict]:
        return self.find_one({'email': email.lower()})

    def find_by_role(self, role: str) -> List[Dict]:
        return self.find_many({'role': role})

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def validate_email(email: str) -> bool:
        return bool(re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email))
