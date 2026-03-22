"""
models/notification.py
----------------------
NotificationModel — notification records in the `notifications` collection.
"""
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId
from pymongo import DESCENDING

from .base import BaseModel


class NotificationModel(BaseModel):
    """Notification model."""

    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["class_id", "target", "message", "sent_by", "created_at"],
            "properties": {
                "class_id":   {"bsonType": "objectId"},
                "target":     {"enum": ["all", "defaulters", "critical", "individual"]},
                "message":    {"bsonType": "string", "minLength": 1},
                "sent_by":    {"bsonType": "objectId"},
                "created_at": {"bsonType": "date"},
                "read":       {"bsonType": "bool"},
                "student_id": {"bsonType": "objectId"},
                "recipients": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"}
                }
            },
        }
    }

    def __init__(self, db):
        super().__init__(db, 'notifications')

    def create_notification(
        self,
        class_id: str,
        target: str,
        message: str,
        sent_by: str,
        student_id: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ) -> Optional[str]:
        doc = {
            'class_id':   ObjectId(class_id),
            'target':     target,
            'message':    message,
            'sent_by':    ObjectId(sent_by),
            'created_at': datetime.now(),
            'read':       False,
        }
        if student_id:
            doc['student_id'] = ObjectId(student_id)
        if recipients is not None:
            doc['recipients'] = recipients
            
        return self.insert_one(doc)

    def find_by_class(self, class_id: str, limit: int = 50) -> List[Dict]:
        return list(
            self.collection
            .find({'class_id': ObjectId(class_id)})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )

    def mark_as_read(self, notification_id: str) -> bool:
        return self.update_one(
            {'_id': ObjectId(notification_id)},
            {'$set': {'read': True}},
        )
