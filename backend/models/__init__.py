from .base import BaseModel
from .user import UserModel
from .student import StudentModel
from .class_model import ClassModel
from .session import SessionModel, AttendanceLogModel, setup_indexes
from .notification import NotificationModel


def get_models(db):
    """Return all model instances keyed by name."""
    return {
        'users': UserModel(db),
        'students': StudentModel(db),
        'classes': ClassModel(db),
        'sessions': SessionModel(db),
        'attendance_logs': AttendanceLogModel(db),
        'notifications': NotificationModel(db),
    }
