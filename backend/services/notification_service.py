"""
services/notification_service.py
---------------------------------
Business logic for creating and queuing notifications.
In production this is where you'd integrate with FCM or another push provider.
"""
from database.connection import get_db
from models import get_models


def send_notification(class_id: str, target: str, message: str, sent_by: str) -> dict:
    """
    Create a notification record.

    Args:
        class_id: Target class ObjectId string.
        target:   'all' | 'defaulters' | 'critical'
        message:  Notification text.
        sent_by:  Teacher user ObjectId string.

    Returns:
        {'ok': True,  'notification_id': str}
        {'ok': False, 'message': str, 'code': int}
    """
    if not all([class_id, target, message, sent_by]):
        return {'ok': False, 'message': 'class_id, target, message and sent_by are required', 'code': 400}

    db = get_db()
    models = get_models(db)
    notification_id = models['notifications'].create_notification(
        class_id=class_id,
        target=target,
        message=message,
        sent_by=sent_by,
    )

    if notification_id:
        # TODO: integrate with FCM / push provider here
        return {'ok': True, 'notification_id': notification_id}
    return {'ok': False, 'message': 'Failed to create notification', 'code': 500}
