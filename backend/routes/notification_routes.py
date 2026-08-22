"""
routes/notification_routes.py
------------------------------
Blueprint: notification management.
URL prefix: /notifications
"""
from flask import Blueprint, request, jsonify

from core.security import token_required, role_required
import services.notification_service as notif_svc

notification_bp = Blueprint('notifications', __name__, url_prefix='/notifications')


@notification_bp.route('/send', methods=['POST'])
@token_required
@role_required('teacher')
def send_notification():
    data = request.get_json(silent=True) or {}
    result = notif_svc.send_notification(
        class_id=data.get('class_id'),
        target=data.get('target'),
        message=data.get('message'),
        sent_by=request.user['user_id'],
        email=data.get('email'),            # New: for individual target
        student_id=data.get('student_id'),  # Legacy fallback (kept for compatibility)
    )
    if result['ok']:
        return jsonify({'status': 'success', 'message': result.get('message', 'Notification sent'),
                        'notification_id': result['notification_id']})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']
