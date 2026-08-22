"""
routes/student_routes.py
------------------------
Blueprint: student-facing self-service endpoints.
URL prefix: /student
All routes require a valid JWT (student or teacher role accepted for profile).
"""
import sys
import logging
from flask import Blueprint, request, jsonify
from bson import ObjectId

from core.security import token_required
from database.connection import get_db
import services.auth_service as auth_svc
import services.analytics_service as aly_svc

student_bp = Blueprint('student', __name__, url_prefix='/student')

# Configure logging to be very visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def _resolve_student_id(user_record: dict):
    """
    Given a user dict (from request.user payload or DB), return the matching 
    students._id string (the face-recognition / attendance-log ID).
    Tries user_id first, then falls back to email.
    """
    db = get_db()
    user_id = user_record.get('user_id')
    email = user_record.get('email')
    
    logger.error(f"\n[DIAGNOSTIC] Resolving student for user: {email} (UID: {user_id})")
    
    # 1. Try by user_id
    if user_id:
        student = db['students'].find_one({'user_id': ObjectId(user_id)})
        if student:
            sid = str(student['_id'])
            logger.error(f"[DIAGNOSTIC] Success: Found by user_id -> {sid}")
            return sid
            
    # 2. Fallback by email
    if email:
        student = db['students'].find_one({'email': email.lower()})
        if student:
            sid = str(student['_id'])
            logger.error(f"[DIAGNOSTIC] Success: Found by email fallback -> {sid}")
            return sid
            
    logger.error(f"[DIAGNOSTIC] ❌ FAILED to resolve student for {email}")
    return None


# ── /student/profile ─────────────────────────────────────────────────────────

@student_bp.route('/profile', methods=['GET'])
@token_required
def get_profile():
    """Return merged users + students profile for the logged-in student."""
    user_id = request.user['user_id']
    profile = auth_svc.get_student_profile(user_id)
    if not profile:
        return jsonify({'status': 'error', 'message': 'Profile not found'}), 404
    return jsonify({'status': 'success', 'data': profile})


@student_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile():
    """Edit name, phone, or department for the logged-in student."""
    user_id = request.user['user_id']
    data = request.get_json(silent=True) or {}
    result = auth_svc.update_student_profile(user_id, data)
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Profile updated'})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /student/report ───────────────────────────────────────────────────────────

@student_bp.route('/report', methods=['GET'])
@token_required
def get_report():
    """
    Overall attendance %  + per-subject breakdown for the logged-in student.
    Resolves users._id → students._id before querying attendance_logs.
    """
    student_id = _resolve_student_id(request.user)
    if not student_id:
        return jsonify({'status': 'success', 'data': {
            'overall_percentage': 0,
            'total_present': 0,
            'total_classes': 0,
            'subjects': [],
        }})
    data = aly_svc.get_student_report(student_id)
    return jsonify({'status': 'success', 'data': data})


# ── /student/change-password ─────────────────────────────────────────────────

@student_bp.route('/change-password', methods=['POST'])
@token_required
def change_password():
    """Verify current password then set a new one."""
    user_id = request.user['user_id']
    data = request.get_json(silent=True) or {}
    result = auth_svc.change_password(
        user_id=user_id,
        old_password=data.get('old_password', ''),
        new_password=data.get('new_password', ''),
    )
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Password updated successfully'})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /student/timeline ────────────────────────────────────────────────────────

@student_bp.route('/timeline', methods=['GET'])
@token_required
def get_timeline():
    """
    Return per-day attendance entries for a calendar month.
    Query params: month (1-12), year (YYYY). Defaults to current month/year.
    """
    from datetime import datetime
    user_id = request.user['user_id']
    now = datetime.now()
    month = int(request.args.get('month', now.month))
    year  = int(request.args.get('year',  now.year))

    if not (1 <= month <= 12) or year < 2000:
        return jsonify({'status': 'error', 'message': 'Invalid month or year'}), 400

    student_id = _resolve_student_id(request.user)
    if not student_id:
        return jsonify({'status': 'success', 'data': []})

    data = aly_svc.get_student_timeline(student_id, month, year)
    return jsonify({'status': 'success', 'data': data})


# ── /student/sessions ────────────────────────────────────────────────────────

@student_bp.route('/sessions', methods=['GET'])
@token_required
def get_session_history():
    """
    Paginated attendance session history for the logged-in student.
    Query params: page (default 1), limit (default 15, max 50).
    """
    user_id = request.user['user_id']
    page  = max(1, int(request.args.get('page', 1)))
    limit = min(50, max(1, int(request.args.get('limit', 15))))

    student_id = _resolve_student_id(request.user)
    if not student_id:
        return jsonify({'status': 'success', 'data': {
            'total': 0, 'page': page, 'limit': limit, 'sessions': []
        }})

    data = aly_svc.get_student_session_history(student_id, page, limit)
    return jsonify({'status': 'success', 'data': data})


# ── /student/debug-db ─────────────────────────────────────────────────────────

@student_bp.route('/debug-db', methods=['GET'])
@token_required
def debug_db():
    """Diagnostic endpoint to see what's wrong with attendance matching."""
    db = get_db()
    user_id = request.user['user_id']
    student = db['students'].find_one({'user_id': ObjectId(user_id)})
    
    diagnostic = {
        'user_id': user_id,
        'student_found': student is not None,
    }
    
    if student:
        face_id = str(student['_id'])
        diagnostic.update({
            'student_id': face_id,
            'roll_no': student.get('roll_no'),
            'name': student.get('name'),
        })
        
        # Count logs with different ID formats
        diagnostic['counts'] = {
            'as_face_id_str': db['attendance_logs'].count_documents({'student_id': face_id}),
            'as_face_id_obj': db['attendance_logs'].count_documents({'student_id': ObjectId(face_id)}),
            'as_user_id_str': db['attendance_logs'].count_documents({'student_id': user_id}),
            'as_roll_no': db['attendance_logs'].count_documents({'student_id': student.get('roll_no')}),
        }
        
    return jsonify({'status': 'success', 'diagnostic': diagnostic})
