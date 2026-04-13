"""
routes/attendance_routes.py
---------------------------
Blueprint: face scan and manual attendance correction.
URL prefix: /  (preserves original /scan and /attendance/* paths)
"""
from flask import Blueprint, request, jsonify

from core.security import token_required, role_required
from services.face_service import face_service
import services.attendance_service as att_svc

attendance_bp = Blueprint('attendance', __name__)


# ── /scan ─────────────────────────────────────────────────────────────────────
# NOTE: @token_required is intentionally kept commented out (matches original behaviour).
# Re-enable for production if all clients send a Bearer token with multipart requests.

@attendance_bp.route('/scan', methods=['POST'])
# @token_required
def scan_attendance():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image sent'}), 400

    image_bytes = request.files['file'].read()
    session_id  = request.form.get('session_id')

    result = face_service.scan_frame(image_bytes, session_id)

    if result['status'] == 'success':
        return jsonify(result)
    return jsonify(result), 400


# ── /attendance/manual ────────────────────────────────────────────────────────

@attendance_bp.route('/attendance/manual', methods=['POST'])
@token_required
@role_required('teacher')
def manual_attendance():
    data = request.get_json(silent=True) or {}
    result = att_svc.manual_mark(
        student_id=data.get('student_id'),
        session_id=data.get('session_id'),
        status=data.get('status', 'Present'),
    )
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Updated'})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /attendance/by-date  (GET) ────────────────────────────────────────────────
# Returns all enrolled students + their Present/Absent status for a given date.
# Query params: class_id=<id>&date=YYYY-MM-DD

@attendance_bp.route('/attendance/by-date', methods=['GET'])
@token_required
def attendance_by_date():
    class_id = request.args.get('class_id', '').strip()
    date_str = request.args.get('date', '').strip()

    if not class_id:
        return jsonify({'status': 'error', 'message': 'class_id is required'}), 400
    if not date_str:
        return jsonify({'status': 'error', 'message': 'date (YYYY-MM-DD) is required'}), 400

    result = att_svc.get_attendance_by_date(class_id, date_str)

    if not result.get('ok'):
        return jsonify({'status': 'error', 'message': result.get('message', 'Failed')}), result.get('code', 400)

    return jsonify({
        'status':   'success',
        'date':     result['date'],
        'sessions': result['sessions'],
        'students': result['students'],
    })

