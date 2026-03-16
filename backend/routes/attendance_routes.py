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
