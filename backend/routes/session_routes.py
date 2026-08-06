"""
routes/session_routes.py
------------------------
Blueprint: attendance session lifecycle.
URL prefix: /sessions
"""
from flask import Blueprint, request, jsonify

from core.security import token_required, role_required
import services.attendance_service as att_svc
from services.face_service import face_service

session_bp = Blueprint('sessions', __name__, url_prefix='/sessions')


@session_bp.route('/create', methods=['POST'])
@token_required
@role_required('teacher')
def create_session():
    data = request.get_json(silent=True) or {}
    result = att_svc.create_session(
        class_id=data.get('class_id'),
        teacher_id=request.user['user_id'],
        location=data.get('location', 'Classroom'),
    )
    if result['ok']:
        return jsonify({'status': 'success', 'session_id': result['session_id']})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


@session_bp.route('/stop', methods=['POST'])
@token_required
@role_required('teacher')
def stop_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    result = att_svc.stop_session(session_id)
    if result['ok']:
        # Free the in-memory dedup set for this session
        face_service.reset_session(session_id)
        return jsonify({'status': 'success', 'message': 'Session stopped'})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']
