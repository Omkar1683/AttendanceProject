"""
routes/attendance_routes.py
---------------------------
Blueprint: face scan and manual attendance correction.
URL prefix: /  (preserves original /scan and /attendance/* paths)

NEW endpoints added (queue-based pipeline):
  POST /scan/enqueue   — fire-and-forget frame enqueue (returns immediately)
  GET  /queue/status   — live counter stats for the dashboard

EXISTING endpoints preserved:
  POST /scan           — now also routes through the queue (backward compat)
  POST /attendance/manual
  GET  /attendance/by-date
"""
from flask import Blueprint, request, jsonify

from core.security import token_required, role_required
from workers.queue_manager import enqueue_frame, get_stats
import services.attendance_service as att_svc

attendance_bp = Blueprint('attendance', __name__)


# ── /scan/enqueue  (NEW — fire and forget) ─────────────────────────────────────
#
# The frontend posts a frame here and gets an instant response.
# Recognition happens asynchronously in a background worker thread.

@attendance_bp.route('/scan/enqueue', methods=['POST'])
def enqueue_scan():
    """
    Receive a camera frame and push it onto the processing queue.

    Form fields:
        file       — multipart image file
        session_id — active session identifier

    Response (always fast, never blocks):
        {"status": "queued"}   — frame accepted
        {"status": "full"}     — queue saturated, frame dropped
        {"status": "error"}    — missing/empty file
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image sent'}), 400

    image_file  = request.files['file']
    image_bytes = image_file.read()
    session_id  = request.form.get('session_id', '')

    if len(image_bytes) == 0:
        return jsonify({'status': 'error', 'message': 'Empty image received'}), 400

    accepted = enqueue_frame(session_id, image_bytes)

    if accepted:
        return jsonify({'status': 'queued'})
    else:
        return jsonify({'status': 'full', 'message': 'Queue is saturated — frame dropped'}), 503


# ── /scan  (PRESERVED — now also uses queue) ──────────────────────────────────
#
# Kept for full backward compatibility with existing app versions.
# Now internally enqueues the frame instead of blocking on recognition.

@attendance_bp.route('/scan', methods=['POST'])
def scan_attendance():
    """
    Scan endpoint.
    If session_id is provided, enqueues the frame for background worker attendance processing.
    If no session_id is provided (e.g. student registration face detection & encoding),
    processes synchronously and returns face encodings.
    """
    print("[Scan] /scan endpoint hit")

    if 'file' not in request.files:
        print("[Scan] ERROR: 'file' key missing from request.files")
        return jsonify({'status': 'error', 'message': 'No image sent'}), 400

    image_file  = request.files['file']
    image_bytes = image_file.read()
    session_id  = request.form.get('session_id', '')

    print(f"[Scan] session_id='{session_id}'  image_size={len(image_bytes)} bytes")

    if len(image_bytes) == 0:
        print("[Scan] ERROR: Empty image received")
        return jsonify({'status': 'error', 'message': 'Empty image received'}), 400

    if session_id:
        accepted = enqueue_frame(session_id, image_bytes)
        if accepted:
            return jsonify({'status': 'queued', 'message': 'Frame accepted for processing'})
        else:
            return jsonify({'status': 'full', 'message': 'Queue full — frame dropped'}), 503

    # Synchronous processing for student registration / face encoding extraction
    from services.face_service import face_service
    result = face_service.scan_frame(image_bytes, session_id=None)
    return jsonify(result)


# ── /queue/status  (NEW) ──────────────────────────────────────────────────────

@attendance_bp.route('/queue/status', methods=['GET'])
def queue_status():
    """
    Return live queue counters + present count for the current session.
    """
    session_id = request.args.get('session_id', '')
    stats = get_stats()
    if session_id:
        from database.connection import get_db
        from models import get_models
        db = get_db()
        if db is not None:
            models = get_models(db)
            logs = models['attendance_logs'].find_by_session(session_id)
            stats['present_count'] = len([l for l in logs if l.get('status') == 'Present'])
            stats['marked_students'] = [
                {'student_id': l.get('student_id'), 'student_name': l.get('student_name')}
                for l in logs if l.get('status') == 'Present'
            ]
    return jsonify(stats)


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
