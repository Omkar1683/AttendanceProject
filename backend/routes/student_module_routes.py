"""
routes/student_module_routes.py
--------------------------------
Blueprint: NEW student module endpoints.
URL prefix: /student

These are additions to the existing student_routes.py endpoints.
No existing endpoints are modified.

New endpoints:
  GET  /student/notifications          — student notifications
  POST /student/notifications/<id>/read — mark notification read
  GET  /student/analytics              — monthly/weekly trends
  GET  /student/export/csv             — download own attendance CSV
  GET  /student/export/pdf             — download own attendance PDF
  GET  /student/classes                — enrolled classes list
"""
import io
from flask import Blueprint, request, jsonify, send_file, Response

from core.security import token_required
import services.student_module_service as sms

student_module_bp = Blueprint('student_module', __name__, url_prefix='/student')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_student_role():
    """Return error response if user is not a student, else None."""
    role = request.user.get('role', '')
    if role != 'student':
        return jsonify({'status': 'error', 'message': 'Student access only'}), 403
    return None


# ── /student/notifications ────────────────────────────────────────────────────

@student_module_bp.route('/notifications', methods=['GET'])
@token_required
def get_notifications():
    """Fetch notifications relevant to the logged-in student."""
    err = _require_student_role()
    if err:
        return err

    page = max(1, int(request.args.get('page', 1)))
    limit = min(50, max(1, int(request.args.get('limit', 30))))

    data = sms.get_student_notifications(request.user['user_id'], page, limit)
    return jsonify({'status': 'success', 'data': data})


@student_module_bp.route('/notifications/<notification_id>/read', methods=['POST'])
@token_required
def mark_notification_read(notification_id):
    """Mark a specific notification as read."""
    err = _require_student_role()
    if err:
        return err

    result = sms.mark_notification_read(notification_id, request.user['user_id'])
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Notification marked as read'})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /student/analytics ────────────────────────────────────────────────────────

@student_module_bp.route('/analytics', methods=['GET'])
@token_required
def get_analytics():
    """Return monthly trend, weekly trend, and present/absent totals."""
    err = _require_student_role()
    if err:
        return err

    data = sms.get_student_analytics(request.user['user_id'])
    return jsonify({'status': 'success', 'data': data})


# ── /student/classes ──────────────────────────────────────────────────────────

@student_module_bp.route('/classes', methods=['GET'])
@token_required
def get_enrolled_classes():
    """Return classes the student is enrolled in."""
    err = _require_student_role()
    if err:
        return err

    data = sms.get_enrolled_classes(request.user['user_id'])
    return jsonify({'status': 'success', 'data': data})


# ── /student/export/csv ───────────────────────────────────────────────────────

@student_module_bp.route('/export/csv', methods=['GET'])
@token_required
def export_csv():
    """Download the student's own attendance as CSV."""
    err = _require_student_role()
    if err:
        return err

    csv_data = sms.export_student_csv(request.user['user_id'])
    if not csv_data:
        return jsonify({'status': 'error', 'message': 'No attendance data to export'}), 404

    return Response(
        csv_data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=my_attendance.csv',
        },
    )


# ── /student/export/pdf ───────────────────────────────────────────────────────

@student_module_bp.route('/export/pdf', methods=['GET'])
@token_required
def export_pdf():
    """Download the student's own attendance as PDF (or HTML fallback)."""
    err = _require_student_role()
    if err:
        return err

    pdf_data = sms.export_student_pdf(request.user['user_id'])
    if not pdf_data:
        return jsonify({'status': 'error', 'message': 'No attendance data to export'}), 404

    # If it's bytes (proper PDF) or string (HTML fallback)
    if isinstance(pdf_data, bytes) and pdf_data[:4] == b'%PDF':
        return send_file(
            io.BytesIO(pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='my_attendance.pdf',
        )
    else:
        # HTML fallback — the client can use expo-print to convert
        return Response(
            pdf_data if isinstance(pdf_data, bytes) else pdf_data.encode('utf-8'),
            mimetype='text/html',
            headers={
                'Content-Disposition': 'attachment; filename=my_attendance.html',
            },
        )
