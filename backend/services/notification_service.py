"""
services/notification_service.py
---------------------------------
Business logic for creating and queuing notifications via Flask-Mail.
"""
from typing import Optional
from flask import current_app
from flask_mail import Message
from bson import ObjectId
import re

from database.connection import get_db
from models import get_models
from .analytics_service import get_defaulters_list


_EMAIL_RE = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def _generate_email_html(student_name: str, message_body: str, attendance_str: str) -> str:
    """Generate HTML body for the email."""
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
          <h2 style="color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px;">AttendAI Notification</h2>
          <p>Dear <strong>{student_name}</strong>,</p>
          <p style="padding: 15px; background-color: #f9fafb; border-left: 4px solid #2563eb; margin: 20px 0;">
            {message_body}
          </p>
          <p><strong>Current Attendance:</strong> {attendance_str}</p>
          <br>
          <p style="font-size: 12px; color: #6b7280; border-top: 1px solid #ddd; padding-top: 10px;">
            This is an automated message from your Attendance Management System. Please do not reply directly to this email.
          </p>
        </div>
      </body>
    </html>
    """


def send_notification(
    class_id: str,
    target: str,
    message: str,
    sent_by: str,
    email: Optional[str] = None,
    # Legacy: still accepted but ignored for individual target
    student_id: Optional[str] = None,
) -> dict:
    """
    Create a notification record and send emails.

    Args:
        class_id:   Target class ObjectId string.
        target:     'all' | 'defaulters' | 'critical' | 'individual'
        message:    Notification text.
        sent_by:    Teacher user ObjectId string.
        email:      Student email for 'individual' target (preferred over student_id).
        student_id: Deprecated — kept for backward compatibility.

    Returns:
        {'ok': True,  'notification_id': str}
        {'ok': False, 'message': str, 'code': int}
    """
    if not all([class_id, target, message, sent_by]):
        return {'ok': False, 'message': 'class_id, target, message and sent_by are required', 'code': 400}

    # For individual target, require email (or fall back to student_id for legacy calls)
    if target == 'individual':
        if not email and not student_id:
            return {'ok': False, 'message': 'email is required for individual notifications', 'code': 400}
        if email:
            email = email.strip().lower()
            if not _is_valid_email(email):
                return {'ok': False, 'message': 'Invalid email format', 'code': 400}

    db = get_db()
    models = get_models(db)

    # ── 1. Fetch Recipients ──────────────────────────────────────────────────
    recipients = []

    if target == 'individual':
        student = None

        # Prefer email lookup
        if email:
            student = db['students'].find_one({'email': email})
            if not student:
                return {
                    'ok': False,
                    'message': f'No student found with email "{email}". Please verify the email address.',
                    'code': 404,
                }
        elif student_id:
            # Legacy fallback: lookup by ObjectId
            try:
                student = db['students'].find_one({'_id': ObjectId(student_id)})
            except Exception:
                pass
            if not student:
                return {'ok': False, 'message': 'Student not found', 'code': 404}

        if student and student.get('email'):
            # Calculate attendance for this student in this class
            attendance_str = "N/A"
            try:
                from .analytics_service import get_student_report
                report = get_student_report(str(student['_id']))
                for subj in report.get('subjects', []):
                    if str(subj.get('class_id')) == str(class_id):
                        attendance_str = f"{subj['percentage']}%"
                        break
            except Exception:
                pass

            recipients.append({
                'email':      student['email'],
                'name':       student.get('name', 'Student'),
                'attendance': attendance_str,
            })
        else:
            return {'ok': False, 'message': 'Student has no email address on file', 'code': 400}

    elif target in ('defaulters', 'critical'):
        # defaulters: < 75%, critical: < 50%
        threshold = 50 if target == 'critical' else 75
        defaulters = get_defaulters_list(class_id, threshold=threshold)
        student_ids = [ObjectId(d['student_id']) for d in defaulters]
        students = list(db['students'].find({'_id': {'$in': student_ids}}))
        student_map = {str(s['_id']): s for s in students}

        for d in defaulters:
            s_doc = student_map.get(d['student_id'])
            if s_doc and s_doc.get('email'):
                recipients.append({
                    'email':      s_doc['email'],
                    'name':       d['name'],
                    'attendance': f"{d['attendance']}%",
                })

    elif target == 'all':
        # Find all students who have attended this class's sessions
        sessions = list(db['sessions'].find({'class_id': ObjectId(class_id)}))
        session_ids = [s['_id'] for s in sessions]
        logs = list(db['attendance_logs'].find({'session_id': {'$in': session_ids}}))
        seen_ids = list({log['student_id'] for log in logs})

        obj_ids = []
        for sid in seen_ids:
            try:
                obj_ids.append(ObjectId(sid))
            except Exception:
                pass

        students = list(db['students'].find({'_id': {'$in': obj_ids}}))
        for student in students:
            if student.get('email'):
                recipients.append({
                    'email':      student['email'],
                    'name':       student.get('name', 'Student'),
                    'attendance': 'Check Dashboard',
                })
    else:
        return {'ok': False, 'message': f'Invalid target: {target}', 'code': 400}

    if not recipients:
        return {'ok': False, 'message': 'No valid student emails found for the selected target', 'code': 404}

    # ── 2. Send Emails ───────────────────────────────────────────────────────
    mail = current_app.extensions.get('mail')
    if not mail:
        return {'ok': False, 'message': 'Email service not configured', 'code': 500}

    sent_emails = []
    success_count = 0

    class_doc = db['classes'].find_one({'_id': ObjectId(class_id)})
    subject_name = class_doc.get('name', 'Class') if class_doc else 'Class'

    for recipient in recipients:
        try:
            msg = Message(
                subject=f"Attendance Alert: {subject_name}",
                recipients=[recipient['email']]
            )
            msg.html = _generate_email_html(
                student_name=recipient['name'],
                message_body=message,
                attendance_str=recipient['attendance'],
            )
            mail.send(msg)
            sent_emails.append(recipient['email'])
            success_count += 1
        except Exception as e:
            print(f"Error sending email to {recipient['email']}: {e}")

    # ── 3. Store Notification Record ─────────────────────────────────────────
    notification_id = models['notifications'].create_notification(
        class_id=class_id,
        target=target,
        message=message,
        sent_by=sent_by,
        student_id=student_id if (target == 'individual' and student_id) else None,
        recipients=sent_emails,
    )

    if notification_id:
        return {
            'ok': True,
            'notification_id': notification_id,
            'message': f'Successfully sent {success_count} email(s)',
        }
    return {'ok': False, 'message': 'Failed to create notification record (emails may have been sent)', 'code': 500}
