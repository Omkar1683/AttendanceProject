"""
services/notification_service.py
---------------------------------
Business logic for creating and queuing notifications via Flask-Mail.
"""
from typing import Optional
from flask import current_app, render_template_string
from flask_mail import Message
from bson import ObjectId

from database.connection import get_db
from models import get_models
from .analytics_service import get_defaulters_list


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

def send_notification(class_id: str, target: str, message: str, sent_by: str, student_id: Optional[str] = None) -> dict:
    """
    Create a notification record.

    Args:
        class_id: Target class ObjectId string.
        target:   'all' | 'defaulters' | 'critical' | 'individual'
        message:  Notification text.
        sent_by:  Teacher user ObjectId string.
        student_id: Optional student ObjectId string for 'individual' target.

    Returns:
        {'ok': True,  'notification_id': str}
        {'ok': False, 'message': str, 'code': int}
    """
    if not all([class_id, target, message, sent_by]):
        return {'ok': False, 'message': 'class_id, target, message and sent_by are required', 'code': 400}

    if target == 'individual' and not student_id:
        return {'ok': False, 'message': 'student_id is required for individual target', 'code': 400}

    db = get_db()
    models = get_models(db)
    
    # ── 1. Fetch Recipients ──────────────────────────────────────────────────
    recipients = []
    
    if target == 'individual':
        student = db['students'].find_one({'_id': ObjectId(student_id)})
        if student and student.get('email'):
            # Calculate simple attendance for this student (fallback: "N/A")
            attendance_str = "N/A"
            try:
                # Reuse the dashboard logic or just a basic stat.
                # In this simplified scenario, if we don't have the exact %, just put "Please check dashboard"
                # (Or we could import analytics_service.get_student_report)
                from .analytics_service import get_student_report
                report = get_student_report(str(student_id))
                # Find this class's percentage
                for subj in report.get('subjects', []):
                    if str(subj['class_id']) == str(class_id):
                        attendance_str = f"{subj['percentage']}%"
                        break
            except Exception:
                pass
                
            recipients.append({
                'email': student['email'],
                'name': student.get('name', 'Student'),
                'attendance': attendance_str
            })
            
    elif target in ('defaulters', 'critical'):
        # defaulters: < 75%, critical: < 50%
        threshold = 50 if target == 'critical' else 75
        defaulters = get_defaulters_list(class_id, threshold=threshold)
        # Fetch emails for these student IDs
        student_ids = [ObjectId(d['student_id']) for d in defaulters]
        students = list(db['students'].find({'_id': {'$in': student_ids}}))
        student_map = {str(s['_id']): s for s in students}
        
        for d in defaulters:
            s_doc = student_map.get(d['student_id'])
            if s_doc and s_doc.get('email'):
                recipients.append({
                    'email': s_doc['email'],
                    'name': d['name'],
                    'attendance': f"{d['attendance']}%"
                })

    elif target == 'all':
        # Find all students who have participated in this class's sessions.
        # An alternative is finding students linked to this class if you have an enrollment schema.
        # Since the app uses face scan logs to record attendance:
        sessions = list(db['sessions'].find({'class_id': ObjectId(class_id)}))
        session_ids = [s['_id'] for s in sessions]
        logs = list(db['attendance_logs'].find({'session_id': {'$in': session_ids}}))
        student_ids = list({log['student_id'] for log in logs})
        
        # Convert to ObjectIds, being careful about string formats
        obj_ids = []
        for sid in student_ids:
            try: obj_ids.append(ObjectId(sid))
            except: pass
            
        students = list(db['students'].find({'_id': {'$in': obj_ids}}))
        for student in students:
            if student.get('email'):
                recipients.append({
                    'email': student['email'],
                    'name': student.get('name', 'Student'),
                    'attendance': 'Check Dashboard'
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
                attendance_str=recipient['attendance']
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
        student_id=student_id if target == 'individual' else None,
        recipients=sent_emails,
    )

    if notification_id:
        return {
            'ok': True, 
            'notification_id': notification_id, 
            'message': f'Successfully sent {success_count} emails'
        }
    return {'ok': False, 'message': 'Failed to create notification record (emails may have been sent)', 'code': 500}
