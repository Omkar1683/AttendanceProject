"""
services/attendance_service.py
-------------------------------
Business logic for session lifecycle and manual attendance correction.
"""
from database.connection import get_db
from models import get_models


def create_session(class_id: str, teacher_id: str, location: str = 'Classroom') -> dict:
    """
    Start (or resume) an attendance session for a class.

    Returns:
        {'ok': True,  'session_id': str}           on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    if not class_id:
        return {'ok': False, 'message': 'Class ID required', 'code': 400}

    db = get_db()
    models = get_models(db)
    session_id = models['sessions'].create_session(
        class_id=class_id,
        teacher_id=teacher_id,
        location=location,
    )

    if session_id:
        return {'ok': True, 'session_id': session_id}
    return {'ok': False, 'message': 'Failed to create session', 'code': 500}


def stop_session(session_id: str) -> dict:
    """
    Mark an active session as completed.

    Returns:
        {'ok': True}
        {'ok': False, 'message': str, 'code': int}
    """
    if not session_id:
        return {'ok': False, 'message': 'Session ID required', 'code': 400}

    db = get_db()
    models = get_models(db)
    success = models['sessions'].end_session(session_id)

    if success:
        return {'ok': True}
    return {'ok': False, 'message': 'Failed to stop session', 'code': 400}


def manual_mark(student_id: str, session_id: str, status: str = 'Present') -> dict:
    """
    Manually set attendance status for a student in a session.

    Args:
        status: 'Present' to mark present, anything else removes the record.
    Returns:
        {'ok': True}
        {'ok': False, 'message': str, 'code': int}
    """
    if not student_id or not session_id:
        return {'ok': False, 'message': 'student_id and session_id are required', 'code': 400}

    db = get_db()
    models = get_models(db)

    if status == 'Present':
        student = models['students'].find_by_id(student_id)
        student_name = student['name'] if student else 'Unknown'
        models['attendance_logs'].mark_attendance(
            session_id=session_id,
            student_id=student_id,
            student_name=student_name,
            status='Present',
            marked_by='Manual',
        )
    else:
        models['attendance_logs'].delete_attendance(session_id, student_id)

    return {'ok': True}
