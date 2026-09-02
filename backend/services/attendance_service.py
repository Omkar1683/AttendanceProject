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
    Mark an active session as completed, then finalize attendance:
    every enrolled student without a record gets marked Absent.

    Returns:
        {'ok': True}
        {'ok': False, 'message': str, 'code': int}
    """
    if not session_id:
        return {'ok': False, 'message': 'Session ID required', 'code': 400}

    db = get_db()
    models = get_models(db)

    # Look up the session to get class_id before ending it
    session_doc = models['sessions'].find_one(
        {'_id': __import__('bson').ObjectId(session_id)}
    )
    if not session_doc:
        return {'ok': False, 'message': 'Session not found', 'code': 404}

    success = models['sessions'].end_session(session_id)
    if not success:
        return {'ok': False, 'message': 'Failed to stop session', 'code': 400}

    # ── Finalize: auto-mark Absent for enrolled students with no record ────
    class_id = str(session_doc['class_id'])
    class_doc = models['classes'].find_by_id(class_id)
    if class_doc:
        enrolled_ids = [str(sid) for sid in (class_doc.get('students') or [])]
        # Get existing attendance records for this session
        existing_logs = models['attendance_logs'].find_by_session(session_id)
        already_marked = {log['student_id'] for log in existing_logs}

        for sid in enrolled_ids:
            if sid not in already_marked:
                # Student has no record at all → mark Absent
                student = models['students'].find_by_id(sid)
                student_name = student['name'] if student else 'Unknown'
                models['attendance_logs'].mark_attendance(
                    session_id=session_id,
                    student_id=sid,
                    student_name=student_name,
                    status='Absent',
                    marked_by='AI',
                )

    return {'ok': True}


def manual_mark(student_id: str, session_id: str, status: str = 'Present') -> dict:
    """
    Manually set attendance status for a student in a session.

    Args:
        status: 'Present' or 'Absent'. Both create/update an actual
                attendance record with marked_by='Manual'.
    Returns:
        {'ok': True}
        {'ok': False, 'message': str, 'code': int}
    """
    if not student_id or not session_id:
        return {'ok': False, 'message': 'student_id and session_id are required', 'code': 400}

    db = get_db()
    models = get_models(db)

    student = models['students'].find_by_id(student_id)
    student_name = student['name'] if student else 'Unknown'

    # Both Present and Absent create a real record with marked_by='Manual'
    models['attendance_logs'].mark_attendance(
        session_id=session_id,
        student_id=student_id,
        student_name=student_name,
        status=status if status in ('Present', 'Absent') else 'Present',
        marked_by='Manual',
    )

    return {'ok': True}


def get_attendance_by_date(class_id: str, date_str: str) -> dict:
    """
    Return all students in a class with their attendance status for a specific date.

    Args:
        class_id:  MongoDB class ID string.
        date_str:  Calendar date string in 'YYYY-MM-DD' format.

    Returns:
        {
            'ok': True,
            'date': 'YYYY-MM-DD',
            'sessions': [{'session_id': str, 'status': str, 'total_scanned': int}],
            'students': [{'student_id': str, 'name': str, 'roll_no': str, 'status': 'Present'|'Absent'}]
        }
    """
    if not class_id or not date_str:
        return {'ok': False, 'message': 'class_id and date are required', 'code': 400}

    db = get_db()
    models = get_models(db)

    # Find all sessions on this date for the class
    sessions = models['sessions'].find_sessions_for_date(class_id, date_str)

    session_ids = [str(s['_id']) for s in sessions]
    session_info = [
        {
            'session_id':    str(s['_id']),
            'status':        s.get('status', 'completed'),
            'total_scanned': s.get('total_scanned', 0),
            'started_at':    s['started_at'].isoformat() if s.get('started_at') else None,
        }
        for s in sessions
    ]

    # Get all students enrolled in this class
    from models.class_model import ClassModel
    cls = ClassModel(db)
    class_doc = cls.find_by_id(class_id)
    # ClassModel stores enrolled students under 'students' (list of ObjectIds)
    enrolled_ids = [str(sid) for sid in (class_doc.get('students') or [])] if class_doc else []

    # Collect all attendance logs for any session on this date
    present_student_ids = set()
    for sid in session_ids:
        logs = models['attendance_logs'].find_by_session(sid)
        for log in logs:
            if log.get('status') == 'Present':
                present_student_ids.add(str(log['student_id']))

    # Build per-student status list
    students_out = []
    for sid in enrolled_ids:
        student = models['students'].find_by_id(sid)
        if not student:
            continue
        students_out.append({
            'student_id': sid,
            'name':       student.get('name', 'Unknown'),
            'roll_no':    student.get('roll_no', ''),
            'status':     'Present' if sid in present_student_ids else 'Absent',
        })

    return {
        'ok':       True,
        'date':     date_str,
        'sessions': session_info,
        'students': students_out,
    }

