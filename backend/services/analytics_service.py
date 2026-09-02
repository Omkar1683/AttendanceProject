"""
services/analytics_service.py
------------------------------
Attendance analytics: daily summaries, defaulter lists, monthly reports, CSV export.
Ported from analytics.py into a service layer — database handle is resolved
via get_db() rather than being passed as an argument.
"""
import io
from datetime import datetime, timedelta

import pandas as pd
from bson import ObjectId

from database.connection import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(present: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((present / total) * 100, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def get_today_summary(class_id: str) -> dict:
    """Today's attendance summary for a class, based on enrolled students."""
    db = get_db()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    # ── Get enrolled students (source of truth) ───────────────────────────
    class_doc = db['classes'].find_one({'_id': ObjectId(class_id)})
    enrolled_ids = set(str(sid) for sid in (class_doc.get('students') or [])) if class_doc else set()
    total_enrolled = len(enrolled_ids)

    if total_enrolled == 0:
        return {'present': 0, 'absent': 0, 'percentage': 0, 'total_students': 0}

    sessions = list(db['sessions'].find({
        'class_id': ObjectId(class_id),
        'date': {'$gte': today, '$lt': tomorrow},
    }))

    if not sessions:
        return {'present': 0, 'absent': total_enrolled, 'percentage': 0, 'total_students': total_enrolled}

    session_ids = [s['_id'] for s in sessions]

    # Get distinct student IDs who are Present AND enrolled
    logs = list(db['attendance_logs'].find({
        'session_id': {'$in': session_ids},
        'status': 'Present',
    }))
    present_ids = set()
    for log in logs:
        sid = str(log['student_id'])
        if sid in enrolled_ids:
            present_ids.add(sid)

    present_count = len(present_ids)
    absent_count = total_enrolled - present_count

    return {
        'present':        present_count,
        'absent':         absent_count,
        'percentage':     _pct(present_count, total_enrolled),
        'total_students': total_enrolled,
    }


def get_defaulters_list(class_id: str, threshold: int = 75) -> list:
    """Students with attendance below *threshold*% — only enrolled students."""
    db = get_db()
    sessions = list(db['sessions'].find({
        'class_id': ObjectId(class_id),
        'status': 'completed',
    }))

    if not sessions:
        return []

    total_classes = len(sessions)
    session_ids = [s['_id'] for s in sessions]

    # Only check enrolled students, not all students on the platform
    class_doc = db['classes'].find_one({'_id': ObjectId(class_id)})
    enrolled_ids = [str(sid) for sid in (class_doc.get('students') or [])] if class_doc else []

    defaulters = []
    for sid in enrolled_ids:
        student = db['students'].find_one({'_id': ObjectId(sid)})
        if not student:
            continue
        present_count = db['attendance_logs'].count_documents({
            'session_id': {'$in': session_ids},
            'student_id': sid,
            'status': 'Present',
        })
        pct = _pct(present_count, total_classes)
        if pct < threshold:
            defaulters.append({
                'student_id': sid,
                'name':       student.get('name', 'Unknown'),
                'attendance': pct,
                'present':    present_count,
                'total':      total_classes,
            })

    defaulters.sort(key=lambda x: x['attendance'])
    return defaulters


def get_monthly_report(class_id: str, month: int, year: int) -> dict:
    """Per-student attendance report for a calendar month — enrolled students only."""
    db = get_db()
    start_date = datetime(year, month, 1)
    end_date = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    sessions = list(db['sessions'].find({
        'class_id': ObjectId(class_id),
        'date': {'$gte': start_date, '$lt': end_date},
        'status': 'completed',
    }))

    if not sessions:
        return {'total_classes': 0, 'students': []}

    total_classes = len(sessions)
    session_ids = [s['_id'] for s in sessions]

    # Only report on enrolled students, not all students on the platform
    class_doc = db['classes'].find_one({'_id': ObjectId(class_id)})
    enrolled_ids = [str(sid) for sid in (class_doc.get('students') or [])] if class_doc else []

    student_reports = []
    for sid in enrolled_ids:
        student = db['students'].find_one({'_id': ObjectId(sid)})
        if not student:
            continue
        present_count = db['attendance_logs'].count_documents({
            'session_id': {'$in': session_ids},
            'student_id': sid,
            'status': 'Present',
        })
        pct = _pct(present_count, total_classes)
        student_reports.append({
            'student_id': sid,
            'name':       student.get('name', 'Unknown'),
            'roll_no':    student.get('roll_no', 'N/A'),
            'attendance': pct,
            'present':    present_count,
            'absent':     total_classes - present_count,
            'total':      total_classes,
            'status':     'Good' if pct >= 75 else 'Defaulter',
        })

    student_reports.sort(key=lambda x: x['name'])

    # Expose session IDs so the mobile app can use them for manual attendance edits
    latest_session_id = str(sessions[-1]['_id']) if sessions else None
    all_session_ids = [str(s['_id']) for s in sessions]

    return {
        'total_classes': total_classes,
        'month': month,
        'year': year,
        'latest_session_id': latest_session_id,
        'sessions': all_session_ids,
        'students': student_reports,
    }


def get_student_report(student_id: str) -> dict:
    """Overall attendance breakdown for one student across all subjects."""
    db = get_db()
    print(f"DEBUG: get_student_report for student_id: {student_id}")

    # Robust matching: try both string and ObjectId just in case
    ids_to_match = [student_id]
    try:
        ids_to_match.append(ObjectId(student_id))
    except:
        pass

    student_logs = list(db['attendance_logs'].find({'student_id': {'$in': ids_to_match}}))
    print(f"DEBUG: Found {len(student_logs)} logs for student")

    if not student_logs:
        return {'overall_percentage': 0, 'total_present': 0, 'total_classes': 0, 'subjects': []}

    session_ids = list({log['session_id'] for log in student_logs})
    sessions = list(db['sessions'].find({'_id': {'$in': session_ids}}))

    class_stats: dict = {}
    for session in sessions:
        cid = str(session['class_id'])
        class_stats.setdefault(cid, {'total': 0, 'present': 0})
        class_stats[cid]['total'] += 1

    for log in student_logs:
        session = next((s for s in sessions if s['_id'] == log['session_id']), None)
        if session and log['status'] == 'Present':
            class_stats[str(session['class_id'])]['present'] += 1

    subjects = []
    total_present = total_classes = 0
    for cid, stats in class_stats.items():
        class_doc = db['classes'].find_one({'_id': ObjectId(cid)})
        if class_doc:
            pct = _pct(stats['present'], stats['total'])
            subjects.append({
                'class_id':     cid,
                'name':         class_doc.get('name', 'Unknown'),
                'total_classes': stats['total'],
                'present':      stats['present'],
                'absent':       stats['total'] - stats['present'],
                'percentage':   pct,
                'status':       'Good' if pct >= 75 else 'Defaulter',
            })
            total_present += stats['present']
            total_classes += stats['total']

    return {
        'overall_percentage': _pct(total_present, total_classes),
        'total_present':      total_present,
        'total_classes':      total_classes,
        'subjects':           subjects,
    }


def export_to_csv(report_data: dict, class_name: str = "Report") -> str:
    """Serialise a monthly report dict to a CSV string."""
    students = report_data.get('students', [])
    df = pd.DataFrame(students)
    cols = ['name', 'roll_no', 'present', 'absent', 'total', 'attendance', 'status']
    df = df[cols]
    df.columns = ['Student Name', 'Roll No', 'Present', 'Absent', 'Total Classes', 'Attendance %', 'Status']
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def get_subject_averages(teacher_id: str) -> list:
    """Average attendance per subject for a teacher's classes."""
    db = get_db()
    classes = list(db['classes'].find({'teacher_id': ObjectId(teacher_id)}))
    result = []

    for cls in classes:
        cid = str(cls['_id'])
        sessions = list(db['sessions'].find({'class_id': ObjectId(cid), 'status': 'completed'}))
        if not sessions:
            continue

        total_classes = len(sessions)
        session_ids = [s['_id'] for s in sessions]
        total_students = cls.get('total_students', 0)
        if total_students == 0:
            continue

        total_present = db['attendance_logs'].count_documents({
            'session_id': {'$in': session_ids},
            'status': 'Present',
        })
        max_possible = total_classes * total_students
        result.append({
            'class_id':     cid,
            'name':         cls.get('name', 'Unknown'),
            'code':         cls.get('code', 'N/A'),
            'average':      round((total_present / max_possible) * 100, 2) if max_possible else 0,
            'total_classes': total_classes,
        })

    return result


# ── Student-facing analytics ────────────────────────────────────────────────

def get_student_timeline(student_id: str, month: int, year: int) -> list:
    """
    Return daily attendance entries for a student in a given month/year,
    joined with session date and class name.  Used by the calendar screen.

    Returns a list of dicts:
        [{'date': 'YYYY-MM-DD', 'time': 'HH:MM', 'subject': str,
          'status': 'Present'|'Absent', 'marked_by': 'AI'|'Manual',
          'confidence': float|None}, ...]
    """
    db = get_db()
    start = datetime(year, month, 1)
    end   = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    ids_to_match = [student_id]
    try:
        ids_to_match.append(ObjectId(student_id))
    except:
        pass

    logs = list(db['attendance_logs'].find({'student_id': {'$in': ids_to_match}}))
    if not logs:
        return []

    session_ids = [l['session_id'] for l in logs]
    sessions_raw = list(db['sessions'].find({
        '_id':  {'$in': session_ids},
        'date': {'$gte': start, '$lt': end},
    }))
    if not sessions_raw:
        return []

    sessions_map = {str(s['_id']): s for s in sessions_raw}

    class_ids = list({s['class_id'] for s in sessions_raw})
    classes_list = list(db['classes'].find({'_id': {'$in': class_ids}}))

    teacher_ids = list({c['teacher_id'] for c in classes_list})
    teachers_map = {
        str(u['_id']): u.get('name', 'Unknown')
        for u in db['users'].find({'_id': {'$in': teacher_ids}})
    }

    classes_map = {}
    for c in classes_list:
        classes_map[str(c['_id'])] = {
            'name': c.get('name', 'Unknown'),
            'faculty': teachers_map.get(str(c['teacher_id']), 'Unknown')
        }

    timeline = []
    for log in logs:
        session = sessions_map.get(str(log['session_id']))
        if not session:
            continue
        cls_info = classes_map.get(str(session['class_id']), {'name': 'Unknown', 'faculty': 'Unknown'})
        timeline.append({
            'date':         session['date'].strftime('%Y-%m-%d'),
            'time':         log['timestamp'].strftime('%H:%M'),
            'subject':      cls_info['name'],
            'faculty_name': cls_info['faculty'],
            'status':       log['status'],
            'marked_by':    log['marked_by'],
            'confidence':   log.get('confidence'),
        })

    timeline.sort(key=lambda x: x['date'])
    return timeline


def get_student_session_history(student_id: str, page: int = 1, limit: int = 15) -> dict:
    """
    Paginated attendance session history for one student.
    Returns: {'total': int, 'page': int, 'limit': int, 'sessions': [...]}
    """
    db = get_db()
    skip = (page - 1) * limit

    ids_to_match = [student_id]
    try:
        ids_to_match.append(ObjectId(student_id))
    except:
        pass

    total = db['attendance_logs'].count_documents({'student_id': {'$in': ids_to_match}})
    logs = list(
        db['attendance_logs']
        .find({'student_id': {'$in': ids_to_match}})
        .sort('timestamp', -1)
        .skip(skip)
        .limit(limit)
    )

    if not logs:
        return {'total': total, 'page': page, 'limit': limit, 'sessions': []}

    session_ids = [l['session_id'] for l in logs]
    sessions_map = {
        str(s['_id']): s
        for s in db['sessions'].find({'_id': {'$in': session_ids}})
    }

    class_ids = list({s['class_id'] for s in sessions_map.values()})
    classes_list = list(db['classes'].find({'_id': {'$in': class_ids}}))

    teacher_ids = list({c['teacher_id'] for c in classes_list})
    teachers_map = {
        str(u['_id']): u.get('name', 'Unknown')
        for u in db['users'].find({'_id': {'$in': teacher_ids}})
    }

    classes_map = {}
    for c in classes_list:
        classes_map[str(c['_id'])] = {
            'name': c.get('name', 'Unknown'),
            'faculty': teachers_map.get(str(c['teacher_id']), 'Unknown')
        }

    result = []
    for log in logs:
        session = sessions_map.get(str(log['session_id']))
        cls_info = classes_map.get(str(session['class_id']), {'name': 'Unknown', 'faculty': 'Unknown'}) if session else {'name': 'Unknown', 'faculty': 'Unknown'}
        result.append({
            'log_id':       str(log['_id']),
            'date':         session['date'].strftime('%Y-%m-%d') if session else '',
            'time':         log['timestamp'].strftime('%H:%M'),
            'subject':      cls_info['name'],
            'faculty_name': cls_info['faculty'],
            'status':       log['status'],
            'marked_by':    log['marked_by'],
            'confidence':   log.get('confidence'),
        })

    return {'total': total, 'page': page, 'limit': limit, 'sessions': result}
