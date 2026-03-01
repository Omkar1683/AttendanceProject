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
    """Today's attendance summary for a class."""
    db = get_db()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    sessions = list(db['sessions'].find({
        'class_id': ObjectId(class_id),
        'date': {'$gte': today, '$lt': tomorrow},
    }))

    if not sessions:
        return {'present': 0, 'absent': 0, 'percentage': 0, 'total_students': 0}

    session_ids = [s['_id'] for s in sessions]
    present_count = db['attendance_logs'].count_documents({
        'session_id': {'$in': session_ids},
        'status': 'Present',
    })

    class_doc = db['classes'].find_one({'_id': ObjectId(class_id)})
    total_students = class_doc.get('total_students', 0) if class_doc else 0
    absent_count = max(total_students - present_count, 0)

    return {
        'present':        present_count,
        'absent':         absent_count,
        'percentage':     _pct(present_count, total_students),
        'total_students': total_students,
    }


def get_defaulters_list(class_id: str, threshold: int = 75) -> list:
    """Students with attendance below *threshold*%."""
    db = get_db()
    sessions = list(db['sessions'].find({
        'class_id': ObjectId(class_id),
        'status': 'completed',
    }))

    if not sessions:
        return []

    total_classes = len(sessions)
    session_ids = [s['_id'] for s in sessions]
    all_students = list(db['students'].find({}))

    defaulters = []
    for student in all_students:
        sid = str(student['_id'])
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
    """Per-student attendance report for a calendar month."""
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
    all_students = list(db['students'].find({}))

    student_reports = []
    for student in all_students:
        sid = str(student['_id'])
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
    return {'total_classes': total_classes, 'month': month, 'year': year, 'students': student_reports}


def get_student_report(student_id: str) -> dict:
    """Overall attendance breakdown for one student across all subjects."""
    db = get_db()
    student_logs = list(db['attendance_logs'].find({'student_id': student_id}))

    if not student_logs:
        return {'overall_percentage': 0, 'subjects': []}

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
