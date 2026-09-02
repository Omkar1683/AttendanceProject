"""
services/student_module_service.py
-----------------------------------
Business logic for student-module-specific features:
  - Notifications (student-facing queries)
  - Analytics aggregation (monthly trends, weekly patterns)
  - CSV/PDF export
  - Enrolled classes lookup

This is a NEW file — it does NOT modify any existing service.
"""
import io
import calendar
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from bson import ObjectId
from database.connection import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(present: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((present / total) * 100, 2)


def _resolve_student_doc(user_id: str) -> Optional[Dict]:
    """Find the students collection doc for a given users._id."""
    db = get_db()
    student = db['students'].find_one({'user_id': ObjectId(user_id)})
    if not student:
        user = db['users'].find_one({'_id': ObjectId(user_id)})
        if user and user.get('email'):
            student = db['students'].find_one({'email': user['email'].lower()})
    return student


def _get_student_id(user_id: str) -> Optional[str]:
    """Resolve users._id → students._id string."""
    student = _resolve_student_doc(user_id)
    return str(student['_id']) if student else None


def _student_id_variants(student_id: str) -> list:
    """Return both string and ObjectId variants for querying attendance_logs."""
    variants = [student_id]
    try:
        variants.append(ObjectId(student_id))
    except Exception:
        pass
    return variants


# ── Enrolled Classes ──────────────────────────────────────────────────────────

def get_enrolled_classes(user_id: str) -> list:
    """
    Find all classes where this student is enrolled.
    Classes store enrolled students in the 'students' array field.
    """
    db = get_db()
    student_id = _get_student_id(user_id)
    if not student_id:
        return []

    classes = list(db['classes'].find({
        'students': ObjectId(student_id)
    }))

    # Also look up teacher names
    teacher_ids = list({c['teacher_id'] for c in classes})
    teachers_map = {
        str(u['_id']): u.get('name', 'Unknown')
        for u in db['users'].find({'_id': {'$in': teacher_ids}})
    }

    return [
        {
            'class_id': str(c['_id']),
            'name': c.get('name', 'Unknown'),
            'code': c.get('code', ''),
            'batch': c.get('batch', ''),
            'department': c.get('department', ''),
            'teacher_name': teachers_map.get(str(c['teacher_id']), 'Unknown'),
            'total_students': c.get('total_students', 0),
        }
        for c in classes
    ]


# ── Notifications ─────────────────────────────────────────────────────────────

def get_student_notifications(user_id: str, page: int = 1, limit: int = 30) -> dict:
    """
    Fetch notifications relevant to the authenticated student.
    A notification is relevant if:
      - It targets a class the student is enrolled in, AND
      - target is 'all', OR
      - target is 'defaulters' or 'critical' (student may be a defaulter), OR
      - target is 'individual' AND student_id matches OR email in recipients
    """
    db = get_db()
    student = _resolve_student_doc(user_id)
    if not student:
        return {'total': 0, 'page': page, 'limit': limit, 'notifications': [], 'unread': 0}

    student_id = str(student['_id'])
    student_email = student.get('email', '')

    # Find classes the student is in
    enrolled = list(db['classes'].find({'students': ObjectId(student_id)}))
    class_ids = [c['_id'] for c in enrolled]

    if not class_ids:
        return {'total': 0, 'page': page, 'limit': limit, 'notifications': [], 'unread': 0}

    # Build query: notifications for enrolled classes
    query = {
        'class_id': {'$in': class_ids},
        '$or': [
            {'target': 'all'},
            {'target': 'defaulters'},
            {'target': 'critical'},
            {'target': 'individual', 'student_id': ObjectId(student_id)},
            {'target': 'individual', 'recipients': student_email},
        ],
    }

    total = db['notifications'].count_documents(query)
    unread = db['notifications'].count_documents({**query, 'read': False})

    skip = (page - 1) * limit
    notifs = list(
        db['notifications']
        .find(query)
        .sort('created_at', -1)
        .skip(skip)
        .limit(limit)
    )

    # Build class name map
    class_map = {str(c['_id']): c.get('name', 'Class') for c in enrolled}

    # Build teacher name map
    teacher_ids = list({c.get('teacher_id') for c in enrolled})
    teacher_map = {
        str(u['_id']): u.get('name', 'Unknown')
        for u in db['users'].find({'_id': {'$in': teacher_ids}})
    }
    class_teacher_map = {
        str(c['_id']): teacher_map.get(str(c.get('teacher_id')), 'Unknown')
        for c in enrolled
    }

    results = []
    for n in notifs:
        cid = str(n.get('class_id', ''))
        results.append({
            'id': str(n['_id']),
            'message': n.get('message', ''),
            'target': n.get('target', ''),
            'class_name': class_map.get(cid, 'Unknown'),
            'teacher_name': class_teacher_map.get(cid, 'Unknown'),
            'created_at': n['created_at'].isoformat() if n.get('created_at') else None,
            'read': n.get('read', False),
        })

    return {
        'total': total,
        'page': page,
        'limit': limit,
        'unread': unread,
        'notifications': results,
    }


def mark_notification_read(notification_id: str, user_id: str) -> dict:
    """Mark a notification as read. Validates that the student has access."""
    db = get_db()

    try:
        notif = db['notifications'].find_one({'_id': ObjectId(notification_id)})
    except Exception:
        return {'ok': False, 'message': 'Invalid notification ID', 'code': 400}

    if not notif:
        return {'ok': False, 'message': 'Notification not found', 'code': 404}

    # Verify student is in the notification's class
    student = _resolve_student_doc(user_id)
    if not student:
        return {'ok': False, 'message': 'Student not found', 'code': 404}

    class_doc = db['classes'].find_one({
        '_id': notif['class_id'],
        'students': student['_id'],
    })
    if not class_doc:
        return {'ok': False, 'message': 'Access denied', 'code': 403}

    db['notifications'].update_one(
        {'_id': ObjectId(notification_id)},
        {'$set': {'read': True}},
    )
    return {'ok': True}


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_student_analytics(user_id: str) -> dict:
    """
    Generate analytics data for the student:
      - monthly_trend: attendance % per month (last 6 months)
      - weekly_trend: attendance by day-of-week
      - present_absent: total present vs absent
    """
    db = get_db()
    student_id = _get_student_id(user_id)
    if not student_id:
        return {'monthly_trend': [], 'weekly_trend': [], 'present_absent': {'present': 0, 'absent': 0, 'total': 0}}

    ids_to_match = _student_id_variants(student_id)

    # Get all attendance logs for this student
    all_logs = list(db['attendance_logs'].find({'student_id': {'$in': ids_to_match}}))
    if not all_logs:
        return {'monthly_trend': [], 'weekly_trend': [], 'present_absent': {'present': 0, 'absent': 0, 'total': 0}}

    # Get session details for these logs
    session_ids = list({log['session_id'] for log in all_logs})
    sessions = list(db['sessions'].find({'_id': {'$in': session_ids}}))
    session_map = {str(s['_id']): s for s in sessions}

    # ── Monthly Trend (last 6 months) ────────────────────────────────────────
    now = datetime.now()
    monthly_trend = []
    for i in range(5, -1, -1):
        m_date = now.replace(day=1) - timedelta(days=i * 30)
        month = m_date.month
        year = m_date.year
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

        month_sessions = [s for s in sessions if start <= s.get('date', datetime.min) < end]
        month_session_ids = {str(s['_id']) for s in month_sessions}

        month_logs = [l for l in all_logs if str(l['session_id']) in month_session_ids]
        present = sum(1 for l in month_logs if l.get('status') == 'Present')
        total = len(month_logs)

        monthly_trend.append({
            'month': month,
            'year': year,
            'present': present,
            'total': total,
            'percentage': _pct(present, total),
        })

    # ── Weekly Trend ─────────────────────────────────────────────────────────
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly = {d: {'present': 0, 'total': 0} for d in day_names}

    for log in all_logs:
        session = session_map.get(str(log['session_id']))
        if session and session.get('date'):
            day_idx = session['date'].weekday()  # Monday=0
            day_name = day_names[day_idx]
            weekly[day_name]['total'] += 1
            if log.get('status') == 'Present':
                weekly[day_name]['present'] += 1

    weekly_trend = [
        {
            'day': d,
            'present': weekly[d]['present'],
            'total': weekly[d]['total'],
            'percentage': _pct(weekly[d]['present'], weekly[d]['total']),
        }
        for d in day_names
        if weekly[d]['total'] > 0
    ]

    # ── Present vs Absent ────────────────────────────────────────────────────
    total_present = sum(1 for l in all_logs if l.get('status') == 'Present')
    total_all = len(all_logs)

    return {
        'monthly_trend': monthly_trend,
        'weekly_trend': weekly_trend,
        'present_absent': {
            'present': total_present,
            'absent': total_all - total_present,
            'total': total_all,
        },
    }


# ── Export ─────────────────────────────────────────────────────────────────────

def export_student_csv(user_id: str) -> Optional[str]:
    """Generate a CSV string of the student's attendance data."""
    db = get_db()
    student = _resolve_student_doc(user_id)
    if not student:
        return None

    student_id = str(student['_id'])
    ids_to_match = _student_id_variants(student_id)

    logs = list(
        db['attendance_logs']
        .find({'student_id': {'$in': ids_to_match}})
        .sort('timestamp', -1)
    )

    if not logs:
        return None

    # Build lookup maps
    session_ids = list({l['session_id'] for l in logs})
    sessions = list(db['sessions'].find({'_id': {'$in': session_ids}}))
    session_map = {str(s['_id']): s for s in sessions}

    class_ids = list({s['class_id'] for s in sessions})
    classes = list(db['classes'].find({'_id': {'$in': class_ids}}))
    class_map = {str(c['_id']): c for c in classes}

    teacher_ids = list({c['teacher_id'] for c in classes})
    teacher_map = {
        str(u['_id']): u.get('name', 'Unknown')
        for u in db['users'].find({'_id': {'$in': teacher_ids}})
    }

    # Build CSV
    lines = ['Date,Time,Subject,Teacher,Status,Marked By']
    for log in logs:
        session = session_map.get(str(log['session_id']), {})
        cls = class_map.get(str(session.get('class_id', '')), {})
        teacher_name = teacher_map.get(str(cls.get('teacher_id', '')), 'Unknown')

        date_str = session.get('date', log.get('timestamp', datetime.min)).strftime('%Y-%m-%d')
        time_str = log.get('timestamp', datetime.min).strftime('%H:%M')
        subject = cls.get('name', 'Unknown')
        status = log.get('status', 'Unknown')
        marked_by = log.get('marked_by', 'Unknown')

        # Escape CSV fields
        lines.append(f'{date_str},{time_str},"{subject}","{teacher_name}",{status},{marked_by}')

    # Add summary section
    import services.analytics_service as aly_svc
    report = aly_svc.get_student_report(student_id)
    lines.append('')
    lines.append('--- ATTENDANCE SUMMARY ---')
    lines.append(f'Student Name,{student.get("name", "Unknown")}')
    lines.append(f'Roll Number,{student.get("roll_no", "N/A")}')
    lines.append(f'Overall Attendance,{report.get("overall_percentage", 0)}%')
    lines.append(f'Total Present,{report.get("total_present", 0)}')
    lines.append(f'Total Classes,{report.get("total_classes", 0)}')

    if report.get('subjects'):
        lines.append('')
        lines.append('Subject,Present,Total,Percentage,Status')
        for subj in report['subjects']:
            lines.append(
                f'"{subj.get("name", "")}",{subj.get("present", 0)},'
                f'{subj.get("total_classes", 0)},{subj.get("percentage", 0)}%,'
                f'{subj.get("status", "")}'
            )

    return '\n'.join(lines)


def export_student_pdf(user_id: str) -> Optional[bytes]:
    """Generate a PDF of the student's attendance data."""
    db = get_db()
    student = _resolve_student_doc(user_id)
    if not student:
        return None

    student_id = str(student['_id'])

    # Get report data
    import services.analytics_service as aly_svc
    report = aly_svc.get_student_report(student_id)

    # Build HTML for PDF
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; color: #333; }}
            h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
            h2 {{ color: #1f2937; margin-top: 30px; }}
            .info {{ margin-bottom: 20px; }}
            .info p {{ margin: 4px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #f3f4f6; font-weight: bold; }}
            .good {{ color: #16a34a; font-weight: bold; }}
            .bad {{ color: #dc2626; font-weight: bold; }}
            .summary {{ background: #eff6ff; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .footer {{ margin-top: 30px; font-size: 11px; color: #9ca3af; border-top: 1px solid #e5e7eb; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>AttendAI — Attendance Report</h1>
        <div class="info">
            <p><strong>Student:</strong> {student.get('name', 'Unknown')}</p>
            <p><strong>Roll No:</strong> {student.get('roll_no', 'N/A')}</p>
            <p><strong>Email:</strong> {student.get('email', 'N/A')}</p>
            <p><strong>Department:</strong> {student.get('department', 'N/A')}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%d %B %Y, %I:%M %p')}</p>
        </div>

        <div class="summary">
            <h2 style="margin-top:0">Overall Attendance</h2>
            <p><strong>Percentage:</strong> <span class="{'good' if report.get('overall_percentage', 0) >= 75 else 'bad'}">{report.get('overall_percentage', 0)}%</span></p>
            <p><strong>Present:</strong> {report.get('total_present', 0)} / {report.get('total_classes', 0)} classes</p>
        </div>

        <h2>Subject-wise Breakdown</h2>
        <table>
            <tr>
                <th>Subject</th>
                <th>Present</th>
                <th>Total</th>
                <th>Absent</th>
                <th>Percentage</th>
                <th>Status</th>
            </tr>
    """
    for subj in report.get('subjects', []):
        pct = subj.get('percentage', 0)
        cls = 'good' if pct >= 75 else 'bad'
        html += f"""
            <tr>
                <td>{subj.get('name', 'Unknown')}</td>
                <td>{subj.get('present', 0)}</td>
                <td>{subj.get('total_classes', 0)}</td>
                <td>{subj.get('absent', 0)}</td>
                <td class="{cls}">{pct}%</td>
                <td class="{cls}">{subj.get('status', '')}</td>
            </tr>
        """

    html += """
        </table>
        <div class="footer">
            <p>This report was automatically generated by AttendAI. Data is based on recorded attendance sessions.</p>
        </div>
    </body>
    </html>
    """

    # Convert HTML to PDF using a simple approach
    # We'll return the HTML and let the client render as PDF, OR
    # use a server-side approach if available
    try:
        # Try using weasyprint if available
        import weasyprint
        pdf_bytes = weasyprint.HTML(string=html).write_pdf()
        return pdf_bytes
    except ImportError:
        # Fall back to returning HTML (client will handle PDF generation)
        return html.encode('utf-8')
