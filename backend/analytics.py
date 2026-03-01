"""
Analytics utilities for attendance calculations, defaulter detection, and reporting
"""
from datetime import datetime, timedelta
import pandas as pd
from bson import ObjectId
import io

def calculate_attendance_percentage(present_count, total_classes):
    """Calculate attendance percentage"""
    if total_classes == 0:
        return 0.0
    return round((present_count / total_classes) * 100, 2)

def get_today_summary(db, class_id):
    """Get today's attendance summary for a class"""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    
    # Find sessions for today
    sessions_col = db['sessions']
    sessions = list(sessions_col.find({
        'class_id': ObjectId(class_id),
        'date': {'$gte': today, '$lt': tomorrow}
    }))
    
    if not sessions:
        return {
            'present': 0,
            'absent': 0,
            'percentage': 0,
            'total_students': 0
        }
    
    # Get attendance logs for today's sessions
    logs_col = db['attendance_logs']
    session_ids = [s['_id'] for s in sessions]
    
    present_count = logs_col.count_documents({
        'session_id': {'$in': session_ids},
        'status': 'Present'
    })
    
    # Get total students from class
    classes_col = db['classes']
    class_doc = classes_col.find_one({'_id': ObjectId(class_id)})
    total_students = class_doc.get('total_students', 0) if class_doc else 0
    
    absent_count = total_students - present_count
    percentage = calculate_attendance_percentage(present_count, total_students)
    
    return {
        'present': present_count,
        'absent': absent_count,
        'percentage': percentage,
        'total_students': total_students
    }

def get_defaulters_list(db, class_id, threshold=75):
    """Get list of students with attendance below threshold"""
    # Get all sessions for this class
    sessions_col = db['sessions']
    sessions = list(sessions_col.find({
        'class_id': ObjectId(class_id),
        'status': 'completed'
    }))
    
    if not sessions:
        return []
    
    total_classes = len(sessions)
    session_ids = [s['_id'] for s in sessions]
    
    # Get all students from class
    classes_col = db['classes']
    class_doc = classes_col.find_one({'_id': ObjectId(class_id)})
    
    # Get students collection
    students_col = db['students']
    all_students = list(students_col.find({}))
    
    # Calculate attendance for each student
    logs_col = db['attendance_logs']
    defaulters = []
    
    for student in all_students:
        student_id = str(student['_id'])
        
        # Count present classes
        present_count = logs_col.count_documents({
            'session_id': {'$in': session_ids},
            'student_id': student_id,
            'status': 'Present'
        })
        
        attendance_percentage = calculate_attendance_percentage(present_count, total_classes)
        
        if attendance_percentage < threshold:
            defaulters.append({
                'student_id': student_id,
                'name': student.get('name', 'Unknown'),
                'attendance': attendance_percentage,
                'present': present_count,
                'total': total_classes
            })
    
    # Sort by attendance (lowest first)
    defaulters.sort(key=lambda x: x['attendance'])
    
    return defaulters

def get_monthly_report(db, class_id, month, year):
    """Get attendance report for a specific month"""
    # Create date range for the month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # Get sessions for this month
    sessions_col = db['sessions']
    sessions = list(sessions_col.find({
        'class_id': ObjectId(class_id),
        'date': {'$gte': start_date, '$lt': end_date},
        'status': 'completed'
    }))
    
    if not sessions:
        return {
            'total_classes': 0,
            'students': []
        }
    
    total_classes = len(sessions)
    session_ids = [s['_id'] for s in sessions]
    
    # Get all students
    students_col = db['students']
    all_students = list(students_col.find({}))
    
    # Calculate attendance for each student
    logs_col = db['attendance_logs']
    student_reports = []
    
    for student in all_students:
        student_id = str(student['_id'])
        
        present_count = logs_col.count_documents({
            'session_id': {'$in': session_ids},
            'student_id': student_id,
            'status': 'Present'
        })
        
        attendance_percentage = calculate_attendance_percentage(present_count, total_classes)
        
        student_reports.append({
            'student_id': student_id,
            'name': student.get('name', 'Unknown'),
            'roll_no': student.get('roll_no', 'N/A'),
            'attendance': attendance_percentage,
            'present': present_count,
            'absent': total_classes - present_count,
            'total': total_classes,
            'status': 'Good' if attendance_percentage >= 75 else 'Defaulter'
        })
    
    # Sort by name
    student_reports.sort(key=lambda x: x['name'])
    
    return {
        'total_classes': total_classes,
        'month': month,
        'year': year,
        'students': student_reports
    }

def get_student_report(db, student_id):
    """Get overall attendance report for a student"""
    # Get all attendance logs for this student
    logs_col = db['attendance_logs']
    student_logs = list(logs_col.find({'student_id': student_id}))
    
    if not student_logs:
        return {
            'overall_percentage': 0,
            'subjects': []
        }
    
    # Group by class/subject
    sessions_col = db['sessions']
    classes_col = db['classes']
    
    # Get unique sessions
    session_ids = list(set([log['session_id'] for log in student_logs]))
    sessions = list(sessions_col.find({'_id': {'$in': session_ids}}))
    
    # Group by class
    class_stats = {}
    for session in sessions:
        class_id = str(session['class_id'])
        if class_id not in class_stats:
            class_stats[class_id] = {
                'total': 0,
                'present': 0
            }
        class_stats[class_id]['total'] += 1
    
    # Count present for each class
    for log in student_logs:
        session = next((s for s in sessions if s['_id'] == log['session_id']), None)
        if session and log['status'] == 'Present':
            class_id = str(session['class_id'])
            class_stats[class_id]['present'] += 1
    
    # Get class details and calculate percentages
    subjects = []
    total_present = 0
    total_classes = 0
    
    for class_id, stats in class_stats.items():
        class_doc = classes_col.find_one({'_id': ObjectId(class_id)})
        if class_doc:
            percentage = calculate_attendance_percentage(stats['present'], stats['total'])
            subjects.append({
                'class_id': class_id,
                'name': class_doc.get('name', 'Unknown'),
                'total_classes': stats['total'],
                'present': stats['present'],
                'absent': stats['total'] - stats['present'],
                'percentage': percentage,
                'status': 'Good' if percentage >= 75 else 'Defaulter'
            })
            total_present += stats['present']
            total_classes += stats['total']
    
    overall_percentage = calculate_attendance_percentage(total_present, total_classes)
    
    return {
        'overall_percentage': overall_percentage,
        'total_present': total_present,
        'total_classes': total_classes,
        'subjects': subjects
    }

def export_to_csv(report_data, class_name):
    """Export attendance report to CSV"""
    students = report_data.get('students', [])
    
    # Create DataFrame
    df = pd.DataFrame(students)
    
    # Reorder columns
    columns = ['name', 'roll_no', 'present', 'absent', 'total', 'attendance', 'status']
    df = df[columns]
    
    # Rename columns for better readability
    df.columns = ['Student Name', 'Roll No', 'Present', 'Absent', 'Total Classes', 'Attendance %', 'Status']
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    return csv_string

def get_subject_averages(db, teacher_id):
    """Get average attendance for all subjects taught by a teacher"""
    classes_col = db['classes']
    classes = list(classes_col.find({'teacher_id': ObjectId(teacher_id)}))
    
    subject_stats = []
    
    for class_doc in classes:
        class_id = str(class_doc['_id'])
        
        # Get all completed sessions
        sessions_col = db['sessions']
        sessions = list(sessions_col.find({
            'class_id': ObjectId(class_id),
            'status': 'completed'
        }))
        
        if not sessions:
            continue
        
        total_classes = len(sessions)
        session_ids = [s['_id'] for s in sessions]
        
        # Calculate average attendance
        logs_col = db['attendance_logs']
        total_students = class_doc.get('total_students', 0)
        
        if total_students > 0:
            total_present = logs_col.count_documents({
                'session_id': {'$in': session_ids},
                'status': 'Present'
            })
            
            # Average = total present / (total classes * total students)
            max_possible = total_classes * total_students
            average_percentage = round((total_present / max_possible) * 100, 2) if max_possible > 0 else 0
            
            subject_stats.append({
                'class_id': class_id,
                'name': class_doc.get('name', 'Unknown'),
                'code': class_doc.get('code', 'N/A'),
                'average': average_percentage,
                'total_classes': total_classes
            })
    
    return subject_stats
