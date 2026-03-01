"""
Simplified MongoDB Models for AttendAI (3-Collection Design)
Based on user-provided schema diagram with enhancements

Collections:
1. users - Unified authentication and student profiles with face encodings
2. classes - Course management with explicit student enrollment
3. attendance - Session-based attendance with embedded student records
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime
from typing import Optional, List, Dict, Any
import re


class BaseModel:
    """Base model class with common CRUD operations"""
    
    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection = db[collection_name]
        self.collection_name = collection_name
    
    def find_by_id(self, doc_id: str) -> Optional[Dict]:
        """Find document by ID"""
        try:
            return self.collection.find_one({'_id': ObjectId(doc_id)})
        except Exception as e:
            print(f"Error finding document: {e}")
            return None
    
    def find_one(self, query: Dict) -> Optional[Dict]:
        """Find single document by query"""
        return self.collection.find_one(query)
    
    def find_many(self, query: Dict = {}, skip: int = 0, limit: int = 0) -> List[Dict]:
        """Find multiple documents"""
        cursor = self.collection.find(query)
        if skip > 0:
            cursor = cursor.skip(skip)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def insert_one(self, document: Dict) -> Optional[str]:
        """Insert single document"""
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error inserting document: {e}")
            return None
    
    def update_one(self, query: Dict, update: Dict) -> bool:
        """Update single document"""
        try:
            result = self.collection.update_one(query, update)
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_one(self, query: Dict) -> bool:
        """Delete single document"""
        try:
            result = self.collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def count(self, query: Dict = {}) -> int:
        """Count documents matching query"""
        return self.collection.count_documents(query)


class UserModel(BaseModel):
    """Unified user model for teachers and students with face recognition"""
    
    def __init__(self, db):
        super().__init__(db, 'users')
    
    def create_user(self, email: str, password_hash: str, role: str, name: str,
                   department: str = None, roll_no: str = None, face_encoding: List[float] = None,
                   phone: str = None, batch: str = None) -> Optional[str]:
        """Create a new user (teacher or student)"""
        
        if face_encoding and len(face_encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")
        
        user_doc = {
            'email': email.lower(),
            'password': password_hash,
            'role': role,
            'name': name,
            'createdAt': datetime.now()
        }
        
        if department:
            user_doc['department'] = department
        
        # Student-specific fields
        if role == 'student':
            if roll_no:
                user_doc['rollNo'] = roll_no
            if face_encoding:
                user_doc['faceEncoding'] = face_encoding
            if phone:
                user_doc['phone'] = phone
            if batch:
                user_doc['batch'] = batch
        
        return self.insert_one(user_doc)
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        """Find user by email"""
        return self.find_one({'email': email.lower()})
    
    def find_by_role(self, role: str) -> List[Dict]:
        """Find all users with specific role"""
        return self.find_many({'role': role})
    
    def find_students_with_faces(self) -> List[Dict]:
        """Get all students who have face encodings"""
        return self.find_many({
            'role': 'student',
            'faceEncoding': {'$exists': True}
        })
    
    def update_face_encoding(self, user_id: str, face_encoding: List[float]) -> bool:
        """Update student's face encoding"""
        if len(face_encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")
        
        return self.update_one(
            {'_id': ObjectId(user_id), 'role': 'student'},
            {'$set': {'faceEncoding': face_encoding}}
        )
    
    def get_students_by_batch(self, batch: str) -> List[Dict]:
        """Get all students in a batch"""
        return self.find_many({'role': 'student', 'batch': batch})


class ClassModel(BaseModel):
    """Class model with student enrollment"""
    
    def __init__(self, db):
        super().__init__(db, 'classes')
    
    def create_class(self, name: str, code: str, teacher_id: str,
                    total_students: int, students: List[str] = None,
                    batch: str = None, department: str = None, 
                    schedule: str = None) -> Optional[str]:
        """Create a new class"""
        
        class_doc = {
            'name': name,
            'code': code.upper(),
            'teacher': ObjectId(teacher_id),
            'students': [ObjectId(sid) for sid in (students or [])],
            'totalStudents': total_students,
            'createdAt': datetime.now()
        }
        
        if batch:
            class_doc['batch'] = batch
        if department:
            class_doc['department'] = department
        if schedule:
            class_doc['schedule'] = schedule
        
        return self.insert_one(class_doc)
    
    def find_by_code(self, code: str) -> Optional[Dict]:
        """Find class by unique code"""
        return self.find_one({'code': code.upper()})
    
    def find_by_teacher(self, teacher_id: str) -> List[Dict]:
        """Find all classes taught by a teacher"""
        return self.find_many({'teacher': ObjectId(teacher_id)})
    
    def find_by_student(self, student_id: str) -> List[Dict]:
        """Find all classes a student is enrolled in"""
        return self.find_many({'students': ObjectId(student_id)})
    
    def enroll_student(self, class_id: str, student_id: str) -> bool:
        """Add student to class"""
        result = self.collection.update_one(
            {
                '_id': ObjectId(class_id),
                'students': {'$ne': ObjectId(student_id)}  # Prevent duplicates
            },
            {
                '$push': {'students': ObjectId(student_id)},
                '$inc': {'totalStudents': 1}
            }
        )
        return result.modified_count > 0
    
    def unenroll_student(self, class_id: str, student_id: str) -> bool:
        """Remove student from class"""
        result = self.collection.update_one(
            {'_id': ObjectId(class_id)},
            {
                '$pull': {'students': ObjectId(student_id)},
                '$inc': {'totalStudents': -1}
            }
        )
        return result.modified_count > 0
    
    def get_enrolled_students(self, class_id: str) -> List[ObjectId]:
        """Get list of enrolled student IDs"""
        cls = self.find_by_id(class_id)
        return cls.get('students', []) if cls else []


class AttendanceModel(BaseModel):
    """Attendance model with embedded session records"""
    
    def __init__(self, db):
        super().__init__(db, 'attendance')
    
    def create_session(self, class_id: str, location: str = "Classroom") -> Optional[str]:
        """Create or resume today's attendance session"""
        
        # Set date to today at midnight
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check for existing session today
        existing = self.find_one({
            'classId': ObjectId(class_id),
            'date': today
        })
        
        if existing:
            # If session is completed, reopen it
            if existing.get('status') == 'completed':
                self.update_one(
                    {'_id': existing['_id']},
                    {'$set': {'status': 'active'}}
                )
            return str(existing['_id'])
        
        # Create new session
        session_doc = {
            'classId': ObjectId(class_id),
            'date': today,
            'startedAt': datetime.now(),
            'location': location,
            'status': 'active',
            'records': [],
            'totalPresent': 0,
            'createdAt': datetime.now()
        }
        
        return self.insert_one(session_doc)
    
    def end_session(self, session_id: str) -> bool:
        """Mark session as completed"""
        return self.update_one(
            {'_id': ObjectId(session_id)},
            {'$set': {
                'status': 'completed',
                'endedAt': datetime.now()
            }}
        )
    
    def mark_attendance(self, session_id: str, student_id: str, student_name: str,
                       status: str = "present", marked_by: str = "AI",
                       confidence: float = None) -> bool:
        """Mark or update attendance for a student"""
        
        # Check if student already marked
        session = self.find_by_id(session_id)
        if not session:
            return False
        
        existing_record = next(
            (r for r in session.get('records', []) if str(r['student']) == student_id),
            None
        )
        
        record = {
            'student': ObjectId(student_id),
            'studentName': student_name,
            'status': status,
            'markedBy': marked_by,
            'timestamp': datetime.now()
        }
        
        if confidence is not None:
            record['confidence'] = confidence
        
        if existing_record:
            # Update existing record
            result = self.collection.update_one(
                {
                    '_id': ObjectId(session_id),
                    'records.student': ObjectId(student_id)
                },
                {'$set': {'records.$': record}}
            )
        else:
            # Add new record
            result = self.collection.update_one(
                {'_id': ObjectId(session_id)},
                {'$push': {'records': record}}
            )
        
        # Recalculate totalPresent
        self._update_total_present(session_id)
        
        return result.modified_count > 0
    
    def _update_total_present(self, session_id: str):
        """Recalculate and update totalPresent count"""
        session = self.find_by_id(session_id)
        if session:
            present_count = sum(
                1 for r in session.get('records', [])
                if r.get('status') == 'present'
            )
            self.update_one(
                {'_id': ObjectId(session_id)},
                {'$set': {'totalPresent': present_count}}
            )
    
    def get_todays_session(self, class_id: str) -> Optional[Dict]:
        """Get today's attendance session for a class"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.find_one({
            'classId': ObjectId(class_id),
            'date': today
        })
    
    def get_student_attendance(self, student_id: str, class_id: str = None) -> List[Dict]:
        """Get attendance history for a student"""
        query = {'records.student': ObjectId(student_id)}
        
        if class_id:
            query['classId'] = ObjectId(class_id)
        
        # Use aggregation to extract only relevant record
        pipeline = [
            {'$match': query},
            {'$project': {
                'classId': 1,
                'date': 1,
                'record': {
                    '$filter': {
                        'input': '$records',
                        'as': 'rec',
                        'cond': {'$eq': ['$$rec.student', ObjectId(student_id)]}
                    }
                }
            }},
            {'$unwind': '$record'},
            {'$sort': {'date': -1}}
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def get_class_attendance_summary(self, class_id: str, start_date: datetime = None,
                                    end_date: datetime = None) -> Dict:
        """Get attendance summary for a class over a date range"""
        query = {
            'classId': ObjectId(class_id),
            'status': 'completed'
        }
        
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lt'] = end_date
            query['date'] = date_filter
        
        sessions = self.find_many(query)
        total_sessions = len(sessions)
        
        # Calculate per-student attendance
        student_stats = {}
        
        for session in sessions:
            for record in session.get('records', []):
                sid = str(record['student'])
                if sid not in student_stats:
                    student_stats[sid] = {
                        'name': record['studentName'],
                        'present': 0,
                        'total': 0
                    }
                student_stats[sid]['total'] += 1
                if record['status'] == 'present':
                    student_stats[sid]['present'] += 1
        
        # Calculate percentages
        for sid, stats in student_stats.items():
            stats['percentage'] = round(
                (stats['present'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                2
            )
        
        return {
            'totalSessions': total_sessions,
            'students': student_stats
        }


def setup_indexes(db):
    """Create all required indexes"""
    print("Creating database indexes for simplified schema...")
    
    # Users indexes
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.users.create_index([("role", ASCENDING)])
    db.users.create_index([("rollNo", ASCENDING)])
    db.users.create_index([("batch", ASCENDING)])
    print("✅ Created users indexes")
    
    # Classes indexes
    db.classes.create_index([("code", ASCENDING)], unique=True)
    db.classes.create_index([("teacher", ASCENDING)])
    db.classes.create_index([("batch", ASCENDING)])
    db.classes.create_index([("students", ASCENDING)])
    print("✅ Created classes indexes")
    
    # Attendance indexes
    db.attendance.create_index([("classId", ASCENDING), ("date", DESCENDING)])
    db.attendance.create_index([("classId", ASCENDING), ("date", ASCENDING)], unique=True)
    db.attendance.create_index([("status", ASCENDING)])
    db.attendance.create_index([("records.student", ASCENDING)])
    print("✅ Created attendance indexes")
    
    print("\n✅ All indexes created successfully!")


def get_models(db):
    """Get all model instances"""
    return {
        'users': UserModel(db),
        'classes': ClassModel(db),
        'attendance': AttendanceModel(db)
    }


if __name__ == '__main__':
    """Setup indexes when run as script"""
    from db_utils import get_db_connection
    
    print("=== AttendAI Simplified Models Setup ===\n")
    
    db = get_db_connection()
    if db is None:
        print("❌ Failed to connect to database")
        exit(1)
    
    # Setup indexes
    setup_indexes(db)
    
    print("\n=== Setup Complete ===")
    print("\nThis is the SIMPLIFIED 3-collection schema.")
    print("Use 'models_simple.py' for this design.")
    print("Use 'models.py' for the original 6-collection design.")
