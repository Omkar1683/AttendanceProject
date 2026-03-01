"""
MongoDB Models and Schema Validation for AttendAI System

This module provides:
- JSON Schema validation for all collections
- Model classes with CRUD operations
- Helper methods for common database operations
- Data validation and sanitization
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
    
    def find_many(self, query: Dict = {}, skip: int = 0, limit: int = 0, projection: Dict = None) -> List[Dict]:
        """Find multiple documents"""
        cursor = self.collection.find(query, projection)
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
    """User model for authentication and profiles"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["email", "password", "role", "name", "created_at"],
            "properties": {
                "email": {
                    "bsonType": "string",
                    "pattern": "^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$",
                    "description": "User email address (unique)"
                },
                "password": {
                    "bsonType": "string",
                    "minLength": 8,
                    "description": "Hashed password (bcrypt)"
                },
                "role": {
                    "enum": ["teacher", "student"],
                    "description": "User role"
                },
                "name": {
                    "bsonType": "string",
                    "minLength": 2,
                    "description": "Full name"
                },
                "department": {
                    "bsonType": "string",
                    "description": "Department/course"
                },
                "roll_no": {
                    "bsonType": "string",
                    "description": "Student roll number (for students)"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Account creation timestamp"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'users')
    
    def create_user(self, email: str, password_hash: str, role: str, 
                   name: str, department: str = None, roll_no: str = None) -> Optional[str]:
        """Create a new user"""
        user_doc = {
            'email': email.lower(),
            'password': password_hash,
            'role': role,
            'name': name,
            'created_at': datetime.now()
        }
        
        if department:
            user_doc['department'] = department
        if roll_no and role == 'student':
            user_doc['roll_no'] = roll_no
        
        return self.insert_one(user_doc)
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        """Find user by email"""
        return self.find_one({'email': email.lower()})
    
    def find_by_role(self, role: str) -> List[Dict]:
        """Find all users with specific role"""
        return self.find_many({'role': role})
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        return bool(re.match(pattern, email))


class StudentModel(BaseModel):
    """Student model for face recognition and registration"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "roll_no", "encoding", "created_at"],
            "properties": {
                "name": {
                    "bsonType": "string",
                    "minLength": 2,
                    "description": "Student full name"
                },
                "roll_no": {
                    "bsonType": "string",
                    "description": "Student roll number"
                },
                "encoding": {
                    "bsonType": "array",
                    "minItems": 128,
                    "maxItems": 128,
                    "items": {
                        "bsonType": "double"
                    },
                    "description": "128-dimensional face encoding"
                },
                "email": {
                    "bsonType": "string",
                    "description": "Student email"
                },
                "phone": {
                    "bsonType": "string",
                    "description": "Contact number"
                },
                "department": {
                    "bsonType": "string",
                    "description": "Department"
                },
                "batch": {
                    "bsonType": "string",
                    "description": "Batch/section"
                },
                "user_id": {
                    "bsonType": "objectId",
                    "description": "Reference to users collection"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Registration timestamp"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'students')
    
    def create_student(self, name: str, roll_no: str, encoding: List[float],
                      email: str = None, phone: str = None, 
                      department: str = None, batch: str = None,
                      user_id: str = None) -> Optional[str]:
        """Register a new student with face encoding"""
        if len(encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")
        
        student_doc = {
            'name': name,
            'roll_no': roll_no,
            'encoding': encoding,
            'created_at': datetime.now()
        }
        
        if email:
            student_doc['email'] = email.lower()
        if phone:
            student_doc['phone'] = phone
        if department:
            student_doc['department'] = department
        if batch:
            student_doc['batch'] = batch
        if user_id:
            student_doc['user_id'] = ObjectId(user_id)
        
        return self.insert_one(student_doc)
    
    def find_by_roll_no(self, roll_no: str) -> Optional[Dict]:
        """Find student by roll number"""
        return self.find_one({'roll_no': roll_no})
    
    def find_by_batch(self, batch: str) -> List[Dict]:
        """Find all students in a batch"""
        return self.find_many({'batch': batch})
    
    def get_all_encodings(self) -> List[Dict]:
        """Get all student face encodings for recognition"""
        return self.find_many({}, projection={'name': 1, 'encoding': 1})
    
    def update_encoding(self, student_id: str, new_encoding: List[float]) -> bool:
        """Update student's face encoding"""
        if len(new_encoding) != 128:
            raise ValueError("Face encoding must have exactly 128 dimensions")
        
        return self.update_one(
            {'_id': ObjectId(student_id)},
            {'$set': {'encoding': new_encoding}}
        )


class ClassModel(BaseModel):
    """Class/Subject model"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "code", "teacher_id", "total_students", "created_at"],
            "properties": {
                "name": {
                    "bsonType": "string",
                    "description": "Class/subject name"
                },
                "code": {
                    "bsonType": "string",
                    "description": "Unique class code"
                },
                "teacher_id": {
                    "bsonType": "objectId",
                    "description": "Reference to users collection"
                },
                "batch": {
                    "bsonType": "string",
                    "description": "Target batch"
                },
                "total_students": {
                    "bsonType": "int",
                    "minimum": 1,
                    "description": "Expected enrollment"
                },
                "department": {
                    "bsonType": "string",
                    "description": "Department"
                },
                "schedule": {
                    "bsonType": "string",
                    "description": "Class timings"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Creation timestamp"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'classes')
    
    def create_class(self, name: str, code: str, teacher_id: str,
                    total_students: int, batch: str = None,
                    department: str = None, schedule: str = None) -> Optional[str]:
        """Create a new class"""
        class_doc = {
            'name': name,
            'code': code.upper(),
            'teacher_id': ObjectId(teacher_id),
            'total_students': total_students,
            'created_at': datetime.now()
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
        return self.find_many({'teacher_id': ObjectId(teacher_id)})
    
    def find_by_batch(self, batch: str) -> List[Dict]:
        """Find all classes for a batch"""
        return self.find_many({'batch': batch})
    
    def update_enrollment(self, class_id: str, total_students: int) -> bool:
        """Update total student count"""
        return self.update_one(
            {'_id': ObjectId(class_id)},
            {'$set': {'total_students': total_students}}
        )


class SessionModel(BaseModel):
    """Attendance session model"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["class_id", "teacher_id", "date", "started_at", "status", "total_scanned"],
            "properties": {
                "class_id": {
                    "bsonType": "objectId",
                    "description": "Reference to classes collection"
                },
                "teacher_id": {
                    "bsonType": "objectId",
                    "description": "Reference to users collection"
                },
                "date": {
                    "bsonType": "date",
                    "description": "Session date"
                },
                "started_at": {
                    "bsonType": "date",
                    "description": "Start timestamp"
                },
                "ended_at": {
                    "bsonType": "date",
                    "description": "End timestamp"
                },
                "status": {
                    "enum": ["active", "completed"],
                    "description": "Session status"
                },
                "total_scanned": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "Present count"
                },
                "location": {
                    "bsonType": "string",
                    "description": "Classroom/location"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'sessions')
    
    def create_session(self, class_id: str, teacher_id: str, 
                      location: str = "Classroom") -> Optional[str]:
        """Create a new attendance session"""
        # Check for active session
        active_session = self.find_one({
            'class_id': ObjectId(class_id),
            'status': 'active'
        })
        
        if active_session:
            return str(active_session['_id'])
        
        session_doc = {
            'class_id': ObjectId(class_id),
            'teacher_id': ObjectId(teacher_id),
            'date': datetime.now(),
            'started_at': datetime.now(),
            'status': 'active',
            'total_scanned': 0,
            'location': location
        }
        
        return self.insert_one(session_doc)
    
    def end_session(self, session_id: str) -> bool:
        """Mark session as completed"""
        return self.update_one(
            {'_id': ObjectId(session_id)},
            {'$set': {
                'status': 'completed',
                'ended_at': datetime.now()
            }}
        )
    
    def increment_scanned(self, session_id: str) -> bool:
        """Increment total scanned count"""
        result = self.collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$inc': {'total_scanned': 1}}
        )
        return result.modified_count > 0
    
    def find_active_session(self, class_id: str) -> Optional[Dict]:
        """Find active session for a class"""
        return self.find_one({
            'class_id': ObjectId(class_id),
            'status': 'active'
        })
    
    def find_by_class(self, class_id: str, status: str = None) -> List[Dict]:
        """Find sessions for a class"""
        query = {'class_id': ObjectId(class_id)}
        if status:
            query['status'] = status
        
        return self.find_many(query)
    
    def find_by_date_range(self, class_id: str, start_date: datetime, 
                          end_date: datetime) -> List[Dict]:
        """Find sessions in date range"""
        return self.find_many({
            'class_id': ObjectId(class_id),
            'date': {'$gte': start_date, '$lt': end_date}
        })


class AttendanceLogModel(BaseModel):
    """Attendance log model"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "student_id", "student_name", "timestamp", "status", "marked_by"],
            "properties": {
                "session_id": {
                    "bsonType": "objectId",
                    "description": "Reference to sessions collection"
                },
                "student_id": {
                    "bsonType": "string",
                    "description": "Reference to students collection"
                },
                "student_name": {
                    "bsonType": "string",
                    "description": "Student name (denormalized)"
                },
                "timestamp": {
                    "bsonType": "date",
                    "description": "Attendance mark time"
                },
                "status": {
                    "enum": ["Present", "Absent"],
                    "description": "Attendance status"
                },
                "marked_by": {
                    "enum": ["AI", "Manual"],
                    "description": "Marking method"
                },
                "confidence": {
                    "bsonType": "double",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "AI confidence score"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'attendance_logs')
    
    def mark_attendance(self, session_id: str, student_id: str, 
                       student_name: str, status: str = "Present",
                       marked_by: str = "AI", confidence: float = None) -> Optional[str]:
        """Mark attendance for a student"""
        # Check for existing record
        existing = self.find_one({
            'session_id': ObjectId(session_id),
            'student_id': student_id
        })
        
        if existing:
            # Update existing record
            update_data = {
                'status': status,
                'timestamp': datetime.now(),
                'marked_by': marked_by
            }
            if confidence is not None:
                update_data['confidence'] = confidence
            
            self.update_one(
                {'_id': existing['_id']},
                {'$set': update_data}
            )
            return str(existing['_id'])
        
        # Create new record
        log_doc = {
            'session_id': ObjectId(session_id),
            'student_id': student_id,
            'student_name': student_name,
            'timestamp': datetime.now(),
            'status': status,
            'marked_by': marked_by
        }
        
        if confidence is not None:
            log_doc['confidence'] = confidence
        
        return self.insert_one(log_doc)
    
    def find_by_session(self, session_id: str) -> List[Dict]:
        """Get all attendance records for a session"""
        return self.find_many({'session_id': ObjectId(session_id)})
    
    def find_by_student(self, student_id: str) -> List[Dict]:
        """Get attendance history for a student"""
        return self.find_many({'student_id': student_id})
    
    def count_present_in_session(self, session_id: str) -> int:
        """Count present students in a session"""
        return self.count({
            'session_id': ObjectId(session_id),
            'status': 'Present'
        })
    
    def delete_attendance(self, session_id: str, student_id: str) -> bool:
        """Remove attendance record"""
        return self.delete_one({
            'session_id': ObjectId(session_id),
            'student_id': student_id
        })


class NotificationModel(BaseModel):
    """Notification model"""
    
    SCHEMA = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["class_id", "target", "message", "sent_by", "created_at"],
            "properties": {
                "class_id": {
                    "bsonType": "objectId",
                    "description": "Reference to classes collection"
                },
                "target": {
                    "enum": ["all", "defaulters", "critical"],
                    "description": "Target audience"
                },
                "message": {
                    "bsonType": "string",
                    "minLength": 1,
                    "description": "Notification text"
                },
                "sent_by": {
                    "bsonType": "objectId",
                    "description": "Reference to users collection"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Creation timestamp"
                },
                "read": {
                    "bsonType": "bool",
                    "description": "Read status"
                }
            }
        }
    }
    
    def __init__(self, db):
        super().__init__(db, 'notifications')
    
    def create_notification(self, class_id: str, target: str, 
                          message: str, sent_by: str) -> Optional[str]:
        """Create a new notification"""
        notif_doc = {
            'class_id': ObjectId(class_id),
            'target': target,
            'message': message,
            'sent_by': ObjectId(sent_by),
            'created_at': datetime.now(),
            'read': False
        }
        
        return self.insert_one(notif_doc)
    
    def find_by_class(self, class_id: str, limit: int = 50) -> List[Dict]:
        """Get notifications for a class"""
        return list(
            self.collection.find({'class_id': ObjectId(class_id)})
            .sort('created_at', DESCENDING)
            .limit(limit)
        )
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        return self.update_one(
            {'_id': ObjectId(notification_id)},
            {'$set': {'read': True}}
        )


def setup_indexes(db):
    """Create all required indexes"""
    print("Creating database indexes...")
    
    # Users indexes
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.users.create_index([("role", ASCENDING)])
    print("✅ Created users indexes")
    
    # Students indexes
    db.students.create_index([("roll_no", ASCENDING)])
    db.students.create_index([("user_id", ASCENDING)])
    print("✅ Created students indexes")
    
    # Classes indexes
    db.classes.create_index([("code", ASCENDING)], unique=True)
    db.classes.create_index([("teacher_id", ASCENDING)])
    db.classes.create_index([("batch", ASCENDING)])
    print("✅ Created classes indexes")
    
    # Sessions indexes
    db.sessions.create_index([("class_id", ASCENDING), ("date", DESCENDING)])
    db.sessions.create_index([("status", ASCENDING)])
    db.sessions.create_index([("teacher_id", ASCENDING)])
    print("✅ Created sessions indexes")
    
    # Attendance logs indexes
    db.attendance_logs.create_index(
        [("session_id", ASCENDING), ("student_id", ASCENDING)], 
        unique=True
    )
    db.attendance_logs.create_index([("student_id", ASCENDING)])
    db.attendance_logs.create_index([("timestamp", DESCENDING)])
    print("✅ Created attendance_logs indexes")
    
    # Notifications indexes
    db.notifications.create_index([("class_id", ASCENDING), ("created_at", DESCENDING)])
    db.notifications.create_index([("target", ASCENDING)])
    print("✅ Created notifications indexes")
    
    print("\n✅ All indexes created successfully!")


def get_models(db):
    """Get all model instances"""
    return {
        'users': UserModel(db),
        'students': StudentModel(db),
        'classes': ClassModel(db),
        'sessions': SessionModel(db),
        'attendance_logs': AttendanceLogModel(db),
        'notifications': NotificationModel(db)
    }


if __name__ == '__main__':
    """Setup indexes when run as script"""
    from pymongo import MongoClient
    from urllib.parse import quote_plus
    
    print("=== AttendAI Database Models Setup ===\n")
    
    # Database Connection Configuration
    RAW_USERNAME = "devesh"
    RAW_PASSWORD = "Devesh_1234"
    
    username = quote_plus(RAW_USERNAME)
    password = quote_plus(RAW_PASSWORD)
    MONGO_URI = f"mongodb+srv://{username}:{password}@attendancecluster.uucwump.mongodb.net/?appName=AttendanceCluster"
    
    try:
        client = MongoClient(MONGO_URI)
        db = client['attendai_db']
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas\n")
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        db = None
    
    if db is None:
        print("❌ Failed to connect to database")
        exit(1)
    
    # Setup indexes
    setup_indexes(db)
    
    print("\n=== Setup Complete ===")

