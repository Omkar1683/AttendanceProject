"""
services/auth_service.py
------------------------
Business logic for user authentication and registration.
Route handlers call these functions; they never touch Flask request/response objects.
"""
from models import get_models
from core.security import hash_password, verify_password, generate_token
from database.connection import get_db


def login(email: str, password: str) -> dict:
    """
    Validate credentials and return a token payload.

    Returns:
        {'ok': True,  'token': str, 'user': dict}  on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    db = get_db()
    models = get_models(db)
    user = models['users'].find_by_email(email)

    if not user:
        return {'ok': False, 'message': 'User not found', 'code': 404}

    if not verify_password(password, user['password']):
        return {'ok': False, 'message': 'Invalid password', 'code': 401}

    token = generate_token(str(user['_id']), user['email'], user['role'])
    return {
        'ok': True,
        'token': token,
        'user': {
            'id':    str(user['_id']),
            'name':  user['name'],
            'email': user['email'],
            'role':  user['role'],
        },
    }


def signup(data: dict) -> dict:
    """
    Register a new user (teacher or student).

    Returns:
        {'ok': True,  'user_id': str}              on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    db = get_db()
    models = get_models(db)

    required = ['email', 'password', 'name', 'role']
    if not all(f in data for f in required):
        return {'ok': False, 'message': 'Missing required fields', 'code': 400}

    if models['users'].find_by_email(data['email']):
        return {'ok': False, 'message': 'Email already registered', 'code': 409}

    user_id = models['users'].create_user(
        email=data['email'],
        password_hash=hash_password(data['password']),
        role=data['role'],
        name=data['name'],
        department=data.get('department'),
        roll_no=data.get('roll_no') if data['role'] == 'student' else None,
    )

    if user_id:
        return {'ok': True, 'user_id': user_id}
    return {'ok': False, 'message': 'Registration failed', 'code': 500}


def register_student(data: dict) -> dict:
    """
    Register a student with face encoding (teacher-only action).

    Returns:
        {'ok': True,  'student_id': str}           on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    db = get_db()
    models = get_models(db)

    required = ['name', 'roll_no', 'encoding']
    if not all(f in data for f in required):
        return {'ok': False, 'message': 'Missing required fields', 'code': 400}

    if len(data['encoding']) != 128:
        return {'ok': False, 'message': 'Face encoding must be 128 dimensions', 'code': 400}

    if models['students'].find_by_roll_no(data['roll_no']):
        return {'ok': False, 'message': 'Student with this roll number already exists', 'code': 409}

    student_id = models['students'].create_student(
        name=data['name'],
        roll_no=data['roll_no'],
        encoding=data['encoding'],
        email=data.get('email'),
        phone=data.get('phone'),
        department=data.get('department'),
        batch=data.get('batch'),
        user_id=data.get('user_id'),
    )

    if student_id:
        return {'ok': True, 'student_id': student_id}
    return {'ok': False, 'message': 'Registration failed', 'code': 500}


def get_classes(teacher_id: str = None) -> list:
    """Return class list, optionally filtered by teacher."""
    db = get_db()
    models = get_models(db)

    if teacher_id:
        classes = models['classes'].find_by_teacher(teacher_id)
    else:
        classes = models['classes'].find_many({})

    return [
        {
            'id':             str(c['_id']),
            'name':           c['name'],
            'code':           c.get('code', ''),
            'batch':          c.get('batch', ''),
            'department':     c.get('department', ''),
            'schedule':       c.get('schedule', ''),
            'total_students': c.get('total_students', 0),
        }
        for c in classes
    ]


def create_class(
    teacher_id: str,
    name: str,
    code: str,
    total_students: int = 1,
    batch: str = None,
    department: str = None,
    schedule: str = None,
) -> dict:
    """
    Create a new class/subject linked to the teacher.

    Returns:
        {'ok': True,  'class_id': str}             on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    if not teacher_id:
        return {'ok': False, 'message': 'Teacher ID required', 'code': 401}
    if not name or not code:
        return {'ok': False, 'message': 'name and code are required', 'code': 400}

    db = get_db()
    models = get_models(db)

    # Prevent duplicate class codes
    if models['classes'].find_by_code(code):
        return {'ok': False, 'message': f'A class with code "{code.upper()}" already exists', 'code': 409}

    class_id = models['classes'].create_class(
        name=name,
        code=code,
        teacher_id=teacher_id,
        total_students=int(total_students),
        batch=batch,
        department=department,
        schedule=schedule,
    )

    if class_id:
        return {'ok': True, 'class_id': class_id}
    return {'ok': False, 'message': 'Failed to create class', 'code': 500}
