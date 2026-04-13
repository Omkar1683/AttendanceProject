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
    # Enrich with student-specific fields (roll_no may live in users or students doc)
    user_id_str = str(user['_id'])
    from bson import ObjectId as _ObjId
    student_doc = db['students'].find_one({'user_id': _ObjId(user_id_str)})

    return {
        'ok': True,
        'token': token,
        'user': {
            'id':         user_id_str,
            'name':       user['name'],
            'email':      user['email'],
            'role':       user['role'],
            'department': user.get('department') or (student_doc.get('department', '') if student_doc else ''),
            'roll_no':    user.get('roll_no') or (student_doc.get('roll_no', '') if student_doc else ''),
            'batch':      student_doc.get('batch', '') if student_doc else '',
        },
    }


def signup(data: dict) -> dict:
    """
    Register a new TEACHER account only.
    Students are created exclusively via register_student().

    Returns:
        {'ok': True,  'user_id': str}              on success
        {'ok': False, 'message': str, 'code': int} on failure
    """
    db = get_db()
    models = get_models(db)

    # Reject student self-registration
    if data.get('role', '').lower() == 'student':
        return {
            'ok': False,
            'message': 'Student accounts cannot be created via signup. '
                       'Students are registered by their teacher.',
            'code': 403,
        }

    required = ['email', 'password', 'name']
    if not all(f in data and data[f] for f in required):
        return {'ok': False, 'message': 'email, password and name are required', 'code': 400}

    if len(data['password']) < 8:
        return {'ok': False, 'message': 'Password must be at least 8 characters', 'code': 400}

    if not models['users'].validate_email(data['email']):
        return {'ok': False, 'message': 'Invalid email format', 'code': 400}

    if models['users'].find_by_email(data['email']):
        return {'ok': False, 'message': 'Email already registered', 'code': 409}

    user_id = models['users'].create_user(
        email=data['email'],
        password_hash=hash_password(data['password']),
        role='teacher',          # Always teacher — hardcoded
        name=data['name'],
        department=data.get('department'),
    )

    if user_id:
        return {'ok': True, 'user_id': user_id}
    return {'ok': False, 'message': 'Registration failed', 'code': 500}


def register_student(data: dict) -> dict:
    """
    Register a student with face encoding (teacher-only action).

    Auto-creates an entry in the `users` collection so the student can log in
    immediately with the default password "12345678".

    Returns:
        {'ok': True,  'student_id': str, 'user_id': str}  on success
        {'ok': False, 'message': str, 'code': int}         on failure
    """
    db = get_db()
    models = get_models(db)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['name', 'roll_no', 'encoding', 'email']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return {
            'ok': False,
            'message': f'Missing required fields: {", ".join(missing)}',
            'code': 400,
        }

    if len(data['encoding']) != 128:
        return {'ok': False, 'message': 'Face encoding must be 128 dimensions', 'code': 400}

    email = data['email'].strip().lower()

    # Validate email format
    if not models['users'].validate_email(email):
        return {'ok': False, 'message': 'Invalid email format', 'code': 400}

    # ── Prevent duplicates ────────────────────────────────────────────────────
    if models['students'].find_by_roll_no(data['roll_no']):
        return {'ok': False, 'message': 'Student with this roll number already exists', 'code': 409}

    if db['students'].find_one({'email': email}):
        return {'ok': False, 'message': 'Student with this email already exists', 'code': 409}

    if models['users'].find_by_email(email):
        return {'ok': False, 'message': 'A user account already exists for this email', 'code': 409}

    # ── 1. Create users entry (so student can log in) ─────────────────────────
    DEFAULT_PASSWORD = '12345678'
    user_id = models['users'].create_user(
        email=email,
        password_hash=hash_password(DEFAULT_PASSWORD),
        role='student',
        name=data['name'],
        department=data.get('department'),
        roll_no=data['roll_no'],
    )
    if not user_id:
        return {'ok': False, 'message': 'Failed to create student login account', 'code': 500}

    # ── 2. Create students entry (face encoding + academic info) ──────────────
    student_id = models['students'].create_student(
        name=data['name'],
        roll_no=data['roll_no'],
        encoding=data['encoding'],
        email=email,
        phone=data.get('phone'),
        department=data.get('department'),
        batch=data.get('batch'),
        user_id=user_id,          # Link to the users document
    )

    if student_id:
        return {'ok': True, 'student_id': student_id, 'user_id': user_id}

    # Rollback: remove the users entry if student insert failed
    try:
        from bson import ObjectId
        db['users'].delete_one({'_id': ObjectId(user_id)})
    except Exception:
        pass

    return {'ok': False, 'message': 'Student registration failed', 'code': 500}


def get_student_profile(user_id: str) -> dict:
    """
    Return a merged profile for a student user.
    Joins `users` doc (auth info) with `students` doc (face + academic info).
    Returns dict or empty dict if not found.
    """
    db = get_db()
    models = get_models(db)

    user = models['users'].find_by_id(user_id)
    if not user:
        return {}

    # Look up the matching students document by user_id FK, with email fallback
    from bson import ObjectId
    student = db['students'].find_one({'user_id': ObjectId(user_id)})
    if not student:
        student = db['students'].find_one({'email': user['email'].lower()})

    profile = {
        'user_id':    str(user['_id']),
        'name':       user.get('name', ''),
        'email':      user.get('email', ''),
        'role':       user.get('role', 'student'),
        'department': user.get('department', ''),
        'phone':      user.get('phone', ''),
        'roll_no':    user.get('roll_no', ''),
        'created_at': user['created_at'].isoformat() if user.get('created_at') else None,
    }

    if student:
        profile['student_id'] = str(student['_id'])
        # Fill in from students doc if not already in users doc
        if not profile['roll_no']:
            profile['roll_no'] = student.get('roll_no', '')
        if not profile['department']:
            profile['department'] = student.get('department', '')
        profile['batch'] = student.get('batch', '')

    return profile


def update_student_profile(user_id: str, data: dict) -> dict:
    """
    Update allowed profile fields for a student.
    Only name, phone, department are editable.
    Returns {'ok': True} or {'ok': False, 'message': str, 'code': int}.
    """
    db = get_db()
    models = get_models(db)

    fields = {k: v for k, v in data.items() if k in ('name', 'phone', 'department') and v}
    if not fields:
        return {'ok': False, 'message': 'No updatable fields provided', 'code': 400}

    models['users'].update_profile(user_id, fields)
    return {'ok': True}   # treat as success — idempotent


def change_password(user_id: str, old_password: str, new_password: str) -> dict:
    """
    Verify old password then replace with a new bcrypt hash.
    Returns {'ok': True} or {'ok': False, 'message': str, 'code': int}.
    """
    db = get_db()
    models = get_models(db)

    if not old_password or not new_password:
        return {'ok': False, 'message': 'Both old and new password are required', 'code': 400}

    if len(new_password) < 8:
        return {'ok': False, 'message': 'New password must be at least 8 characters', 'code': 400}

    user = models['users'].find_by_id(user_id)
    if not user:
        return {'ok': False, 'message': 'User not found', 'code': 404}

    if not verify_password(old_password, user['password']):
        return {'ok': False, 'message': 'Current password is incorrect', 'code': 401}

    new_hash = hash_password(new_password)
    models['users'].update_password(user_id, new_hash)
    return {'ok': True}


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
            'students':       [str(s) for s in c.get('students', [])],
        }
        for c in classes
    ]


def create_class(
    teacher_id: str,
    name: str,
    code: str,
    student_ids: list = None,
    total_students: int = None,
    batch: str = None,
    department: str = None,
    schedule: str = None,
) -> dict:
    """
    Create a new class/subject linked to the teacher.
    Accepts an optional list of student ObjectId strings to enroll immediately.

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

    # Determine total_students: prefer explicit value, fall back to count of student_ids
    student_ids = student_ids or []
    if total_students is None:
        total_students = max(len(student_ids), 1)

    class_id = models['classes'].create_class(
        name=name,
        code=code,
        teacher_id=teacher_id,
        total_students=int(total_students),
        student_ids=student_ids,
        batch=batch,
        department=department,
        schedule=schedule,
    )

    if class_id:
        return {'ok': True, 'class_id': class_id}
    return {'ok': False, 'message': 'Failed to create class', 'code': 500}


def get_students_by_batch(batch: str = None) -> list:
    """
    Return all registered students, optionally filtered by batch.
    Used by the frontend multi-select picker when creating a class.

    Returns:
        [{'id', 'name', 'email', 'roll_no', 'batch', 'department'}, ...]
    """
    db = get_db()
    query = {}
    if batch:
        query['batch'] = batch

    students = list(db['students'].find(
        query,
        projection={'name': 1, 'email': 1, 'roll_no': 1, 'batch': 1, 'department': 1}
    ))

    return [
        {
            'id':         str(s['_id']),
            'name':       s.get('name', ''),
            'email':      s.get('email', ''),
            'roll_no':    s.get('roll_no', ''),
            'batch':      s.get('batch', ''),
            'department': s.get('department', ''),
        }
        for s in students
    ]
