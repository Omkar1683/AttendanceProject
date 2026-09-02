"""
routes/auth_routes.py
---------------------
Blueprint: authentication, user registration, student registration, class listing.
URL prefix: /  (endpoints keep their original top-level paths)
"""
from flask import Blueprint, request, jsonify

from core.security import token_required, role_required
import services.auth_service as auth_svc
from services.face_service import face_service

auth_bp = Blueprint('auth', __name__)


# ── /login ────────────────────────────────────────────────────────────────────

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    if not data.get('email') or not data.get('password'):
        return jsonify({'status': 'error', 'message': 'Missing credentials'}), 400

    result = auth_svc.login(data['email'], data['password'])
    if result['ok']:
        return jsonify({'status': 'success', 'token': result['token'], 'user': result['user']})
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /signup ───────────────────────────────────────────────────────────────────

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    result = auth_svc.signup(data)
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Teacher account created successfully',
                        'user_id': result['user_id']}), 201
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /students/register ────────────────────────────────────────────────────────

@auth_bp.route('/students/register', methods=['POST'])
@token_required
@role_required('teacher')
def register_student():
    data = request.get_json(silent=True) or {}
    result = auth_svc.register_student(data)
    if result['ok']:
        face_service.load_faces()   # Refresh in-memory cache
        return jsonify({
            'status': 'success',
            'message': 'Student registered successfully. Default password: 12345678',
            'student_id': result['student_id'],
            'user_id':    result['user_id'],
        }), 201
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /students — list all students (teacher only) ──────────────────────────────

@auth_bp.route('/students', methods=['GET'])
@token_required
@role_required('teacher')
def list_students():
    """
    Return all registered students for the multi-select class picker.
    Optional query param: ?batch=MCA-A
    """
    batch = request.args.get('batch', '').strip() or None
    students = auth_svc.get_students_by_batch(batch)
    return jsonify({'status': 'success', 'data': students})


# ── /classes ──────────────────────────────────────────────────────────────────

@auth_bp.route('/classes', methods=['GET'])
@token_required
def get_classes():
    teacher_id = request.args.get('teacher_id')
    # Fall back to current user's id from JWT if not provided
    if not teacher_id:
        teacher_id = request.user.get('user_id')
    print(f"🔍 GET /classes — teacher_id={teacher_id}")
    classes = auth_svc.get_classes(teacher_id)
    print(f"🔍 Returning {len(classes)} classes")
    return jsonify({'status': 'success', 'data': classes})


# ── /classes/create ───────────────────────────────────────────────────────────

@auth_bp.route('/classes/create', methods=['POST'])
@token_required
@role_required('teacher')
def create_class():
    data = request.get_json(silent=True) or {}
    # teacher_id from JWT token (set on request.user by @token_required)
    teacher_id = request.user.get('user_id')
    result = auth_svc.create_class(
        teacher_id=teacher_id,
        name=data.get('name'),
        code=data.get('code'),
        student_ids=data.get('student_ids', []),
        total_students=data.get('total_students'),
        batch=data.get('batch'),
        department=data.get('department'),
        schedule=data.get('schedule'),
    )
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Class created successfully',
                        'class_id': result['class_id']}), 201
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── /classes/<class_id>  (UPDATE) ─────────────────────────────────────────────

@auth_bp.route('/classes/<class_id>', methods=['PUT'])
@token_required
@role_required('teacher')
def update_class(class_id):
    """
    Update an existing class's enrolled students and/or batch.
    Teacher ownership is verified via JWT — a teacher can only
    modify their own classes.
    """
    from bson import ObjectId
    data = request.get_json(silent=True) or {}
    teacher_id = request.user.get('user_id')

    db = __import__('database.connection', fromlist=['get_db']).get_db()
    models = __import__('models', fromlist=['get_models']).get_models(db)

    # ── Verify class exists ───────────────────────────────────────────────
    class_doc = models['classes'].find_by_id(class_id)
    if not class_doc:
        return jsonify({'status': 'error', 'message': 'Class not found'}), 404

    # ── Verify teacher ownership ──────────────────────────────────────────
    if str(class_doc.get('teacher_id')) != teacher_id:
        return jsonify({'status': 'error', 'message': 'Forbidden — not your class'}), 403

    # ── Validate student IDs ──────────────────────────────────────────────
    student_ids = data.get('students', [])
    if student_ids:
        valid_count = db['students'].count_documents({
            '_id': {'$in': [ObjectId(sid) for sid in student_ids]}
        })
        if valid_count != len(student_ids):
            return jsonify({
                'status': 'error',
                'message': f'Some student IDs are invalid ({valid_count}/{len(student_ids)} found)',
            }), 400

    # ── Update ────────────────────────────────────────────────────────────
    batch = data.get('batch')
    models['classes'].update_class_details(
        class_id=class_id,
        student_ids=student_ids,
        batch=batch,
    )

    return jsonify({
        'status': 'success',
        'message': 'Class updated',
        'total_students': len(student_ids),
    })


# ── / (health check) ─────────────────────────────────────────────────────────

@auth_bp.route('/', methods=['GET'])
def home():
    return "AttendAI API is running 🚀"
