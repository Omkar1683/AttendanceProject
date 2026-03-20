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
        return jsonify({'status': 'success', 'message': 'User registered successfully',
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
        return jsonify({'status': 'success', 'message': 'Student registered successfully',
                        'student_id': result['student_id']}), 201
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


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
        total_students=data.get('total_students', 1),
        batch=data.get('batch'),
        department=data.get('department'),
        schedule=data.get('schedule'),
    )
    if result['ok']:
        return jsonify({'status': 'success', 'message': 'Class created successfully',
                        'class_id': result['class_id']}), 201
    return jsonify({'status': 'error', 'message': result['message']}), result['code']


# ── / (health check) ─────────────────────────────────────────────────────────

@auth_bp.route('/', methods=['GET'])
def home():
    return "AttendAI API is running 🚀"
