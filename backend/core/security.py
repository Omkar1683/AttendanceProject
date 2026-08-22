"""
core/security.py
----------------
JWT token generation/verification and Flask route decorators
for authentication and role-based access control.
"""
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify


# Secret key — loaded from config at app-startup via init_security()
_SECRET_KEY: str = os.getenv('JWT_SECRET', 'change-me-in-production')
_JWT_EXPIRY_DAYS: int = 7


def init_security(secret_key: str, expiry_days: int = 7) -> None:
    """Called by create_app() to inject config values."""
    global _SECRET_KEY, _JWT_EXPIRY_DAYS
    _SECRET_KEY = secret_key
    _JWT_EXPIRY_DAYS = expiry_days


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> bytes:
    """Hash a plain-text password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)


def verify_password(password: str, hashed) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def generate_token(user_id: str, email: str, role: str) -> str:
    """Generate a signed JWT token containing user identity."""
    payload = {
        'user_id': str(user_id),
        'email': email,
        'role': role,
        'exp': datetime.utcnow() + timedelta(days=_JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm='HS256')


def verify_token(token: str) -> dict | None:
    """Decode and verify a JWT token; returns payload or None."""
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ── Route decorators ──────────────────────────────────────────────────────────

def token_required(f):
    """Decorator: require a valid Bearer JWT in the Authorization header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                return jsonify({'status': 'error', 'message': 'Invalid token format'}), 401

        if not token:
            return jsonify({'status': 'error', 'message': 'Token is missing'}), 401

        payload = verify_token(token)
        if not payload:
            return jsonify({'status': 'error', 'message': 'Invalid or expired token'}), 401

        request.user = payload
        return f(*args, **kwargs)
    return decorated


def role_required(required_role: str):
    """Decorator: require the authenticated user to have a specific role."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'user'):
                return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
            if request.user.get('role') != required_role:
                return jsonify({'status': 'error', 'message': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
