"""
Authentication utilities for JWT token generation and validation
"""
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
import os

# Secret key for JWT (use environment variable in production)
SECRET_KEY = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')

def hash_password(password):
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def verify_password(password, hashed):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def generate_token(user_id, email, role):
    """Generate JWT token with user information"""
    payload = {
        'user_id': str(user_id),
        'email': email,
        'role': role,
        'exp': datetime.utcnow() + timedelta(days=7)  # Token expires in 7 days
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def verify_token(token):
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to protect routes with JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'status': 'error', 'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'status': 'error', 'message': 'Token is missing'}), 401
        
        # Verify token
        payload = verify_token(token)
        if not payload:
            return jsonify({'status': 'error', 'message': 'Invalid or expired token'}), 401
        
        # Add user info to request context
        request.user = payload
        return f(*args, **kwargs)
    
    return decorated

def role_required(required_role):
    """Decorator to check user role"""
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
