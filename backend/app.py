"""
app.py — Application Factory
=============================
Usage
-----
Development:
    python app.py

Production (Gunicorn with threading worker):
    gunicorn --worker-class=gthread --threads=4 wsgi:app

Environment variables:
    APP_ENV          development | production   (default: development)
    JWT_SECRET       Long random string         (REQUIRED in production)
    MONGO_USERNAME   MongoDB Atlas username
    MONGO_PASSWORD   MongoDB Atlas password
    MONGO_HOST       Atlas cluster host
    MONGO_DB_NAME    Database name              (default: attendai_db)
    MONGO_APP_NAME   Atlas app name
    FACE_MATCH_THRESHOLD  0.0-1.0              (default: 0.50)
"""
import os
from flask import Flask
from flask_cors import CORS
from flask_mail import Mail

from core.config import get_config
from core.security import init_security
from database.connection import init_db
from services.face_service import face_service
from services.socket_service import init_socketio
from workers.attendance_worker import start_workers

# ── Blueprints ────────────────────────────────────────────────────────────────
from routes.auth_routes         import auth_bp
from routes.session_routes      import session_bp
from routes.attendance_routes   import attendance_bp
from routes.analytics_routes    import analytics_bp
from routes.notification_routes import notification_bp
from routes.student_routes      import student_bp
from routes.student_module_routes import student_module_bp

mail = Mail()


def create_app(config_name: str = None) -> Flask:
    """
    Application factory.

    Args:
        config_name: 'development' | 'production'
                     Falls back to APP_ENV env var, then 'development'.
    """
    app = Flask(__name__)
    CORS(
        app,
        expose_headers=["Content-Disposition", "Content-Type"],
        supports_credentials=True,
    )

    # ── Load config ───────────────────────────────────────────────────────────
    Config = get_config(config_name)
    app.config.from_object(Config)

    # ── Initialize Mail ───────────────────────────────────────────────────────
    mail.init_app(app)

    # ── Initialize WebSocket (Flask-SocketIO) ─────────────────────────────────
    init_socketio(app)

    # ── Boot security layer ───────────────────────────────────────────────────
    init_security(
        secret_key=Config.JWT_SECRET,
        expiry_days=Config.JWT_EXPIRY_DAYS,
    )

    # ── Boot database ─────────────────────────────────────────────────────────
    db = init_db(Config.get_mongo_uri(), Config.MONGO_DB_NAME)
    if db is None:
        print("WARNING: Running WITHOUT database connection")

    # ── Boot face recognition cache ───────────────────────────────────────────
    face_service.match_threshold = Config.FACE_MATCH_THRESHOLD
    if db is not None:
        face_service.load_faces()

    # ── Start background queue workers ────────────────────────────────────────
    start_workers()

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(auth_bp)           # /login, /signup, /students/register, /classes
    app.register_blueprint(session_bp)        # /sessions/create, /sessions/stop
    app.register_blueprint(attendance_bp)     # /scan, /scan/enqueue, /queue/status
    app.register_blueprint(analytics_bp)      # /analytics/*, /reports/*
    app.register_blueprint(notification_bp)   # /notifications/send
    app.register_blueprint(student_bp)        # /student/* (self-service student APIs)
    app.register_blueprint(student_module_bp)  # /student/* (student module: notifications, analytics, export)

    return app


if __name__ == '__main__':
    import services.socket_service as socket_service
    app = create_app()
    # Use socketio.run() instead of app.run() so WebSocket handshakes work
    socket_service.socketio.run(app, host='0.0.0.0', port=5000, debug=app.config.get('DEBUG', True), allow_unsafe_werkzeug=True)