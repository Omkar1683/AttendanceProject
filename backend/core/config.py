"""
core/config.py
--------------
Application configuration classes.
Set APP_ENV environment variable to switch between environments:
  - APP_ENV=development  (default)
  - APP_ENV=production
"""
import os
from urllib.parse import quote_plus


class BaseConfig:
    """Shared settings across all environments."""
    APP_NAME = "AttendAI"
    DEBUG = False
    TESTING = False

    # JWT
    JWT_SECRET = os.getenv('JWT_SECRET', 'change-me-in-production-use-a-long-random-string')
    JWT_EXPIRY_DAYS = int(os.getenv('JWT_EXPIRY_DAYS', 7))

    # MongoDB
    MONGO_USERNAME = os.getenv('MONGO_USERNAME', 'devesh')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', 'Devesh_1234')
    MONGO_HOST     = os.getenv('MONGO_HOST', 'attendancecluster.uucwump.mongodb.net')
    MONGO_DB_NAME  = os.getenv('MONGO_DB_NAME', 'attendai_db')
    MONGO_APP_NAME = os.getenv('MONGO_APP_NAME', 'AttendanceCluster')

    @classmethod
    def get_mongo_uri(cls) -> str:
        username = quote_plus(cls.MONGO_USERNAME)
        password = quote_plus(cls.MONGO_PASSWORD)
        return (
            f"mongodb+srv://{username}:{password}@{cls.MONGO_HOST}"
            f"/?appName={cls.MONGO_APP_NAME}"
        )

    # Face recognition
    FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', 0.50))

    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')

    # Mail
    MAIL_SERVER   = 'smtp.gmail.com'
    MAIL_PORT     = 587
    MAIL_USE_TLS  = True
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')   # Gmail address
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')   # Gmail App Password
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_USERNAME')


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False
    # In production, these MUST come from actual environment variables
    JWT_SECRET = os.getenv('JWT_SECRET')


# Config registry — select via APP_ENV env var
configs = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
}

def get_config(env: str = None):
    env = env or os.getenv('APP_ENV', 'development')
    return configs.get(env, DevelopmentConfig)
