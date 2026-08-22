"""
wsgi.py — Production WSGI Entry Point
======================================
Use this file with a production WSGI server such as Gunicorn:

    gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4

For development, use app.py directly:

    python app.py
"""
from app import create_app

app = create_app('production')
