"""
services/socket_service.py
--------------------------
Flask-SocketIO wrapper.

Provides:
  - init_socketio(app)  — call once from create_app()
  - emit_attendance(...)  — broadcast a recognition result to all clients
                            watching a specific session room
  - emit_queue_status(stats)  — broadcast queue counter updates

Room naming:
  Each attendance session gets its own SocketIO room: "session_<session_id>".
  The React Native client joins this room after calling /sessions/create.
"""
from flask_socketio import SocketIO, emit, join_room

# Module-level singleton — assigned by init_socketio()
socketio: SocketIO = None


def init_socketio(app) -> SocketIO:
    """
    Create and attach a SocketIO instance to the Flask app.

    async_mode='threading' keeps compatibility with the standard Flask
    development server and Gunicorn threaded workers without requiring
    eventlet or gevent monkey-patching.
    """
    global socketio
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False,
    )

    # ── Built-in events ───────────────────────────────────────────────────────

    @socketio.on('connect')
    def on_connect():
        print(f"[SocketIO] Client connected: {socketio}")

    @socketio.on('join_session')
    def on_join_session(data):
        """
        Client sends: { "session_id": "<id>" }
        Server puts the client into room "session_<id>".
        """
        session_id = data.get('session_id', '')
        room = f"session_{session_id}"
        join_room(room)
        print(f"[SocketIO] Client joined room: {room}")
        emit('joined', {'room': room})

    @socketio.on('disconnect')
    def on_disconnect():
        print("[SocketIO] Client disconnected")

    return socketio


def emit_attendance(
    session_id: str,
    student_id: str,
    student_name: str,
    confidence: float,
    present_count: int,
    worker_name: str,
) -> None:
    """
    Broadcast a recognition event to all clients in the session room.

    Payload:
        {
            "event":        "student_recognized",
            "student_id":   "<id>",
            "student_name": "Omkar",
            "confidence":   0.87,
            "present_count": 5,
            "worker":       "Worker-2"
        }
    """
    if socketio is None:
        return

    room = f"session_{session_id}"
    payload = {
        'event':        'student_recognized',
        'student_id':   student_id,
        'student_name': student_name,
        'confidence':   round(confidence, 3),
        'present_count': present_count,
        'worker':       worker_name,
    }
    socketio.emit('attendance_update', payload, room=room)
    socketio.emit('attendance_update', payload)


def emit_queue_status(stats: dict) -> None:
    """Broadcast queue counter stats to all connected clients."""
    if socketio is None:
        return
    socketio.emit('queue_status', stats)
