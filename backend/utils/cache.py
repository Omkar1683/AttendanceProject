"""
utils/cache.py
--------------
Session-level in-memory cache that tracks which students have already
been marked Present in the current session.

Prevents duplicate DB writes when the same student appears in
multiple consecutive frames.

Structure:
    present_students = {
        "session_id_1": {"student_id_a", "student_id_b"},
        "session_id_2": {"student_id_c"},
    }
"""
import threading

_lock = threading.Lock()
_present_students: dict[str, set[str]] = {}


def is_already_marked(session_id: str, student_id: str) -> bool:
    """Return True if this student is already marked Present in this session."""
    with _lock:
        return student_id in _present_students.get(session_id, set())


def mark_student(session_id: str, student_id: str) -> None:
    """Record that this student has been marked Present in this session."""
    with _lock:
        if session_id not in _present_students:
            _present_students[session_id] = set()
        _present_students[session_id].add(student_id)


def get_present_count(session_id: str) -> int:
    """Return how many unique students are marked Present so far."""
    with _lock:
        return len(_present_students.get(session_id, set()))


def get_present_ids(session_id: str) -> set[str]:
    """Return a copy of the present student IDs for this session."""
    with _lock:
        return set(_present_students.get(session_id, set()))


def clear_session(session_id: str) -> None:
    """Remove session cache when a session ends."""
    with _lock:
        _present_students.pop(session_id, None)
