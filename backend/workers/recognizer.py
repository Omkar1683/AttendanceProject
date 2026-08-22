"""
workers/recognizer.py
---------------------
Thin orchestration layer called by each worker thread.

Responsibilities:
  1. Frame dedup   — skip if identical bytes already being processed
  2. Student dedup — skip DB write if student already marked this session
  3. Face recognition — delegates to existing face_service.scan_frame()
  4. SocketIO emit  — broadcasts result to the session room

This module does NOT change any recognition logic. It only wraps
the existing face_service pipeline with queue infrastructure.
"""
import time

from services.face_service import face_service
from services import socket_service
from utils import frame_hash, cache
from workers import queue_manager
from database.connection import get_db
from models import get_models


def process_frame(job, worker_name: str) -> None:
    """
    Full processing pipeline for a single FrameJob.

    Args:
        job:         A FrameJob(session_id, image_bytes, timestamp).
        worker_name: Human-readable name, e.g. "Worker-1".
    """
    session_id  = job.session_id
    image_bytes = job.image_bytes

    print(f"[{worker_name}] Processing frame for session={session_id}  "
          f"size={len(image_bytes)} bytes")

    # ── 1. Frame-level dedup (SHA-256) ────────────────────────────────────────
    frame_h = frame_hash.try_acquire(image_bytes)
    if frame_h is None:
        print(f"[{worker_name}] Skipped duplicate frame (hash collision)")
        return

    t_start = time.perf_counter()

    try:
        # ── 2. Run existing recognition pipeline ──────────────────────────────
        #    face_service.scan_frame already handles:
        #      - image decode & resize
        #      - face_locations + face_encodings
        #      - distance matching against known_encodings
        #    We pass session_id=None here because we handle DB writes below
        #    ourselves (using the session cache), giving us finer control.
        result = face_service.scan_frame(image_bytes, session_id=None)

        elapsed = time.perf_counter() - t_start
        print(f"[{worker_name}] Recognition done in {elapsed:.2f}s  "
              f"status={result.get('status')}")

        if result.get('status') != 'success':
            # "No faces detected" or decode error — not a failure
            print(f"[{worker_name}] {result.get('message', 'No result')}")
            return

        # ── 3. Process each recognised face ───────────────────────────────────
        db     = get_db()
        models = get_models(db) if db is not None else None

        for person in result.get('people', []):
            name       = person.get('name', 'Unknown')
            student_id = person.get('student_id')
            status     = person.get('status', 'Absent')

            if status != 'Present' or not student_id:
                print(f"[{worker_name}] Unknown face — skipping DB write")
                continue

            # ── 3a. Session-level student dedup ───────────────────────────────
            if cache.is_already_marked(session_id, student_id):
                print(f"[{worker_name}] {name} already marked this session — skip")
                continue

            # ── 3b. Write attendance to DB ────────────────────────────────────
            if models:
                # Compute confidence from the encoding distance stored by
                # face_service (face_service stores it in the person dict
                # only when we ask it; we fall back to 0.0 if absent).
                confidence = person.get('confidence', 0.0)

                new_id = models['attendance_logs'].mark_attendance(
                    session_id=session_id,
                    student_id=student_id,
                    student_name=name,
                    status='Present',
                    marked_by='AI',
                    confidence=round(confidence, 4),
                )
                if new_id:
                    models['sessions'].increment_scanned(session_id)

            # ── 3c. Update session cache ──────────────────────────────────────
            cache.mark_student(session_id, student_id)
            present_count = cache.get_present_count(session_id)

            print(f"[{worker_name}] Recognized {name}  "
                  f"(total present this session: {present_count})")

            # ── 3d. Emit WebSocket event ──────────────────────────────────────
            socket_service.emit_attendance(
                session_id=session_id,
                student_id=student_id,
                student_name=name,
                confidence=confidence,
                present_count=present_count,
                worker_name=worker_name,
            )

            # ── 3e. Notify student client ─────────────────────────────────
            socket_service.emit_student_attendance_update(
                student_id=student_id,
                data={
                    'event': 'attendance_marked',
                    'session_id': session_id,
                    'status': 'Present',
                    'marked_by': 'AI',
                    'confidence': round(confidence, 3),
                },
            )

    except Exception as exc:
        print(f"[{worker_name}] ERROR processing frame: {exc}")
        raise   # re-raise so the worker can mark_failed()

    finally:
        # Always release the frame hash, even on error
        frame_hash.release(frame_h)
