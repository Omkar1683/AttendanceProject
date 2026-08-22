"""
workers/queue_manager.py
------------------------
Central frame queue and job counters.

Provides a single shared Queue instance used by:
  - attendance_routes.py  (producers — enqueue incoming frames)
  - attendance_worker.py  (consumers — dequeue and process)

Counter semantics:
  queued     — frames currently sitting in the queue waiting to be picked up
  processing — frames currently being processed by a worker
  completed  — frames successfully processed (recognised or "no face")
  failed     — frames that raised an unhandled exception during processing
"""
from queue import Queue, Full
from dataclasses import dataclass, field
from datetime import datetime
import threading

# ── Queue ──────────────────────────────────────────────────────────────────────

# maxsize=100 — if the queue fills up (workers can't keep up),
# new frames are dropped rather than blocking the HTTP request.
frame_queue: Queue = Queue(maxsize=100)

# ── Job dataclass ──────────────────────────────────────────────────────────────

@dataclass
class FrameJob:
    session_id:  str
    image_bytes: bytes
    timestamp:   datetime = field(default_factory=datetime.utcnow)


# ── Counters ───────────────────────────────────────────────────────────────────

_counter_lock = threading.Lock()
_counters = {
    'queued':     0,
    'processing': 0,
    'completed':  0,
    'failed':     0,
}


def _inc(key: str, delta: int = 1) -> None:
    with _counter_lock:
        _counters[key] += delta


def get_stats() -> dict:
    """Return a snapshot of the current queue counters."""
    with _counter_lock:
        return dict(_counters)


# ── Producer API ───────────────────────────────────────────────────────────────

def enqueue_frame(session_id: str, image_bytes: bytes) -> bool:
    """
    Add a frame job to the queue.

    Returns:
        True  — frame successfully enqueued.
        False — queue is full; frame dropped.
    """
    job = FrameJob(session_id=session_id, image_bytes=image_bytes)
    try:
        frame_queue.put_nowait(job)
        _inc('queued')
        return True
    except Full:
        print("[QueueManager] Queue full — frame dropped")
        return False


# ── Consumer helpers (called by workers) ──────────────────────────────────────

def mark_processing() -> None:
    """Call when a worker picks up a job."""
    _inc('queued', -1)
    _inc('processing')


def mark_completed() -> None:
    """Call when a worker finishes a job successfully."""
    _inc('processing', -1)
    _inc('completed')


def mark_failed() -> None:
    """Call when a worker encounters an unrecoverable error on a job."""
    _inc('processing', -1)
    _inc('failed')
