"""
utils/frame_hash.py
-------------------
Thread-safe SHA-256 deduplication for incoming frames.

If two workers receive the exact same frame (same bytes),
only the first worker processes it. The second sees the hash
already in the set and skips immediately.

Usage:
    h = try_acquire(image_bytes)
    if h is None:
        return  # duplicate — skip
    try:
        process(image_bytes)
    finally:
        release(h)
"""
import hashlib
import threading

_lock = threading.Lock()
_processing: set[str] = set()


def try_acquire(image_bytes: bytes) -> str | None:
    """
    Compute SHA-256 of image_bytes and attempt to claim it.

    Returns:
        The hex digest string if the frame is new (caller should process it).
        None if the frame hash already exists (duplicate — caller should skip).
    """
    h = hashlib.sha256(image_bytes).hexdigest()
    with _lock:
        if h in _processing:
            return None
        _processing.add(h)
    return h


def release(h: str) -> None:
    """Remove the hash from the processing set once done (or on error)."""
    with _lock:
        _processing.discard(h)


def current_count() -> int:
    """Return how many frames are currently being deduplicated."""
    with _lock:
        return len(_processing)
