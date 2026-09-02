"""
services/face_service.py
------------------------
Manages the in-memory face-encoding cache and processes scan frames.
The FaceService singleton is initialised once by create_app() and then
injected into scan routes via the Flask application context.
"""
import numpy as np
import cv2
import face_recognition

from database.connection import get_db
from models import get_models


class FaceService:
    """Singleton that owns the in-memory face encoding cache."""

    def __init__(self, match_threshold: float = 0.50):
        self.match_threshold = match_threshold
        self.known_encodings: list[np.ndarray] = []
        self.known_names:     list[str] = []
        self.known_ids:       list[str] = []

        # Per-session dedup: {session_id: set(student_id, ...)}
        self._session_marked: dict[str, set] = {}

    # ── Cache management ──────────────────────────────────────────────────────

    def load_faces(self) -> int:
        """Load all student face encodings from DB into memory. Returns count loaded."""
        db = get_db()
        if db is None:
            print("[FaceService] ERROR: No DB connection - cannot load faces")
            return 0

        models = get_models(db)
        students = models['students'].get_all_encodings()

        self.known_encodings = []
        self.known_names     = []
        self.known_ids       = []

        for student in students:
            encoding_list = student.get('encoding', [])
            if len(encoding_list) == 128:
                self.known_encodings.append(np.array(encoding_list))
                self.known_names.append(student.get('name', 'Unknown'))
                self.known_ids.append(str(student['_id']))

        print(f"[FaceService] Loaded {len(self.known_encodings)} face(s)")
        return len(self.known_encodings)

    # ── Frame processing ──────────────────────────────────────────────────────

    def scan_frame(self, image_bytes: bytes, session_id: str = None) -> dict:
        """
        Detect and identify faces in *image_bytes*.

        Key decisions:
          - Resize proportionally only when the image is very wide/tall (>1000px).
            Aggressive fixed-size resize (e.g. 640x480) distorts portrait frames
            from mobile cameras and shrinks faces below the detection threshold.
          - number_of_times_to_upsample=2: upscales twice before HOG, making it
            possible to detect faces that occupy only a small part of the frame.
          - Per-session in-memory set prevents redundant DB writes.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            session_id:  If provided, attendance is marked in the DB.

        Returns:
            {'status': 'success', 'people': [...], 'count': int}
            {'status': 'error',   'message': str}
        """
        print("[Scan] Received scan request")

        # ── 1. Validate & decode image ────────────────────────────────────────
        if not image_bytes:
            print("[Scan] ERROR: Empty image bytes received")
            return {'status': 'error', 'message': 'Invalid image'}

        img_array = np.frombuffer(image_bytes, np.uint8)

        if img_array.size == 0:
            print("[Scan] ERROR: img_array is empty after frombuffer")
            return {'status': 'error', 'message': 'Invalid image'}

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("[Scan] ERROR: cv2.imdecode returned None - corrupted or unsupported image")
            return {'status': 'error', 'message': 'Corrupted image'}

        # ── 2. Proportional resize — cap longer edge at 640px for high speed ──
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"[Scan] Resized: ({w}x{h}) -> ({new_w}x{new_h})")

        print(f"[Scan] Image shape: {img.shape}  (H x W x C)")

        # ── 3. Convert colour space ───────────────────────────────────────────
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── 4. Detect faces (upsample=1 for speed & accuracy) ─────────────────
        face_locations = face_recognition.face_locations(
            rgb_img,
            number_of_times_to_upsample=1,
        )
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        print(f"[Scan] Faces detected: {len(face_locations)} | Encodings: {len(face_encodings)}")

        if not face_encodings:
            return {'status': 'error', 'message': 'No faces detected'}

        # ── 5. Prepare DB + dedup state ───────────────────────────────────────
        db     = get_db()
        models = get_models(db) if db is not None else None
        skip_matching = (len(self.known_encodings) == 0)

        if session_id:
            if session_id not in self._session_marked:
                self._session_marked[session_id] = set()
            already_marked = self._session_marked[session_id]
        else:
            already_marked = None

        detected = []

        # ── 6. Match each detected face ───────────────────────────────────────
        for face_encoding in face_encodings:
            name       = "Unknown"
            status     = "Absent"
            student_id = None

            if not skip_matching:
                distances     = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_idx      = int(np.argmin(distances))
                best_distance = float(distances[best_idx])

                print(f"[Scan] Best match distance: {best_distance:.4f} (threshold={self.match_threshold})")

                if best_distance < self.match_threshold:
                    name       = self.known_names[best_idx]
                    student_id = self.known_ids[best_idx]
                    status     = "Present"

                    print(f"[Scan] MATCH: {name} (distance={best_distance:.3f})")

                    if session_id and models:
                        if already_marked is None or student_id not in already_marked:
                            new_id = models['attendance_logs'].mark_attendance(
                                session_id=session_id,
                                student_id=student_id,
                                student_name=name,
                                status='Present',
                                marked_by='AI',
                                confidence=round(1.0 - best_distance, 4),
                            )
                            if new_id and models:
                                models['sessions'].increment_scanned(session_id)
                            if already_marked is not None:
                                already_marked.add(student_id)
                        else:
                            print(f"[Scan] {name} already marked this session - skipping DB write")
                else:
                    print(f"[Scan] No match above threshold (distance={best_distance:.3f})")

            detected.append({
                'name':       name,
                'status':     status,
                'student_id': student_id,
                'encoding':   face_encoding.tolist(),
            })

        return {'status': 'success', 'people': detected, 'count': len(detected)}

    def reset_session(self, session_id: str) -> None:
        """Clear the in-memory dedup set when a session ends."""
        self._session_marked.pop(session_id, None)


# Module-level singleton — create_app() calls face_service.load_faces() at startup
face_service = FaceService()
