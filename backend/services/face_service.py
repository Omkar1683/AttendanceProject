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

    # ── Cache management ──────────────────────────────────────────────────────

    def load_faces(self) -> int:
        """Load all student face encodings from DB into memory. Returns count loaded."""
        db = get_db()
        if db is None:
            print("❌ FaceService: No DB connection — cannot load faces")
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

        print(f"✅ FaceService: loaded {len(self.known_encodings)} face(s)")
        return len(self.known_encodings)

    # ── Frame processing ──────────────────────────────────────────────────────

    def scan_frame(self, image_bytes: bytes, session_id: str = None) -> dict:
        """
        Detect and identify faces in *image_bytes*.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            session_id:  If provided, attendance is marked in the DB.

        Returns:
            {'status': 'success', 'people': [...], 'count': int}
            {'status': 'error',   'message': str}
        """
        # Decode image
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {'status': 'error', 'message': 'Could not decode image'}

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            return {'status': 'error', 'message': 'No faces detected'}

        db     = get_db()
        models = get_models(db) if db is not None else None
        skip_matching = (len(self.known_encodings) == 0)
        detected = []

        for face_encoding in face_encodings:
            name       = "Unknown"
            status     = "Absent"
            student_id = None

            if not skip_matching:
                distances        = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_idx         = int(np.argmin(distances))
                best_distance    = float(distances[best_idx])

                if best_distance < self.match_threshold:
                    name       = self.known_names[best_idx]
                    student_id = self.known_ids[best_idx]
                    status     = "Present"

                    # Mark attendance if a session is active
                    if session_id and models:
                        new_id = models['attendance_logs'].mark_attendance(
                            session_id=session_id,
                            student_id=student_id,
                            student_name=name,
                            status='Present',
                            marked_by='AI',
                            confidence=round(1.0 - best_distance, 4),
                        )
                        # Only increment scanned count for genuinely new records
                        if new_id and models:
                            models['sessions'].increment_scanned(session_id)

            detected.append({
                'name':       name,
                'status':     status,
                'student_id': student_id,
                'encoding':   face_encoding.tolist(),
            })

        return {'status': 'success', 'people': detected, 'count': len(detected)}


# Module-level singleton — create_app() calls face_service.load_faces() at startup
face_service = FaceService()
