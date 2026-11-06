"""
Face-recognition engine for Missing Person Finder.

Uses InsightFace buffalo_l for face detection and embeddings, OpenCV for
image/video processing, and PostgreSQL for storing/searching embeddings.

Expected tables:
  missing_persons(user_id, name, age, description, photo_path, embedding, match_status)
  cctv_embeddings(video_name, camera_name, frame_number, embedding, location, frame_path)

The embedding columns must accept a PostgreSQL array (for example REAL[]/DOUBLE PRECISION[])
or another type psycopg2 can adapt a Python list to.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis

from database import get_connection

try:
    from email_service import send_match_alert
except ImportError:
    # Email alerts are optional; face matching should still work without this module.
    send_match_alert = None


# ---------------- SETTINGS ----------------

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))
TWIN_THRESHOLD = float(os.getenv("TWIN_THRESHOLD", "0.80"))

BASE_DIR = Path(__file__).resolve().parent
CCTV_FRAME_FOLDER = BASE_DIR / "static" / "cctv_frames"
CCTV_FRAME_FOLDER.mkdir(parents=True, exist_ok=True)

# Render services generally do not have a GPU. Set INSIGHTFACE_CTX_ID=0 only
# on a machine with a correctly configured CUDA environment.
CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))

# Load buffalo_l once when this module is imported.

face_app = None

def get_face_app():
    global face_app

    if face_app is None:
        print("Loading InsightFace buffalo_l model...")
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
        print("InsightFace model loaded successfully!")

    return face_app


# ---------------- PREPROCESSING ----------------

def preprocess_frame(frame):
    """Apply CLAHE to improve local contrast while preserving colour."""
    if frame is None:
        return None

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ---------------- EMBEDDING GENERATION ----------------

def _normalize_embedding(embedding):
    """Return a unit-length float32 embedding, or None if invalid."""
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm == 0:
        return None

    return vector / norm


def create_embeddings_from_frame(frame, min_face_size=40):
    """
    Return normalized embeddings for every sufficiently large detected face.
    Each embedding is a 512-dimensional buffalo_l feature vector.
    """
    processed = preprocess_frame(frame)
    if processed is None:
        return []

    faces = get_face_app().get(processed)
    embeddings = []

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
            continue

        embedding = _normalize_embedding(face.embedding)
        if embedding is not None:
            embeddings.append(embedding)

    return embeddings


def create_embedding_from_image(image_path):
    """
    Create one embedding from an image.

    When multiple faces are present, the largest detected face is selected,
    matching the behaviour of the original project.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    processed = preprocess_frame(image)
    faces = face_app.get(processed)

    if not faces:
        return None

    largest_face = max(
        faces,
        key=lambda face: max(0, face.bbox[2] - face.bbox[0])
                         * max(0, face.bbox[3] - face.bbox[1])
    )

    x1, y1, x2, y2 = largest_face.bbox.astype(int)
    if (x2 - x1) < 80 or (y2 - y1) < 80:
        return None

    return _normalize_embedding(largest_face.embedding)


# ---------------- LOCATION / ALERT ----------------

def get_current_location():
    """Return approximate IP-based location for the match alert."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            return "Location not available"

        city = data.get("city")
        region = data.get("regionName")
        country = data.get("country")
        latitude = data.get("lat")
        longitude = data.get("lon")

        return (
            f"{city}, {region}, {country} "
            f"(Lat: {latitude}, Lon: {longitude})"
        )
    except requests.RequestException:
        return "Location not available"
    except (ValueError, TypeError):
        return "Location not available"


def _send_alert_safely(user_id):
    """Send an optional notification without breaking a successful match."""
    if send_match_alert is None:
        return

    try:
        send_match_alert(user_id, get_current_location())
    except Exception as exc:
        print("Match alert could not be sent:", exc)


# ---------------- DATABASE HELPERS ----------------

def _rows_as_embeddings(records):
    """Yield valid normalized DB embeddings and their associated metadata."""
    for record in records:
        embedding_data, location, camera_name, frame_path = record

        embedding = _normalize_embedding(embedding_data)
        if embedding is None:
            continue

        yield embedding, location, camera_name, frame_path


def _find_best_cctv_match(query_embedding):
    """Find the highest cosine-similarity CCTV embedding in the database."""
    query_embedding = _normalize_embedding(query_embedding)
    if query_embedding is None:
        return 0.0, None, None, None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT embedding, location, camera_name, frame_path
                FROM cctv_embeddings
                WHERE embedding IS NOT NULL
            """)
            records = cur.fetchall()
    finally:
        conn.close()

    best_score = -1.0
    best_location = None
    best_camera = None
    best_frame = None

    for db_embedding, location, camera_name, frame_path in _rows_as_embeddings(records):
        if db_embedding.shape != query_embedding.shape:
            continue

        similarity = float(np.dot(db_embedding, query_embedding))
        if similarity > best_score:
            best_score = similarity
            best_location = location
            best_camera = camera_name
            best_frame = frame_path

    if best_score < 0:
        best_score = 0.0

    return best_score, best_location, best_camera, best_frame


# ---------------- STORE MISSING PERSON ----------------

def store_missing_person(user_id, name, age, description, photo_path):
    embedding = create_embedding_from_image(photo_path)

    if embedding is None:
        return False, "No suitable face detected. Please upload a clear face image."

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO missing_persons
                    (user_id, name, age, description, photo_path, embedding, match_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                name,
                age,
                description,
                str(photo_path),
                embedding.tolist(),
                "Not Found",
            ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return True, "Details uploaded successfully"


# ---------------- SEARCH USER IMAGE ----------------

def search_face(user_id):
    """
    Search the user's registered missing-person embedding against CCTV data.

    Always returns exactly:
      (result, score, location, camera_name, frame_path)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, embedding
                FROM missing_persons
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT 1
            """, (user_id,))
            record = cur.fetchone()
    finally:
        conn.close()

    if not record or record[1] is None:
        return "No Data Found", 0.0, None, None, None

    person_id, stored_embedding = record
    test_embedding = _normalize_embedding(stored_embedding)

    if test_embedding is None:
        return "No Data Found", 0.0, None, None, None

    score, location, camera_name, frame_path = _find_best_cctv_match(test_embedding)

    if score >= SIMILARITY_THRESHOLD and camera_name and frame_path:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE missing_persons
                    SET match_status = 'Found'
                    WHERE id = %s
                """, (person_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        _send_alert_safely(user_id)
        return "MATCH", round(score, 4), location, camera_name, frame_path

    return "No Match", round(score, 4), None, None, None


# ---------------- ADMIN IMAGE SEARCH ----------------

def search_face_for_admin(image_path):
    embedding = create_embedding_from_image(image_path)

    if embedding is None:
        return "No Face Detected", 0.0, None, None, None

    score, location, camera_name, frame_path = _find_best_cctv_match(embedding)

    if score >= SIMILARITY_THRESHOLD and camera_name and frame_path:
        return "MATCH", round(score, 4), location, camera_name, frame_path

    return "No Match", round(score, 4), None, None, None


# ---------------- STORE CCTV VIDEO EMBEDDINGS ----------------

def store_video_embeddings(video_path, camera_name, location):
    """
    Extract approximately three samples per second from a video.
    Every detected face gets its own database embedding row.

    The saved frame path is relative to the project root so it can be served
    by Flask's static route when the frame is under static/.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "Unable to open video. Check the video format."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    # At least one frame interval; avoids modulo-by-zero for low-FPS videos.
    frame_interval = max(1, int(round(fps / 3.0)))

    video_name = Path(str(video_path)).name
    frame_number = 0
    stored_count = 0
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    embeddings = create_embeddings_from_frame(frame)

                    if embeddings:
                        frame_filename = (
                            f"{Path(video_name).stem}_{camera_name}_"
                            f"{frame_number}.jpg"
                        )
                        # Sanitize camera name for a safe filename.
                        safe_camera = "".join(
                            ch if ch.isalnum() or ch in ("-", "_") else "_"
                            for ch in str(camera_name)
                        )[:60]
                        frame_filename = (
                            f"{Path(video_name).stem}_{safe_camera}_"
                            f"{frame_number}.jpg"
                        )
                        frame_path = CCTV_FRAME_FOLDER / frame_filename

                        if not cv2.imwrite(str(frame_path), frame):
                            frame_number += 1
                            continue

                        relative_frame_path = frame_path.relative_to(BASE_DIR).as_posix()

                        # Store each face separately; do not discard all but the last face.
                        for embedding in embeddings:
                            cur.execute("""
                                INSERT INTO cctv_embeddings
                                    (video_name, camera_name, frame_number,
                                     embedding, location, frame_path)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (
                                video_name,
                                camera_name,
                                frame_number,
                                embedding.tolist(),
                                location,
                                relative_frame_path,
                            ))
                            stored_count += 1

                frame_number += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cap.release()
        conn.close()

    if stored_count == 0:
        return "Video processed, but no suitable faces were detected."

    return f"Video processed successfully. Stored {stored_count} face embeddings."


# ---------------- SEARCH IMAGE IN CCTV ----------------

def search_image_in_cctv_db(image_path):
    """
    Return exactly (camera_name, location, frame_path).
    Returns (None, None, None) when no suitable match exists.
    """
    test_embedding = create_embedding_from_image(image_path)

    if test_embedding is None:
        return None, None, None

    score, location, camera_name, frame_path = _find_best_cctv_match(test_embedding)

    if score >= SIMILARITY_THRESHOLD and camera_name and frame_path:
        return camera_name, location, frame_path

    return None, None, None


# ---------------- POSSIBLE TWIN / SIMILAR-FACE SEARCH ----------------

def detect_possible_twin(image_path, twin_threshold=None):
    """
    Return CCTV records whose cosine similarity meets the higher twin threshold.

    This is only a possible-similarity flag, not proof that two people are twins.
    """
    if twin_threshold is None:
        twin_threshold = TWIN_THRESHOLD

    query_embedding = create_embedding_from_image(image_path)
    if query_embedding is None:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT embedding, location, camera_name, frame_path
                FROM cctv_embeddings
                WHERE embedding IS NOT NULL
            """)
            records = cur.fetchall()
    finally:
        conn.close()

    possible_twins = []

    for db_embedding, location, camera_name, frame_path in _rows_as_embeddings(records):
        if db_embedding.shape != query_embedding.shape:
            continue

        similarity = float(np.dot(db_embedding, query_embedding))
        if similarity >= float(twin_threshold):
            possible_twins.append({
                "similarity": round(similarity, 4),
                "location": location,
                "camera": camera_name,
                "frame": frame_path,
            })

    possible_twins.sort(key=lambda item: item["similarity"], reverse=True)
    return possible_twins
