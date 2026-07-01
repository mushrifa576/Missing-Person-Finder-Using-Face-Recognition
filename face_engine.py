import os
import numpy as np
import cv2
import requests
from database import get_connection
from email_service import send_match_alert
from insightface.app import FaceAnalysis

# ================= SETTINGS =================
SIMILARITY_THRESHOLD = 0.40

# ================= LOAD INSIGHTFACE MODEL =================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # 0 = GPU , -1 = CPU


# ================= PREPROCESS =================
def preprocess_frame(frame):
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


# ================= CREATE EMBEDDING (COMMON FUNCTION) =================
def create_embedding_from_image(image_path):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        frame = preprocess_frame(frame)

        faces = app.get(frame)

        if len(faces) == 0:
            return None

        # Select largest face (better for group photos)
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

        x1, y1, x2, y2 = face.bbox.astype(int)
        face_width = x2 - x1
        face_height = y2 - y1

        # Skip very small faces
        if face_width < 80 or face_height < 80:
            return None

        embedding = face.embedding

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as e:
        print("Embedding error:", e)
        return None
# ================= GET CURRENT LOCATION =================
def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()

        city = data.get("city")
        region = data.get("regionName")
        country = data.get("country")
        lat = data.get("lat")
        lon = data.get("lon")

        return f"{city}, {region}, {country} (Lat: {lat}, Lon: {lon})"
    except:
        return "Location not available"


# ================= STORE MISSING PERSON =================
def store_missing_person(user_id, name, age, description, photo_path):
    embedding = create_embedding_from_image(photo_path)

    if embedding is None:
        return False, "No face detected"

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO missing_persons
        (user_id, name, age, description, photo_path, embedding, match_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id, name, age, description,
        photo_path,
        embedding.tolist(),
        "Not Found"
    ))

    conn.commit()
    cur.close()
    conn.close()

    return True, "Details uploaded successfully"


# ================= SEARCH USER IMAGE =================
def search_face(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, embedding FROM missing_persons WHERE user_id=%s", (user_id,))
    record = cur.fetchone()

    if not record:
        cur.close()
        conn.close()
        return "No Data Found", 0, None

    person_id = record[0]
    test_embedding = np.array(record[1], dtype=np.float32)
    test_embedding /= np.linalg.norm(test_embedding)

    cur.execute("SELECT embedding, location, camera_name, frame_path FROM cctv_embeddings")
    records = cur.fetchall()

    best_score = -1
    best_location = None
    best_camera = None
    best_frame = None

    for embedding_data, location, camera_name, frame_path in records:
        db_embedding = np.array(embedding_data, dtype=np.float32)
        db_embedding /= np.linalg.norm(db_embedding)

        similarity = float(np.dot(db_embedding, test_embedding))

        if similarity > best_score:
            best_score = similarity
            best_location = location
            best_camera = camera_name
            best_frame = frame_path

    cur.close()
    conn.close()

    if best_score >= SIMILARITY_THRESHOLD:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE missing_persons SET match_status='Found' WHERE id=%s", (person_id,))
        conn.commit()
        cur.close()
        conn.close()

        location = get_current_location()
        send_match_alert(user_id, location)

        return "MATCH", round(best_score, 4), best_location, best_camera, best_frame
       

    return "No Match", round(best_score, 4), None, None, None

#==================Search admin=================
def search_face_for_admin(image_path):

    embedding = create_embedding_from_image(image_path)
    if embedding is None:
        return "No Face Detected", 0, None, None, None

    embedding = np.array(embedding, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT embedding, location, camera_name, frame_path FROM cctv_embeddings")
    records = cur.fetchall()

    cur.close()
    conn.close()

    best_score = -1
    best_location = None
    best_camera = None
    best_frame = None

    for embedding_data, location, camera_name, frame_path in records:

        db_embedding = np.array(embedding_data, dtype=np.float32)

        norm = np.linalg.norm(db_embedding)
        if norm == 0:
            continue

        db_embedding /= norm

        similarity = float(np.dot(db_embedding, embedding))

        if similarity > best_score:
            best_score = similarity
            best_location = location
            best_camera = camera_name
            best_frame = frame_path

    if best_score >= SIMILARITY_THRESHOLD:
        return "MATCH", round(best_score, 4), best_location, best_camera, best_frame

    return "No Match", round(best_score, 4), None, None, None
# ================= STORE VIDEO EMBEDDINGS (1 FPS) =================
def store_video_embeddings(video_path, camera_name, location):

    conn = get_connection()
    cursor = conn.cursor()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    frame_interval = int(fps/3)  # 3 frame per second
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:

            frame = preprocess_frame(frame)
            faces = app.get(frame)

            if len(faces) == 0:
                frame_number += 1
                continue

            for face in faces:
                embedding = face.embedding

                embedding = embedding / np.linalg.norm(embedding)

            frame_folder = "static/cctv_frames"
            os.makedirs(frame_folder, exist_ok=True)

            frame_path = f"{frame_folder}/{camera_name}_{frame_number}.jpg"
           

            cv2.imwrite(frame_path, frame)

            cursor.execute("""
            INSERT INTO cctv_embeddings
            (video_name, camera_name, frame_number, embedding, location, frame_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (
            video_path,
            camera_name,
            frame_number,
            embedding.tolist(),
            location,
            frame_path
            ))

        frame_number += 1

    conn.commit()
    cursor.close()
    conn.close()
    cap.release()

    return "Video embeddings stored successfully"


# ================= SEARCH IMAGE IN CCTV =================
def search_image_in_cctv_db(image_path):

    test_embedding = create_embedding_from_image(image_path)
    if test_embedding is None:
        return None, None

    test_embedding = np.array(test_embedding, dtype=np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT embedding, location, camera_name, frame_path FROM cctv_embeddings")
    records = cur.fetchall()

    best_score = -1
    best_location = None
    best_camera = None

    best_frame = None

    for embedding_data, location, camera_name, frame_path in records:
        db_embedding = np.array(embedding_data, dtype=np.float32)
        db_embedding /= np.linalg.norm(db_embedding)

        similarity = float(np.dot(db_embedding, test_embedding))
        print("Similarity:", similarity)

        if similarity > best_score:
            best_score = similarity
            best_location = location
            best_camera = camera_name
            best_frame = frame_path
    print("Best Score:", best_score)

   
    if best_score >= SIMILARITY_THRESHOLD:
        return best_camera, best_location, best_frame

    return None, None ,None



# ================= TWIN DETECTION =================

def detect_possible_twin(image_path, twin_threshold=0.80):

    embedding = create_embedding_from_image(image_path)

    if embedding is None:
        return []

    embedding = np.array(embedding, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT embedding, location, camera_name, frame_path
        FROM cctv_embeddings
    """)

    records = cur.fetchall()

    cur.close()
    conn.close()

    possible_twins = []

    for embedding_data, location, camera_name, frame_path in records:

        db_embedding = np.array(embedding_data, dtype=np.float32)

        norm = np.linalg.norm(db_embedding)
        if norm == 0:
            continue

        db_embedding /= norm

        similarity = float(np.dot(db_embedding, embedding))

        # Higher threshold for twin detection
        if similarity >= twin_threshold:

            possible_twins.append({
                "similarity": round(similarity,4),
                "location": location,
                "camera": camera_name,
                "frame": frame_path
            })

    return possible_twins

    