import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- CONFIG ---------------- #

EMBEDDINGS_FOLDER = "face_embeddings"
YOLO_MODEL = "yolov8n.pt"
SIMILARITY_THRESHOLD = 0.40

# ---------------------------------------- #

# Load YOLO
face_detector = YOLO(YOLO_MODEL)


# ========== LOAD STORED EMBEDDINGS ========== #

def load_embeddings(folder_path):
    embeddings_db = {}

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            person_name = file.replace(".npy", "")
            path = os.path.join(folder_path, file)

            embeddings = np.load(path)
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )

            embeddings_db[person_name] = embeddings

    return embeddings_db


# ========== FACE DETECTION ========== #

def detect_face(image):
    results = face_detector(image)

    if len(results[0].boxes) == 0:
        return None

    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box[:4])

    return image[y1:y2, x1:x2]


# ========== EMBEDDING CREATION ========== #

def get_embedding(face_img):
    rep = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )

    embedding = np.asarray(rep[0]["embedding"], dtype="float32")
    embedding /= np.linalg.norm(embedding)
    return embedding


# ========== FACE MATCHING ========== #

def recognize_face(test_embedding, embeddings_db):

    best_match = "No Match"
    best_score = 0

    for person_name, db_embeddings in embeddings_db.items():

        similarities = np.dot(db_embeddings, test_embedding)
        max_similarity = np.max(similarities)

        if max_similarity > best_score:
            best_score = max_similarity
            best_match = person_name

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match, float(best_score)

    return "No Match", float(best_score)


# ========== MAIN EXECUTION ========== #

if __name__ == "__main__":

    print("Loading stored embeddings...")
    embeddings_db = load_embeddings(EMBEDDINGS_FOLDER)

    print("Processing test image...")
    image = cv2.imread("test.jpg")

    if image is None:
        print("Test image not found!")
        exit()

    face = detect_face(image)

    if face is None:
        print("No face detected!")
        exit()

    test_embedding = get_embedding(face)

    name, similarity = recognize_face(test_embedding, embeddings_db)

    if name != "No Match":
        print(f"\n✅ Match Found!")
        print(f"Person: {name}")
        print(f"Similarity Score: {similarity:.4f}")
    else:
        print("\n❌ No Match Found")
        print(f"Highest Similarity: {similarity:.4f}")