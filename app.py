import os
import uuid
from pathlib import Path as FilePath

from flask import flash, redirect, url_for
from flask import Flask, request, session, flash, redirect, url_for, render_template
from flask import get_flashed_messages

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    render_template_string,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from database import get_connection
from face_engine import (
    store_missing_person,
    search_face,
    search_face_for_admin,
    store_video_embeddings,
    search_image_in_cctv_db,
    detect_possible_twin,
)

# Load local .env values. On Render, configure these as environment variables.
load_dotenv()

BASE_DIR = FilePath(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "missing-person-finder-development-key"
)

# CCTV videos can be large. Adjust this limit to suit your storage plan.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def _extension(filename):
    """Return a lowercase file extension without the leading dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def save_upload(file_storage, allowed_extensions):
    """Validate and save an uploaded file with a collision-resistant name.

    Returns the relative path stored in the database, e.g.
    'static/uploads/uuid_photo.jpg'.
    """
    if file_storage is None or not file_storage.filename:
        raise ValueError("No file selected.")

    original_name = secure_filename(file_storage.filename)
    ext = _extension(original_name)

    if not original_name or ext not in allowed_extensions:
        raise ValueError("Unsupported file type.")

    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    destination = UPLOAD_FOLDER / unique_name
    file_storage.save(destination)

    return f"static/uploads/{unique_name}"


def static_file_url(path):
    """Convert a stored static path into a URL usable by an HTML img/video tag."""
    if not path:
        return ""
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("/"):
        normalized = normalized[1:]
    if normalized.startswith("static/"):
        normalized = normalized[len("static/"):]
    return url_for("static", filename=normalized)


def is_admin():
    return session.get("role") == "admin"


def is_user():
    return session.get("role") == "user"


def verify_password_and_upgrade(table, username, supplied_password, stored_password):
    """Verify modern hashes and support old plaintext records during migration.

    Existing plaintext passwords are upgraded to Werkzeug hashes after a
    successful login. The table argument is restricted to known table names.
    """
    if not stored_password:
        return False

    is_hash = str(stored_password).startswith(("pbkdf2:", "scrypt:"))
    if is_hash:
        try:
            return check_password_hash(stored_password, supplied_password)
        except (ValueError, TypeError):
            return False

    # Legacy plaintext comparison: temporary compatibility for existing users.
    if supplied_password != stored_password:
        return False

    if table not in {"users", "admins"}:
        return True

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"UPDATE {table} SET password=%s WHERE username=%s",
            (generate_password_hash(supplied_password), username),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return True


def draw_bounding_box(frame_path, target_embedding):
    """Draw a box around the detected face closest to the target embedding.

    Note: the score below is Euclidean distance, not cosine distance.
    """
    image = cv2.imread(str(frame_path))
    if image is None:
        return frame_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return frame_path

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    best_face = None
    best_score = float("inf")

    for (x, y, w, h) in faces:
        face_img = image[y:y + h, x:x + w]
        try:
            representations = DeepFace.represent(
                img_path=face_img,
                model_name="ArcFace",
                enforce_detection=False,
            )
            embedding = np.asarray(representations[0]["embedding"])
            score = np.linalg.norm(
                np.asarray(target_embedding) - embedding
            )

            if score < best_score:
                best_score = score
                best_face = (x, y, w, h)
        except Exception:
            # Skip a face if the model cannot generate its representation.
            continue

    if best_face is not None:
        x, y, w, h = best_face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    frame_path = str(frame_path)
    output_path = (
        frame_path[:-4] + "_boxed.jpg"
        if frame_path.lower().endswith(".jpg")
        else frame_path + "_boxed.jpg"
    )
    cv2.imwrite(output_path, image)
    return output_path


@app.errorhandler(413)
def file_too_large(_error):
    return "File is too large. Please upload a smaller file.", 413


@app.route("/")
def index():
    return render_template("index.html")



# ---------------- USER REGISTRATION ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Username, email and password are required.", "error")
            return redirect(url_for("register"))

        conn = get_connection()
        cur = conn.cursor()

        try:
            # Check whether username already exists
            cur.execute(
                "SELECT id FROM users WHERE username = %s",
                (username,)
            )

            if cur.fetchone():
                flash(
                    "Username already exists. Please choose another username.",
                    "error"
                )
                return redirect(url_for("register"))

            # Check whether email already exists
            cur.execute(
                "SELECT id FROM users WHERE email = %s",
                (email,)
            )

            if cur.fetchone():
                flash(
                    "Email address is already registered. Please use another email.",
                    "error"
                )
                return redirect(url_for("register"))

            # Register new user
            cur.execute(
                """
                INSERT INTO users (username, email, password)
                VALUES (%s, %s, %s)
                """,
                (
                    username,
                    email,
                    generate_password_hash(password)
                )
            )

            conn.commit()

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            conn.rollback()

            # Handle database uniqueness errors safely
            if getattr(e, "pgcode", None) == "23505":
                constraint = getattr(
                    getattr(e, "diag", None),
                    "constraint_name",
                    ""
                )

                if "username" in constraint:
                    flash("Username already exists. Please choose another.", "error")
                elif "email" in constraint:
                    flash("Email already registered. Please use another.", "error")
                else:
                    flash("This account information is already registered.", "error")

                return redirect(url_for("register"))

            # Unexpected errors should still be visible in the terminal
            raise

        finally:
            cur.close()
            conn.close()

    return render_template("register.html")

# ---------------- USER LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, password FROM users WHERE username=%s",
                (username,),
            )
            user = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if user and verify_password_and_upgrade(
            "users", username, password, user[1]
        ):
            session.clear()
            session["user_id"] = user[0]
            session["role"] = "user"
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, password FROM admins WHERE username=%s",
                (username,),
            )
            admin = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if admin and verify_password_and_upgrade(
            "admins", username, password, admin[1]
        ):
            session.clear()
            session["admin_id"] = admin[0]
            session["role"] = "admin"
            return redirect("/admin_dashboard")

        return render_template(
            "admin_login.html", error="Invalid username or password."
        )

    return render_template("admin_login.html")


# ---------------- USER DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if not is_user():
        return redirect("/")
    return render_template("dashboard.html")


# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not is_admin():
        return redirect("/admin_login")
    return render_template("admin_dashboard.html")


# ---------------- USER UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not is_user():
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM missing_persons WHERE user_id=%s",
            (session["user_id"],),
        )
        existing = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if existing:
        return "Details already uploaded."

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        description = request.form.get("description", "").strip()

        try:
            path = save_upload(
                request.files.get("photo"), ALLOWED_IMAGE_EXTENSIONS
            )
        except ValueError as exc:
            return str(exc), 400

        success, message = store_missing_person(
            session["user_id"], name, age, description, path
        )
        return message

    return render_template("upload.html")


# ---------------- USER CCTV SEARCH ----------------
@app.route("/search")
def search():
    if not is_user():
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT photo_path FROM missing_persons WHERE user_id=%s",
            (session["user_id"],),
        )
        data = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if not data:
        return "No uploaded image found."

    filepath = data[0]
    result, score, location, camera_name, frame_path = search_face(
        session["user_id"]
    )

    if result == "MATCH":
        return render_template_string(
            """
            <h2>Possible Match Found</h2>
            <p><strong>Camera:</strong> {{ camera_name }}</p>
            <p><strong>Location:</strong> {{ location }}</p>
            <p><strong>Similarity score:</strong> {{ score }}</p>
            <div style="display:flex;gap:40px;justify-content:center;flex-wrap:wrap">
              <div><h3>Uploaded Image</h3>
                <img src="{{ uploaded_url }}" width="250"></div>
              <div><h3>Matched CCTV Frame</h3>
                <img src="{{ frame_url }}" width="350"></div>
            </div>
            <br><a href="/dashboard">Back</a>
            """,
            camera_name=camera_name,
            location=location,
            score=score,
            uploaded_url=static_file_url(filepath),
            frame_url=static_file_url(frame_path),
        )

    return render_template_string(
        """
        <h2>No Match Found</h2>
        <div style="text-align:center">
          <h3>Uploaded Image</h3>
          <img src="{{ uploaded_url }}" width="250">
        </div>
        <br><a href="/dashboard">Back</a>
        """,
        uploaded_url=static_file_url(filepath),
    )


# ---------------- USER STATUS ----------------
@app.route("/status")
def status():
    if not is_user():
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT name, match_status
            FROM missing_persons
            WHERE user_id=%s
            """,
            (session["user_id"],),
        )
        result = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    return render_template("status.html", data=result)


# ---------------- ADMIN UPLOAD AND SEARCH ----------------
@app.route("/admin_upload_search", methods=["GET", "POST"])
def admin_upload_search():
    if not is_admin():
        return redirect("/admin_login")

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()

        try:
            filepath = save_upload(
                request.files.get("photo"), ALLOWED_IMAGE_EXTENSIONS
            )
        except ValueError as exc:
            return str(exc), 400

        store_missing_person(None, name, age, "", filepath)

        result, score, location, camera_name, frame_path = (
            search_face_for_admin(filepath)
        )
        twins = detect_possible_twin(filepath)

        if result == "MATCH":
            conn = get_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    UPDATE missing_persons
                    SET match_status='Found'
                    WHERE name=%s
                    """,
                    (name,),
                )
                conn.commit()
            finally:
                cur.close()
                conn.close()

            return render_template_string(
                """
                <!doctype html>
                <html><head><title>Possible Match Found</title>
                <style>
                  body {font-family:Arial,sans-serif;text-align:center;background:#f4f6f9}
                  .container {width:80%;margin:auto}
                  .info {margin-bottom:30px;font-size:18px}
                  .image-container {display:flex;justify-content:center;gap:40px;flex-wrap:wrap}
                  .image-box {background:white;padding:15px;border-radius:10px;
                    border:2px solid #ddd;box-shadow:0 4px 10px rgba(0,0,0,.1)}
                  img {max-width:100%;height:auto;border-radius:8px}
                  a {display:inline-block;margin-top:30px;padding:10px 20px;
                    background:#007BFF;color:white;text-decoration:none;border-radius:6px}
                </style></head><body>
                <div class="container">
                  <h2>Possible Match Found</h2>
                  <div class="info">
                    <p><strong>Camera Name:</strong> {{ camera_name }}</p>
                    <p><strong>Location:</strong> {{ location }}</p>
                    <p><strong>Similarity Score:</strong> {{ score }}</p>
                  </div>
                  <div class="image-container">
                    <div class="image-box"><h3>Uploaded Image</h3>
                      <img src="{{ uploaded_url }}" width="250"></div>
                    <div class="image-box"><h3>Matched CCTV Frame</h3>
                      <img src="{{ frame_url }}" width="350"></div>
                  </div>
                  <a href="/admin_dashboard">Back to Dashboard</a>
                </div></body></html>
                """,
                camera_name=camera_name,
                location=location,
                score=score,
                uploaded_url=static_file_url(filepath),
                frame_url=static_file_url(frame_path),
            )

        return render_template_string(
            """
            <h2>No Match Found</h2>
            <p><strong>Similarity Score:</strong> {{ score }}</p>
            <h3>Uploaded Image</h3>
            <img src="{{ uploaded_url }}" width="200">
            <br><a href="/admin_dashboard">Back to Dashboard</a>
            """,
            score=score,
            uploaded_url=static_file_url(filepath),
        )

    return render_template("admin_upload_search.html")


# ---------------- ADMIN CCTV VIDEO UPLOAD ----------------
@app.route("/admin_upload_video", methods=["GET", "POST"])
def admin_upload_video():
    if not is_admin():
        return redirect("/admin_login")

    if request.method == "POST":
        camera_name = request.form.get("camera_name", "").strip()
        cctv_location = request.form.get("location", "").strip()

        try:
            video_path = save_upload(
                request.files.get("video"), ALLOWED_VIDEO_EXTENSIONS
            )
        except ValueError as exc:
            return str(exc), 400

        message = store_video_embeddings(
            video_path, camera_name, cctv_location
        )

        return render_template_string(
            '<h2>{{ message }}</h2><a href="/admin_dashboard">Back</a>',
            message=message,
        )

    return render_template("admin_upload_video.html")


# ---------------- ADMIN IMAGE SEARCH IN CCTV ----------------
@app.route("/search_cctv", methods=["GET", "POST"])
def search_cctv():
    if not is_admin():
        return redirect("/admin_login")

    if request.method == "POST":
        try:
            image_path = save_upload(
                request.files.get("photo"), ALLOWED_IMAGE_EXTENSIONS
            )
        except ValueError as exc:
            return str(exc), 400

        camera_name, location, frame_path = search_image_in_cctv_db(
            image_path
        )

        if camera_name:
            return render_template_string(
                """
                <h2>Possible Match Found</h2>
                <p><strong>Camera:</strong> {{ camera_name }}</p>
                <p><strong>Location:</strong> {{ location }}</p>
                <div style="display:flex;gap:40px;justify-content:center;flex-wrap:wrap">
                  <div><h3>Uploaded Image</h3>
                    <img src="{{ uploaded_url }}" width="250"></div>
                  <div><h3>Matched CCTV Frame</h3>
                    <img src="{{ frame_url }}" width="350"></div>
                </div>
                <br><a href="/admin_dashboard">Back</a>
                """,
                camera_name=camera_name,
                location=location,
                uploaded_url=static_file_url(image_path),
                frame_url=static_file_url(frame_path),
            )

        return '<h2>No Match Found</h2><a href="/admin_dashboard">Back</a>'

    return render_template("search_cctv.html")


# ---------------- ADMIN VIEW USERS ----------------
@app.route("/view_users")
def view_users():
    if not is_admin():
        return redirect("/admin_login")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, username FROM users")
        users = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return render_template("view_users.html", users=users)


# ---------------- ADMIN VIEW CASES ----------------
@app.route("/view_cases")
def view_cases():
    if not is_admin():
        return redirect("/admin_login")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, name, age, match_status
            FROM missing_persons
            """
        )
        cases = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return render_template("view_cases.html", cases=cases)


# ---------------- ADMIN ANALYTICS ----------------
@app.route("/analytics")
def analytics():
    if not is_admin():
        return redirect("/admin_login")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM missing_persons")
        total_cases = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*)
            FROM missing_persons
            WHERE match_status='Found'
            """
        )
        total_found = cur.fetchone()[0]
    finally:
        cur.close()
        conn.close()

    return render_template(
        "analytics.html",
        total_users=total_users,
        total_cases=total_cases,
        total_found=total_found,
    )


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/admin_logout")
def admin_logout():
    session.clear()
    return redirect("/")



@app.route("/test_flash")
def test_flash():
    flash("Flash messages are working!", "success")

    print("SESSION DATA:", dict(session))

    messages = get_flashed_messages(with_categories=True)

    print("FLASH MESSAGES:", messages)

    return render_template_string("""
        <h2>Flash Test</h2>

        {% for category, message in messages %}
            <p style="color: green; font-size: 24px;">
                {{ message }}
            </p>
        {% endfor %}
    """, messages=messages)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
