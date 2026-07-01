from flask import Flask, render_template, request, redirect, session
from database import get_connection
import os
from flask import flash
import cv2
from deepface import DeepFace
import numpy as np

'''
def draw_bounding_box(frame_path, target_embedding):

    image = cv2.imread(frame_path)

    if image is None:
        return frame_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    best_face = None
    best_score = 999

    for (x, y, w, h) in faces:

        face_img = image[y:y+h, x:x+w]

        try:
            rep = DeepFace.represent(
                img_path=face_img,
                model_name="ArcFace",
                enforce_detection=False
            )

            embedding = np.array(rep[0]["embedding"])

            # cosine distance
            score = np.linalg.norm(target_embedding - embedding)

            if score < best_score:
                best_score = score
                best_face = (x, y, w, h)

        except:
            continue

    if best_face is not None:
        x, y, w, h = best_face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)

    boxed_path = frame_path.replace(".jpg", "_boxed.jpg")
    cv2.imwrite(boxed_path, image)

    return boxed_path
'''

from face_engine import (
    store_missing_person,
    search_face,
    search_face_for_admin,
    store_video_embeddings,
    search_image_in_cctv_db,
    detect_possible_twin
)


app = Flask(__name__)
app.secret_key = "secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============= Home ================
# @app.route("/")
#def home():
 #   return render_template("home.html")
# ================= INDEX =================
@app.route("/")
def index():
    return render_template("index.html")


# ================= USER REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (username, email, password)
            VALUES (%s, %s, %s)
        """, (username, email, password))
        conn.commit()
        cur.close()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


# ================= USER LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username=%s AND password=%s",
                    (username, password))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user[0]
            session["role"] = "user"
            return redirect("/dashboard")

    return render_template("login.html")


# ================= ADMIN LOGIN =================
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM admins WHERE username=%s AND password=%s",
                    (username, password))
        admin = cur.fetchone()
        cur.close()
        conn.close()

        if admin:
            session.clear()
            session["admin_id"] = admin[0]
            session["role"] = "admin"
            return redirect("/admin_dashboard")

    return render_template("admin_login.html")


# ================= USER DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if session.get("role") != "user":
        return redirect("/")
    return render_template("dashboard.html")


# ================= ADMIN DASHBOARD =================
@app.route("/admin_dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect("/")
    return render_template("admin_dashboard.html")


# ================= USER UPLOAD =================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if session.get("role") != "user":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM missing_persons WHERE user_id=%s",
                (session["user_id"],))
    existing = cur.fetchone()

    if existing:
        cur.close()
        conn.close()
        return "Details already uploaded."

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        description = request.form["description"]
        photo = request.files["photo"]

        path = os.path.join(UPLOAD_FOLDER, photo.filename)
        photo.save(path)

        success, msg = store_missing_person(
            session["user_id"], name, age, description, path
        )

        cur.close()
        conn.close()

        return msg

    cur.close()
    conn.close()
    return render_template("upload.html")


# ================= USER SEARCH =================
@app.route("/search")
def search():

    if session.get("role") != "user":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT photo_path FROM missing_persons WHERE user_id=%s",
                (session["user_id"],))
    data = cur.fetchone()

    cur.close()
    conn.close()

    if not data:
        return "No uploaded image found."

    filepath = data[0]

    result, score, location, camera_name, frame_path = search_face(session["user_id"])

    if result == "MATCH":

        return f"""
        <h2>Match Found</h2>

        <p><strong>Camera:</strong> {camera_name}</p>
        <p><strong>Location:</strong> {location}</p>
        <p><strong>Similarity Score:</strong> {score}</p>

        <div style="display:flex; gap:40px; justify-content:center">

            <div>
                <h3>Uploaded Image</h3>
                <img src="/{filepath}" width="250">
            </div>

            <div>
                <h3>Matched CCTV Frame</h3>
                <img src="/{frame_path}" width="350">
            </div>

        </div>

        <br><br>
        <a href='/dashboard'>Back</a>
        """
    else:
        return f"""
        <h2>No Match Found</h2>

        <div style="text-align:center">
            <h3>Uploaded Image</h3>
            <img src="/{filepath}" width="250">
        </div>

        <br><br>
        <a href='/dashboard'>Back</a>
        """
# ================= STATUS =================
@app.route("/status")
def status():
    if session.get("role") != "user":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, match_status FROM missing_persons WHERE user_id=%s",
                (session["user_id"],))
    result = cur.fetchone()
    cur.close()
    conn.close()

    return render_template("status.html", data=result)


# ================= ADMIN UPLOAD & SEARCH =================
@app.route("/admin_upload_search", methods=["GET", "POST"])
def admin_upload_search():

    if session.get("role") != "admin":
        return redirect("/admin_login")

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        file = request.files["photo"]

        if file.filename == "":
            return "No file selected"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Store case
        store_missing_person(None, name, age, "", filepath)

        # Search in CCTV embeddings
      
        result, score, location, camera_name, frame_path = search_face_for_admin(filepath)
        # Detect possible twins
        twins = detect_possible_twin(filepath)

        # ================= MATCH FOUND =================
        if result == "MATCH":
           

            conn = get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE missing_persons
                SET match_status = 'Found'
                WHERE name = %s
            """, (name,))

            conn.commit()
            cur.close()
            conn.close()

            return f"""
            <html>
            <head>
            <style>

            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f4f6f9;
            }}

            .container {{
                width: 80%;
                margin: auto;
            }}

            .info {{
                margin-bottom: 30px;
                font-size: 18px;
            }}

            .image-container {{
                display: flex;
                justify-content: center;
                gap: 40px;
            }}

            .image-box {{
                background: white;
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #ddd;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }}

            .image-box img {{
                border-radius: 8px;
            }}

            a {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background: #007BFF;
                color: white;
                text-decoration: none;
                border-radius: 6px;
            }}

            a:hover {{
                background: #0056b3;
            }}

            </style>
            </head>

            <body>

            <div class="container">

            <h2>Match Found</h2>

            <div class="info">
            <p><strong>Camera Name:</strong> {camera_name}</p>
            <p><strong>Location:</strong> {location}</p>
            <p><strong>Similarity Score:</strong> {score}</p>
            </div>

            <div class="image-container">

            <div class="image-box">
            <h3>Uploaded Image</h3>
            <img src="/{filepath}" width="250">
            </div>

            <div class="image-box">
            <h3>Matched CCTV Frame</h3>
            <img src="/{frame_path}" width="350">
            </div>

            </div>

            <a href='/admin_dashboard'>Back to Dashboard</a>

            </div>

            </body>
            </html>
            """
        # ================= NO MATCH =================
        else:

            return f"""
            <h2>No Match Found</h2>

            <p><strong>Similarity Score:</strong> {score}</p>

            <h3>Uploaded Image</h3>
            <img src="/{filepath}" width="200">

            <br><br>
            <a href='/admin_dashboard'>Back to Dashboard</a>
            """

    return render_template("admin_upload_search.html")


# ================= ADMIN CCTV UPLOAD =================
@app.route("/admin_upload_video", methods=["GET", "POST"])
def admin_upload_video():

    if session.get("role") != "admin":
        return redirect("/")

    if request.method == "POST":

        camera_name = request.form["camera_name"]
        cctv_location = request.form["location"]
        video = request.files["video"]

        if video.filename == "":
            return "No video selected"

        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        message = store_video_embeddings(
            video_path,
            camera_name,
            cctv_location
        )

        return f"""
        <h2>{message}</h2>
        <a href='/admin_dashboard'>Back</a>
        """

    return render_template("admin_upload_video.html")


# ================= SEARCH IMAGE IN CCTV =================
@app.route("/search_cctv", methods=["GET", "POST"])
def search_cctv():

    if request.method == "POST":

        image = request.files["photo"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        camera_name, location, frame_path = search_image_in_cctv_db(image_path)

        if camera_name:
            return f"""
                <h2>Match Found</h2>

                <p><strong>Camera:</strong> {camera_name}</p>
                <p><strong>Location:</strong> {location}</p>

                <div style="display:flex; gap:40px; justify-content:center">

                    <div>
                        <h3>Uploaded Image</h3>
                        <img src="/{image_path}" width="250">
                    </div>

                    <div>
                        <h3>Matched CCTV Frame</h3>
                        <img src="/{frame_path}" width="350">
                    </div>

                </div>

                <br><br>
                <a href='/admin_dashboard'>Back</a>
                """
        else:
            return """
            <h2>No Match Found</h2>
            <a href='/admin_dashboard'>Back</a>
            """

    return render_template("search_cctv.html")


# ================= VIEW USERS =================
@app.route("/view_users")
def view_users():
    if session.get("role") != "admin":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, username FROM users")
    users = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("view_users.html", users=users)

# ================= VIEW CASES =================
@app.route("/view_cases")
def view_cases():
    if session.get("role") != "admin":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, age, match_status 
        FROM missing_persons
    """)
    cases = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("view_cases.html", cases=cases)

# ================= ANALYTICS =================
@app.route("/analytics")
def analytics():
    if session.get("role") != "admin":
        return redirect("/")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM missing_persons")
    total_cases = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM missing_persons WHERE match_status='Found'")
    total_found = cur.fetchone()[0]

    cur.close()
    conn.close()

    return render_template("analytics.html",
                           total_users=total_users,
                           total_cases=total_cases,
                           total_found=total_found)
# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/admin_logout")
def admin_logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
