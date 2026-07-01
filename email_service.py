import smtplib
from email.mime.text import MIMEText
from database import get_connection

SENDER_EMAIL = "mubashirtkm2@gmail.com"
SENDER_PASSWORD = "ncbbvdmtljogaiwb"  # use Gmail App Password

def send_match_alert(user_id, location):

    conn = get_connection()
    cur = conn.cursor()

    # Get user email
    cur.execute("SELECT email FROM users WHERE id=%s", (user_id,))
    user_email = cur.fetchone()[0]

    # Get admin emails
    cur.execute("SELECT email FROM admins")
    admin_emails = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    recipients = [user_email] + admin_emails

    # Email body with location
    message_body = f"""
    🚨 MATCH FOUND ALERT 🚨

    The missing person has been detected.

    📍 Location: {location}

    Please take necessary action immediately.
    """
    message = MIMEText(message_body)
    message["Subject"] = "🚨 Missing Person Match Found"
    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(recipients)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, message.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print("Email failed:", e)