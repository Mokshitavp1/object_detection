import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def send_alert(snapshot_path: Path | None, detection_event: dict) -> None:
    """Send a Gmail SMTP alert for a detection event. Never raises."""
    smtp_user  = os.environ.get("SMTP_USER", "")
    smtp_pass  = os.environ.get("SMTP_PASSWORD", "")
    recipient  = os.environ.get("ALERT_RECIPIENT", "")

    if not all([smtp_user, smtp_pass, recipient]):
        print("[WARNING] Email alert skipped — SMTP_USER, SMTP_PASSWORD, or ALERT_RECIPIENT not set.")
        return

    msg = MIMEMultipart("mixed")
    msg["From"]    = smtp_user
    msg["To"]      = recipient
    msg["Subject"] = f"Detection Alert — {detection_event['class']} at {detection_event['timestamp']}"

    body = "\n".join(f"{k}: {v}" for k, v in detection_event.items())
    msg.attach(MIMEText(body, "plain"))

    if snapshot_path is not None and snapshot_path.exists():
        image_data = MIMEImage(snapshot_path.read_bytes(), _subtype="jpeg")
        image_data.add_header("Content-Disposition", "attachment", filename=snapshot_path.name)
        msg.attach(image_data)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Email error: {e}")
