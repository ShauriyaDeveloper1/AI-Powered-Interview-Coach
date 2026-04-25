import base64
import hashlib
import io
import json
import os
import queue
import re
import secrets
import smtplib
import tempfile
import threading
import time
import uuid
import importlib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import cv2
import nltk
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file, session
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename

from ats_checker import ALLOWED_EXTENSIONS, analyze_resume_file

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

_firebase_available = False
firebase_auth = None
_firebase_app = None

try:
    import firebase_admin  # type: ignore
    from firebase_admin import auth as firebase_auth  # type: ignore
    from firebase_admin import credentials  # type: ignore

    _svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    _svc_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

    if _svc_json:
        cred = credentials.Certificate(json.loads(_svc_json))
        _firebase_app = firebase_admin.initialize_app(cred)
        _firebase_available = True
    elif _svc_path and os.path.exists(_svc_path):
        cred = credentials.Certificate(_svc_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firebase_available = True
except Exception:
    _firebase_available = False
    firebase_auth = None
    _firebase_app = None

try:
    from openai import OpenAI

    _OPENAI_NEW = True
except ImportError:
    import openai

    _OPENAI_NEW = False

# MoviePy v1 exposes `moviepy.editor` and mutating setters like `.set_duration()`.
# MoviePy v2 removed `moviepy.editor` and uses immutable-style `.with_duration()`.
try:
    import moviepy.editor as mp  # type: ignore
except Exception:
    try:
        import moviepy as mp  # type: ignore
    except Exception:
        mp = None
from PIL import Image, ImageDraw


# Prefer bundled NLTK data (repo-local) when available.
_LOCAL_NLTK_DATA = Path(__file__).resolve().parent / "nltk_data"
try:
    if _LOCAL_NLTK_DATA.exists():
        local_path = str(_LOCAL_NLTK_DATA)
        if local_path not in nltk.data.path:
            nltk.data.path.insert(0, local_path)
except Exception:
    pass


# Ensure necessary NLTK resources are available.
def ensure_nltk_resources():
    # Avoid blocking startup on hosted environments unless explicitly enabled.
    auto_download = os.getenv("NLTK_AUTO_DOWNLOAD", "0").strip() == "1"
    # If NLTK_DATA is set, prefer downloading to its first path segment.
    nltk_data_dir = os.getenv("NLTK_DATA")
    download_dir = None
    if nltk_data_dir:
        download_dir = nltk_data_dir.split(os.pathsep)[0]
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except (LookupError, OSError):
            if auto_download:
                try:
                    if download_dir:
                        nltk.download(
                            name,
                            quiet=True,
                            download_dir=download_dir,
                            force=True,
                        )
                    else:
                        nltk.download(name, quiet=True, force=True)
                except Exception:
                    pass


ensure_nltk_resources()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "interview-coach-dev-secret")

# Hugging Face Spaces runs the app inside an iframe. Use cross-site secure
# cookies there so session auth survives the login redirect/fetch cycle.
if os.getenv("SPACE_ID"):
    app.config.update(
        SESSION_COOKIE_SAMESITE="None",
        SESSION_COOKIE_SECURE=True,
    )

USER_DB_FILE = "users.json"
REPORTS_DIR = "reports"

_email_otp_store = {}
_OTP_TTL_SECONDS = 10 * 60

api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
openai_base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
freepik_api_key = os.getenv("FREEPIK_API_KEY")

client = None
if _OPENAI_NEW and api_key:
    try:
        client = OpenAI(api_key=api_key, base_url=openai_base_url)
    except Exception:
        client = None
elif api_key:
    openai.api_key = api_key
    openai.api_base = openai_base_url


QUESTION_BANK = [
    "Tell me about yourself.",
    "What is your greatest strength?",
    "What is your greatest weakness?",
    "Why do you want to work for this company?",
    "Where do you see yourself in five years?",
    "Describe a challenging situation at work and how you handled it.",
    "Why should we hire you?",
    "What are your salary expectations?",
    "Do you have any questions for us?",
]

_generated_videos = {}
_posture_states = {}

# RL integration (optional at runtime if rl dependencies are unavailable).
_rl_env = None
_rl_agent = None
_session_episodes = {}
_rl_available = False
_openenv_env = None

try:
    from rl_interview_coach import (
        InterviewCoachEnv,
        QLearningAgent,
        TaskBank,
        TaskType,
        Action,
        FeedbackStrategy,
    )

    _rl_available = True
except Exception:
    _rl_available = False


def init_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"


def validate_username(username):
    if len(username) < 8:
        return False, "Username must be at least 8 characters"
    return True, "Username is valid"


def _load_users():
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users):
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def user_exists(username):
    return username in _load_users()


def add_user(username, password, profile=None, email_verified=False):
    users = _load_users()
    record = {
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "email_verified": bool(email_verified),
        "profile": profile or {},
    }
    users[username] = record
    _save_users(users)


def authenticate_user(username, password):
    users = _load_users()
    if username not in users:
        return False
    record = users[username]
    if isinstance(record, str):
        # Backward compatible: older DB stored just the password hash.
        return record == hash_password(password)
    if isinstance(record, dict):
        return (record.get("password_hash") or "") == hash_password(password)
    return False


def _is_valid_email(email: str) -> bool:
    email = (email or "").strip()
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email))


def _normalize_phone(phone: str) -> str:
    phone = (phone or "").strip()
    phone = re.sub(r"[^0-9+]", "", phone)
    return phone


def _otp_hash(email: str, otp: str) -> str:
    secret = os.getenv("OTP_SECRET") or app.secret_key
    payload = f"{email.strip().lower()}::{otp.strip()}::{secret}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _send_email_otp(email: str, otp: str) -> tuple[bool, str]:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT") or "587")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM") or smtp_user

    subject = "Your Interview Coach AI verification code"
    body = (
        "Your verification code is:\n\n"
        f"{otp}\n\n"
        "This code expires in 10 minutes."
    )

    # Dev fallback: if SMTP isn't configured, log the OTP.
    if not smtp_host or not smtp_user or not smtp_pass or not smtp_from:
        print(f"[DEV] Email OTP for {email}: {otp}")
        return True, "OTP generated (dev mode). Check the server console for the code."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "OTP sent to your email."
    except Exception as exc:
        print(f"Failed to send OTP email: {exc}")
        return False, "Unable to send OTP email. Please try again later."


def _firebase_upsert_user(email: str, full_name: str) -> str | None:
    """Create (or update) a Firebase Auth user if Firebase is configured."""

    if not _firebase_available or not firebase_auth or not _firebase_app:
        return None

    email = (email or "").strip()
    full_name = (full_name or "").strip()
    if not email:
        return None

    try:
        user = firebase_auth.get_user_by_email(email)
        firebase_auth.update_user(user.uid, display_name=full_name or user.display_name)
        return user.uid
    except Exception:
        try:
            user = firebase_auth.create_user(
                email=email,
                display_name=full_name or None,
                email_verified=True,
            )
            return user.uid
        except Exception:
            return None


def get_user_reports_path(username):
    return os.path.join(REPORTS_DIR, f"{username}_reports.json")


def save_interview_report(username, report_data):
    reports_file = get_user_reports_path(username)
    if os.path.exists(reports_file):
        with open(reports_file, "r", encoding="utf-8") as f:
            reports = json.load(f)
    else:
        reports = []

    report_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report_data["date"] = time.strftime("%Y-%m-%d")
    reports.append(report_data)

    with open(reports_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)


def get_user_reports(username):
    reports_file = get_user_reports_path(username)
    if not os.path.exists(reports_file):
        return []
    with open(reports_file, "r", encoding="utf-8") as f:
        return json.load(f)


class VideoTransformer:
    def __init__(self):
        self.posture_status = "Unknown"
        self.head_position = "Unknown"
        self.posture_history = []
        self.slouch_counter = 0
        self.posture_feedback = "Analyzing your posture..."
        self.last_face_seen = time.time()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face_and_shoulders(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None, None

        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        face_center_x = x + w // 2
        face_bottom_y = y + h
        shoulder_y = face_bottom_y + h // 2
        shoulder_width = w * 2.5
        left_shoulder_x = max(0, int(face_center_x - shoulder_width // 2))
        right_shoulder_x = min(img.shape[1], int(face_center_x + shoulder_width // 2))

        return (x, y, w, h), (left_shoulder_x, right_shoulder_x, shoulder_y)

    def analyze_posture(self, img):
        face_rect, shoulder_points = self.detect_face_and_shoulders(img)
        if face_rect is None or shoulder_points is None:
            if time.time() - self.last_face_seen > 3:
                self.posture_feedback = "Face not detected. Sit facing camera with good lighting."
                self.posture_status = "Not Visible"
            return self.posture_status

        x, _, w, _ = face_rect
        left_shoulder_x, right_shoulder_x, _ = shoulder_points

        face_center_x = x + w // 2
        shoulders_center_x = (left_shoulder_x + right_shoulder_x) // 2
        horizontal_offset = abs(face_center_x - shoulders_center_x)

        if horizontal_offset < w * 0.2:
            self.head_position = "Centered"
        else:
            self.head_position = "Tilted" if face_center_x < shoulders_center_x else "Leaning"

        if self.head_position == "Centered":
            self.posture_status = "Good"
            self.slouch_counter = max(0, self.slouch_counter - 1)
            self.posture_feedback = "Great posture! Keep it up."
        else:
            self.posture_status = "Needs Improvement"
            self.slouch_counter += 1
            self.posture_feedback = "Center your head above your shoulders."

        if self.slouch_counter > 5:
            self.posture_feedback = "You are slouching. Sit up straight for a professional appearance."

        self.last_face_seen = time.time()
        self.posture_history.append(self.posture_status)
        if len(self.posture_history) > 10:
            self.posture_history.pop(0)

        return self.posture_status

    def posture_recommendations(self):
        history = self.posture_history[-10:]
        if not history:
            return ["Ensure your full face is visible to begin analysis."]

        improvements = []
        needs_impr_ratio = history.count("Needs Improvement") / len(history)
        if needs_impr_ratio > 0.6:
            improvements.append("Keep shoulders level and sit upright.")
        if self.head_position in ("Tilted", "Leaning"):
            improvements.append("Center your head above your shoulders to appear attentive.")
        if self.slouch_counter > 3:
            improvements.append("Roll shoulders back and reduce slouching.")
        if self.posture_status == "Good" and not improvements:
            improvements.append("Maintain current posture. Great alignment.")
        return improvements


class InterviewCoach:
    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception:
            self.sentiment_analyzer = None
        try:
            self.stopwords = set(stopwords.words("english"))
        except Exception:
            self.stopwords = set()

        self.professional_words = {
            "accomplished",
            "achieved",
            "analyzed",
            "coordinated",
            "created",
            "delivered",
            "developed",
            "enhanced",
            "executed",
            "improved",
            "initiated",
            "launched",
            "managed",
            "optimized",
            "organized",
            "planned",
            "resolved",
            "spearheaded",
            "streamlined",
            "success",
        }

        self.filler_words = {
            "um",
            "uh",
            "like",
            "you know",
            "actually",
            "basically",
            "literally",
            "sort of",
            "kind of",
            "so",
            "well",
            "just",
            "stuff",
            "things",
        }

    def transcribe_audio(self, audio_file):
        if self.recognizer is None:
            return ""
        if audio_file is None:
            return ""
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio_data)
            except Exception:
                return ""

    def analyze_tone(self, text):
        if not text:
            return {
                "score": 0,
                "sentiment": "neutral",
                "feedback": "No speech detected to analyze tone.",
            }

        if self.sentiment_analyzer is None:
            return {
                "score": 0,
                "sentiment": "neutral",
                "feedback": "Tone analyzer is unavailable in this runtime. Continuing with neutral scoring.",
            }

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores["compound"]

        if compound_score >= 0.05:
            sentiment = "positive"
        elif compound_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        if sentiment == "positive":
            feedback = "Your tone is positive and enthusiastic, which is great for an interview."
            if compound_score > 0.5:
                feedback += " Avoid sounding overly enthusiastic to keep authenticity."
        elif sentiment == "negative":
            feedback = "Your tone comes across as somewhat negative. Emphasize strengths and achievements."
        else:
            feedback = "Your tone is neutral. Add a bit more enthusiasm when discussing achievements."

        return {"score": compound_score, "sentiment": sentiment, "feedback": feedback}

    def analyze_word_choice(self, text):
        if not text:
            return {
                "professional_word_count": 0,
                "filler_word_count": 0,
                "professional_words_used": [],
                "filler_words_used": [],
                "feedback": "No speech detected to analyze word choice.",
            }

        words = nltk.word_tokenize(text.lower())
        professional_words_used = [word for word in words if word in self.professional_words]
        filler_words_used = [filler for filler in self.filler_words if filler in text.lower()]

        feedback = ""
        if professional_words_used:
            feedback += (
                f"Good use of professional language: {', '.join(professional_words_used[:3])}. "
            )
        else:
            feedback += "Consider using more professional vocabulary. "

        if filler_words_used:
            feedback += f"Reduce filler words like {', '.join(filler_words_used[:3])}."
        else:
            feedback += "Great job minimizing filler words."

        return {
            "professional_word_count": len(professional_words_used),
            "filler_word_count": len(filler_words_used),
            "professional_words_used": professional_words_used,
            "filler_words_used": filler_words_used,
            "feedback": feedback,
        }

    def analyze_confidence(self, text, tone_analysis):
        if not text:
            return {"confidence_score": 0, "feedback": "No speech detected to analyze confidence."}

        confidence_score = 5
        sentiment_score = tone_analysis["score"]
        if sentiment_score > 0:
            confidence_score += sentiment_score * 2
        elif sentiment_score < -0.2:
            confidence_score -= abs(sentiment_score) * 2

        hesitation_patterns = [
            r"\bI think\b",
            r"\bmaybe\b",
            r"\bpossibly\b",
            r"\bperhaps\b",
            r"\bI guess\b",
            r"\bsort of\b",
            r"\bkind of\b",
            r"\bI hope\b",
            r"\bI\'m not sure\b",
            r"\bI don\'t know\b",
        ]

        hesitation_count = sum(len(re.findall(pattern, text.lower())) for pattern in hesitation_patterns)
        confidence_score -= hesitation_count * 0.5

        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = (
            np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0
        )

        if avg_sentence_length > 20:
            confidence_score += 1
        elif avg_sentence_length < 8:
            confidence_score -= 1

        confidence_score = max(0, min(10, confidence_score))

        if confidence_score >= 8:
            feedback = "You sound very confident."
        elif confidence_score >= 6:
            feedback = "You sound reasonably confident."
        elif confidence_score >= 4:
            feedback = "Confidence is moderate. Speak more assertively."
        else:
            feedback = "Work on reducing hesitant phrases and speaking with conviction."

        return {"confidence_score": confidence_score, "feedback": feedback}

    def analyze_text_input(self, text):
        tone_analysis = self.analyze_tone(text)
        word_choice_analysis = self.analyze_word_choice(text)
        confidence_analysis = self.analyze_confidence(text, tone_analysis)

        return {
            "tone": tone_analysis,
            "word_choice": word_choice_analysis,
            "confidence": confidence_analysis,
            "text": text,
        }

    def provide_comprehensive_feedback(self, analysis_results):
        tone = analysis_results["tone"]
        word_choice = analysis_results["word_choice"]
        confidence = analysis_results["confidence"]
        posture = analysis_results.get(
            "posture", {"status": "Not analyzed", "feedback": "No posture analysis available."}
        )

        feedback = []
        feedback.append("INTERVIEW RESPONSE EVALUATION")
        feedback.append("")
        feedback.append(f"Tone: {tone['sentiment']} (Score: {tone['score']:.2f})")
        feedback.append(f"Tone Feedback: {tone['feedback']}")
        feedback.append("")
        feedback.append(f"Professional Words: {word_choice['professional_word_count']}")
        feedback.append(f"Filler Words: {word_choice['filler_word_count']}")
        feedback.append(f"Word Choice Feedback: {word_choice['feedback']}")
        feedback.append("")
        feedback.append(f"Confidence Score: {confidence['confidence_score']:.1f}/10")
        feedback.append(f"Confidence Feedback: {confidence['feedback']}")
        feedback.append("")
        feedback.append(f"Posture Status: {posture['status']}")
        feedback.append(f"Posture Feedback: {posture['feedback']}")
        feedback.append("")

        improvement_areas = []
        if tone["score"] < 0:
            improvement_areas.append("Use more positive language")
        if word_choice["filler_word_count"] > 3:
            improvement_areas.append("Reduce filler words")
        if word_choice["professional_word_count"] < 2:
            improvement_areas.append("Increase professional vocabulary")
        if confidence["confidence_score"] < 5:
            improvement_areas.append("Project higher confidence")
        if posture["status"] not in ("Good", "Not analyzed"):
            improvement_areas.append("Improve posture and body language")

        if improvement_areas:
            feedback.append("Areas to Improve:")
            feedback.extend([f"- {area}" for area in improvement_areas])
        else:
            feedback.append("Great job. Keep practicing.")

        return "\n".join(feedback)


def create_progress_report_pdf(username, reports, start_date=None, end_date=None):
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Interview Performance Report for {username}", styles["Title"]))
    elements.append(Spacer(1, 20))

    if start_date and end_date:
        filtered_reports = [r for r in reports if start_date <= r["date"] <= end_date]
    else:
        filtered_reports = reports

    if not filtered_reports:
        elements.append(Paragraph("No data available for the selected date range.", styles["Normal"]))
    else:
        num_sessions = len(filtered_reports)
        
        # Calculate averages safely, supporting both legacy and new session-based formats
        total_conf = 0
        total_sent = 0
        valid_conf_count = 0
        valid_sent_count = 0
        
        for r in filtered_reports:
            if r.get("type") == "session":
                turns = r.get("turns", [])
                for t in turns:
                    analysis = t.get("analysis", {})
                    if "confidence" in analysis:
                        total_conf += analysis["confidence"].get("confidence_score", 0)
                        valid_conf_count += 1
                    if "tone" in analysis:
                        total_sent += analysis["tone"].get("score", 0)
                        valid_sent_count += 1
            else:
                if "analysis" in r:
                    total_conf += r["analysis"].get("confidence", {}).get("confidence_score", 0)
                    total_sent += r["analysis"].get("tone", {}).get("score", 0)
                    valid_conf_count += 1
                    valid_sent_count += 1

        avg_confidence = (total_conf / valid_conf_count) if valid_conf_count > 0 else 0
        avg_sentiment = (total_sent / valid_sent_count) if valid_sent_count > 0 else 0

        elements.append(Paragraph(f"Sessions Analyzed: {num_sessions}", styles["Normal"]))
        elements.append(Paragraph(f"Average Confidence Score: {avg_confidence:.1f}/10", styles["Normal"]))
        elements.append(Paragraph(f"Average Tone Score: {avg_sentiment:.2f}", styles["Normal"]))
        elements.append(Spacer(1, 20))

        table_data = [["Date", "Question/Session", "Avg Confidence", "Avg Tone", "Prof Words", "Filler Words"]]
        for report in filtered_reports:
            if report.get("type") == "session":
                turns = report.get("turns", [])
                if not turns:
                    continue
                q = report.get("root_question", "Session")[:30] + "..."
                
                t_conf = sum(t["analysis"].get("confidence", {}).get("confidence_score", 0) for t in turns if "analysis" in t)
                t_tone = sum(t["analysis"].get("tone", {}).get("score", 0) for t in turns if "analysis" in t)
                prof_w = sum(t["analysis"].get("word_choice", {}).get("professional_word_count", 0) for t in turns if "analysis" in t)
                fill_w = sum(t["analysis"].get("word_choice", {}).get("filler_word_count", 0) for t in turns if "analysis" in t)
                
                table_data.append([
                    report.get("date", "Unknown"),
                    q,
                    f"{(t_conf / len(turns)):.1f}",
                    f"{(t_tone / len(turns)):.2f}",
                    str(prof_w),
                    str(fill_w),
                ])
            else:
                q = report.get("question", "")[:30] + "..."
                table_data.append([
                    report.get("date", "Unknown"),
                    q,
                    f"{report.get('analysis', {}).get('confidence', {}).get('confidence_score', 0):.1f}",
                    f"{report.get('analysis', {}).get('tone', {}).get('score', 0):.2f}",
                    str(report.get("analysis", {}).get("word_choice", {}).get("professional_word_count", 0)),
                    str(report.get("analysis", {}).get("word_choice", {}).get("filler_word_count", 0)),
                ])

        table = Table(table_data, colWidths=[80, 150, 80, 80, 80, 80])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        elements.append(table)

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def _fallback_transcript(role, experience, additional_details, interview_type):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"Interviewer: Welcome to this {interview_type.lower()} interview for the {role} role.\n"
        f"Candidate: Thank you. My {experience.lower()} background aligns well with the position.\n"
        "Interviewer: Could you summarize your relevant experience?\n"
        f"Candidate: {additional_details} These experiences have shaped my approach.\n"
        f"[Locally generated fallback at {ts} due to API unavailability]"
    )


def generate_interview_transcript(role, experience, additional_details, interview_type):
    model_engine = "gpt-3.5-turbo"
    prompt = (
        f"Generate a {interview_type} mock interview script for a {experience} {role} candidate. "
        f"Keep it concise and realistic. Additional details: {additional_details}"
    )
    messages = [
        {"role": "system", "content": "You generate realistic mock interview transcripts."},
        {"role": "user", "content": prompt},
    ]

    if not api_key:
        return _fallback_transcript(role, experience, additional_details, interview_type)

    try:
        if _OPENAI_NEW and client:
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                max_tokens=1024,
                n=1,
                temperature=0.7,
            )
            return response.choices[0].message.content

        legacy_resp = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages,
            max_tokens=1024,
            n=1,
            temperature=0.7,
        )
        return legacy_resp.choices[0].message.content
    except Exception:
        return _fallback_transcript(role, experience, additional_details, interview_type)


def generate_audio_for_video(script, lang="en"):
    cleaned_script = " ".join(line.strip() for line in script.splitlines() if line.strip())
    if not cleaned_script:
        raise ValueError("Script is empty")

    # Primary path: gTTS (network-based) for natural voice output.
    if gTTS is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
                audio_file = temp.name
            tts = gTTS(text=cleaned_script, lang=lang, slow=False)
            tts.save(audio_file)
            return audio_file
        except Exception:
            pass

    # Fallback path: pyttsx3 (offline system TTS) when gTTS is unavailable.
    try:
        pyttsx3 = importlib.import_module("pyttsx3")
    except Exception as exc:
        raise RuntimeError("No text-to-speech engine available") from exc

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        audio_file = temp.name

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(cleaned_script, audio_file)
        engine.runAndWait()
        return audio_file
    except Exception:
        if os.path.exists(audio_file):
            os.unlink(audio_file)
        raise


def generate_image(prompt, api_key):
    if not api_key:
        return None

    api_url = "https://api.freepik.com/v1/ai/text-to-image"
    headers = {"x-freepik-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": f"{prompt}, professional photography, high quality, realistic, corporate setting",
        "negative_prompt": "cartoon, anime, drawing, sketch, low quality, blurry",
        "guidance_scale": 7,
        "num_images": 1,
        "image": {"size": "landscape_16_9"},
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return None

        result = response.json()
        if not result.get("data"):
            return None

        image_data = result["data"][0]
        if "base64" not in image_data:
            return None

        image_bytes = base64.b64decode(image_data["base64"])
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None


def create_placeholder_image(prompt, size=(1280, 720)):
    """Create a local placeholder image when external image generation is unavailable."""
    width, height = size
    image = Image.new("RGB", size, color=(20, 36, 60))
    draw = ImageDraw.Draw(image)

    # Add simple layered bands for visual texture.
    draw.rectangle([0, int(height * 0.55), width, height], fill=(15, 26, 43))
    draw.rectangle([0, int(height * 0.78), width, height], fill=(8, 14, 24))

    title = "AI-Powered Mock Interview"
    subtitle = prompt[:90] + ("..." if len(prompt) > 90 else "")
    hint = "Using local fallback visuals"

    draw.text((50, 70), title, fill=(240, 245, 255))
    draw.text((50, 120), subtitle, fill=(205, 219, 240))
    draw.text((50, height - 60), hint, fill=(160, 180, 210))
    return image


def create_video_with_images_and_audio(images, audio_file, script, output_video):
    if mp is None:
        raise RuntimeError("MoviePy is not installed")

    def _with_duration(clip, seconds: float):
        if hasattr(clip, "with_duration"):
            return clip.with_duration(seconds)
        return clip.set_duration(seconds)

    def _with_audio(clip, audio):
        if audio is None:
            return clip
        if hasattr(clip, "with_audio"):
            return clip.with_audio(audio)
        return clip.set_audio(audio)

    audio_clip = None
    lines = [line for line in script.split("\n") if ":" in line and line.split(":", 1)[1].strip()]

    if not lines:
        raise ValueError("No valid dialogue lines found")

    if audio_file and os.path.exists(audio_file):
        audio_clip = mp.AudioFileClip(audio_file)
        total_audio_duration = audio_clip.duration
    else:
        # Fallback duration if TTS is unavailable.
        total_audio_duration = max(6, len(lines) * 4)

    duration_per_clip = total_audio_duration / len(lines)
    clips = []

    for line in lines:
        speaker, _ = line.split(":", 1)
        background = images[0] if "Interviewer" in speaker else images[1]
        img_array = np.array(background.convert("RGB"))
        clip = _with_duration(mp.ImageClip(img_array), duration_per_clip)
        clips.append(clip)

    video = mp.concatenate_videoclips(clips, method="compose")
    final_clip = _with_audio(video, audio_clip)
    final_clip.write_videofile(
        output_video,
        codec="libx264",
        fps=24,
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        logger=None,
    )


def login_required():
    username = session.get("username")
    if not username:
        return None, (jsonify({"error": "Unauthorized"}), 401)
    return username, None


_PROFILE_FIELDS = [
    "full_name",
    "email",
    "phone",
    "university",
    "college_year",
    "degree",
    "major",
    "linkedin",
    "about",
]


def _get_user_profile(username: str) -> dict:
    users = _load_users()
    record = users.get(username)
    if isinstance(record, dict):
        profile = record.get("profile")
        if isinstance(profile, dict):
            return profile
    return {}


def _upsert_user_profile(username: str, updates: dict) -> dict:
    """Update only profile fields for a user without re-validating full signup/profile."""

    if not isinstance(updates, dict):
        return _get_user_profile(username)

    users = _load_users()
    record = users.get(username)

    # Backward compatible: older DB stored just the password hash as a string.
    if isinstance(record, str):
        record = {
            "password_hash": record,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "email_verified": True,
            "profile": {},
        }
    if not isinstance(record, dict):
        record = {"profile": {}}

    profile = record.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    profile.update(updates)
    record["profile"] = profile
    users[username] = record
    _save_users(users)
    return profile


def _calculate_gamification(username: str, grade: float, is_boss: bool = False) -> dict:
    profile = _get_user_profile(username)
    gamification = profile.get("gamification")
    if not isinstance(gamification, dict):
        gamification = {"xp": 0, "streak": 0, "badges": [], "level": 1, "last_grade": 0.0}
    
    base_xp = int(grade * 100)
    bonus_xp = 0
    
    last_grade = float(gamification.get("last_grade", 0.0))
    streak = int(gamification.get("streak", 0))
    
    if grade > last_grade and grade > 0.5:
        streak += 1
    elif grade < 0.5:
        streak = 0
        
    if streak >= 3:
        bonus_xp += 30
        
    if is_boss and grade > 0.7:
        bonus_xp += 100
        
    if grade >= 0.9:
        bonus_xp += 20
        
    earned_xp = base_xp + bonus_xp
    gamification["xp"] = int(gamification.get("xp", 0)) + earned_xp
    gamification["last_grade"] = grade
    gamification["streak"] = streak
    
    old_level = int(gamification.get("level", 1))
    new_level = int(gamification["xp"] / 100) + 1
    gamification["level"] = new_level
    
    badges = gamification.get("badges", [])
    new_badges = []
    
    if new_level >= 2 and "Intermediate" not in badges:
        new_badges.append("Intermediate")
    if new_level >= 3 and "Advanced" not in badges:
        new_badges.append("Advanced")
    if new_level >= 4 and "Interview Ready 🔥" not in badges:
        new_badges.append("Interview Ready 🔥")
        
    if grade >= 0.9 and "Communication Pro" not in badges:
        new_badges.append("Communication Pro")
        
    if new_badges:
        badges.extend(new_badges)
        gamification["badges"] = badges
    
    _upsert_user_profile(username, {"gamification": gamification})
    
    return {
        "earned_xp": earned_xp,
        "total_xp": gamification["xp"],
        "level": new_level,
        "level_up": new_level > old_level,
        "streak": streak,
        "new_badges": new_badges,
        "is_boss": is_boss
    }

def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))


def _safe_mean(values) -> float:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return (sum(nums) / len(nums)) if nums else 0.0


def _get_ats_history(profile: dict) -> list[dict]:
    hist = profile.get("ats_history") if isinstance(profile, dict) else None
    if isinstance(hist, list):
        # Filter to dict records only.
        return [h for h in hist if isinstance(h, dict)]
    return []


def _record_ats_score(username: str, result: dict) -> None:
    try:
        score100 = int(result.get("score"))
    except Exception:
        score100 = 0
    score01 = _clamp01(score100 / 100.0)
    profile = _get_user_profile(username)
    hist = _get_ats_history(profile)
    hist.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "score": score01,
        }
    )
    # Keep latest 25 runs.
    hist = hist[-25:]
    _upsert_user_profile(username, {"ats_history": hist, "last_ats_score": score01})


_PERSONALITIES = {"strict", "friendly", "faang"}
_TRAINING_MODES = {"normal", "fix_weakness"}
_SKILLS = {"dsa", "system_design", "communication"}


def _get_coach_settings(profile: dict) -> dict:
    profile = profile if isinstance(profile, dict) else {}
    personality = str(profile.get("coach_personality") or "friendly").strip().lower()
    if personality not in _PERSONALITIES:
        personality = "friendly"
    adaptive_personality = bool(profile.get("adaptive_personality"))

    training_mode = str(profile.get("training_mode") or "normal").strip().lower()
    if training_mode not in _TRAINING_MODES:
        training_mode = "normal"

    target_skill = str(profile.get("target_skill") or "auto").strip().lower()
    if target_skill != "auto" and target_skill not in _SKILLS:
        target_skill = "auto"

    return {
        "coach_personality": personality,
        "adaptive_personality": adaptive_personality,
        "training_mode": training_mode,
        "target_skill": target_skill,
    }


def _token_count(text: str) -> int:
    try:
        return len(nltk.word_tokenize(text or ""))
    except Exception:
        return len(re.findall(r"[A-Za-z']+", text or ""))


def _dimension_scores_from_report(report: dict) -> dict:
    """Return normalized (0..1) scores for confidence, clarity, technical_depth."""
    analysis = report.get("analysis") if isinstance(report, dict) else None
    analysis = analysis if isinstance(analysis, dict) else {}

    conf_raw = (((analysis.get("confidence") or {}) if isinstance(analysis.get("confidence"), dict) else {}).get(
        "confidence_score"
    ))
    try:
        confidence = _clamp01(float(conf_raw) / 10.0)
    except Exception:
        confidence = 0.0

    filler = 0
    try:
        filler = int(
            (((analysis.get("word_choice") or {}) if isinstance(analysis.get("word_choice"), dict) else {}).get(
                "filler_word_count"
            ))
            or 0
        )
    except Exception:
        filler = 0

    answer_text = (report.get("answer") or "") if isinstance(report, dict) else ""
    tokens = max(_token_count(answer_text), 1)
    try:
        sentences = len(nltk.sent_tokenize(answer_text)) if answer_text else 0
    except Exception:
        sentences = len(re.findall(r"[.!?]+", answer_text or "")) + (1 if (answer_text or "").strip() else 0)
    sentences = max(sentences, 1)
    avg_sentence_len = tokens / sentences

    # Clarity: reward moderate sentence length and low filler usage.
    filler_penalty = _clamp01(filler / 8.0) * 0.6
    length_bonus = 0.0
    if 10 <= avg_sentence_len <= 22:
        length_bonus = 0.25
    elif avg_sentence_len < 7:
        length_bonus = 0.05
    else:
        length_bonus = 0.12
    clarity = _clamp01((0.75 - filler_penalty) + length_bonus)

    pro_words = 0
    try:
        pro_words = int(
            (((analysis.get("word_choice") or {}) if isinstance(analysis.get("word_choice"), dict) else {}).get(
                "professional_word_count"
            ))
            or 0
        )
    except Exception:
        pro_words = 0
    metrics = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_text or ""))
    technical_depth = _clamp01((pro_words / 6.0) * 0.7 + _clamp01(metrics / 6.0) * 0.3)

    return {
        "confidence": confidence,
        "clarity": clarity,
        "technical_depth": technical_depth,
    }


def _infer_skill_from_question(question: str) -> str:
    q = (question or "").strip().lower()
    if any(k in q for k in ("data structure", "algorithm", "complexity", "leetcode", "array", "tree", "graph")):
        return "dsa"
    if any(k in q for k in ("system design", "design", "architecture", "scal", "distributed")):
        return "system_design"
    # Default bucket for most behavioral/general questions.
    return "communication"


def _compute_skill_breakdown(reports: list[dict]) -> dict:
    by_skill = {"dsa": [], "system_design": [], "communication": []}
    for r in reports:
        if not isinstance(r, dict):
            continue
        skill = str(r.get("task_skill") or "").strip().lower()
        if skill not in _SKILLS:
            skill = _infer_skill_from_question(r.get("question") or "")
        dims = _dimension_scores_from_report(r)
        # Communication leans more on clarity+confidence.
        if skill == "communication":
            score = 0.6 * dims["clarity"] + 0.4 * dims["confidence"]
        else:
            score = 0.55 * dims["technical_depth"] + 0.45 * dims["clarity"]
        by_skill[skill].append(_clamp01(score))

    scores = {k: _safe_mean(v) for k, v in by_skill.items()}
    weakest = min(scores, key=scores.get) if scores else "communication"

    def level(value: float) -> str:
        if value < 0.5:
            return "weak"
        if value < 0.75:
            return "medium"
        return "strong"

    return {
        "scores": scores,
        "levels": {k: level(v) for k, v in scores.items()},
        "weakest": weakest,
    }


def _answer_score_from_report(report: dict) -> float:
    if not isinstance(report, dict):
        return 0.0

    grade = report.get("grade")
    if isinstance(grade, (int, float)):
        return _clamp01(float(grade))

    analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else {}
    tone = 0.0
    try:
        tone = float(((analysis.get("tone") or {}) if isinstance(analysis.get("tone"), dict) else {}).get("score") or 0.0)
    except Exception:
        tone = 0.0
    tone01 = _clamp01(tone * 0.5 + 0.5)
    dims = _dimension_scores_from_report(report)
    return _clamp01(0.45 * tone01 + 0.35 * dims["clarity"] + 0.2 * dims["technical_depth"])


def _compute_readiness(reports: list[dict], profile: dict) -> dict:
    hist = _get_ats_history(profile)
    resume_start = None
    resume_end = None
    if hist:
        resume_start = _clamp01(float(hist[0].get("score") or 0.0))
        resume_end = _clamp01(float(hist[-1].get("score") or 0.0))
    else:
        last = profile.get("last_ats_score") if isinstance(profile, dict) else None
        if isinstance(last, (int, float)):
            resume_start = _clamp01(float(last))
            resume_end = _clamp01(float(last))

    if reports:
        start_report = reports[0]
        end_report = reports[-1]
        answer_start = _answer_score_from_report(start_report)
        answer_end = _answer_score_from_report(end_report)
        dims_start = _dimension_scores_from_report(start_report)
        dims_end = _dimension_scores_from_report(end_report)
        conf_start = dims_start["confidence"]
        conf_end = dims_end["confidence"]
    else:
        answer_start = None
        answer_end = None
        conf_start = None
        conf_end = None

    def mean_available(values):
        vals = [v for v in values if isinstance(v, (int, float))]
        return _clamp01(sum(vals) / len(vals)) if vals else 0.0

    readiness_start = mean_available([resume_start, answer_start, conf_start])
    readiness_end = mean_available([resume_end, answer_end, conf_end])

    return {
        "resume": {"start": resume_start if resume_start is not None else 0.0, "end": resume_end if resume_end is not None else 0.0},
        "answer": {"start": answer_start if answer_start is not None else 0.0, "end": answer_end if answer_end is not None else 0.0},
        "confidence": {"start": conf_start if conf_start is not None else 0.0, "end": conf_end if conf_end is not None else 0.0},
        "readiness": {"start": readiness_start, "end": readiness_end},
    }


def _compute_improvement_scorecard(reports: list[dict]) -> dict:
    if not reports:
        blank = {"confidence": 0.0, "clarity": 0.0, "technical_depth": 0.0}
        return {"before": blank, "after": blank}

    before_slice = reports[:3]
    after_slice = reports[-3:]

    def avg_dims(slice_reports):
        dims = [_dimension_scores_from_report(r) for r in slice_reports if isinstance(r, dict)]
        return {
            "confidence": _safe_mean([d["confidence"] for d in dims]),
            "clarity": _safe_mean([d["clarity"] for d in dims]),
            "technical_depth": _safe_mean([d["technical_depth"] for d in dims]),
        }

    return {"before": avg_dims(before_slice), "after": avg_dims(after_slice)}


def _compute_feedback_effectiveness(reports: list[dict]) -> dict:
    """Average grade improvement attributed to each coach action within RL episodes.

    Falls back to illustrative defaults when we don't have enough RL data.
    """

    actions = {"hint": [], "example": [], "follow_up": []}
    # Group by episode id then compute per-attempt improvements.
    by_episode = {}
    for r in reports:
        if not isinstance(r, dict):
            continue
        ep = (r.get("rl_episode_id") or "").strip()
        if not ep:
            continue
        by_episode.setdefault(ep, []).append(r)

    for ep_reports in by_episode.values():
        # Preserve original order as stored.
        for idx in range(1, len(ep_reports)):
            prev = ep_reports[idx - 1]
            cur = ep_reports[idx]
            prev_grade = prev.get("grade")
            cur_grade = cur.get("grade")
            if not isinstance(prev_grade, (int, float)) or not isinstance(cur_grade, (int, float)):
                continue
            improvement = float(cur_grade) - float(prev_grade)
            coach_action = str(cur.get("coach_action") or "").strip().lower()
            if coach_action == "give_hint":
                actions["hint"].append(improvement)
            elif coach_action == "give_example":
                actions["example"].append(improvement)
            elif coach_action == "ask_follow_up":
                actions["follow_up"].append(improvement)

    effectiveness = {
        "hint": _safe_mean(actions["hint"]),
        "example": _safe_mean(actions["example"]),
        "follow_up": _safe_mean(actions["follow_up"]),
    }

    # If no RL data, provide the illustrative sample numbers from the spec screenshots.
    if not any(actions.values()):
        effectiveness = {"hint": 0.10, "example": 0.30, "follow_up": 0.20}

    return effectiveness


@app.route("/")
def index():
    firebase_web_config = {
        "apiKey": os.getenv("FIREBASE_WEB_API_KEY") or "",
        "authDomain": os.getenv("FIREBASE_WEB_AUTH_DOMAIN") or "",
        "projectId": os.getenv("FIREBASE_WEB_PROJECT_ID") or "",
        "appId": os.getenv("FIREBASE_WEB_APP_ID") or "",
        "storageBucket": os.getenv("FIREBASE_WEB_STORAGE_BUCKET") or "",
        "messagingSenderId": os.getenv("FIREBASE_WEB_MESSAGING_SENDER_ID") or "",
    }

    # Only expose config if the essentials exist.
    if not (firebase_web_config["apiKey"] and firebase_web_config["authDomain"] and firebase_web_config["projectId"]):
        firebase_web_config = None

    return render_template(
        "index.html",
        firebase_enabled=bool(_firebase_available),
        firebase_web_config=firebase_web_config,
    )


def _find_username_by_email(email: str) -> str | None:
    email_l = (email or "").strip().lower()
    if not email_l:
        return None
    users = _load_users()
    for username, record in users.items():
        profile = (record or {}).get("profile") or {}
        if (profile.get("email") or "").strip().lower() == email_l:
            return username
    return None


@app.get("/api/meta")
def api_meta():
    return jsonify(
        {
            "questions": QUESTION_BANK,
            "has_api_key": bool(api_key),
            "has_freepik_key": bool(freepik_api_key),
        }
    )


@app.get("/api/me")
def api_me():
    username = session.get("username")
    if not username:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": username, "profile": _get_user_profile(username)})


@app.post("/api/profile")
def api_update_profile():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request"}), 400

    current_profile = _get_user_profile(username)

    def pick(key: str) -> str:
        if key in data:
            value = data.get(key)
        else:
            value = current_profile.get(key)
        if value is None:
            return ""
        return str(value).strip()

    full_name = pick("full_name")
    email = pick("email")
    phone = _normalize_phone(pick("phone"))
    university = pick("university")
    college_year_raw = pick("college_year")
    degree = pick("degree")
    major = pick("major")
    linkedin = pick("linkedin")
    about = pick("about")

    # Match signup validation rules for required profile fields.
    if not full_name or not email or not phone or not university or not college_year_raw or not degree or not about:
        return jsonify({"error": "Please fill in all profile fields"}), 400

    if not _is_valid_email(email):
        return jsonify({"error": "Please enter a valid email address"}), 400

    if len(phone) < 8:
        return jsonify({"error": "Please enter a valid phone number"}), 400

    if not college_year_raw.isdigit() or not (1 <= int(college_year_raw) <= 8):
        return jsonify({"error": "Current year of college must be a number (1-8)"}), 400

    updated_profile = {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "university": university,
        "college_year": int(college_year_raw),
        "degree": degree,
        "major": major,
        "linkedin": linkedin,
        "about": about,
    }

    users = _load_users()
    record = users.get(username)

    # Backward compatible: older DB stored just the password hash as a string.
    if isinstance(record, str):
        record = {
            "password_hash": record,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "email_verified": True,
            "profile": {},
        }
    if not isinstance(record, dict):
        record = {"profile": {}}

    profile = record.get("profile")
    if not isinstance(profile, dict):
        profile = {}

    profile.update(updated_profile)
    record["profile"] = profile
    users[username] = record
    _save_users(users)

    return jsonify({"ok": True, "profile": profile})


@app.post("/api/auth/send-email-otp")
def api_send_email_otp():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip()
    if not _is_valid_email(email):
        return jsonify({"error": "Please enter a valid email address"}), 400

    otp = _generate_otp()
    _email_otp_store[email.lower()] = {
        "otp_hash": _otp_hash(email, otp),
        "expires_at": time.time() + _OTP_TTL_SECONDS,
        "created_at": time.time(),
    }

    ok, msg = _send_email_otp(email, otp)
    if not ok:
        return jsonify({"error": msg}), 500
    return jsonify({"ok": True, "message": msg})


@app.post("/api/signup")
def api_signup():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip()
    phone = _normalize_phone(data.get("phone") or "")
    university = (data.get("university") or "").strip()
    college_year = (data.get("college_year") or "").strip()
    degree = (data.get("degree") or "").strip()
    major = (data.get("major") or "").strip()
    linkedin = (data.get("linkedin") or "").strip()
    about = (data.get("about") or "").strip()
    otp = (data.get("otp") or "").strip()

    if not username or not password or not confirm_password:
        return jsonify({"error": "Please fill in username and password fields"}), 400

    # Profile fields (requested full signup data).
    if not full_name or not email or not phone or not university or not college_year or not degree or not about:
        return jsonify({"error": "Please fill in all profile fields"}), 400

    if not _is_valid_email(email):
        return jsonify({"error": "Please enter a valid email address"}), 400

    if len(phone) < 8:
        return jsonify({"error": "Please enter a valid phone number"}), 400

    if not college_year.isdigit() or not (1 <= int(college_year) <= 8):
        return jsonify({"error": "Current year of college must be a number (1-8)"}), 400

    # OTP verification.
    if not otp:
        return jsonify({"error": "Please enter the email OTP"}), 400

    otp_record = _email_otp_store.get(email.lower())
    if not otp_record:
        return jsonify({"error": "OTP not found. Please click Send OTP."}), 400
    if float(otp_record.get("expires_at") or 0) < time.time():
        return jsonify({"error": "OTP expired. Please request a new OTP."}), 400
    if (otp_record.get("otp_hash") or "") != _otp_hash(email, otp):
        return jsonify({"error": "Invalid OTP"}), 400

    valid_username, username_msg = validate_username(username)
    if not valid_username:
        return jsonify({"error": username_msg}), 400

    valid_password, password_msg = validate_password(password)
    if not valid_password:
        return jsonify({"error": password_msg}), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    if user_exists(username):
        return jsonify({"error": "Username already exists"}), 400

    profile = {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "university": university,
        "college_year": int(college_year),
        "degree": degree,
        "major": major,
        "linkedin": linkedin,
        "about": about,
    }

    firebase_uid = _firebase_upsert_user(email=email, full_name=full_name)
    if firebase_uid:
        profile["firebase_uid"] = firebase_uid
    add_user(username, password, profile=profile, email_verified=True)

    # OTP was used successfully; remove it.
    try:
        del _email_otp_store[email.lower()]
    except Exception:
        pass
    return jsonify({"ok": True, "message": "Account created successfully"})


@app.post("/api/login")
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Please provide username and password"}), 400

    if not authenticate_user(username, password):
        return jsonify({"error": "Invalid username or password"}), 401

    session["username"] = username
    return jsonify({"ok": True, "username": username})


@app.post("/api/auth/firebase/session")
def api_firebase_session_login():
    """Create a Flask session from a Firebase ID token (email-link sign-in)."""

    if not _firebase_available or not firebase_auth:
        return jsonify({"error": "Firebase is not configured on the server"}), 400

    data = request.get_json(force=True)
    id_token = (data.get("id_token") or "").strip()
    if not id_token:
        return jsonify({"error": "Missing id_token"}), 400

    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except Exception:
        return jsonify({"error": "Invalid Firebase token"}), 401

    email = (decoded.get("email") or "").strip()
    email_verified = bool(decoded.get("email_verified"))
    firebase_uid = (decoded.get("uid") or "").strip()
    if not email:
        return jsonify({"error": "Firebase token missing email"}), 400
    if not email_verified:
        return jsonify({"error": "Email is not verified"}), 401

    username = _find_username_by_email(email)
    if not username:
        return jsonify({"error": "No local account found for this email. Please Sign Up first."}), 404

    # Sync firebase uid into the local record if needed.
    try:
        users = _load_users()
        record = users.get(username) or {}
        profile = record.get("profile") or {}
        if firebase_uid and profile.get("firebase_uid") != firebase_uid:
            profile["firebase_uid"] = firebase_uid
        record["profile"] = profile
        record["email_verified"] = True
        users[username] = record
        _save_users(users)
    except Exception:
        pass

    session["username"] = username
    return jsonify({"ok": True, "username": username})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.post("/api/transcribe")
def api_transcribe():
    username, err = login_required()
    if err:
        return err

    if sr is None:
        return jsonify({"error": "SpeechRecognition not installed"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]
    suffix = os.path.splitext(secure_filename(audio.filename or "audio.wav"))[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        audio_path = temp.name
        audio.save(audio_path)

    coach = InterviewCoach()
    text = coach.transcribe_audio(audio_path)

    try:
        os.unlink(audio_path)
    except Exception:
        pass

    return jsonify({"transcription": text})


def _build_analysis_payload(question, answer, posture=None):
    coach = InterviewCoach()
    analysis = coach.analyze_text_input(answer)
    if posture:
        analysis["posture"] = posture
    feedback = coach.provide_comprehensive_feedback(analysis)

    # Optional: enrich with offline/local ML model outputs (T5 + emotion classifier).
    # This is best-effort and never blocks the core app flow.
    try:
        from interview_coach_models.ml_answer_grader import get_ml_answer_grader

        ml_grader = get_ml_answer_grader()
    except Exception:
        ml_grader = None

    if ml_grader is not None:
        try:
            ml_out = ml_grader.grade(question=question or "", answer=answer or "", keywords=[])
        except Exception:
            ml_out = None

        if isinstance(ml_out, dict) and ml_out:
            analysis["local_ml"] = {
                "grade": ml_out.get("grade"),
                "emotion": ml_out.get("emotion"),
                "feedback": ml_out.get("ml_feedback"),
            }

            append_ml = (os.getenv("INTERVIEW_COACH_APPEND_LOCAL_ML_FEEDBACK") or "").strip().lower()
            if append_ml in {"1", "true", "yes", "on"}:
                ml_text = str(ml_out.get("ml_feedback") or "").strip()
                if ml_text:
                    feedback = f"{feedback}\n\nLOCAL ML FEEDBACK\n{ml_text}"
    return {
        "question": question,
        "answer": answer,
        "analysis": analysis,
        "feedback": feedback,
    }


def _style_feedback(personality: str, analysis: dict, base_feedback: str) -> str:
    personality = (personality or "").strip().lower()
    if personality not in _PERSONALITIES:
        personality = "friendly"

    dims = _dimension_scores_from_report({"analysis": analysis, "answer": analysis.get("text") if isinstance(analysis, dict) else ""})
    confidence = dims.get("confidence", 0.0)
    clarity = dims.get("clarity", 0.0)
    technical = dims.get("technical_depth", 0.0)

    if personality == "strict":
        return (
            "STRICT INTERVIEWER\n"
            f"Confidence: {confidence:.2f} | Clarity: {clarity:.2f} | Technical Depth: {technical:.2f}\n"
            "Fix these items in your next attempt:\n"
            f"{base_feedback}"
        )

    if personality == "faang":
        # Keep it concise and technical.
        improvement_areas = []
        try:
            filler = int(((analysis.get("word_choice") or {}).get("filler_word_count") or 0))
        except Exception:
            filler = 0
        if filler >= 3:
            improvement_areas.append("reduce filler words")
        if clarity < 0.6:
            improvement_areas.append("structure answer (STAR / clear narrative)")
        if technical < 0.55:
            improvement_areas.append("add concrete metrics + technical detail")
        if confidence < 0.6:
            improvement_areas.append("speak more assertively")

        top_fixes = "; ".join(improvement_areas[:3]) if improvement_areas else "keep the same structure; add one concrete example"
        return (
            "FAANG INTERVIEWER NOTES\n"
            f"Signals: confidence={confidence:.2f}, clarity={clarity:.2f}, depth={technical:.2f}\n"
            f"Next iteration: {top_fixes}.\n\n"
            f"{base_feedback}"
        )

    # friendly
    return (
        "FRIENDLY MENTOR\n"
        f"Confidence: {confidence:.2f} | Clarity: {clarity:.2f} | Technical Depth: {technical:.2f}\n"
        f"{base_feedback}\n\n"
        "You’re improving — take one feedback point and retry."
    )


def _get_personality_stats(profile: dict) -> dict:
    stats = profile.get("personality_stats") if isinstance(profile, dict) else None
    if not isinstance(stats, dict):
        return {p: {"count": 0, "total_improvement": 0.0} for p in _PERSONALITIES}

    normalized = {}
    for p in _PERSONALITIES:
        raw = stats.get(p)
        if isinstance(raw, dict):
            try:
                count = int(raw.get("count") or 0)
            except Exception:
                count = 0
            try:
                total = float(raw.get("total_improvement") or 0.0)
            except Exception:
                total = 0.0
        else:
            count = 0
            total = 0.0
        normalized[p] = {"count": max(0, count), "total_improvement": float(total)}
    return normalized


def _recommended_personality(profile: dict) -> str:
    stats = _get_personality_stats(profile)
    best = "friendly"
    best_avg = float("-inf")
    for p, rec in stats.items():
        count = int(rec.get("count") or 0)
        total = float(rec.get("total_improvement") or 0.0)
        avg = (total / count) if count else 0.0
        if avg > best_avg:
            best_avg = avg
            best = p
    return best


def _pick_personality_for_attempt(profile: dict) -> str:
    settings = _get_coach_settings(profile)
    if not settings.get("adaptive_personality"):
        return settings.get("coach_personality") or "friendly"

    # Epsilon-greedy over personalities.
    epsilon = 0.2
    if secrets.randbelow(1000) < int(epsilon * 1000):
        return list(_PERSONALITIES)[secrets.randbelow(len(_PERSONALITIES))]
    return _recommended_personality(profile)


def _update_personality_stats(username: str, personality: str, improvement: float) -> None:
    personality = (personality or "").strip().lower()
    if personality not in _PERSONALITIES:
        return
    profile = _get_user_profile(username)
    stats = _get_personality_stats(profile)
    rec = stats.get(personality) or {"count": 0, "total_improvement": 0.0}
    rec["count"] = int(rec.get("count") or 0) + 1
    rec["total_improvement"] = float(rec.get("total_improvement") or 0.0) + float(improvement or 0.0)
    stats[personality] = rec
    _upsert_user_profile(username, {"personality_stats": stats})


def _normalize_thread_turns(raw_turns):
    if not isinstance(raw_turns, list):
        return []

    normalized = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or item.get("transcription") or "").strip()
        if not q or not a:
            continue
        normalized.append({"question": q, "answer": a})

    # Keep prompts compact.
    return normalized[-8:]


def _sanitize_followup_question(text: str) -> str:
    question = (text or "").strip().replace("\n", " ")
    question = re.sub(r"\s+", " ", question).strip()
    question = question.strip("\"'`“””‘’ ")
    if not question:
        return "Can you walk me through a specific example and the outcome?"
    if not question.endswith("?"):
        question = f"{question}?"
    # Keep it short and interview-like.
    if len(question) > 240:
        question = question[:237].rstrip() + "...?"
    return question


def _heuristic_followup_question(root_question: str, turns) -> str:
    root = (root_question or "").lower()
    last_answer = (turns[-1]["answer"] if turns else "")
    asked_questions = [t.get("question", "").strip() for t in turns if isinstance(t, dict)]
    asked_set = {q for q in asked_questions if q}

    def pick(candidates):
        for q in candidates:
            if q not in asked_set:
                return q
        return None

    if "tell me about yourself" in root:
        candidate = pick(
            [
                "Which part of your background is most relevant to this role, and why?",
                "Tell me about a recent project you're most proud of. What was your role and impact?",
                "What kind of work environment helps you do your best work, and why?",
                "What would you say is your biggest strength, and one area you're actively improving?",
                "Why are you interested in this role, specifically?",
            ]
        )
        if candidate:
            return candidate

    if "greatest strength" in root or "your greatest strength" in root or root.strip().endswith("strength?"):
        candidate = pick(
            [
                "Can you share a specific example where that strength made a measurable impact?",
                "How do you apply that strength when you're under pressure or facing tight deadlines?",
                "How would your teammates describe that strength, and why?",
            ]
        )
        if candidate:
            return candidate

    if "greatest weakness" in root or "your greatest weakness" in root or root.strip().endswith("weakness?"):
        candidate = pick(
            [
                "What steps are you taking to improve that weakness, and what progress have you seen so far?",
                "Can you share an example where that weakness showed up, and how you handled it?",
                "What systems or habits do you use today to prevent that weakness from impacting results?",
            ]
        )
        if candidate:
            return candidate

    # Keyword-based probe from the candidate's last answer.
    tokens = re.findall(r"[A-Za-z][A-Za-z']{2,}", (last_answer or "").lower())
    stop = set()
    try:
        stop = set(stopwords.words("english"))
    except Exception:
        stop = set()
    common = {
        "also",
        "really",
        "very",
        "just",
        "like",
        "think",
        "know",
        "because",
        "about",
        "into",
        "from",
        "with",
        "that",
        "this",
        "there",
        "their",
        "have",
        "been",
        "were",
        "when",
        "what",
        "where",
        "which",
        "your",
    }
    keywords = [t for t in tokens if t not in stop and t not in common]
    topic = keywords[0] if keywords else "that"

    # Use a rotating set of probes and avoid repeats.
    probes = [
        f"You mentioned {topic}. Can you describe a specific situation, what you did, and the result?",
        "What was the biggest challenge in that situation, and how did you overcome it?",
        "How did you measure success, and what did you learn from the outcome?",
        "If you had to do it again, what would you do differently and why?",
    ]
    candidate = pick(probes)
    if candidate:
        return candidate

    return "Can you walk me through a specific example and the outcome?"


def _openai_followup_question(root_question: str, turns) -> str | None:
    if not api_key:
        return None

    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_NAME")
        or os.getenv("OPENAI_CHAT_MODEL")
        or "gpt-4o-mini"
    )
    dialogue = []
    for t in turns:
        dialogue.append(f"Q: {t['question']}\nA: {t['answer']}")
    conversation = "\n\n".join(dialogue) if dialogue else ""
    asked_questions = [t.get("question", "").strip() for t in turns if isinstance(t, dict) and t.get("question")]
    asked_block = "\n".join(f"- {q}" for q in asked_questions[-8:])

    system = (
        "You are an HR interviewer running a realistic mock interview. "
        "Given the initial question and the candidate's answers so far, ask exactly ONE concise follow-up question. "
        "Do not repeat any previously asked question. "
        "Return only the question text (no bullets, no preface, no feedback)."
    )
    user = (
        f"Initial question: {root_question}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Previously asked questions:\n{asked_block}\n\n"
        "Ask the next follow-up question."
    )

    try:
        if _OPENAI_NEW and client is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=80,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content

        if not _OPENAI_NEW:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=80,
            )
            content = (resp["choices"][0]["message"]["content"] or "").strip()
            return content
    except Exception:
        return None

    return None


def _generate_followup_question(root_question: str, raw_turns) -> str:
    turns = _normalize_thread_turns(raw_turns)
    root = (root_question or "").strip() or (turns[0]["question"] if turns else "")

    # Prefer LLM if configured, otherwise fall back.
    candidate = _openai_followup_question(root, turns)
    if not candidate:
        candidate = _heuristic_followup_question(root, turns)

    return _sanitize_followup_question(candidate)


@app.post("/api/practice/text")
def api_practice_text():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not question or not answer:
        return jsonify({"error": "Question and answer are required"}), 400

    profile = _get_user_profile(username)
    settings = _get_coach_settings(profile)
    personality_used = settings.get("coach_personality") or "friendly"

    payload = _build_analysis_payload(question, answer)
    payload["feedback"] = _style_feedback(personality_used, payload.get("analysis") or {}, payload.get("feedback") or "")
    payload["coach_personality_used"] = personality_used
    root_question = (data.get("root_question") or question).strip()
    thread_turns = data.get("thread_turns")
    if not thread_turns:
        thread_turns = [{"question": question, "answer": answer}]
    payload["follow_up_question"] = _generate_followup_question(root_question, thread_turns)
    dims = _dimension_scores_from_report({"analysis": payload["analysis"]})
    grade = (dims.get("confidence", 0) + dims.get("clarity", 0) + dims.get("technical_depth", 0)) / 3.0
    is_boss = len(thread_turns) > 0 and len(thread_turns) % 4 == 0
    payload["gamification"] = _calculate_gamification(username, grade, is_boss)
    if is_boss:
        payload["follow_up_question"] = "🔥 FINAL BOSS QUESTION (Hard): " + payload["follow_up_question"]

    return jsonify(payload)


@app.post("/api/practice/audio")
def api_practice_audio():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    transcription = (data.get("transcription") or "").strip()
    if not question or not transcription:
        return jsonify({"error": "Question and transcription are required"}), 400

    profile = _get_user_profile(username)
    settings = _get_coach_settings(profile)
    personality_used = settings.get("coach_personality") or "friendly"

    payload = _build_analysis_payload(question, transcription)
    payload["feedback"] = _style_feedback(personality_used, payload.get("analysis") or {}, payload.get("feedback") or "")
    payload["coach_personality_used"] = personality_used
    root_question = (data.get("root_question") or question).strip()
    thread_turns = data.get("thread_turns")
    if not thread_turns:
        thread_turns = [{"question": question, "answer": transcription}]
    payload["follow_up_question"] = _generate_followup_question(root_question, thread_turns)
    dims = _dimension_scores_from_report({"analysis": payload["analysis"]})
    grade = (dims.get("confidence", 0) + dims.get("clarity", 0) + dims.get("technical_depth", 0)) / 3.0
    is_boss = len(thread_turns) > 0 and len(thread_turns) % 4 == 0
    payload["gamification"] = _calculate_gamification(username, grade, is_boss)
    if is_boss:
        payload["follow_up_question"] = "🔥 FINAL BOSS QUESTION (Hard): " + payload["follow_up_question"]

    return jsonify(payload)


@app.post("/api/posture/frame")
def api_posture_frame():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True)
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Image frame is required"}), 400

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    frame_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image frame"}), 400

    state = _posture_states.get(username)
    if state is None:
        state = VideoTransformer()
        _posture_states[username] = state

    status = state.analyze_posture(frame)
    return jsonify(
        {
            "status": status,
            "feedback": state.posture_feedback,
            "recommendations": state.posture_recommendations(),
        }
    )


@app.post("/api/practice/video")
def api_practice_video():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    transcription = (data.get("transcription") or "").strip()
    posture = data.get("posture") or {}

    if not question or not transcription:
        return jsonify({"error": "Question and transcription are required"}), 400

    posture_data = {
        "status": posture.get("status", "Not analyzed"),
        "feedback": posture.get("feedback", "No posture analysis available."),
    }

    profile = _get_user_profile(username)
    settings = _get_coach_settings(profile)
    personality_used = settings.get("coach_personality") or "friendly"

    payload = _build_analysis_payload(question, transcription, posture=posture_data)
    payload["feedback"] = _style_feedback(personality_used, payload.get("analysis") or {}, payload.get("feedback") or "")
    payload["coach_personality_used"] = personality_used
    root_question = (data.get("root_question") or question).strip()
    thread_turns = data.get("thread_turns")
    if not thread_turns:
        thread_turns = [{"question": question, "answer": transcription}]
    payload["follow_up_question"] = _generate_followup_question(root_question, thread_turns)
    dims = _dimension_scores_from_report({"analysis": payload["analysis"]})
    grade = (dims.get("confidence", 0) + dims.get("clarity", 0) + dims.get("technical_depth", 0)) / 3.0
    is_boss = len(thread_turns) > 0 and len(thread_turns) % 4 == 0
    payload["gamification"] = _calculate_gamification(username, grade, is_boss)
    if is_boss:
        payload["follow_up_question"] = "🔥 FINAL BOSS QUESTION (Hard): " + payload["follow_up_question"]

    return jsonify(payload)


@app.get("/api/reports")
def api_reports():
    username, err = login_required()
    if err:
        return err

    reports = get_user_reports(username)
    return jsonify({"reports": reports})


@app.get("/api/progress")
def api_progress():
    username, err = login_required()
    if err:
        return err

    reports = get_user_reports(username)
    if not reports:
        return jsonify({"sessions": 0, "timeline": []})

    timeline = []
    for report in reports:
        timeline.append(
            {
                "timestamp": report["timestamp"],
                "confidence": report["analysis"]["confidence"]["confidence_score"],
                "tone": report["analysis"]["tone"]["score"],
                "filler_words": report["analysis"]["word_choice"]["filler_word_count"],
                "professional_words": report["analysis"]["word_choice"]["professional_word_count"],
            }
        )

    return jsonify({"sessions": len(reports), "timeline": timeline})


@app.post("/api/ats/check")
def api_ats_check():
    username, err = login_required()
    if err:
        return err

    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    resume = request.files["resume"]
    filename = secure_filename(resume.filename or "resume")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Upload PDF, DOCX, or TXT."}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp:
        path = temp.name
        resume.save(path)

    try:
        result = analyze_resume_file(path, filename)
        try:
            _record_ats_score(username, result)
        except Exception:
            pass
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"ATS check failed: {exc}"}), 500
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@app.get("/api/coach/summary")
def api_coach_summary():
    username, err = login_required()
    if err:
        return err

    reports = get_user_reports(username)
    profile = _get_user_profile(username)
    settings = _get_coach_settings(profile)

    readiness = _compute_readiness(reports, profile)
    skills = _compute_skill_breakdown(reports)
    effectiveness = _compute_feedback_effectiveness(reports)
    scorecard = _compute_improvement_scorecard(reports)

    stats = _get_personality_stats(profile)
    recommended = _recommended_personality(profile)

    return jsonify(
        {
            "ok": True,
            "settings": {
                **settings,
                "recommended_personality": recommended,
                "personality_stats": stats,
            },
            "readiness": readiness,
            "skills": skills,
            "effectiveness": effectiveness,
            "scorecard": scorecard,
        }
    )


@app.post("/api/coach/settings")
def api_coach_settings():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request"}), 400

    updates = {}
    if "coach_personality" in data:
        p = str(data.get("coach_personality") or "").strip().lower()
        if p not in _PERSONALITIES:
            return jsonify({"error": "Invalid coach_personality"}), 400
        updates["coach_personality"] = p

    if "adaptive_personality" in data:
        updates["adaptive_personality"] = bool(data.get("adaptive_personality"))

    if "training_mode" in data:
        m = str(data.get("training_mode") or "").strip().lower()
        if m not in _TRAINING_MODES:
            return jsonify({"error": "Invalid training_mode"}), 400
        updates["training_mode"] = m

    if "target_skill" in data:
        s = str(data.get("target_skill") or "").strip().lower()
        if s != "auto" and s not in _SKILLS:
            return jsonify({"error": "Invalid target_skill"}), 400
        updates["target_skill"] = s

    profile = _upsert_user_profile(username, updates)
    return jsonify({"ok": True, "settings": _get_coach_settings(profile)})


@app.get("/api/reports/pdf")
def api_reports_pdf():
    username, err = login_required()
    if err:
        return err

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    reports = get_user_reports(username)
    pdf_data = create_progress_report_pdf(username, reports, start_date, end_date)
    return send_file(
        io.BytesIO(pdf_data),
        as_attachment=True,
        download_name=f"{username}_interview_report.pdf",
        mimetype="application/pdf",
    )


@app.post("/api/mock-interview")
def api_mock_interview():
    username, err = login_required()
    if err:
        return err

    data = request.get_json(force=True)
    role = (data.get("role") or "Data Analyst").strip()
    experience = (data.get("experience") or "Entry-level").strip()
    interview_type = (data.get("interview_type") or "Standard").strip()
    additional_details = (data.get("additional_details") or "").strip()

    transcript = generate_interview_transcript(role, experience, additional_details, interview_type)
    audio_file = None
    warning_messages = []

    try:
        audio_file = generate_audio_for_video(transcript)
    except Exception:
        warning_messages.append("Audio narration generation failed; using silent fallback.")

    prompts = [
        "A professional interviewer in a modern office setting, clear and realistic",
        "A confident candidate being interviewed in a corporate environment",
    ]
    images = []
    for prompt in prompts:
        image = generate_image(prompt, freepik_api_key) if freepik_api_key else None
        if image is None:
            image = create_placeholder_image(prompt)
            warning_messages.append("Image generation unavailable; using local placeholder images.")
        images.append(image)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        output_video = temp.name

    try:
        create_video_with_images_and_audio(images, audio_file, transcript, output_video)
    except Exception:
        # Keep transcript usable even when video pipeline is unavailable (e.g., FFmpeg missing).
        try:
            if os.path.exists(output_video):
                os.unlink(output_video)
        except Exception:
            pass
        warning_messages.append(
            "Video generation is unavailable on this machine. Install FFmpeg to enable downloadable videos."
        )
        return jsonify(
            {
                "ok": True,
                "video_id": None,
                "video_available": False,
                "transcript": transcript,
                "warning": " ".join(dict.fromkeys(warning_messages)),
            }
        )
    finally:
        try:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
        except Exception:
            pass

    video_id = str(uuid.uuid4())
    _generated_videos[video_id] = output_video
    response = {"ok": True, "video_id": video_id, "video_available": True, "transcript": transcript}
    if warning_messages:
        response["warning"] = " ".join(dict.fromkeys(warning_messages))
    return jsonify(response)


@app.get("/api/mock-interview/<video_id>")
def api_mock_interview_download(video_id):
    username, err = login_required()
    if err:
        return err

    path = _generated_videos.get(video_id)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Video not found"}), 404

    return send_file(path, as_attachment=True, download_name=f"{username}_mock_interview.mp4")


def init_rl_system():
    """Initialize RL environment and agent when RL dependencies are available."""
    global _rl_env, _rl_agent, _openenv_env
    if not _rl_available:
        return

    _rl_env = InterviewCoachEnv(max_attempts=5, target_grade=0.80)
    _openenv_env = InterviewCoachEnv(max_attempts=5, target_grade=0.80)
    _rl_agent = QLearningAgent(learning_rate=0.15)

    checkpoint = Path("models/agent_checkpoint.json")
    if checkpoint.exists():
        try:
            _rl_agent.load(checkpoint)
        except Exception:
            pass


@app.post("/api/rl/new-session")
def api_rl_new_session():
    username, err = login_required()
    if err:
        return err

    if not _rl_available or _rl_env is None:
        return jsonify({"error": "RL module is unavailable in this runtime"}), 503

    data = request.get_json(force=True) or {}
    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    profile = _get_user_profile(username)
    settings = _get_coach_settings(profile)
    training_mode = str(data.get("training_mode") or settings.get("training_mode") or "normal").strip().lower()
    if training_mode not in _TRAINING_MODES:
        training_mode = "normal"
    target_skill = str(data.get("target_skill") or settings.get("target_skill") or "auto").strip().lower()
    if target_skill != "auto" and target_skill not in _SKILLS:
        target_skill = "auto"

    session_key = f"{username}_rl_session"
    if session_key in _session_episodes:
        del _session_episodes[session_key]

    difficulty_map = {
        "easy": TaskType.EASY,
        "medium": TaskType.MEDIUM,
        "hard": TaskType.HARD,
    }

    tasks = TaskBank.get_tasks_by_difficulty(difficulty_map[difficulty])
    if not tasks:
        return jsonify({"error": f"No {difficulty} tasks available"}), 404

    if training_mode == "fix_weakness":
        if target_skill == "auto":
            target_skill = _compute_skill_breakdown(get_user_reports(username)).get("weakest") or "communication"
        filtered = [t for t in tasks if _infer_skill_from_question(t.question) == target_skill]
        if filtered:
            tasks = filtered

    task = tasks[0]
    task_skill = _infer_skill_from_question(task.question)
    obs = _rl_env.reset(task)
    _session_episodes[session_key] = {
        "task": task,
        "observation": obs,
        "attempt": 0,
        "total_reward": 0.0,
        "grades": [],
        "strategies_used": [],
        "task_skill": task_skill,
        "training_mode": training_mode,
        "target_skill": target_skill,
    }

    return jsonify(
        {
            "ok": True,
            "task_id": task.task_id,
            "question": task.question,
            "description": task.description,
            "difficulty": task.difficulty.value,
            "target_grade": task.target_grade,
            "max_attempts": task.max_attempts,
            "keywords": task.keywords,
            "task_skill": task_skill,
            "training_mode": training_mode,
            "target_skill": target_skill,
        }
    )


@app.post("/api/rl/practice/text")
def api_rl_practice_text():
    username, err = login_required()
    if err:
        return err

    if not _rl_available or _rl_env is None:
        return jsonify({"error": "RL module is unavailable in this runtime"}), 503

    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    task_difficulty = (data.get("task_difficulty") or "medium").strip().lower()
    use_agent_feedback = data.get("use_agent_feedback", True)

    if not question or not answer:
        return jsonify({"error": "Question and answer are required"}), 400

    if task_difficulty not in ["easy", "medium", "hard"]:
        task_difficulty = "medium"

    difficulty_map = {
        "easy": TaskType.EASY,
        "medium": TaskType.MEDIUM,
        "hard": TaskType.HARD,
    }

    profile = _get_user_profile(username)
    personality_used = _pick_personality_for_attempt(profile)

    payload = _build_analysis_payload(question, answer)
    payload["feedback"] = _style_feedback(personality_used, payload.get("analysis") or {}, payload.get("feedback") or "")
    payload["coach_personality_used"] = personality_used
    root_question = (data.get("root_question") or question).strip()
    thread_turns = data.get("thread_turns")
    if not thread_turns:
        thread_turns = [{"question": question, "answer": answer}]
    payload["follow_up_question"] = _generate_followup_question(root_question, thread_turns)
    approx_grade = payload["analysis"].get("tone", {}).get("score", 0.0) * 0.5 + 0.5
    approx_grade = min(max(approx_grade, 0.0), 1.0)

    session_key = f"{username}_rl_session"
    if session_key not in _session_episodes:
        tasks = TaskBank.get_tasks_by_difficulty(difficulty_map[task_difficulty])
        task = tasks[0] if tasks else None
        obs = _rl_env.reset(task)
        _session_episodes[session_key] = {
            "task": task,
            "observation": obs,
            "attempt": 0,
            "total_reward": 0.0,
            "grades": [],
            "strategies_used": [],
        }

    session_state = _session_episodes[session_key]
    session_state["attempt"] += 1
    prev_grade_for_improvement = session_state["grades"][-1] if session_state["grades"] else 0.0

    if use_agent_feedback and _rl_agent:
        current_obs = session_state["observation"]
        best_action = _rl_agent.choose_action(current_obs, use_epsilon_greedy=False)
        action = Action(strategy=best_action, confidence=0.95)
        result = _rl_env.step(action, answer)

        session_state["observation"] = result.observation
        session_state["total_reward"] += result.reward.total
        session_state["strategies_used"].append(best_action.value)

        # Prefer the environment's grader score when available.
        try:
            grade = float(result.info.get("grade", approx_grade))
        except Exception:
            grade = approx_grade
        grade = min(max(grade, 0.0), 1.0)

        try:
            rl_feedback = _rl_env._generate_feedback(
                best_action,
                answer,
                result.info.get("grade", grade),
                result.info.get("grade_details", {}),
            )
        except Exception:
            rl_feedback = f"Recommended strategy: {best_action.value}"
        rl_strategy = best_action.value
        episode_done = result.done
        reward = result.reward.total
        success = result.reward.success

        if episode_done:
            _rl_agent.episode_complete()
            try:
                _rl_agent.save(Path("models/agent_checkpoint.json"))
            except Exception:
                pass
    else:
        rl_strategy = "none"
        rl_feedback = "RL coaching disabled"
        grade = approx_grade
        reward = 0.0
        episode_done = False
        success = False

    session_state["grades"].append(grade)
    improvement = float(grade) - float(prev_grade_for_improvement)
    if session_state["attempt"] > 1:
        try:
            _update_personality_stats(username, personality_used, improvement)
        except Exception:
            pass

    # Agent brain visualization (based on the pre-step observation).
    weak_skill = _compute_skill_breakdown(get_user_reports(username)).get("weakest") or "communication"
    coach_action = "give_hint"
    reason = "Small nudge to improve the next attempt"
    example_text = ""
    try:
        if current_obs.keyword_recall < 0.4:
            coach_action = "give_example"
            reason = "Low keyword recall"
        elif current_obs.structure_score < 0.55:
            coach_action = "ask_follow_up"
            reason = "Low conceptual clarity / structure"
    except Exception:
        pass

    try:
        task = session_state.get("task")
        if coach_action == "give_example" and task and getattr(task, "examples", None):
            example_text = str(task.examples[0])
    except Exception:
        example_text = ""

    payload["agent_brain"] = {
        "state": {
            "score": float(getattr(current_obs, "current_grade", 0.0)) if use_agent_feedback and _rl_agent else float(grade),
            "weak": weak_skill,
        },
        "action": coach_action,
        "reason": reason,
    }
    if example_text:
        payload["agent_brain"]["example"] = example_text

    is_boss = len(thread_turns) > 0 and len(thread_turns) % 4 == 0
    payload["gamification"] = _calculate_gamification(username, grade, is_boss)
    if is_boss:
        final_follow_up = "🔥 FINAL BOSS QUESTION (Hard): " + final_follow_up
        
    attempts = max(int(session_state.get("attempt") or 0), 1)
    raw_total_reward = float(session_state.get("total_reward") or 0.0)
    total_reward_normalized = min(max(raw_total_reward / attempts, 0.0), 1.0)

    return jsonify(
        {
            **payload,
            "grade": grade,
            "rl_strategy": rl_strategy,
            "rl_feedback": rl_feedback,
            "reward": reward,
            "episode_done": episode_done,
            "episode_success": success,
            "session_progress": {
                "attempt": session_state["attempt"],
                # Keep raw cumulative reward for debugging/analytics.
                "raw_total_reward": raw_total_reward,
                # Normalized total reward stays within [0, 1] for UI.
                "total_reward": total_reward_normalized,
                "avg_grade": sum(session_state["grades"]) / len(session_state["grades"]),
                "strategies_used": session_state["strategies_used"],
            },
        }
    )


@app.get("/api/rl/session-status")
def api_rl_session_status():
    username, err = login_required()
    if err:
        return err

    session_key = f"{username}_rl_session"
    if session_key not in _session_episodes:
        return jsonify(
            {
                "has_active_session": False,
                "message": "No active session. Call /api/rl/new-session first.",
            }
        )

    session_state = _session_episodes[session_key]
    grades = session_state["grades"]
    attempts = max(int(session_state.get("attempt") or 0), 1)
    raw_total_reward = float(session_state.get("total_reward") or 0.0)
    total_reward_normalized = min(max(raw_total_reward / attempts, 0.0), 1.0)
    return jsonify(
        {
            "has_active_session": True,
            "task_id": session_state["task"].task_id,
            "question": session_state["task"].question,
            "attempt": session_state["attempt"],
            "max_attempts": session_state["task"].max_attempts,
            "raw_total_reward": raw_total_reward,
            "total_reward": total_reward_normalized,
            "average_grade": sum(grades) / len(grades) if grades else 0.0,
            "max_grade": max(grades) if grades else 0.0,
            "strategies_used": session_state["strategies_used"],
            "session_complete": session_state["attempt"] >= session_state["task"].max_attempts,
        }
    )


@app.post("/api/rl/end-session")
def api_rl_end_session():
    username, err = login_required()
    if err:
        return err

    session_key = f"{username}_rl_session"
    if session_key not in _session_episodes:
        return jsonify({"error": "No active session"}), 400

    session_state = _session_episodes[session_key]
    grades = session_state["grades"]
    summary = {
        "task_id": session_state["task"].task_id,
        "attempts": session_state["attempt"],
        "max_attempts": session_state["task"].max_attempts,
        "grades": grades,
        "average_grade": sum(grades) / len(grades) if grades else 0.0,
        "max_grade": max(grades) if grades else 0.0,
        "improvement": (grades[-1] - grades[0]) if len(grades) > 1 else 0.0,
        "total_reward": session_state["total_reward"],
        "strategies_used": session_state["strategies_used"],
        "target_grade": session_state["task"].target_grade,
        "success": (max(grades) if grades else 0.0) >= session_state["task"].target_grade,
    }
    del _session_episodes[session_key]
    return jsonify(summary)


@app.get("/api/rl/agent-stats")
def api_rl_agent_stats():
    if not _rl_available or _rl_agent is None:
        return jsonify({"error": "RL agent is unavailable in this runtime"}), 503

    stats = _rl_agent.get_stats()
    return jsonify(
        {
            "agent_type": "Q-Learning",
            "state_space_size": 270,
            "action_space_size": 3,
            "actions": ["strict", "moderate", "hint"],
            "learning_rate": _rl_agent.learning_rate,
            "discount_factor": _rl_agent.discount_factor,
            "epsilon": _rl_agent.epsilon,
            "episodes_trained": stats.get("episodes_trained", 0),
            "unique_states": stats.get("unique_states", 0),
            "q_table_summary": _rl_agent.get_q_table_summary(),
        }
    )


def _openenv_error(message, status=400):
    return jsonify({"error": message}), status


@app.get("/health")
def openenv_health():
    return jsonify({"ok": True, "service": "interview-coach-openenv"})


@app.post("/reset")
def openenv_reset():
    if not _rl_available or _openenv_env is None:
        return _openenv_error("OpenEnv environment is unavailable in this runtime", 503)

    data = request.get_json(silent=True) or {}
    task_id = (data.get("task_id") or "").strip()

    try:
        task = TaskBank.get_task(task_id) if task_id else None
        obs = _openenv_env.reset(task)
    except Exception as exc:
        return _openenv_error(str(exc), 400)

    return jsonify(obs.model_dump(mode="json"))


@app.get("/state")
def openenv_state():
    if not _rl_available or _openenv_env is None:
        return _openenv_error("OpenEnv environment is unavailable in this runtime", 503)

    try:
        obs = _openenv_env.state()
    except Exception as exc:
        return _openenv_error(str(exc), 400)

    return jsonify(obs.model_dump(mode="json"))


@app.post("/api/save_report")
def api_save_report():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    report_data = {
        "type": "session",
        "root_question": data.get("root_question", "Practice Session"),
        "turns": data.get("turns", []),
    }
    save_interview_report(session["username"], report_data)
    return jsonify({"status": "success"})


@app.post("/step")
def openenv_step():
    if not _rl_available or _openenv_env is None:
        return _openenv_error("OpenEnv environment is unavailable in this runtime", 503)

    data = request.get_json(silent=True) or {}
    action_data = data.get("action") if isinstance(data.get("action"), dict) else data

    try:
        strategy = action_data.get("strategy")
        confidence = action_data.get("confidence", 0.95)
        response_text = action_data.get("response_text")

        if not strategy:
            return _openenv_error("action.strategy is required")
        if not response_text or not str(response_text).strip():
            return _openenv_error("action.response_text is required")

        action = Action(
            strategy=FeedbackStrategy(strategy),
            confidence=float(confidence),
            response_text=str(response_text),
        )
        result = _openenv_env.step(action)
    except ValueError as exc:
        return _openenv_error(str(exc), 400)
    except Exception as exc:
        return _openenv_error(str(exc), 500)

    return jsonify(result.model_dump(mode="json"))


init_db()
init_rl_system()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
