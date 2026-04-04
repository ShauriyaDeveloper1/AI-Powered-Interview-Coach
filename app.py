import base64
import hashlib
import io
import json
import os
import queue
import re
import tempfile
import threading
import time
import uuid
import importlib
from datetime import datetime
from pathlib import Path

import cv2
import nltk
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file, session
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI

    _OPENAI_NEW = True
except ImportError:
    import openai

    _OPENAI_NEW = False

import moviepy.editor as mp
from PIL import Image, ImageDraw


# Ensure necessary NLTK resources are available.
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


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

openai_api_key = os.getenv("OPENAI_API_KEY")
freepik_api_key = os.getenv("FREEPIK_API_KEY")

client = None
if _OPENAI_NEW and openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key, base_url="https://api.aimlapi.com/v1")
    except Exception:
        client = None
elif openai_api_key:
    openai.api_key = openai_api_key
    openai.api_base = "https://api.aimlapi.com/v1"


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


def add_user(username, password):
    users = _load_users()
    users[username] = hash_password(password)
    _save_users(users)


def authenticate_user(username, password):
    users = _load_users()
    return username in users and users[username] == hash_password(password)


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
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(stopwords.words("english"))

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
        avg_confidence = (
            sum(r["analysis"]["confidence"]["confidence_score"] for r in filtered_reports) / num_sessions
        )
        avg_sentiment = sum(r["analysis"]["tone"]["score"] for r in filtered_reports) / num_sessions

        elements.append(Paragraph(f"Sessions Analyzed: {num_sessions}", styles["Normal"]))
        elements.append(Paragraph(f"Average Confidence Score: {avg_confidence:.1f}/10", styles["Normal"]))
        elements.append(Paragraph(f"Average Tone Score: {avg_sentiment:.2f}", styles["Normal"]))
        elements.append(Spacer(1, 20))

        table_data = [["Date", "Question", "Confidence", "Tone", "Professional", "Filler"]]
        for report in filtered_reports:
            q = report["question"][:30] + "..." if len(report["question"]) > 30 else report["question"]
            table_data.append(
                [
                    report["date"],
                    q,
                    f"{report['analysis']['confidence']['confidence_score']:.1f}",
                    f"{report['analysis']['tone']['score']:.2f}",
                    str(report["analysis"]["word_choice"]["professional_word_count"]),
                    str(report["analysis"]["word_choice"]["filler_word_count"]),
                ]
            )

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

    if not openai_api_key:
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
        clip = mp.ImageClip(img_array).set_duration(duration_per_clip)
        clips.append(clip)

    video = mp.concatenate_videoclips(clips, method="compose")
    final_clip = video.set_audio(audio_clip) if audio_clip else video
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


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    return jsonify(
        {
            "questions": QUESTION_BANK,
            "has_openai_key": bool(openai_api_key),
            "has_freepik_key": bool(freepik_api_key),
        }
    )


@app.get("/api/me")
def api_me():
    username = session.get("username")
    if not username:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": username})


@app.post("/api/signup")
def api_signup():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    if not username or not password or not confirm_password:
        return jsonify({"error": "Please fill in all fields"}), 400

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

    add_user(username, password)
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
    return {
        "question": question,
        "answer": answer,
        "analysis": analysis,
        "feedback": feedback,
    }


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

    payload = _build_analysis_payload(question, answer)
    save_interview_report(
        username,
        {"question": question, "answer": answer, "analysis": payload["analysis"]},
    )
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

    payload = _build_analysis_payload(question, transcription)
    save_interview_report(
        username,
        {"question": question, "answer": transcription, "analysis": payload["analysis"]},
    )
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

    payload = _build_analysis_payload(question, transcription, posture=posture_data)
    save_interview_report(
        username,
        {
            "question": question,
            "answer": transcription,
            "analysis": payload["analysis"],
        },
    )
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

    data = request.get_json(force=True)
    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

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

    task = tasks[0]
    obs = _rl_env.reset(task)
    _session_episodes[session_key] = {
        "task": task,
        "observation": obs,
        "attempt": 0,
        "total_reward": 0.0,
        "grades": [],
        "strategies_used": [],
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
        }
    )


@app.post("/api/rl/practice/text")
def api_rl_practice_text():
    username, err = login_required()
    if err:
        return err

    if not _rl_available or _rl_env is None:
        return jsonify({"error": "RL module is unavailable in this runtime"}), 503

    data = request.get_json(force=True)
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

    payload = _build_analysis_payload(question, answer)
    grade = payload["analysis"].get("tone", {}).get("score", 0.0) * 0.5 + 0.5
    grade = min(max(grade, 0.0), 1.0)

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
    session_state["grades"].append(grade)

    if use_agent_feedback and _rl_agent:
        current_obs = session_state["observation"]
        best_action = _rl_agent.choose_action(current_obs, use_epsilon_greedy=False)
        action = Action(strategy=best_action, confidence=0.95)
        result = _rl_env.step(action, answer)

        session_state["observation"] = result.observation
        session_state["total_reward"] += result.reward.total
        session_state["strategies_used"].append(best_action.value)

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
        reward = 0.0
        episode_done = False
        success = False

    save_interview_report(
        username,
        {
            "question": question,
            "answer": answer,
            "analysis": payload["analysis"],
            "rl_strategy": rl_strategy,
            "grade": grade,
        },
    )

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
                "total_reward": session_state["total_reward"],
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
    return jsonify(
        {
            "has_active_session": True,
            "task_id": session_state["task"].task_id,
            "question": session_state["task"].question,
            "attempt": session_state["attempt"],
            "max_attempts": session_state["task"].max_attempts,
            "total_reward": session_state["total_reward"],
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
