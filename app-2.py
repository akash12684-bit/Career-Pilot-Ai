"""
CareerPilot AI — Flask backend (multi-user)
=============================================

Serves the frontend, handles registration/login/logout, stores each user's
profile and AI results in SQLite (scoped by user_id), extracts text from
uploaded resumes, and calls Gemini for all AI features.

Environment variables:
    GEMINI_API_KEY     — your Gemini API key (required for AI features)
    FLASK_SECRET_KEY   — secret key for signing session cookies (required in production)

Run with:
    GEMINI_API_KEY=your_key FLASK_SECRET_KEY=your_secret python app.py
"""

import os
import re
import json
import uuid
import sqlite3
import logging
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document

# ---------------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerpilot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "careerpilot.db")

# Uploads live outside anything Flask serves publicly — there is no route
# that reads from this folder and hands files back to the browser.
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Only these are accepted, because these are the only formats we can
# actually extract text from below (pypdf and python-docx).
ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx"}
MIN_RESUME_TEXT_LENGTH = 30  # below this, we treat extraction as a failure

# Keep the model name in one place so it's easy to update later.
MODEL = "gemini-3.7-flash"

app = Flask(__name__, static_folder=None)

flask_secret = os.environ.get("FLASK_SECRET_KEY")
if not flask_secret:
    flask_secret = "dev-only-insecure-secret-CHANGE-ME"
    logger.warning(
        "FLASK_SECRET_KEY is not set. Using an insecure development fallback. "
        "Set a real FLASK_SECRET_KEY before deploying this anywhere other than your own machine."
    )
app.secret_key = flask_secret

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload size
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Note: SESSION_COOKIE_SECURE is left off so cookies still work over plain
# http on localhost. If you deploy this behind HTTPS, set
# app.config["SESSION_COOKIE_SECURE"] = True.


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

if client is None:
    logger.warning("GEMINI_API_KEY is not set. AI features will return HTTP 503 until it is.")


class AIServiceError(Exception):
    """Raised whenever Gemini fails, is unreachable, or returns something
    we can't trust. Routes catch this and return HTTP 503 — they never save
    or return the broken result as if it were real data."""
    pass


def _matches_expected_type(value, expected):
    """Type check used to validate Gemini's JSON output field-by-field."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        # bool is technically an int subclass in Python — exclude it here
        # so a stray true/false doesn't pass as a valid score/percentage.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "list":
        return isinstance(value, list)
    return True  # unknown expected type — don't block on it


def ask_gemini_json(prompt, field_types=None):
    """
    Call Gemini and return the response parsed as JSON.
    Uses Gemini's built-in JSON output mode (response_mime_type) so we don't
    have to rely purely on fragile manual string parsing.

    `field_types` is an optional dict like {"reply": "string"} that doubles
    as both the required-fields list (its keys) and a type check for each
    field's value ("string", "number", or "list").

    Raises AIServiceError (never returns a fake/partial result) if:
      - the API key is missing
      - the request to Gemini fails
      - the response isn't valid JSON
      - the response is missing any field in `field_types`
      - any field's value doesn't match its expected type in `field_types`
    """
    if client is None:
        raise AIServiceError("AI service is temporarily unavailable. Please try again.")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
    except Exception:
        logger.exception("Gemini request failed")
        raise AIServiceError("AI service is temporarily unavailable. Please try again.")

    # response_mime_type="application/json" should already give us clean JSON,
    # but we still guard against stray formatting just in case.
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                result = None

    if not isinstance(result, dict):
        logger.error("Could not parse Gemini response as JSON: %s", text[:300])
        raise AIServiceError("AI service is temporarily unavailable. Please try again.")

    if field_types:
        missing = [f for f in field_types if f not in result]
        if missing:
            logger.error("Gemini response missing fields %s. Raw: %s", missing, text[:300])
            raise AIServiceError("AI service is temporarily unavailable. Please try again.")

        bad_types = [
            f for f, expected in field_types.items()
            if not _matches_expected_type(result[f], expected)
        ]
        if bad_types:
            logger.error(
                "Gemini response has wrong type for fields %s. Raw: %s", bad_types, text[:300]
            )
            raise AIServiceError("AI service is temporarily unavailable. Please try again.")

    return result


# ---------------------------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------------------------
def extract_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        logger.exception("PDF text extraction failed")
        return ""


def extract_docx_text(file_path):
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception:
        logger.exception("DOCX text extraction failed")
        return ""


def extract_resume_text(file_path, extension):
    if extension == "pdf":
        return extract_pdf_text(file_path)
    if extension == "docx":
        return extract_docx_text(file_path)
    return ""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, table):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def column_exists(conn, table, column):
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c["name"] == column for c in cols)


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )

    # New per-user profile table. Note this is named "profiles" (plural) —
    # an old single-row "profile" table from an earlier version of this app
    # (if it exists) is a different table and is left untouched below.
    c.execute(
        """CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            current_role TEXT,
            target_role TEXT,
            skills TEXT,
            experience TEXT,
            interests TEXT,
            resume_filename TEXT,
            resume_path TEXT,
            resume_text TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS career_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS skill_gap (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS roadmap (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_text TEXT,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS interview_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            answer TEXT,
            feedback_json TEXT,
            score INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )

    conn.commit()

    # --- simple migration -------------------------------------------------
    # If any of these tables already existed from an older version of this
    # app (without a user_id column, or without resume_text), add the
    # missing column instead of dropping the table. Old rows are preserved.
    for table in ["career_analysis", "skill_gap", "roadmap", "resume_analysis", "interview_results", "chat_history"]:
        if table_exists(conn, table) and not column_exists(conn, table, "user_id"):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
            logger.info("[migration] Added missing 'user_id' column to existing '%s' table.", table)

    if table_exists(conn, "profiles") and not column_exists(conn, "profiles", "resume_text"):
        conn.execute("ALTER TABLE profiles ADD COLUMN resume_text TEXT")
        logger.info("[migration] Added missing 'resume_text' column to 'profiles'.")

    conn.commit()

    if table_exists(conn, "profile"):
        logger.info(
            "[migration] Found the old single-user 'profile' table from a previous version of this app. "
            "It has been left in place (not deleted) but is no longer used — every account now gets its "
            "own row in the new 'profiles' table instead."
        )

    conn.close()


init_db()


def ensure_profile_row(conn, user_id):
    """Make sure a profile row exists for this user (created automatically at registration,
    but this is a safe fallback in case it's ever missing)."""
    row = conn.execute("SELECT id FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.execute("INSERT INTO profiles (user_id) VALUES (?)", (user_id,))
        conn.commit()


def get_profile_row(user_id):
    conn = get_db()
    ensure_profile_row(conn, user_id)
    row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def profile_context_text(user):
    p = get_profile_row(user["id"])
    return (
        f"Name: {user.get('name', '')}\n"
        f"Current role: {p.get('current_role') or ''}\n"
        f"Target role: {p.get('target_role') or ''}\n"
        f"Skills: {p.get('skills') or ''}\n"
        f"Experience: {p.get('experience') or ''}\n"
        f"Interests: {p.get('interests') or ''}"
    )


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
def get_current_user():
    """Return {'id', 'name', 'email'} for the logged-in user, or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db()
    row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def login_required(view_func):
    """Decorator: blocks the route with a 401 unless the user is logged in."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Please log in first."}), 401
        return view_func(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# Static routes (serve the frontend) — no login required
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


# ---------------------------------------------------------------------------
# Auth: register / login / logout / me
# ---------------------------------------------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({"error": "An account with this email already exists."}), 409

    password_hash = generate_password_hash(password)
    cursor = conn.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
        (name, email, password_hash),
    )
    user_id = cursor.lastrowid

    # Every new user gets their own empty profile row right away.
    conn.execute("INSERT INTO profiles (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()

    session.clear()
    session["user_id"] = user_id

    return jsonify({"success": True, "name": name})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session.clear()
    session["user_id"] = user["id"]

    return jsonify({"success": True, "name": user["name"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/me", methods=["GET"])
def me():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Please log in first."}), 401
    return jsonify(user)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@app.route("/api/profile", methods=["GET", "POST"])
@login_required
def profile_route():
    user = get_current_user()
    conn = get_db()
    ensure_profile_row(conn, user["id"])

    if request.method == "GET":
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user["id"],)).fetchone()
        conn.close()
        result = dict(row) if row else {}
        result["name"] = user["name"]
        # Don't ship the full resume text down with every profile fetch —
        # the frontend only needs to know a resume exists, via /api/resume.
        result.pop("resume_text", None)
        return jsonify(result)

    data = request.get_json(silent=True) or {}

    new_name = (data.get("name") or "").strip()
    target_role = (data.get("target_role") or "").strip()
    skills = (data.get("skills") or "").strip()
    experience = (data.get("experience") or "").strip()
    current_role = (data.get("current_role") or "").strip()
    interests = (data.get("interests") or "").strip()

    if not target_role:
        conn.close()
        return jsonify({"error": "Target role is required."}), 400

    # The frontend's profile form also sends "name" — let it update the
    # account's display name too, since that's what the user expects.
    if new_name:
        conn.execute("UPDATE users SET name = ? WHERE id = ?", (new_name, user["id"]))

    conn.execute(
        """
        UPDATE profiles
        SET current_role = ?, target_role = ?, skills = ?, experience = ?, interests = ?, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
        """,
        (current_role, target_role, skills, experience, interests, user["id"]),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Resume upload / fetch / delete
# ---------------------------------------------------------------------------
def allowed_resume_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_RESUME_EXTENSIONS


@app.route("/api/resume/upload", methods=["POST"])
@login_required
def resume_upload():
    user = get_current_user()

    if "resume" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_resume_file(file.filename):
        return jsonify({"error": "Only PDF and DOCX files are allowed."}), 400

    safe_name = secure_filename(file.filename)
    extension = safe_name.rsplit(".", 1)[1].lower()
    unique_name = f"user{user['id']}_{uuid.uuid4().hex[:8]}_{safe_name}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(save_path)
    except Exception:
        logger.exception("Failed to save uploaded resume")
        return jsonify({"error": "Could not save the file. Please try again."}), 500

    extracted_text = extract_resume_text(save_path, extension)

    if not extracted_text or len(extracted_text) < MIN_RESUME_TEXT_LENGTH:
        # Don't keep a broken record — remove the file we just saved.
        try:
            os.remove(save_path)
        except OSError:
            logger.exception("Failed to clean up unreadable resume file")
        return jsonify({
            "error": "Could not read this file. Please upload a text-based PDF or DOCX (not a scanned image)."
        }), 400

    conn = get_db()
    ensure_profile_row(conn, user["id"])
    conn.execute(
        """
        UPDATE profiles
        SET resume_filename = ?, resume_path = ?, resume_text = ?, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
        """,
        (file.filename, unique_name, extracted_text, user["id"]),
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Resume uploaded and processed successfully.", "filename": file.filename})


@app.route("/api/resume", methods=["GET"])
@login_required
def resume_get():
    user = get_current_user()
    conn = get_db()
    row = conn.execute(
        "SELECT resume_filename, resume_text, updated_at FROM profiles WHERE user_id = ?", (user["id"],)
    ).fetchone()
    conn.close()

    if not row or not row["resume_filename"]:
        return jsonify({"resume_filename": None, "has_text": False})

    return jsonify({
        "resume_filename": row["resume_filename"],
        "updated_at": row["updated_at"],
        "has_text": bool(row["resume_text"]),
    })


@app.route("/api/resume", methods=["DELETE"])
@login_required
def resume_delete():
    user = get_current_user()
    conn = get_db()
    row = conn.execute("SELECT resume_path FROM profiles WHERE user_id = ?", (user["id"],)).fetchone()

    if row and row["resume_path"]:
        # resume_path was generated by the server (never taken directly from
        # user input), so this is safe from path traversal.
        file_path = os.path.join(UPLOAD_FOLDER, row["resume_path"])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logger.exception("Failed to delete resume file from disk")

    conn.execute(
        """
        UPDATE profiles
        SET resume_filename = NULL, resume_path = NULL, resume_text = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
        """,
        (user["id"],),
    )
    # The old resume analysis results were based on a resume that no longer
    # exists, so they go too — but only this user's rows.
    conn.execute("DELETE FROM resume_analysis WHERE user_id = ?", (user["id"],))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Career analysis
# ---------------------------------------------------------------------------
CAREER_FIELD_TYPES = {
    "best_fit_career": "string",
    "match_percentage": "number",
    "alternative_careers": "list",
    "strengths": "list",
    "weaknesses": "list",
    "missing_skills": "list",
    "recommendations": "list",
}


@app.route("/api/career-analysis", methods=["POST"])
@login_required
def career_analysis():
    user = get_current_user()
    profile = get_profile_row(user["id"])

    if not (profile.get("target_role") or "").strip():
        return jsonify({"error": "Please add a target role to your profile first."}), 400

    prompt = f"""You are a career guidance AI. Analyze this person's profile against the SPECIFIC
target role they entered — accept any role they name, do not force a predefined list.

Profile:
{profile_context_text(user)}

Return JSON with exactly this structure:
{{
  "best_fit_career": "string",
  "match_percentage": 0,
  "alternative_careers": [{{"title": "string", "match_percentage": 0}}],
  "strengths": ["string"],
  "weaknesses": ["string"],
  "missing_skills": ["string"],
  "recommendations": ["string"]
}}

Judge how well their current skills and experience fit their stated target role specifically.
Do not invent or mention salary figures."""

    try:
        result = ask_gemini_json(prompt, CAREER_FIELD_TYPES)
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 503

    conn = get_db()
    conn.execute(
        "INSERT INTO career_analysis (user_id, result_json) VALUES (?, ?)",
        (user["id"], json.dumps(result)),
    )
    conn.commit()
    conn.close()

    return jsonify(result)


# ---------------------------------------------------------------------------
# Skill gap
# ---------------------------------------------------------------------------
SKILL_GAP_FIELD_TYPES = {
    "existing_skills": "list",
    "missing_skills": "list",
    "learning_order": "list",
}


@app.route("/api/skill-gap", methods=["POST"])
@login_required
def skill_gap():
    user = get_current_user()
    profile = get_profile_row(user["id"])

    if not (profile.get("target_role") or "").strip():
        return jsonify({"error": "Please add a target role to your profile first."}), 400

    prompt = f"""You are a career guidance AI. Analyze the skill gap between this profile and
their target role (accept any role they entered).

Profile:
{profile_context_text(user)}

Return JSON with exactly this structure:
{{
  "existing_skills": ["string"],
  "missing_skills": [{{"skill": "string", "priority": "High", "explanation": "string"}}],
  "learning_order": ["string"]
}}

"priority" must be one of: High, Medium, Low. "learning_order" lists topics/skills in the
order they should be learned, specific to this person's target role."""

    try:
        result = ask_gemini_json(prompt, SKILL_GAP_FIELD_TYPES)
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 503

    conn = get_db()
    conn.execute(
        "INSERT INTO skill_gap (user_id, result_json) VALUES (?, ?)",
        (user["id"], json.dumps(result)),
    )
    conn.commit()
    conn.close()

    return jsonify(result)


# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------
ROADMAP_FIELD_TYPES = {
    "total_weeks": "number",
    "phases": "list",
}


@app.route("/api/roadmap", methods=["POST"])
@login_required
def roadmap():
    user = get_current_user()
    profile = get_profile_row(user["id"])

    if not (profile.get("target_role") or "").strip():
        return jsonify({"error": "Please add a target role to your profile first."}), 400

    prompt = f"""You are a career guidance AI. Build an 8-12 week personalized learning roadmap
for this specific person to reach their target role, based on their current skills, likely
skill gaps, experience level, and interests. Two people with different profiles should get
different roadmaps — do not give a generic one-size-fits-all plan.

Profile:
{profile_context_text(user)}

Return JSON with exactly this structure:
{{
  "total_weeks": 10,
  "phases": [{{"weeks": "Weeks 1-2", "title": "string", "details": "string"}}]
}}"""

    try:
        result = ask_gemini_json(prompt, ROADMAP_FIELD_TYPES)
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 503

    conn = get_db()
    conn.execute(
        "INSERT INTO roadmap (user_id, result_json) VALUES (?, ?)",
        (user["id"], json.dumps(result)),
    )
    conn.commit()
    conn.close()

    return jsonify(result)


# ---------------------------------------------------------------------------
# Resume analyzer (pasted text OR previously uploaded resume)
# ---------------------------------------------------------------------------
RESUME_FIELD_TYPES = {
    "overall_score": "number",
    "strengths": "list",
    "weaknesses": "list",
    "ats_observations": "list",
    "missing_keywords": "list",
    "missing_skills": "list",
    "improvement_suggestions": "list",
}


def clamp_number(value, low, high, default=0):
    """Safely convert an AI-returned value to an int, clamped to [low, high]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN check
        return default
    return max(low, min(high, int(round(value))))


@app.route("/api/resume-analysis", methods=["POST"])
@login_required
def resume_analysis():
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    resume_text = (data.get("resume_text") or "").strip()

    # B: no resume_text sent — fall back to the user's stored uploaded resume.
    used_stored_resume = False
    if not resume_text:
        profile = get_profile_row(user["id"])
        resume_text = (profile.get("resume_text") or "").strip()
        used_stored_resume = True

    # C: still nothing — tell the user what to do.
    if not resume_text or len(resume_text) < MIN_RESUME_TEXT_LENGTH:
        if used_stored_resume:
            return jsonify({"error": "Please upload a resume or paste your resume text first."}), 400
        return jsonify({"error": "Please paste more of your resume text — it looks too short to analyze."}), 400

    prompt = f"""You are a resume review AI. Analyze this resume against the person's target role.
This is an ESTIMATED compatibility score, NOT an official ATS score — never call it official.

Profile:
{profile_context_text(user)}

Resume:
{resume_text}

Return JSON with exactly this structure:
{{
  "overall_score": 0,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "ats_observations": ["string"],
  "missing_keywords": ["string"],
  "missing_skills": ["string"],
  "improvement_suggestions": ["string"]
}}

"overall_score" must be a whole number from 0 to 100."""

    try:
        result = ask_gemini_json(prompt, RESUME_FIELD_TYPES)
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 503

    result["overall_score"] = clamp_number(result.get("overall_score"), 0, 100)

    conn = get_db()
    conn.execute(
        "INSERT INTO resume_analysis (user_id, resume_text, result_json) VALUES (?, ?, ?)",
        (user["id"], resume_text, json.dumps(result)),
    )
    conn.commit()
    conn.close()

    return jsonify(result)


# ---------------------------------------------------------------------------
# Mock interview
# ---------------------------------------------------------------------------
INTERVIEW_START_FIELD_TYPES = {"questions": "list"}
INTERVIEW_EVAL_FIELD_TYPES = {"score": "number", "improvement_suggestions": "list"}


@app.route("/api/mock-interview", methods=["POST"])
@login_required
def mock_interview():
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action == "start":
        profile = get_profile_row(user["id"])
        if not (profile.get("target_role") or "").strip():
            return jsonify({"error": "Please add a target role to your profile first."}), 400

        prompt = f"""You are an interviewer AI. Generate 5 mock interview questions tailored
specifically to this person's target role.

Profile:
{profile_context_text(user)}

Return JSON with exactly this structure:
{{
  "questions": ["string", "string", "string", "string", "string"]
}}"""
        try:
            result = ask_gemini_json(prompt, INTERVIEW_START_FIELD_TYPES)
        except AIServiceError as e:
            return jsonify({"error": str(e)}), 503

        if not isinstance(result.get("questions"), list) or len(result["questions"]) == 0:
            return jsonify({"error": "AI service is temporarily unavailable. Please try again."}), 503

        return jsonify(result)

    if action == "evaluate":
        question = (data.get("question") or "").strip()
        answer = (data.get("answer") or "").strip()

        if not question:
            return jsonify({"error": "Missing interview question."}), 400
        if len(answer) < 3:
            return jsonify({"error": "Please write a more complete answer before submitting."}), 400

        prompt = f"""You are an interview coach AI. Evaluate this candidate's answer to a mock
interview question for their target role.

Profile:
{profile_context_text(user)}

Question: {question}
Answer: {answer}

Return JSON with exactly this structure:
{{
  "technical_accuracy": "string",
  "completeness": "string",
  "communication": "string",
  "missing_points": ["string"],
  "improvement_suggestions": ["string"],
  "score": 0
}}

"score" is out of 10."""

        try:
            result = ask_gemini_json(prompt, INTERVIEW_EVAL_FIELD_TYPES)
        except AIServiceError as e:
            return jsonify({"error": str(e)}), 503

        # Safely convert to a number and clamp to the 0-10 range this
        # feature expects, regardless of what Gemini actually returned.
        score_value = clamp_number(result.get("score", 0), 0, 10)
        result["score"] = score_value

        conn = get_db()
        conn.execute(
            "INSERT INTO interview_results (user_id, question, answer, feedback_json, score) VALUES (?, ?, ?, ?, ?)",
            (user["id"], question, answer, json.dumps(result), score_value),
        )
        conn.commit()
        conn.close()

        return jsonify(result)

    return jsonify({"error": "Invalid action. Use 'start' or 'evaluate'."}), 400


# ---------------------------------------------------------------------------
# AI assistant / chat
# ---------------------------------------------------------------------------
CHAT_FIELD_TYPES = {"reply": "string"}


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    conn = get_db()
    history_rows = conn.execute(
        "SELECT role, message FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT 6",
        (user["id"],),
    ).fetchall()
    history_rows = list(reversed(history_rows))
    history_text = "\n".join([f"{r['role']}: {r['message']}" for r in history_rows])

    prompt = f"""You are CareerPilot AI's career assistant — a professional career mentor. Give
specific, practical, personalized advice based on the profile below. Never give generic
chatbot filler; always tie your answer back to this person's actual situation.

Profile:
{profile_context_text(user)}

Recent conversation:
{history_text}

User's new message: {message}

Return JSON with exactly this structure:
{{
  "reply": "string"
}}

Keep the reply concrete and no more than 4-5 sentences."""

    try:
        result = ask_gemini_json(prompt, CHAT_FIELD_TYPES)
    except AIServiceError as e:
        conn.close()
        return jsonify({"error": str(e)}), 503

    reply = result["reply"]

    conn.execute(
        "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
        (user["id"], "user", message),
    )
    conn.execute(
        "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
        (user["id"], "assistant", reply),
    )
    conn.commit()
    conn.close()

    return jsonify({"reply": reply})


# ---------------------------------------------------------------------------
# Friendly error handlers (never leak internal exception details)
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File is too large. Maximum upload size is 5 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
