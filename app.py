import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from extract_audio import pull_audio_from_video, check_if_audio_file, check_if_video_file
from transcribe import (
    load_whisper_model, transcribe_audio,
    save_transcript_to_json, save_transcript_readable,
)
from extract_frames import detect_diagram_segments, extract_all_diagram_frames
from image_analysis import (
    upload_all_frames, analyze_all_frames, analyze_image_with_groq,
    generate_summary, chat_with_lecture,
)

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma",
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("[info] preloading whisper model at startup...")
whisper_model = load_whisper_model("base", device="cpu")

# ── in-memory session store (keyed by filename) ──
# stores transcript + diagram data so the /chat endpoint can access it
lecture_sessions = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════
#  ROUTES
# ═══════════════════════════════════

@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def handle_upload():
    if "lecture_file" not in request.files:
        return jsonify({"error": "no file selected"}), 400

    uploaded_file = request.files["lecture_file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "no file selected"}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "unsupported file format"}), 400

    safe_name = secure_filename(uploaded_file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    uploaded_file.save(save_path)

    return jsonify({
        "message": "file uploaded successfully",
        "filename": safe_name,
        "filepath": save_path,
    })


@app.route("/transcribe", methods=["POST"])
def handle_transcription():
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "no filename provided"}), 400

    filename = data["filename"]
    source_file = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(source_file):
        return jsonify({"error": "file not found on server"}), 404

    base_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(OUTPUT_FOLDER, base_name)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # ── audio extraction ──
    audio_to_transcribe = None
    if check_if_audio_file(source_file):
        audio_to_transcribe = source_file
    elif check_if_video_file(source_file):
        extracted = pull_audio_from_video(source_file, output_dir)
        if extracted is None:
            return jsonify({"error": "audio extraction failed. is ffmpeg installed?"}), 500
        audio_to_transcribe = extracted
    else:
        return jsonify({"error": "unsupported file type"}), 400

    # ── whisper transcription ──
    transcript_data = transcribe_audio(whisper_model, audio_to_transcribe)
    if transcript_data is None:
        return jsonify({"error": "transcription failed"}), 500

    elapsed_time = round(time.time() - start_time, 1)

    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_transcript_to_json(transcript_data, json_path)
    save_transcript_readable(transcript_data, txt_path)

    # ── diagram extraction + auto analysis ──
    diagrams_data = []
    if check_if_video_file(source_file):
        diagram_segments = detect_diagram_segments(transcript_data["segments"])
        if diagram_segments:
            frames = extract_all_diagram_frames(source_file, diagram_segments, output_dir)
            if frames:
                frames = upload_all_frames(frames)
                frames = analyze_all_frames(frames)   # auto-analyze with AI
                for frame in frames:
                    rel_path = os.path.relpath(frame["frame_path"], OUTPUT_FOLDER)
                    frame["local_url"] = f"/output/{rel_path}"
                    frame["frame_rel_path"] = rel_path
                diagrams_data = frames

    # ── store in session for chat ──
    lecture_sessions[filename] = {
        "transcript": transcript_data,
        "diagrams": diagrams_data,
    }

    return jsonify({
        "message": "transcription complete",
        "time_taken": elapsed_time,
        "language": transcript_data["language"],
        "language_confidence": transcript_data["language_confidence"],
        "total_segments": transcript_data["total_segments"],
        "segments": transcript_data["segments"],
        "full_text": transcript_data["full_text"],
        "diagrams": diagrams_data,
    })


@app.route("/output/<path:filepath>")
def serve_output_file(filepath):
    return send_from_directory(OUTPUT_FOLDER, filepath)


@app.route("/analyze-diagram", methods=["POST"])
def handle_analyze_diagram():
    data = request.get_json()
    if not data or "frame_rel_path" not in data:
        return jsonify({"error": "no frame path provided"}), 400

    actual_path = os.path.join(OUTPUT_FOLDER, data["frame_rel_path"])
    if not os.path.exists(actual_path):
        return jsonify({"error": "frame image not found"}), 404

    analysis = analyze_image_with_groq(actual_path)
    return jsonify({"analysis": analysis, "frame_rel_path": data["frame_rel_path"]})


@app.route("/summary", methods=["POST"])
def handle_summary():
    """Generate a lecture summary from transcript + diagram analyses."""
    data = request.get_json()
    filename = data.get("filename", "") if data else ""

    session = lecture_sessions.get(filename)
    if not session:
        return jsonify({"error": "no lecture data found – transcribe first"}), 400

    summary = generate_summary(
        session["transcript"]["full_text"],
        session["diagrams"],
    )
    return jsonify({"summary": summary})


@app.route("/chat", methods=["POST"])
def handle_chat():
    """
    Chat endpoint — takes a question + filename, finds relevant transcript
    segments and diagram analyses, sends to Groq LLM, returns answer + sources.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "no question provided"}), 400

    filename = data.get("filename", "")
    question = data["question"].strip()

    if not question:
        return jsonify({"error": "empty question"}), 400

    session = lecture_sessions.get(filename)
    if not session:
        return jsonify({"error": "no lecture data found – transcribe first"}), 400

    result = chat_with_lecture(
        question,
        session["transcript"]["segments"],
        session["diagrams"],
    )
    return jsonify(result)


if __name__ == "__main__":
    print("\n starting LectureIQ web server...")
    print(" open http://localhost:5000 in browser\n")
    app.run(debug=False, port=5000)
