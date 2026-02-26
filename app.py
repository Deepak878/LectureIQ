import os
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from extract_audio import pull_audio_from_video, check_if_audio_file, check_if_video_file
from transcribe import load_whisper_model, transcribe_audio, save_transcript_to_json, save_transcript_readable


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   

# creating folders if they dont exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# loading whisper model once at startup so we dont reload every request
print("[info] preloading whisper model at startup...")
whisper_model = load_whisper_model("base", device="cpu")


def allowed_file(filename):
    """checking if uploaded file has a valid extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def handle_upload():
    """handles file upload from the frontend"""

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
        "message": "file uploaded succesfully",
        "filename": safe_name,
        "filepath": save_path
    })


@app.route("/transcribe", methods=["POST"])
def handle_transcription():
    """runs the full asr pipeline on the uploaded file"""

    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "no filename provided"}), 400

    filename = data["filename"]
    source_file = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(source_file):
        return jsonify({"error": "file not found on server"}), 404

    # setting up output dir for this file
    base_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(OUTPUT_FOLDER, base_name)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # extract audio if its a video file
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

    # running whisper transcription
    transcript_data = transcribe_audio(whisper_model, audio_to_transcribe)

    if transcript_data is None:
        return jsonify({"error": "transcription failed"}), 500

    elapsed_time = round(time.time() - start_time, 1)

    #  saving outputs to disk too
    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_transcript_to_json(transcript_data, json_path)
    save_transcript_readable(transcript_data, txt_path)

    # sending back to frontend
    return jsonify({
        "message": "transcription complete",
        "time_taken": elapsed_time,
        "language": transcript_data["language"],
        "language_confidence": transcript_data["language_confidence"],
        "total_segments": transcript_data["total_segments"],
        "segments": transcript_data["segments"],
        "full_text": transcript_data["full_text"],
    })


if __name__ == "__main__":
    print("\n starting LectureIQ web server...")
    print(" open http://localhost:5000 in  browser\n")
    app.run(debug=False, port=5000)
