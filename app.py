import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from extract_audio import pull_audio_from_video, check_if_audio_file, check_if_video_file
from transcribe import load_whisper_model, transcribe_audio, save_transcript_to_json, save_transcript_readable
from vectorize import (
    load_embedding_model,
    load_reranker_model,
    load_qdrant_client,
    setup_collection,
    embed_and_store,
    search_transcript,
)
from rag import build_context_from_segments, build_system_prompt
from llm_client import get_llm_client
from study_generator import generate_flashcards, generate_quiz
from visual_cue import process_visual_cues, find_visual_cues


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# preloading models once after starting
print("[info] preloading whisper model...")
whisper_model = load_whisper_model("base", device="cpu")
print("[info] preloading embedding model...")
embedding_model = load_embedding_model()
print("[info] preloading reranker model...")
reranker_model = load_reranker_model()
print("[info] connecting to qdrant...")
qdrant_client = load_qdrant_client()
setup_collection(qdrant_client)

# loading LLM client  groq from .env
llm_client = None
try:
    llm_client = get_llm_client()
    print(f"[info] LLM ready: {llm_client.provider_name}")
except Exception as e:
    print(f"[warn] LLM not available: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/api/lectures", methods=["GET"])
def list_lectures():
    """list all available lecture files in uploads folder"""
    lectures = []

    # checking uploads folder for media files
    for f in os.listdir(UPLOAD_FOLDER):
        if allowed_file(f):
            filepath = os.path.join(UPLOAD_FOLDER, f)
            base_name = os.path.splitext(f)[0]
            transcript_json = os.path.join(OUTPUT_FOLDER, base_name, f"{base_name}_transcript.json")
            has_transcript = os.path.exists(transcript_json)

            lectures.append({
                "filename": f,
                "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 1),
                "has_transcript": has_transcript,
            })

    return jsonify({"lectures": lectures})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """serve uploaded video/audio files for the HTML5 player"""
    return send_from_directory(UPLOAD_FOLDER, filename)


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
        "filepath": save_path
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

    # checking if transcript already exists
    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        # also make sure vectors are stored
        embed_and_store(json_path, embedding_model, qdrant_client)
        return jsonify({
            "message": "transcript loaded from cache",
            "cached": True,
            "time_taken": 0,
            "language": transcript_data["language"],
            "language_confidence": transcript_data.get("language_confidence", 0),
            "total_segments": transcript_data["total_segments"],
            "segments": transcript_data["segments"],
            "full_text": transcript_data["full_text"],
        })

    start_time = time.time()

    # extracting audio if needed
    audio_to_transcribe = None
    if check_if_audio_file(source_file):
        audio_to_transcribe = source_file
    elif check_if_video_file(source_file):
        extracted, err = pull_audio_from_video(source_file, output_dir)
        if extracted is None:
            msg = err or "audio extraction failed. is ffmpeg installed?"
            return jsonify({"error": msg}), 500
        audio_to_transcribe = extracted
    else:
        return jsonify({"error": "unsupported file type"}), 400

    transcript_data = transcribe_audio(whisper_model, audio_to_transcribe)
    if transcript_data is None:
        return jsonify({"error": "transcription failed"}), 500

    elapsed_time = round(time.time() - start_time, 1)

    # saving outputs
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_transcript_to_json(transcript_data, json_path)
    save_transcript_readable(transcript_data, txt_path)

    #vectorizing
    embed_and_store(json_path, embedding_model, qdrant_client)

    return jsonify({
        "message": "transcription complete",
        "cached": False,
        "time_taken": elapsed_time,
        "language": transcript_data["language"],
        "language_confidence": transcript_data["language_confidence"],
        "total_segments": transcript_data["total_segments"],
        "segments": transcript_data["segments"],
        "full_text": transcript_data["full_text"],
    })


@app.route("/search", methods=["POST"])
def handle_search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "no query provided"}), 400

    query = data["query"]
    top_k = data.get("top_k", 5)

    results = search_transcript(
        query, embedding_model, qdrant_client,
        top_k=top_k, reranker=reranker_model
    )

    matches = []
    for r in results:
        matches.append({
            "text": r.payload["text"],
            "start": r.payload["start"],
            "end": r.payload["end"],
            "score": round(r.score, 3),
        })

    return jsonify({"query": query, "matches": matches})


def load_all_segments(filename):
    """load all transcript segments for a given filename"""
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(OUTPUT_FOLDER, base_name, f"{base_name}_transcript.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f).get("segments", [])


@app.route("/ask", methods=["POST"])
def handle_ask():
    if not llm_client:
        return jsonify({"error": "LLM API key not configured"}), 500

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "no question provided"}), 400

    question = data["question"]
    filename = data.get("filename")

    # always send full transcript so the LLM has complete lecture context
    all_segments = load_all_segments(filename) if filename else None

    if not all_segments:
        return jsonify({"error": "no transcript found. transcribe a lecture first."}), 404

    context_text = build_context_from_segments(all_segments)
    system_prompt = build_system_prompt()
    user_message = f"Full lecture transcript:\n{context_text}\n\nQuestion: {question}"

    answer = llm_client.generate(system_prompt, user_message, temperature=0.1, max_tokens=1024)

    # use search to find the most relevant timestamps to show as sources
    results = search_transcript(
        question, embedding_model, qdrant_client,
        top_k=3, reranker=reranker_model
    )
    source_segments = [
        {"text": r.payload["text"], "start": r.payload["start"], "end": r.payload["end"]}
        for r in results[:3]
    ]

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": source_segments,
    })


@app.route("/generate-flashcards", methods=["POST"])
def handle_flashcards():
    if not llm_client:
        return jsonify({"error": "LLM API key not configured"}), 500

    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "no filename provided"}), 400

    filename = data["filename"]
    count = data.get("count", 8)

    # loading transcript
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(OUTPUT_FOLDER, base_name, f"{base_name}_transcript.json")

    if not os.path.exists(json_path):
        return jsonify({"error": "transcript not found. transcribe the lecture first."}), 404

    with open(json_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    flashcards = generate_flashcards(llm_client, transcript_data["segments"], count=count)

    return jsonify({"flashcards": flashcards, "count": len(flashcards)})


@app.route("/generate-quiz", methods=["POST"])
def handle_quiz():
    if not llm_client:
        return jsonify({"error": "LLM API key not configured"}), 500

    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "no filename provided"}), 400

    filename = data["filename"]
    count = data.get("count", 5)

    #loading transcript
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(OUTPUT_FOLDER, base_name, f"{base_name}_transcript.json")

    if not os.path.exists(json_path):
        return jsonify({"error": "transcript not found. transcribe the lecture first."}), 404

    with open(json_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    quiz = generate_quiz(llm_client, transcript_data["segments"], count=count)

    return jsonify({"questions": quiz, "count": len(quiz)})


@app.route("/frames/<path:filepath>")
def serve_frame(filepath):
    """serve captured frame images from output folder"""
    return send_from_directory(OUTPUT_FOLDER, filepath)


@app.route("/detect-visual-cues", methods=["POST"])
def handle_visual_cues():
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "no filename provided"}), 400

    filename = data["filename"]
    source_file = os.path.join(UPLOAD_FOLDER, filename)

    if not check_if_video_file(source_file):
        return jsonify({"error": "visual cue detection requires a video file, not audio"}), 400

    base_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(OUTPUT_FOLDER, base_name)
    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")

    if not os.path.exists(json_path):
        return jsonify({"error": "transcript not found. transcribe the lecture first."}), 404

    # return cached results if already processed
    cues_path = os.path.join(output_dir, "visual_cues.json")
    if os.path.exists(cues_path):
        with open(cues_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return jsonify({**cached, "cached": True})

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return jsonify({"error": "Gemini API key not configured. Set GEMINI_API_KEY in environment."}), 500

    with open(json_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    results = process_visual_cues(
        source_file, transcript_data["segments"], output_dir, gemini_key
    )

    # embed visual cue descriptions into qdrant so they show up in search
    if results["cues"]:
        from qdrant_client.models import PointStruct
        cue_texts = [c["description"] for c in results["cues"]]
        cue_embeddings = embedding_model.encode(cue_texts)
        base_id = 10000  # offset to avoid colliding with transcript segment IDs
        points = []
        for j, (cue, emb) in enumerate(zip(results["cues"], cue_embeddings)):
            points.append(PointStruct(
                id=base_id + j,
                vector=emb.tolist(),
                payload={
                    "text": f"[Visual] {cue['description']}",
                    "start": cue["timestamp"],
                    "end": cue["timestamp"] + 5,
                    "source_file": filename,
                    "type": "visual_cue",
                },
            ))
        qdrant_client.upsert(collection_name="lecture_segments", points=points)
        print(f"[visual] indexed {len(points)} visual cue descriptions in qdrant")

    return jsonify({**results, "cached": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n starting LectureIQ web server...")
    print(f" open http://localhost:{port} in browser\n")
    app.run(debug=False, host="0.0.0.0", port=port)
