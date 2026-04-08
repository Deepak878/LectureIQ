"""
image_analysis.py - Cloudinary Upload, AI Vision Analysis & Lecture Chat

MODELS USED:
  - Vision analysis: meta-llama/llama-4-scout-17b-16e-instruct (multimodal)
  - Chat answers:    llama-3.3-70b-versatile (text, fast)
"""

import os
import re
import base64
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

# ─── Groq model config ───
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHAT_MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════
#  CLOUDINARY
# ══════════════════════════════════════

def init_cloudinary():
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if not all([cloud_name, api_key, api_secret]):
        return False
    if cloud_name == "your_cloud_name_here" or api_key == "your_api_key_here":
        return False

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    return True


def upload_to_cloudinary(image_path, folder="lectureiq/diagrams"):
    try:
        result = cloudinary.uploader.upload(
            image_path, folder=folder, resource_type="image"
        )
        url = result.get("secure_url")
        print(f"[info] uploaded to cloudinary: {url}")
        return url
    except Exception as e:
        print(f"[error] cloudinary upload failed: {e}")
        return None


def upload_all_frames(frames_list):
    if not init_cloudinary():
        print("[warn] skipping cloudinary upload – credentials not configured")
        return frames_list

    print(f"[info] uploading {len(frames_list)} frames to cloudinary...")
    for frame in frames_list:
        url = upload_to_cloudinary(frame["frame_path"])
        if url:
            frame["cloudinary_url"] = url

    ok = sum(1 for f in frames_list if f.get("cloudinary_url"))
    print(f"[info] cloudinary upload complete: {ok}/{len(frames_list)} succeeded")
    return frames_list


# ══════════════════════════════════════
#  VISION ANALYSIS  (Groq – Llama 4 Scout)
# ══════════════════════════════════════

def _get_groq_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return None, "GROQ_API_KEY not configured in .env"
    return Groq(api_key=api_key), None


def analyze_image_with_groq(image_path):
    """Analyze a single diagram frame using Groq vision model."""
    client, err = _get_groq_client()
    if err:
        return f"Error: {err}"
    if not os.path.exists(image_path):
        return f"Error: Image not found: {image_path}"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a frame from an educational lecture video. "
                                "Analyze it carefully.\n"
                                "If it contains a diagram, chart, graph, code snippet, "
                                "data structure, formula, flowchart, or any educational visual:\n"
                                "1. Describe what the visual shows\n"
                                "2. Explain the key concepts illustrated\n"
                                "3. Note important labels or annotations\n"
                                "4. Summarize what students should learn\n\n"
                                "If it just shows a person talking with no visual aid, say: "
                                "'No educational diagram detected – shows the lecturer speaking.'"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        print(f"[info] vision analysis done ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"[error] vision analysis failed: {e}")
        return f"Analysis failed: {e}"


def analyze_all_frames(frames_list):
    """Auto-analyze every extracted frame. Called during the pipeline."""
    client, err = _get_groq_client()
    if err:
        print(f"[warn] skipping auto-analysis – {err}")
        return frames_list

    import time

    for i, frame in enumerate(frames_list):
        print(f"[info] auto-analyzing diagram {i + 1}/{len(frames_list)}...")
        frame["analysis"] = analyze_image_with_groq(frame["frame_path"])
        # small pause to stay within Groq free-tier rate limits
        if i < len(frames_list) - 1:
            time.sleep(1.5)

    done = sum(1 for f in frames_list if f.get("analysis") and not f["analysis"].startswith("Error"))
    print(f"[info] auto-analysis complete: {done}/{len(frames_list)} succeeded")
    return frames_list


# ══════════════════════════════════════
#  LECTURE SUMMARY
# ══════════════════════════════════════

def generate_summary(transcript_text, diagrams):
    """Create a concise lecture summary using Groq LLM."""
    client, err = _get_groq_client()
    if err:
        return f"Error: {err}"

    # Include diagram info in the context
    diagram_context = ""
    for d in (diagrams or []):
        if d.get("analysis") and not d["analysis"].startswith("Error"):
            diagram_context += f"\n[Visual at {d.get('timestamp', '?')}s]: {d['analysis'][:300]}\n"

    # Truncate transcript to fit context window
    max_chars = 12000
    trunc = transcript_text[:max_chars]
    if len(transcript_text) > max_chars:
        trunc += "\n... [transcript truncated]"

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "text" if False else "content": (
                        "You are a helpful study assistant. Given a lecture transcript "
                        "and information about diagrams/visuals shown during the lecture, "
                        "produce a clear, well-structured summary covering all key topics, "
                        "concepts, examples, and takeaways. Use bullet points and headers."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Lecture Transcript\n{trunc}\n\n"
                        f"## Visual Content Descriptions\n{diagram_context or 'None detected.'}\n\n"
                        "Please summarize this lecture concisely."
                    ),
                },
            ],
            max_tokens=1500,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[error] summary generation failed: {e}")
        return f"Summary generation failed: {e}"


# ══════════════════════════════════════
#  CHAT / Q&A  (RAG-style)
# ══════════════════════════════════════

def _simple_relevance(query, text):
    """Very lightweight relevance scoring – counts query-word hits."""
    q_words = set(re.findall(r"\w{3,}", query.lower()))
    t_lower = text.lower()
    return sum(1 for w in q_words if w in t_lower)


def chat_with_lecture(question, transcript_segments, diagrams):
    """
    Answer a user question by:
    1. Finding the most relevant transcript segments (keyword match)
    2. Finding relevant diagram analyses
    3. Sending context + question to Groq LLM
    4. Returning the answer + sources
    """
    client, err = _get_groq_client()
    if err:
        return {"answer": f"Error: {err}", "sources": []}

    # ── score transcript segments ──
    scored_segs = []
    for seg in transcript_segments:
        score = _simple_relevance(question, seg["text"])
        if score > 0:
            scored_segs.append((score, seg))
    scored_segs.sort(key=lambda x: x[0], reverse=True)
    top_segs = scored_segs[:8]

    # ── score diagrams ──
    scored_diags = []
    for d in (diagrams or []):
        analysis = d.get("analysis", "") or ""
        transcript = d.get("transcript_text", "") or ""
        score = _simple_relevance(question, analysis + " " + transcript)
        if score > 0:
            scored_diags.append((score, d))
    scored_diags.sort(key=lambda x: x[0], reverse=True)
    top_diags = scored_diags[:3]

    # ── build context ──
    context_parts = []
    source_list = []

    for _, seg in top_segs:
        ts = _fmt(seg["start"])
        context_parts.append(f"[{ts}] {seg['text']}")
        source_list.append({
            "type": "transcript",
            "timestamp": ts,
            "start": seg["start"],
            "text": seg["text"][:120],
        })

    for _, d in top_diags:
        ts = _fmt(d.get("timestamp", d.get("start", 0)))
        analysis = (d.get("analysis") or "")[:500]
        context_parts.append(
            f"[DIAGRAM at {ts}] Transcript: {d.get('transcript_text', '')[:200]}\n"
            f"  AI Description: {analysis}"
        )
        source_list.append({
            "type": "diagram",
            "timestamp": ts,
            "start": d.get("timestamp", d.get("start", 0)),
            "text": d.get("transcript_text", "")[:120],
            "image_url": d.get("cloudinary_url") or d.get("local_url", ""),
            "analysis_snippet": analysis[:200],
            "diagram_index": d.get("index", 0),
        })

    context_block = "\n".join(context_parts) if context_parts else "No directly relevant content found."

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful lecture assistant. A student is asking a question "
                        "about a lecture they watched. Below is relevant context extracted from "
                        "the lecture transcript and any diagrams/visuals shown.\n\n"
                        "Rules:\n"
                        "- Answer based ONLY on the provided context.\n"
                        "- If a diagram is relevant, mention that a visual was shown and describe it.\n"
                        "- Include timestamps like [HH:MM:SS] when referencing specific parts.\n"
                        "- Be concise but thorough.\n"
                        "- If the context doesn't cover the question, say so honestly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Relevant Lecture Context\n{context_block}\n\n"
                        f"## Student Question\n{question}"
                    ),
                },
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        print(f"[error] chat failed: {e}")
        answer = f"Sorry, an error occurred: {e}"

    return {"answer": answer, "sources": source_list}


def _fmt(seconds):
    s = float(seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img = sys.argv[1]
        print(f"\nAnalyzing: {img}\n" + "-" * 40)
        result = analyze_image_with_groq(img)
        print(f"\n{result}")
    else:
        print("usage: python image_analysis.py <image_path>")
