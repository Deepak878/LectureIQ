import os
import subprocess
import json
from google import genai
from PIL import Image


VISUAL_CUE_PHRASES = [
    "look at this",
    "as you can see",
    "take a look",
    "if you look at",
    "you can see",
    "this diagram",
    "this equation",
    "this formula",
    "this chart",
    "this graph",
    "this table",
    "this figure",
    "this slide",
    "on the slide",
    "on this slide",
    "on the screen",
    "shown here",
    "right here",
    "over here",
    "let me show",
    "here we see",
    "looking at this",
    "notice the",
    "as shown",
    "refer to",
    "this image",
    "this picture",
    "pay attention to",
    "the screen shows",
    "written here",
    "drawn here",
]

# for minimum gap between cues in seconds and avoiding capturing almost duplicate frames
MIN_CUE_GAP = 10


def find_visual_cues(segments):
    """scanning transcript segments for verbal cue phrases that hint at visual content"""

    cues = []
    last_timestamp = -999

    for seg in segments:
        text_lower = seg["text"].lower()
        for phrase in VISUAL_CUE_PHRASES:
            if phrase in text_lower:
                #skipping if too close to previous cue
                if seg["start"] - last_timestamp < MIN_CUE_GAP:
                    break
                cues.append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "phrase_matched": phrase,
                })
                last_timestamp = seg["start"]
                break

    return cues


def extract_frame(video_path, timestamp_seconds, output_path):
    """grabbing a single frame from the video at given timestamp using ffmpeg"""

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp_seconds),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"[warn] frame extraction failed at {timestamp_seconds}s")
            return False
        return os.path.exists(output_path)
    except FileNotFoundError:
        print("[error] ffmpeg not found")
        return False


def describe_frame(image_path, api_key, context_text=""):
    """sending captured frame to Gemini Vision and getting a description of what's shown"""

    client = genai.Client(api_key=api_key)

    img = Image.open(image_path)

    prompt = (
        "This is a screenshot from a lecture video. "
        f'The professor was saying: "{context_text}" when this was captured.\n\n'
        "Describe what is visible in this frame. Focus on diagrams, equations, "
        "charts, code, flowcharts, or any text on screen. Be concise (2-3 sentences). "
        "If nothing meaningful is visible besides the professor, say 'No visual content detected'."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, img],
    )
    return response.text


def process_visual_cues(video_path, segments, output_dir, gemini_api_key):
    """pipeline: find cues, extract frames, use gemini to understand and return cue results"""

    cues = find_visual_cues(segments)

    if not cues:
        return {"cues": [], "total_cues": 0}

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    results = []

    for i, cue in enumerate(cues):
        timestamp = cue["start"]
        frame_filename = f"frame_{int(timestamp)}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)

        print(f"[visual] frame {i+1}/{len(cues)} at {timestamp:.1f}s — matched: \"{cue['phrase_matched']}\"")

        # extracting frame from video
        if not extract_frame(video_path, timestamp, frame_path):
            continue

        # sending to gemini for description
        try:
            description = describe_frame(frame_path, gemini_api_key, cue["text"])
        except Exception as e:
            print(f"[warn] gemini failed for frame at {timestamp}s: {e}")
            description = "Could not analyze this frame"

        # building relative path for serving via flask
        base_name = os.path.basename(output_dir)
        relative_path = f"{base_name}/frames/{frame_filename}"

        results.append({
            "timestamp": timestamp,
            "transcript_text": cue["text"],
            "phrase_matched": cue["phrase_matched"],
            "frame_url": f"/frames/{relative_path}",
            "description": description,
        })

    # saving to disk to avoid reprocess
    results_file = os.path.join(output_dir, "visual_cues.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"cues": results, "total_cues": len(results)}, f, indent=2, ensure_ascii=False)

    return {"cues": results, "total_cues": len(results)}
