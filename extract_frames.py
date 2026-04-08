"""
extract_frames.py - Diagram/Visual Content Detection & Frame Extraction

HOW IT WORKS:
=============
1. KEYWORD DETECTION: Scans transcript segments for phrases that indicate
   visual content is being shown (e.g., "as you can see", "this diagram",
   "look at this", "on the screen", etc.)

2. DEDUPLICATION: Groups nearby detections within 10 seconds of each other
   to avoid extracting multiple frames of the same visual.

3. FRAME EXTRACTION: Uses ffmpeg to extract a single JPEG frame from the
   video at the midpoint of each detected segment. This captures the moment
   the instructor is most likely showing/discussing the visual.

4. OUTPUT: Frames are saved to output/<video_name>/diagrams/ folder as
   high-quality JPEG images named diagram_1_30s.jpg, diagram_2_95s.jpg, etc.
"""

import os
import subprocess


# Phrases that indicate visual content is being displayed/shown
# These are intentionally focused on "showing" language rather than
# just topic mentions to reduce false positives
DIAGRAM_KEYWORDS = [
    # direct references to visuals
    "diagram", "figure", "chart", "flowchart", "illustration",
    "screenshot", "this slide", "this image", "this picture",
    # pointing/showing language
    "look at this", "look at that", "as you can see",
    "shown here", "shown on", "this shows", "let me show",
    "let me draw", "take a look", "here we have", "as shown",
    # screen/board references
    "on the screen", "on screen", "on the board", "whiteboard",
    # code/example visibility
    "code here", "code on screen", "example here",
    "you can see here", "right here", "over here",
]


def detect_diagram_segments(transcript_segments):
    """
    Scans transcript segments for mentions of visual/diagram content.

    HOW: Iterates through each segment's text, checking for any of the
    DIAGRAM_KEYWORDS. Only one match per segment (first match wins).
    Then merges detections within 10 seconds of each other to avoid
    duplicate frames of the same visual.

    Args:
        transcript_segments: list of dicts with keys: start, end, text

    Returns:
        list of merged detections, each with: start, end, text, keyword_matched
    """
    raw_detections = []

    for seg in transcript_segments:
        text_lower = seg["text"].lower()
        for keyword in DIAGRAM_KEYWORDS:
            if keyword in text_lower:
                raw_detections.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "keyword_matched": keyword
                })
                break  # one keyword match per segment is enough

    if not raw_detections:
        print("[info] no diagram/visual references detected in transcript")
        return []

    # deduplicate: merge detections within 10 seconds of each other
    # this prevents extracting 5 frames from the same diagram discussion
    merged = [raw_detections[0].copy()]
    for det in raw_detections[1:]:
        if det["start"] - merged[-1]["end"] < 10:
            # extend the previous detection window
            merged[-1]["end"] = det["end"]
            merged[-1]["text"] += " ... " + det["text"]
        else:
            merged.append(det.copy())

    print(f"[info] detected {len(merged)} visual content references "
          f"(from {len(raw_detections)} raw keyword matches)")
    return merged


def extract_frame_at_timestamp(video_path, timestamp_sec, output_path):
    """
    Uses ffmpeg to grab a single frame from the video.

    HOW: Runs ffmpeg with -ss (seek) to jump to the exact timestamp,
    then -vframes 1 to grab just one frame, saved as JPEG with
    quality level 2 (high quality). The -y flag overwrites existing files.

    Args:
        video_path: path to source video file
        timestamp_sec: time in seconds to extract frame at
        output_path: where to save the JPEG frame

    Returns:
        output_path if successful, None if failed
    """
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp_sec),   # seek to timestamp
        "-i", video_path,            # input video
        "-vframes", "1",             # grab exactly 1 frame
        "-q:v", "2",                 # jpeg quality (2 = high quality)
        "-y",                        # overwrite if exists
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )

        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"[info] extracted frame at {timestamp_sec:.1f}s -> {output_path}")
                return output_path

        print(f"[warn] frame extraction failed at {timestamp_sec:.1f}s")
        return None

    except subprocess.TimeoutExpired:
        print(f"[error] ffmpeg timed out extracting frame at {timestamp_sec:.1f}s")
        return None
    except FileNotFoundError:
        print("[error] ffmpeg not found - make sure ffmpeg is installed")
        return None
    except Exception as e:
        print(f"[error] frame extraction error: {e}")
        return None


def extract_all_diagram_frames(video_path, diagram_segments, output_dir):
    """
    Extracts video frames for all detected diagram/visual segments.

    HOW:
    1. Creates a 'diagrams' subfolder inside the output directory
    2. For each detected segment, calculates the midpoint timestamp
       (middle of start-end range, where the visual is most likely shown)
    3. Calls ffmpeg to extract a single frame at that timestamp
    4. Returns a list of successfully extracted frame info dicts

    Args:
        video_path: path to the source video file
        diagram_segments: list from detect_diagram_segments()
        output_dir: base output directory (e.g., output/video_name/)

    Returns:
        list of dicts with: index, frame_path, frame_filename, timestamp,
        start, end, transcript_text, keyword_matched, cloudinary_url, analysis
    """
    diagrams_dir = os.path.join(output_dir, "diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)

    extracted_frames = []

    for i, seg in enumerate(diagram_segments):
        # extract frame at the midpoint of the segment
        mid_timestamp = (seg["start"] + seg["end"]) / 2
        frame_filename = f"diagram_{i + 1}_{int(mid_timestamp)}s.jpg"
        frame_path = os.path.join(diagrams_dir, frame_filename)

        result = extract_frame_at_timestamp(video_path, mid_timestamp, frame_path)

        if result:
            extracted_frames.append({
                "index": i + 1,
                "frame_path": result,
                "frame_filename": frame_filename,
                "timestamp": round(mid_timestamp, 2),
                "start": seg["start"],
                "end": seg["end"],
                "transcript_text": seg["text"],
                "keyword_matched": seg["keyword_matched"],
                "cloudinary_url": None,   # filled by image_analysis.py
                "analysis": None          # filled on-demand via /analyze-diagram
            })

    print(f"[info] extracted {len(extracted_frames)}/{len(diagram_segments)} diagram frames")
    print(f"[info] frames saved to: {diagrams_dir}")
    return extracted_frames


if __name__ == "__main__":
    # quick test: python extract_frames.py <video_path> <timestamp>
    import sys
    if len(sys.argv) >= 3:
        video = sys.argv[1]
        ts = float(sys.argv[2])
        out = f"test_frame_{int(ts)}s.jpg"
        extract_frame_at_timestamp(video, ts, out)
    else:
        print("usage: python extract_frames.py <video_path> <timestamp_seconds>")
