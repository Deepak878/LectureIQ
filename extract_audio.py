import os
import subprocess
import sys
import json
from pathlib import Path

def pull_audio_from_video(video_filepath, output_dir=None):
    """
    taking a video file and extracting audio track as .wav
    returning path to audio file
    """
    video_path = Path(video_filepath)
    
    if not video_path.exists():
        print(f"[error] file is not found: {video_filepath}")
        return None, "file not found"

    # getting the filename using Path for cleaner logic
    base_name = video_path.stem
    output_path = Path(output_dir) if output_dir else video_path.parent
    audio_output_path = output_path / f"{base_name}_audio.wav"

    # checking if we already extracted before and skip
    if audio_output_path.exists():
        print(f"audio already extracted in path as : {audio_output_path}")
        return str(audio_output_path), None

    print(f" extracting audio from: {video_filepath}")
    print(f" saving to: {audio_output_path}")

    # check if the video actually has an audio stream
    audio_streams = _probe_audio_streams(video_filepath)
    if audio_streams is None:
        return None, "ffprobe not found. please install FFmpeg (includes ffprobe) and add it to PATH."
    if not audio_streams:
        return None, "no audio stream found in this file. try a file that includes audio or re-encode it with audio."

    # using ffmpeg to extract audio
    ffmpeg_command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(audio_output_path)
    ]

    try:
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f" error : failed :\n{result.stderr}")
            return None, _clean_ffmpeg_error(result.stderr)

        print(f" audio extracted succesfully")
        return str(audio_output_path), None

    except FileNotFoundError:
        print("ffmpeg isnot installed")
        return None, "ffmpeg not found. please install it and add it to PATH."


def _probe_audio_streams(video_filepath):
    """return list of audio stream indexes or None if ffprobe missing"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        str(video_filepath),
    ]
    try:
        # Optimization: Simplified return logic
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return [s.get("index") for s in data.get("streams", [])]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _clean_ffmpeg_error(stderr_text):
    """best-effort cleanup of noisy ffmpeg stderr for user display"""
    if not stderr_text:
        return "unknown ffmpeg error"
    # Optimization: Generator expression for memory efficiency on large logs
    lines = [l.strip() for l in stderr_text.splitlines() if l.strip()]
    return " | ".join(lines[-6:])


def check_if_audio_file(filepath):
    """checking if given file is already an audio file so we dont need to extract"""
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    return Path(filepath).suffix.lower() in audio_extensions


def check_if_video_file(filepath):
    """checking if file is a standrd video format"""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    return Path(filepath).suffix.lower() in video_extensions

def get_audio_duration(filepath):
    """
    Returns the duration of the audio/video file in seconds.
    Useful for LectureIQ to estimate transcription time or segmenting audio.
    """
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except (FileNotFoundError, ValueError):
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pull_audio_from_video(sys.argv[1])
    else:
        print("i/p comand: python extract_audio.py <video_file_path>")