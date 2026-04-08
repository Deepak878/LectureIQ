import os
import subprocess
import sys


def pull_audio_from_video(video_filepath, output_dir=None):
    """
    taking a video file and extracting  audio track as .wav
    returning path to audio file
    """

    if not os.path.exists(video_filepath):
        print(f"[error] file is not found: {video_filepath}")
        return None

    # getting the filename 
    base_name = os.path.splitext(os.path.basename(video_filepath))[0]

    if output_dir is None:
        output_dir = os.path.dirname(video_filepath)

    # saving as wav formats
    audio_output_path = os.path.join(output_dir, f"{base_name}_audio.wav")

    # checking if we already extracted before and skip
    if os.path.exists(audio_output_path):
        print(f"audio already extracted in path as : {audio_output_path}")
        
        return audio_output_path

    print(f" extracting audio from: {video_filepath}")
    print(f" saving to: {audio_output_path}")

    # using ffmpeg to extract audio
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_filepath,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_output_path
    ]

    try:
        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f" error : failed :\n{result.stderr}")
            return None

        print(f" audio extracted succesfully")
        return audio_output_path

    except FileNotFoundError:
        print("ffmpeg isnot installed")
        
        return None


def check_if_audio_file(filepath):
    """checking if given file is already an audio file so we dont need to extract"""
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]
    file_ext = os.path.splitext(filepath)[1].lower()
    return file_ext in audio_extensions


def check_if_video_file(filepath):
    """checking  if file is a standrd video format"""
    video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"]
    file_ext = os.path.splitext(filepath)[1].lower()
    return file_ext in video_extensions


if __name__ == "__main__":
    # quick test run
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        pull_audio_from_video(test_path)
    else:
        print("i/p comand: python extract_audio.py <video_file_path>")
