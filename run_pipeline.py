import os
import sys
import time
from extract_audio import pull_audio_from_video, check_if_audio_file, check_if_video_file
from transcribe import load_whisper_model, transcribe_audio, save_transcript_to_json, save_transcript_readable


WHISPER_MODEL_SIZE = "base"   
DEVICE = "cpu"                   
OUTPUT_FOLDER = "output"


def get_file_path_from_user():
    """asking for lecture file path"""

    print("\n" + "=" * 50)
    print("  LectureIQ")
    print("=" * 50)
    print()

    # checking if path was passed as command line arg
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"using file from argument: {file_path}")
    else:
        file_path = input("enter path to your lecture video or audio file: ").strip()

    # removing quotes if path contains
    file_path = file_path.strip('"').strip("'")

    # trying absolute path
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f" error: can't find file: {file_path}")
        return None

    return file_path


def setup_output_folder(source_filepath):
    """creating output folder based on the input file name"""

    base_name = os.path.splitext(os.path.basename(source_filepath))[0]
    output_path = os.path.join(OUTPUT_FOLDER, base_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f" created output directory: {output_path}")

    return output_path


def run_asr_pipeline():

    #  getting input file 
    source_file = get_file_path_from_user()
    if source_file is None:
        return

    # setting up where to save outputs
    output_dir = setup_output_folder(source_file)

    # looking if we need audio extraction 
    audio_file_to_transcribe = None

    if check_if_audio_file(source_file):
        # already an audio file, no extraction needed
        print(f" input is already an audio file")
        audio_file_to_transcribe = source_file

    elif check_if_video_file(source_file):
        # extract audio from video first
        print(f"input is a video file, extracting audio...")
        extracted_audio = pull_audio_from_video(source_file, output_dir)
        if extracted_audio is None:
            print(" error : audio extraction failed")
            return
        audio_file_to_transcribe = extracted_audio

    else:
        print(f" error:  unsupported file format: {os.path.splitext(source_file)[1]}")
        print("  Use supported video format: .mp4, .mkv, .avi, .mov, .webm, .flv, .wmv")
        print("  Use supported audio format : .mp3, .wav, .flac, .ogg, .m4a, .aac, .wma")
        return

    # loading model and running transcription 
    print()
    start_time = time.time()

    whisper_model = load_whisper_model(WHISPER_MODEL_SIZE, DEVICE)
    transcript_data = transcribe_audio(whisper_model, audio_file_to_transcribe)

    elapsed = time.time() - start_time

    if transcript_data is None:
        print(" error: transcription failed")
        return

    #  saving all outputs 
    base_name = os.path.splitext(os.path.basename(source_file))[0]

    # saving json version
    json_output = os.path.join(output_dir, f"{base_name}_transcript.json")
    save_transcript_to_json(transcript_data, json_output)

    # saving readable text version
    txt_output = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_transcript_readable(transcript_data, txt_output)

    print()
    print("=" * 50)
    print("  Pipeline complete")
    print("=" * 50)
    print(f"  source file  : {source_file}")
    print(f"  language      : {transcript_data['language']}")
    print(f"  segments      : {transcript_data['total_segments']}")
    print(f"  time taken    : {elapsed:.1f} seconds")
    print(f"  json output   : {json_output}")
    print(f"  text output   : {txt_output}")
    print("=" * 50)


if __name__ == "__main__":
    run_asr_pipeline()
