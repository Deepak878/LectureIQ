import os
import json
from faster_whisper import WhisperModel


def load_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    """
    loading whisper base model locally
    """

    if device == "cuda":
        compute_type = "float16"

    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # print(f" model loaded succesfully")
    return whisper_model


def transcribe_audio(whisper_model, audio_filepath, language=None):
    # running transcription on audio file
    
    if not os.path.exists(audio_filepath):
        print(f" error : audio file isnot found: {audio_filepath}")
        return None

    print(f"[info] starting transcription of: {audio_filepath}")

    transcription_options = {
        "beam_size": 5,
        "vad_filter": True,         # for silence portions
        "vad_parameters": dict(
            min_silence_duration_ms=500,
        ),
    }

    if language:
        transcription_options["language"] = language

    segments_generator, detection_info = whisper_model.transcribe(
        audio_filepath,
        **transcription_options
    )

    # detecting the language  of  lecture 
    detected_lang = detection_info.language
    lang_confidence = detection_info.language_probability
    print(f"[info] detected language: {detected_lang} (confidence: {lang_confidence:.2f})")

    # collecting all segments into a list with timestamps
    transcript_segments = []
    full_transcript_text = ""

    for seg in segments_generator:
        segment_data = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        transcript_segments.append(segment_data)
        full_transcript_text += seg.text.strip() + " "

        timestamp_display = format_timestamp(seg.start)
        print(f"  [{timestamp_display}] {seg.text.strip()}")

    print(f"\n transcription complete, total segments: {len(transcript_segments)}")

    transcript_result = {
        "language": detected_lang,
        "language_confidence": round(lang_confidence, 2),
        "total_segments": len(transcript_segments),
        "segments": transcript_segments,
        "full_text": full_transcript_text.strip(),
    }

    return transcript_result


def format_timestamp(seconds):
    # converting seconds to standard HH:MM:SS format
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def save_transcript_to_json(transcript_data, output_filepath):
    # saving full transcript data as json 

    with open(output_filepath, "w", encoding="utf-8") as json_file:
        json.dump(transcript_data, json_file, indent=2, ensure_ascii=False)

    print(f" saved transcript json at : {output_filepath}")


def save_transcript_readable(transcript_data, output_filepath):
    # saving readable version of the transcript with timestamps 
    

    with open(output_filepath, "w", encoding="utf-8") as txt_file:
        txt_file.write("=" * 60 + "\n")
        txt_file.write("  Lecture Transcript : LectureIQ\n")
        txt_file.write("=" * 60 + "\n\n")
        txt_file.write(f"language: {transcript_data['language']}\n")
        txt_file.write(f"total segments: {transcript_data['total_segments']}\n")
        txt_file.write("-" * 60 + "\n\n")

        for seg in transcript_data["segments"]:
            start_stamp = format_timestamp(seg["start"])
            end_stamp = format_timestamp(seg["end"])
            txt_file.write(f"[{start_stamp} -> {end_stamp}]\n")
            txt_file.write(f"{seg['text']}\n\n")

        txt_file.write("-" * 60 + "\n")
        txt_file.write("\n--- FULL TEXT ---\n\n")
        txt_file.write(transcript_data["full_text"])
        txt_file.write("\n")

    print(f" saving readable transcript at : {output_filepath}")


if __name__ == "__main__":
    # testing
    import sys
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
        model = load_whisper_model("base")
        result = transcribe_audio(model, test_audio)
        if result:
            save_transcript_readable(result, "test_transcript.txt")
    else:
        print(" to use : python transcribe.py <audio_file_path>")
