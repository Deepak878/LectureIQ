"""
Flashcard and quiz generation from lecture transcripts.
Uses the LLM abstraction layer (llm_client.py) so it works with any provider.
"""

import json
from transcribe import format_timestamp


def build_transcript_context(segments, max_segments=50):
    """take transcript segments and build a readable text block for the LLM"""
    # use up to max_segments to stay within token limits
    selected = segments[:max_segments]
    lines = []
    for seg in selected:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


def generate_flashcards(llm, segments, count=8):
    """generate flashcards from transcript segments using the LLM"""

    context = build_transcript_context(segments)

    system_prompt = (
        "You are an academic study assistant. Generate flashcards from the lecture transcript provided. "
        "Each flashcard should have a clear question on the front and a concise answer on the back. "
        "Focus on key concepts, definitions, and important facts from the lecture. "
        "Return ONLY valid JSON array, no other text."
    )

    user_message = f"""Lecture transcript:
{context}

Generate exactly {count} flashcards from this lecture content.
Return as a JSON array with this exact format:
[
  {{"front": "What is ...?", "back": "It is ...", "timestamp": "00:01:30"}},
  ...
]
The timestamp should reference where in the lecture this concept was discussed.
Return ONLY the JSON array, nothing else."""

    raw = llm.generate(system_prompt, user_message, temperature=0.3, max_tokens=2048)

    # parse JSON from response (handle markdown code blocks)
    return _parse_json_response(raw)


def generate_quiz(llm, segments, count=5):
    """generate multiple choice quiz from transcript segments using the LLM"""

    context = build_transcript_context(segments)

    system_prompt = (
        "You are an academic quiz generator. Create multiple choice questions based on the lecture transcript provided. "
        "Each question should test understanding of key concepts from the lecture. "
        "Make the wrong options plausible but clearly incorrect. "
        "Return ONLY valid JSON array, no other text."
    )

    user_message = f"""Lecture transcript:
{context}

Generate exactly {count} multiple choice questions from this lecture.
Return as a JSON array with this exact format:
[
  {{
    "question": "What is ...?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct": 0,
    "explanation": "Brief explanation of why this is correct",
    "timestamp": "00:05:12"
  }},
  ...
]
"correct" is the zero-based index of the correct option (0=A, 1=B, 2=C, 3=D).
The timestamp should reference where in the lecture this was discussed.
Return ONLY the JSON array, nothing else."""

    raw = llm.generate(system_prompt, user_message, temperature=0.4, max_tokens=3000)

    return _parse_json_response(raw)


def _parse_json_response(raw_text):
    """extract JSON array from LLM response, handling markdown code blocks"""
    text = raw_text.strip()

    # strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # find the JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        print(f"[warn] could not find JSON array in LLM response: {text[:200]}")
        return []

    json_str = text[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[warn] JSON parse error: {e}")
        print(f"[warn] raw: {json_str[:300]}")
        return []
