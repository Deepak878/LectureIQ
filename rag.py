import os
from groq import Groq
from dotenv import load_dotenv
from transcribe import format_timestamp

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def load_llm_client():
    """loading groq client using api key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(" error: GROQ_API_KEY not set in .env file")
    client = Groq(api_key=api_key)
    print(f"[info] groq client ready, model: {GROQ_MODEL}")
    return client


def build_context_from_segments(segments):
    """formats retrieved transcript segments into a readable context block for the llm"""
    context_parts = []
    for seg in segments:
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        context_parts.append(f"[{start_ts} -> {end_ts}]: {seg['text']}")
    return "\n".join(context_parts)


def build_system_prompt(metadata=None):
    """builds system prompt, injecting lecture metadata if provided"""

    base = (
        "You are a lecture assistant helping students understand academic content. "
        "You are given timestamped transcript segments from a specific lecture. "
    )

    # injecting course context if provided — helps ground the llm in the right domain
    if metadata:
        context_lines = []
        if metadata.get("course"):
            context_lines.append(f"Course: {metadata['course']}")
        if metadata.get("topic"):
            context_lines.append(f"Topic: {metadata['topic']}")
        if metadata.get("subtopic"):
            context_lines.append(f"Subtopic: {metadata['subtopic']}")

        if context_lines:
            base += (
                "This lecture is from:\n"
                + "\n".join(f"  - {line}" for line in context_lines)
                + "\n\n"
            )

    base += (
        "Rules:\n"
        "1. Answer primarily using the provided transcript segments.\n"
        "2. If the answer is clearly in the transcript, cite the timestamp.\n"
        "3. If the transcript only partially covers it, you may supplement with your own knowledge "
        "but explicitly say: 'Based on my knowledge (not in this transcript):'\n"
        "4. If the topic is completely absent from the transcript, say so clearly. Do not guess or fabricate.\n"
        "5. Keep answers concise and academic in tone."
    )

    return base


def ask_lecture(question, context_segments, llm_client, metadata=None):
    """main rag function: takes question + retrieved segments -> returns llm answer"""

    context_text = build_context_from_segments(context_segments)

    system_prompt = build_system_prompt(metadata)

    user_message = (
        f"Lecture transcript context:\n{context_text}\n\nQuestion: {question}"
    )

    print(f"[info] sending question to groq: '{question}'")

    response = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    print(f"[info] got answer from groq ({len(answer)} chars)")
    return answer