
# LectureIQ

LectureIQ turns lecture recordings into searchable, timestamped transcripts so we don't have to scrub through hours of video just to find one thing the professor said.

Here you can just upload a lecture video or audio file and it gives you a clean transcript with timestamps. You can search by meaning (not just keywords), ask questions about the lecture, and generate flashcards or quizzes from the content.

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

You need a `GROQ_API_KEY` in a `.env` file for the Q&A and study tools to work. It is free at:  console.groq.com.

## What it uses

- faster-whisper for transcription (runs locally, no API needed)
- sentence-transformers for semantic search
- Qdrant as the vector database (local, no server)
- Groq API with LLaMA 3.3 70B for answering questions and generating study material
- FFmpeg for pulling audio from video files
- Flask for the web interface
