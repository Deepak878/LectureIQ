FROM python:3.12-slim

# ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# installing python depends first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copying app source
COPY . .

# creating runtime dirs
RUN mkdir -p uploads output qdrant_db

# huggingface spaces injects PORT=7860
ENV PORT=7860
EXPOSE 7860

# gunicorn:
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 app:app"]
