# syntax=docker/dockerfile:1

# ---- Stage 1: build the React frontend into frontend/dist ------------------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # runs prebuild (VAD assets) + vite build -> dist/

# ---- Stage 2: Python runtime (LITE: chat + memory, no GPU/torch) -----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMPANION_DB_PATH=/data/companion.db \
    CHROMA_PATH=/data/chroma_db
WORKDIR /app

# curl is used by the entrypoint to wait for Ollama and pull models.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Core deps only — the lazy-imported voice/ML libs are intentionally absent,
# so /stt, /tts and /ws/call report "voice unavailable" and everything else
# (streaming chat, memory, personas, proactive, journal) runs.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# App code + assets. Runtime data (companion.db, chroma_db) lives on /data.
COPY app/ ./app/
COPY config.yaml ./
COPY personas/ ./personas/
COPY media/ ./media/
COPY --from=frontend /fe/dist ./frontend/dist
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /data

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
