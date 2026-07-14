#!/usr/bin/env sh
# Wait for the Ollama service, pull the models the lite experience needs, then
# start the API. Makes `docker compose up` a true one-command start.
set -e

OLLAMA="${OLLAMA_BASE_URL:-http://ollama:11434}"
CHAT_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

echo "companion: waiting for Ollama at $OLLAMA ..."
until curl -sf "$OLLAMA/api/tags" >/dev/null 2>&1; do
  sleep 2
done
echo "companion: Ollama is up."

pull() {
  echo "companion: ensuring model '$1' (first run downloads it) ..."
  curl -sf "$OLLAMA/api/pull" -d "{\"name\":\"$1\"}" >/dev/null || {
    echo "companion: WARNING could not pull '$1', chat/memory may be limited."
  }
}
pull "$CHAT_MODEL"
pull "$EMBED_MODEL"

echo "companion: starting API on :8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
