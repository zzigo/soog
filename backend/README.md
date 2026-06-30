# SOOG Backend (Flask)

A Flask-based API that powers SOOG. It provides:

- `/api/generate`: Calls Ollama chat completion and optionally returns code or a rendered matplotlib plot.
- `/api/predict`: Runs a small multimodal model (DistilBERT + CLIP + attention head) on text + image.
- `/api/somap/graph`: Builds a knowledge graph from markdown notes in `backend/soopub` (folders, notes, tags, wikilinks).
- `/api/version`: Returns backend version from `version.txt`.
- `/api/health`: Lightweight health check with version and device info.
- `/api/ollama/verify`: Verifies Ollama connectivity and model availability.
- `/log`: HTML view of structured JSON logs.

## Requirements

- Python 3.10+
- macOS or Linux recommended
- Ollama installed where the backend runs
- Optional: NVIDIA GPU + CUDA for faster Torch inference
- Optional: Redis if you plan to use the cache utilities (not wired by default in `app.py` yet)

Python dependencies are listed in `requirements.txt` (includes torch, transformers, matplotlib, trimesh, etc.).

## Environment variables

Copy `.env.example` to `.env` and fill in values:

- `OLLAMA_BASE_URL`: Ollama endpoint (default `http://localhost:11434`).
- `OLLAMA_MODEL`: Model name (default `qwen2.5:7b-instruct`).
- `OLLAMA_API_KEY`: Optional; only needed if your Ollama endpoint is behind auth proxy.
- `PORT`: Port for Flask (default 10000).
- `HF_HOME`: Optional path for Hugging Face cache (defaults to `./.cache/huggingface`).

## Running locally (fish shell)

```fish
# 0) Start Ollama and pull model (one time pull)
ollama serve
ollama pull qwen2.5:7b-instruct

# 1) Create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate.fish

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set env (if you didn’t create .env yet)
set -gx OLLAMA_BASE_URL "http://localhost:11434"
set -gx OLLAMA_MODEL "qwen2.5:7b-instruct"
set -gx PORT 10000
# 4) Run the server
python app.py
```

### Ubuntu/VPN deployment (Ollama installed on same server)

Set `/opt/soog/backend/.env` on the VPN Ubuntu host:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
PORT=10000
```

Then ensure model/service are ready on Ubuntu:

```bash
ollama pull qwen2.5:7b-instruct
sudo systemctl restart soog-backend.service
```

### Or use the Makefile (works from any shell)

```fish
# install deps into venv/
make install

# quick import sanity check
make check

# run the server on default port 10000
make run

# override port
make run PORT=11000

# if your venv is corrupted, rebuild and reinstall
make reenv
make install
```

### One-shot dev script

```fish
chmod +x scripts/dev.sh
PORT=10000 bash scripts/dev.sh
```

The server listens on `0.0.0.0:$PORT` (default 10000). Health check:

- `GET http://localhost:10000/api/health`

## Notes

- `/api/generate` tries Ollama OpenAI-compatible endpoint (`/v1/chat/completions`) first, then native Ollama (`/api/chat`).
- Heavy models: The `/api/predict` endpoint loads DistilBERT and CLIP and an attention head; first call can be slow due to downloads. Models are cached under `./.cache/huggingface`.
- Checkpoints: If `modeltrainer/outputModel/multimodal_model_final.pth` exists, it will be loaded.
- Logs: Structured JSON logs are written to `backend/logs/app.json`. View logs at `/log`.
- CORS: Enabled for `/api/*` and `/log`.

## Troubleshooting

- Ollama not reachable:
  - Check `/api/health` and verify `ollama.base_url` and `ollama.model`.
  - Call `/api/ollama/verify` (or legacy alias `/api/deepseek/verify`) to test model listing.
  - On Ubuntu server, check `ollama serve` is running and listening on `127.0.0.1:11434`.
- Configured model missing:
  - Run `ollama pull qwen2.5:7b-instruct`.
- Torch/transformers install:
  - On macOS with ARM, some combos may be slow or need extra wheels; you can temporarily disable `/api/predict` if not needed.
- Port conflict:
  - If 5000 is in use, use `PORT=10000` (the default in `app.py`) and run again.

Security: If you use `OLLAMA_API_KEY` behind a proxy, keep it only in `backend/.env` and never expose it to frontend env files.

## Optional: Redis cache

`cache_*.py` implements a Redis-based cache manager and decorators. If you want to enable caching in routes, start a Redis server locally and wire the decorators in `app.py` accordingly. Redis defaults to `localhost:6379` DB 0/1 per `cache_config.py`.
