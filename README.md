# LocalSight CCTV Scrubber

[![CI](https://github.com/kanearvin/localsight-cctv/actions/workflows/ci.yml/badge.svg)](https://github.com/kanearvin/localsight-cctv/actions/workflows/ci.yml)

Privacy-first, local-first search for exported home CCTV footage (`.mp4`, `.mkv`, `.dav`).
LocalSight remuxes the file to browser-playable H.264, drops static frames with OpenCV motion
detection (optionally masked to a zone you draw), groups motion into events, and lets you ask
natural-language questions ("delivery parcel", "person near car") via a local or BYOK vision model.

Raw video and the SQLite index never leave `./workspace/` on your machine.

## Prerequisites
- **FFmpeg** (`ffmpeg` + `ffprobe` on PATH) — macOS: `brew install ffmpeg`
- **Python 3.11+**
- **Node 20+**

## Quick start

```bash
# 1. Backend  (http://127.0.0.1:8000)
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 2. Frontend (http://localhost:5173) — in a second terminal
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, load a clip, optionally click **Draw Detection Zone** → place ≥ 3 points →
**Re-scan with zone**, then type a query and press **Find**. Click any event card (or a timeline
marker) to jump the player to that moment.

## Choosing an AI mode

### Offline — Ollama (100% local, default)
```bash
brew install ollama          # or the desktop app from https://ollama.com
ollama serve                 # keep running; listens on http://localhost:11434
ollama pull qwen2.5vl:3b     # in another terminal, ~3 GB one-time download (default)
# Optional, needs ≥16 GB RAM to run at usable speed: ollama pull qwen2.5vl:7b
```
Select **Offline Ollama** in the header. The header badge turns green when Ollama is reachable.
If Ollama isn't running or the model isn't pulled, **Find** shows an error explaining what to run.

Pick the model from the dropdown next to the provider (it lists everything you've pulled). Local
inference is sequential and each contact sheet may take a while on CPU; the per-sheet timeout is
300 s and can be changed with `LOCALSIGHT_OLLAMA_TIMEOUT=<seconds>`.

### Cloud BYOK — Gemini
Select **Cloud BYOK**, choose a model — **Gemini 3.5 Flash-Lite** (default, fastest/cheapest) or
**Gemini 4 Hydrogen** (more capable) — and paste your own Gemini API key. (`gemini-2.5-flash` is
deprecated by Google and returns 404 for new keys.)

### Cloud BYOK — OpenRouter (free models)
1. Create a free key at https://openrouter.ai/keys.
2. Select **Cloud BYOK · OpenRouter**. The model dropdown is loaded live from OpenRouter's public
   model list and shows only **free (`:free`) models that accept images** (default
   `google/gemma-4-26b-a4b-it:free`); models without JSON mode are labelled.
3. Free models are rate-limited (requests per minute and per day), so LocalSight sends 2 sheets at a
   time and automatically retries HTTP 429 responses. Long videos may hit the daily cap.

### Cloud BYOK — Cloudflare Workers AI
1. In the Cloudflare dashboard, copy your **account ID** (32 hex characters) and create an API token
   with the **Workers AI** permission at https://dash.cloudflare.com/profile/api-tokens.
2. Select **Cloud BYOK · Cloudflare Workers AI**, choose a vision model (default
   `@cf/meta/llama-4-scout-17b-16e-instruct`; also Mistral Small 3.1 and Gemma 3 12B), and enter the
   account ID and token.
3. The free tier includes 10,000 Neurons/day; LocalSight uses Cloudflare's OpenAI-compatible
   `/ai/v1/chat/completions` endpoint.

For every cloud provider, keys/tokens/account IDs are kept only in the browser tab's
`sessionStorage` and sent per request; the backend never writes them to disk, and only the
1440×540 contact-sheet JPEGs are transmitted. The key lives only in this browser tab's
`sessionStorage` (cleared when the tab closes) and is never written to the backend's disk.
Only the compressed 1440×540 2×3 contact-sheet JPEGs are sent to Google — never the raw video.

## How it works
1. `ingest.py` — `ffprobe` the codec; stream-copy H.264 MP4s, transcode everything else
   (`.dav`, H.265, `.mkv`) to H.264 via `ffmpeg`. Sample at 1 FPS, blur, mask to the ROI polygon,
   and keep frames whose changed-pixel ratio ≥ 0.8%.
2. `events.py` — cluster motion frames with a 4 s gap tolerance and render a 2×3 timestamp-burned
   contact sheet per event.
3. `vlm_client.py` — send each sheet + your query to Ollama (one at a time) or Gemini (concurrently); malformed
   replies or timeouts mark only that event as failed.
4. `db.py` — videos, ROI and events persist in `workspace/localsight.db`, so indexes survive restarts.

### API
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | `{status, ffmpeg, ollama_reachable}` |
| POST | `/api/index` | multipart `file` (+ optional `roi_json`, 0–1000 coords) → events |
| POST | `/api/videos/{id}/reindex` | `{roi: [{x,y}…] \| null}` → re-scan stored video without re-uploading |
| GET | `/api/videos/{id}` | stored video + events (used to restore the UI after a restart) |
| POST | `/api/search` | `{video_id, query, provider, api_key?, ollama_model?}` → ranked results |
| GET | `/api/video/{id}` | playable MP4 with HTTP Range support |

## Tests
```bash
cd backend && source .venv/bin/activate && pytest -v   # synthetic-video pipeline + API tests
cd frontend && npm run build                            # must exit 0
```
Set `LOCALSIGHT_WORKSPACE=/some/dir` to store data somewhere other than `./workspace`.

## CI/CD
- **CI** (`.github/workflows/ci.yml`) runs on every push to `main` and every PR: backend `pytest`
  on Python 3.11 and 3.14 with FFmpeg installed (FFmpeg tests are required, never skipped), and
  frontend `npm ci && npm run build` on Node 22.
- **Release** (`.github/workflows/release.yml`): pushing a `vX.Y.Z` tag re-runs CI, then publishes a
  GitHub Release with a self-contained bundle (backend + prebuilt frontend) and SHA-256 checksums.
  Tags with a hyphen (e.g. `v1.0.0-rc.1`) are marked as pre-releases. There is no cloud deploy —
  LocalSight runs on the user's machine.
  ```bash
  git tag v1.0.0 && git push origin v1.0.0
  ```
- **Dependabot** opens weekly update PRs for pip, npm and GitHub Actions.

### Running a release bundle (single command)
When `frontend/dist` exists, the backend serves the UI itself, so no Node is needed:
```bash
tar -xzf localsight-cctv-v1.0.0.tar.gz && cd localsight-cctv-v1.0.0/backend
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000   # open http://127.0.0.1:8000
```
