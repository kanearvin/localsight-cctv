# LocalSight CCTV Scrubber

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
# install from https://ollama.com, then:
ollama pull qwen2.5-vl:7b
ollama serve            # listens on http://localhost:11434
```
Select **Offline Ollama** in the header. The header badge turns green when Ollama is reachable.

### Cloud BYOK — Gemini 2.5 Flash
Select **Cloud BYOK** and paste your own Gemini API key. The key lives only in this browser tab's
`sessionStorage` (cleared when the tab closes) and is never written to the backend's disk.
Only the compressed 1440×540 2×3 contact-sheet JPEGs are sent to Google — never the raw video.

## How it works
1. `ingest.py` — `ffprobe` the codec; stream-copy H.264 MP4s, transcode everything else
   (`.dav`, H.265, `.mkv`) to H.264 via `ffmpeg`. Sample at 1 FPS, blur, mask to the ROI polygon,
   and keep frames whose changed-pixel ratio ≥ 0.8%.
2. `events.py` — cluster motion frames with a 4 s gap tolerance and render a 2×3 timestamp-burned
   contact sheet per event.
3. `vlm_client.py` — send each sheet + your query to Ollama or Gemini concurrently; malformed
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
