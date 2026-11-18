# CLAUDE.md — LocalSight CCTV Scrubber

## Project Context & Mission
You are building **LocalSight**, a privacy-first, local-first web application that ingests exported residential CCTV files (`.mp4`, `.mkv`, `.dav`), masks out background street/tree noise via user-drawn Region-of-Interest (ROI) polygons, clusters active motion frames into $2 \times 3$ contact sheets locally, and searches hours of footage in seconds using natural-language Vision-Language Model (VLM) queries.

* **Source of Truth:** Always consult `SPEC.md` for exact file paths, SQLite schemas, REST API contracts, and code implementations, and `PRD.md` for product guardrails and acceptance criteria.

---

## Core Architectural Guardrails (Non-Negotiable)
1. **Local-First Privacy:** Raw CCTV video files and SQLite indexes MUST stay strictly on `localhost` (`./workspace/`). Never upload raw video files to external servers. In BYOK mode, only compressed $2 \times 3$ JPEG contact sheets (`1440x540`) are transmitted to the Gemini API.
2. **Two-Stage Funnel:** Never send raw 30 FPS frames directly to a VLM. Always run local OpenCV frame-differencing at `1.0 FPS` with optional `roi_polygon` masking (`ingest.py`) -> temporal event clustering (`events.py`) -> $2 \times 3$ timestamp-burned contact sheets before calling Ollama or Gemini (`vlm_client.py`).
3. **Zero Config Complexity:** Do not introduce Docker, Redis, Celery, or YAML configuration files. Keep persistence in local SQLite (`./workspace/localsight.db`) and background processing in `asyncio` thread pools (`loop.run_in_executor`).
4. **Universal DVR Playback:** Always remux proprietary `.dav` or H.265 streams into browser-playable H.264 `.mp4` via `ffmpeg` (`ensure_browser_mp4`) so HTML5 `<video>` scrubbing works reliably in all browsers.

---

## Repository Map
```text
localsight-cctv/
├── CLAUDE.md                # This instruction file
├── PRD.md                   # Product requirements & metrics
├── SPEC.md                  # Technical specification & starter code
├── backend/
│   ├── requirements.txt
│   ├── main.py              # FastAPI server (/api/health, /api/index, /api/search, /api/video/{id})
│   ├── db.py                # SQLite schema & CRUD helpers
│   ├── ingest.py            # FFmpeg remuxer & OpenCV ROI motion extraction
│   ├── events.py            # Motion clustering & 2x3 timestamp grid generator
│   ├── vlm_client.py        # Async structured JSON adapters (Ollama & Gemini Flash)
│   └── tests/
│       └── test_pipeline.py # Pytest suite with synthetic OpenCV video fixtures
├── frontend/
│   └── src/
│       ├── App.tsx          # Main UI (video player, SVG ROI polygon overlay, search timeline)
│       ├── types.ts         # TypeScript interfaces matching FastAPI responses
│       └── api.ts           # Typed API client
└── workspace/               # Gitignored local storage for remuxed MP4s, grids & SQLite DB
```

---

## Commands & Workflow

### Backend Setup & Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Run Backend Tests (Required Before Marking Tasks Done)
```bash
cd backend
pytest -v
```

### Frontend Setup, Dev & Build Check
```bash
cd frontend
npm install
npm run dev        # Runs on http://localhost:5173
npm run build      # Must exit with 0 TypeScript/Vite errors
```

---

## Coding Standards & Conventions
* **Python (`backend/`):**
  * Use Python 3.11+ type hints (`list[str]`, `str | None`) and Pydantic v2 models for all API payloads and VLM structured outputs.
  * Keep CPU-bound OpenCV/FFmpeg work off the FastAPI async event loop using `loop.run_in_executor`.
  * Wrap individual VLM contact-sheet calls in `try/except` inside `asyncio.gather` so a single malformed JSON response or timeout never aborts the entire video search.
  * Parameterize all SQLite queries in `db.py` and ensure foreign keys (`PRAGMA foreign_keys = ON`) are enabled.
* **TypeScript / React (`frontend/`):**
  * Store user-entered Gemini BYOK API keys in `sessionStorage` only—never hardcode keys or persist them to backend disk.
  * Normalize SVG ROI polygon coordinates to a `0–1000` coordinate space in the UI and convert to native video pixel dimensions on the backend.

---

## Step-by-Step Build Execution Plan
When asked to scaffold or implement the project, execute in this exact order:
1. **Scaffold Backend:** Create `backend/requirements.txt`, `db.py`, `ingest.py`, `events.py`, `vlm_client.py`, and `main.py` per `SPEC.md`. Ensure `./workspace` is added to `.gitignore`.
2. **Verify with Pytest:** Write and run `backend/tests/test_pipeline.py` testing:
   * Synthetic video motion detection with vs. without an ROI polygon mask.
   * $2 \times 3$ contact sheet JPEG generation (`1440x540`).
   * FastAPI endpoints (`/api/health`, `/api/index`, `/api/search`) with mocked VLM responses.
3. **Scaffold Frontend:** Initialize Vite + React + TypeScript + Tailwind in `frontend/`, implement `types.ts`, `api.ts`, and `App.tsx`, and verify `npm run build` passes cleanly.
4. **Document:** Create a concise `README.md` with quick-start instructions for both Offline Ollama (`ollama pull qwen2.5-vl:7b`) and Cloud BYOK modes.