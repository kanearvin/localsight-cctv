# SPEC.md — LocalSight CCTV Scrubber (MVP v1.0)

## 1. Project Overview & Architecture
Build **LocalSight**, a local-first web application that allows homeowners and small residential landlords to ingest exported CCTV video files (`.mp4`, `.mkv`, `.dav`), mask out background noise using a Region-of-Interest (ROI) polygon, extract active motion events locally, and search hours of footage in seconds using natural-language Vision-Language Model (VLM) queries.

### End-to-End Local-First Pipeline
```text
[Raw CCTV File: .mp4 / .dav / .mkv]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Local Ingestion & Motion/ROI Filtering (CPU)    │
│ • FFmpeg remuxes .dav/.h265 -> browser-playable .mp4     │
│ • Samples at 1 FPS + applies user-drawn ROI polygon mask │
│ • OpenCV frame differencing drops ~90–95% static frames  │
└──────┬───────────────────────────────────────────────────┘
       │  (Outputs: ~200–500 active motion keyframes + timestamps)
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Local Visual Indexing & Event Grouping          │
│ • Groups consecutive motion frames into "Event Clips"    │
│ • Extracts up to 6 representative keyframes per event    │
└──────┬───────────────────────────────────────────────────┘
       │  (Outputs: ~30–80 distinct Event Storyboards)
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: Semantic VLM Reasoning (Hybrid: Local or BYOK)  │
│ • Stitches 4–6 keyframes per event into a 2x3 image grid │
│ • Sends grid + user prompt to VLM (Qwen2.5-VL / Gemini)  │
│ • Returns JSON: { timestamp, match_score, description }  │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
[Interactive Web UI: Timeline Scrubber + Timestamp Jump Cards]
```

### Latency & Cost Benchmark (6-Hour 1080p Home Video)
| Pipeline Stage | Processing Time (Standard Laptop) | Compute / API Cost |
| :--- | :--- | :--- |
| **1. FFmpeg + OpenCV 1 FPS Motion Scan** | ~18–25 seconds | $0.00 (Local CPU) |
| **2. Event Clustering & Contact-Sheet Generation** | ~2 seconds | $0.00 (Local CPU) |
| **3. VLM Analysis (45 Event Grids via Gemini Flash)** | ~4–6 seconds (async parallel) | ~$0.01 – $0.02 total |
| **3 (Alt). Local Ollama (`qwen2.5-vl:7b` on M-series Mac)** | ~35–50 seconds | $0.00 (100% Offline) |

---

## 2. Tech Stack & Directory Structure
* **Backend:** Python 3.11+, FastAPI, Uvicorn, OpenCV (`opencv-python-headless`), NumPy, Pillow, Pydantic v2, HTTPX, SQLite (`sqlite3`).
* **System Dependency:** `ffmpeg` and `ffprobe` installed on host PATH.
* **Frontend:** React 18+, TypeScript, Vite, Tailwind CSS, `lucide-react`.
* **Local AI Runtime:** Ollama running at `http://localhost:11434` with `qwen2.5-vl:7b` (optional offline mode).

```text
localsight-cctv/
├── PRD.md
├── SPEC.md
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── main.py              # FastAPI app, CORS, REST endpoints, background thread pool
│   ├── db.py                # SQLite persistence for indexed videos & event metadata
│   ├── ingest.py            # FFmpeg .dav/.h265 remuxer & OpenCV ROI motion filtering
│   ├── events.py            # Temporal motion clustering & 2x3 timestamp grid builder
│   ├── vlm_client.py        # Async structured-output adapters for Ollama & Gemini Flash
│   └── tests/
│       └── test_pipeline.py # Pytest suite with synthetic video fixtures
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx          # Main workspace (upload, player, ROI canvas, search, timeline)
│       ├── types.ts         # Shared TypeScript interfaces matching API schemas
│       └── api.ts           # Typed fetch client for /api/index, /api/search, /api/video
└── workspace/               # Runtime directory (gitignored) for remuxed MP4s & event grids
```

---

## 3. Backend Specifications (`backend/`)

### 3.1 `db.py` — SQLite Persistence
Persist video indexing metadata in `./workspace/localsight.db` so indexed videos survive server restarts.
* **Table `videos`:**
  * `video_id TEXT PRIMARY KEY`
  * `original_filename TEXT NOT NULL`
  * `playable_mp4_path TEXT NOT NULL`
  * `duration_sec REAL`
  * `roi_json TEXT`
  * `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
* **Table `events`:**
  * `id INTEGER PRIMARY KEY AUTOINCREMENT`
  * `video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE`
  * `event_index INTEGER NOT NULL`
  * `start_sec REAL NOT NULL`
  * `end_sec REAL NOT NULL`
  * `start_str TEXT NOT NULL`
  * `end_str TEXT NOT NULL`
  * `frame_count INTEGER NOT NULL`
  * `contact_sheet_path TEXT NOT NULL`

### 3.2 `ingest.py` — Video Remuxing & Motion Extraction
* `format_timestamp(seconds: float) -> str`: Formats seconds into `HH:MM:SS`.
* `ensure_browser_mp4(input_path: str, output_dir: str) -> str`:
  * Probes codec via `ffprobe`. If already H.264 `.mp4`, stream-copies (`-c:v copy -an`) for instant availability; otherwise transcodes `.dav`/`.h265`/`.mkv` using `ffmpeg -y -i <in> -c:v libx264 -preset ultrafast -crf 26 -an <out>`.
* `extract_motion_frames(...) -> list[MotionFrame]`:
  * Samples frames at `sample_fps=1.0`, converts to grayscale, applies `cv2.GaussianBlur(gray, (21, 21), 0)`.
  * If `roi_polygon` ($\ge 3$ points) is provided, builds binary mask via `cv2.fillPoly` and applies `cv2.bitwise_and` so background motion outside the zone is ignored.
  * Computes `cv2.absdiff` against the previous sampled frame; retains frames where `changed_pixels / active_area >= min_changed_ratio` (`0.008`).

### 3.3 `events.py` — Event Clustering & Contact-Sheet Builder
* `cluster_into_events(motion_frames: list[MotionFrame], gap_tolerance_sec: float = 4.0) -> list[list[MotionFrame]]`:
  * Groups sequential `MotionFrame` items separated by $\le 4.0\text{s}$ into discrete event clusters.
* `build_contact_sheets(clusters: list[list[MotionFrame]], output_dir: str, tile_size: tuple[int, int] = (480, 270)) -> list[VideoEvent]`:
  * Samples up to 6 evenly spaced frames per cluster (`np.linspace`).
  * Resizes each tile to `480x270`, burns a high-contrast `HH:MM:SS` badge onto the top-left corner, pads missing slots with black frames, and writes a `1440x540` $2 \times 3$ JPEG grid (`quality=85`).

### 3.4 `vlm_client.py` — Structured VLM Adapters
* **Pydantic Output Schema (`EventSearchResult`):**
  ```python
  class EventSearchResult(BaseModel):
      matched: bool
      confidence: float          # 0.0 to 1.0
      best_timestamp: str        # HH:MM:SS burned into matching tile
      summary: str               # 1-sentence objective summary
      tags: list[str]            # 2-4 lowercase tags (e.g., ["person", "parcel"])
  ```
* Supports `analyze_contact_sheet_ollama` (`POST http://localhost:11434/api/chat`, `format="json"`) and `analyze_contact_sheet_gemini` (`gemini-2.5-flash`, `responseMimeType="application/json"`).
* Catches malformed JSON or network timeouts per contact sheet gracefully without failing the full search batch.

### 3.5 `main.py` — REST API Contract
1. **`GET /api/health`** -> `{ "status": "ok", "ffmpeg": bool, "ollama_reachable": bool }`
2. **`POST /api/index`** (`multipart/form-data`: `file`, optional `roi_json`) -> Remuxes video, runs ROI motion extraction & contact-sheet clustering in `loop.run_in_executor`, persists to SQLite, and returns `{ video_id, playable_url, event_count, events }`.
3. **`POST /api/search`** (`application/json`: `{ video_id, query, provider, api_key?, ollama_model? }`) -> Evaluates cached contact sheets concurrently via `asyncio.Semaphore` (`4` for Ollama, `10` for Gemini) and returns `{ results }` sorted by `(matched DESC, confidence DESC)`.
4. **`GET /api/video/{video_id}`** -> Serves `playable_mp4_path` via `FileResponse` with HTTP Range request support.

---

## 4. Complete Starter Implementation Code

### `backend/requirements.txt`
```text
fastapi
uvicorn[standard]
python-multipart
opencv-python-headless
numpy
pillow
pydantic>=2.0.0
httpx
pytest
```

### `backend/ingest.py`
```python
import subprocess
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MotionFrame:
    timestamp_sec: float
    timestamp_str: str
    frame: np.ndarray
    motion_score: float

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def ensure_browser_mp4(input_path: str, output_dir: str = "./workspace") -> str:
    """Remuxes proprietary .dav or raw H.265 streams into standard H.264 .mp4."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    in_file = Path(input_path)
    out_file = Path(output_dir) / f"{in_file.stem}_playable.mp4"

    if out_file.exists():
        return str(out_file)

    cmd = [
        "ffmpeg", "-y", "-i", str(in_file),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-an", str(out_file)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_file)

def extract_motion_frames(
    video_path: str,
    sample_fps: float = 1.0,
    roi_polygon: list[tuple[int, int]] | None = None,
    pixel_diff_threshold: int = 25,
    min_changed_ratio: float = 0.008
) -> list[MotionFrame]:
    """
    Scans video at `sample_fps`, applies optional ROI polygon mask,
    and keeps only frames with significant pixel changes.
    """
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(native_fps / sample_fps)))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_mask = None
    active_area = width * height
    if roi_polygon and len(roi_polygon) >= 3:
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(roi_polygon, dtype=np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)
        active_area = max(1, int(np.count_nonzero(roi_mask)))

    prev_gray = None
    motion_frames: list[MotionFrame] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if roi_mask is not None:
                gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(delta, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
                changed_pixels = int(np.count_nonzero(thresh))
                motion_ratio = changed_pixels / active_area

                if motion_ratio >= min_changed_ratio:
                    ts_sec = frame_idx / native_fps
                    motion_frames.append(
                        MotionFrame(
                            timestamp_sec=ts_sec,
                            timestamp_str=format_timestamp(ts_sec),
                            frame=frame.copy(),
                            motion_score=round(motion_ratio, 4)
                        )
                    )

            prev_gray = gray
        frame_idx += 1

    cap.release()
    return motion_frames
```

### `backend/events.py`
```python
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from ingest import MotionFrame

@dataclass
class VideoEvent:
    event_id: int
    start_sec: float
    end_sec: float
    start_str: str
    end_str: str
    contact_sheet_path: str
    frame_count: int

def cluster_into_events(
    motion_frames: list[MotionFrame],
    gap_tolerance_sec: float = 4.0
) -> list[list[MotionFrame]]:
    """Groups contiguous motion frames separated by <= gap_tolerance_sec into single events."""
    if not motion_frames:
        return []

    clusters: list[list[MotionFrame]] = [[motion_frames[0]]]
    for mf in motion_frames[1:]:
        if mf.timestamp_sec - clusters[-1][-1].timestamp_sec <= gap_tolerance_sec:
            clusters[-1].append(mf)
        else:
            clusters.append([mf])
    return clusters

def build_contact_sheets(
    clusters: list[list[MotionFrame]],
    output_dir: str = "./workspace/events",
    tile_size: tuple[int, int] = (480, 270)
) -> list[VideoEvent]:
    """Stitches up to 6 evenly spaced frames per event into a 2x3 labeled contact sheet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    events: list[VideoEvent] = []

    for idx, cluster in enumerate(clusters, start=1):
        indices = np.linspace(0, len(cluster) - 1, min(6, len(cluster)), dtype=int)
        sampled = [cluster[i] for i in indices]

        tiles = []
        for mf in sampled:
            tile = cv2.resize(mf.frame, tile_size)
            cv2.rectangle(tile, (8, 8), (135, 38), (0, 0, 0), -1)
            cv2.putText(
                tile, mf.timestamp_str, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2
            )
            tiles.append(tile)

        while len(tiles) < 6:
            tiles.append(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))

        row1 = np.hstack(tiles[0:3])
        row2 = np.hstack(tiles[3:6])
        grid = np.vstack([row1, row2])

        sheet_path = str(Path(output_dir) / f"event_{idx:03d}.jpg")
        cv2.imwrite(sheet_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])

        events.append(
            VideoEvent(
                event_id=idx,
                start_sec=cluster[0].timestamp_sec,
                end_sec=cluster[-1].timestamp_sec,
                start_str=cluster[0].timestamp_str,
                end_str=cluster[-1].timestamp_str,
                contact_sheet_path=sheet_path,
                frame_count=len(cluster)
            )
        )

    return events
```

### `backend/vlm_client.py`
```python
import base64
import httpx
from pathlib import Path
from pydantic import BaseModel, Field

class EventSearchResult(BaseModel):
    matched: bool = Field(description="True if the contact sheet clearly shows what the user asked for.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0.")
    best_timestamp: str = Field(description="Exact HH:MM:SS timestamp burned onto the matching frame.")
    summary: str = Field(description="Concise 1-sentence description of what happened in the frames.")
    tags: list[str] = Field(description="2 to 4 short lowercase keyword tags (e.g. person, car, parcel).")

SYSTEM_PROMPT = """You are a home CCTV forensic assistant analyzing a 2x3 sequence of video frames.
Each tile has a burned-in timestamp (HH:MM:SS) in the top-left corner.
Evaluate the frames strictly against the user's search query.
Return ONLY valid JSON matching this schema:
{
  "matched": boolean,
  "confidence": float (0.0 to 1.0),
  "best_timestamp": "HH:MM:SS",
  "summary": "1-sentence description of the activity",
  "tags": ["tag1", "tag2"]
}"""

def _encode_image_b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

async def analyze_contact_sheet_ollama(
    image_path: str,
    user_query: str,
    model: str = "qwen2.5-vl:7b",
    ollama_url: str = "http://localhost:11434/api/chat"
) -> EventSearchResult:
    """100% local offline VLM inference via Ollama."""
    img_b64 = _encode_image_b64(image_path)
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User query: '{user_query}'. Analyze this CCTV contact sheet.",
                "images": [img_b64]
            }
        ]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(ollama_url, json=payload)
        resp.raise_for_status()
        raw_text = resp.json()["message"]["content"]
        return EventSearchResult.model_validate_json(raw_text)

async def analyze_contact_sheet_gemini(
    image_path: str,
    user_query: str,
    api_key: str,
    model: str = "gemini-2.5-flash"
) -> EventSearchResult:
    """Fast cloud BYOK inference via Gemini Flash REST API."""
    img_b64 = _encode_image_b64(image_path)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {"text": f"User query: '{user_query}'. Analyze this CCTV contact sheet."},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return EventSearchResult.model_validate_json(raw_text)
```

### `backend/main.py`
```python
import asyncio
import json
import shutil
import uuid
import cv2
import httpx
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingest import ensure_browser_mp4, extract_motion_frames
from events import cluster_into_events, build_contact_sheets, VideoEvent
from vlm_client import (
    analyze_contact_sheet_ollama,
    analyze_contact_sheet_gemini,
    EventSearchResult,
)

app = FastAPI(title="LocalSight CCTV API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = Path("./workspace")
WORKSPACE.mkdir(exist_ok=True)
VIDEO_STORE: dict[str, dict] = {}

class SearchRequest(BaseModel):
    video_id: str
    query: str
    provider: Literal["ollama", "gemini"] = "ollama"
    api_key: str | None = None
    ollama_model: str = "qwen2.5-vl:7b"

def _run_indexing_pipeline(
    raw_path: str,
    video_dir: Path,
    roi_polygon: list[tuple[int, int]] | None
) -> tuple[str, list[VideoEvent]]:
    playable_mp4 = ensure_browser_mp4(raw_path, output_dir=str(video_dir))
    motion_frames = extract_motion_frames(
        playable_mp4,
        sample_fps=1.0,
        roi_polygon=roi_polygon
    )
    clusters = cluster_into_events(motion_frames, gap_tolerance_sec=4.0)
    events = build_contact_sheets(clusters, output_dir=str(video_dir / "events"))
    return playable_mp4, events

@app.get("/api/health")
async def health_check():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ffmpeg": ffmpeg_ok, "ollama_reachable": ollama_ok}

@app.post("/api/index")
async def index_video(
    file: UploadFile = File(...),
    roi_json: str | None = Form(default=None)
):
    video_id = uuid.uuid4().hex[:10]
    video_dir = WORKSPACE / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    raw_path = video_dir / (file.filename or "upload.mp4")
    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    roi_polygon = None
    if roi_json:
        raw_pts = json.loads(roi_json)
        if len(raw_pts) >= 3:
            cap = cv2.VideoCapture(str(raw_path))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            roi_polygon = [(int(p["x"] * w / 1000), int(p["y"] * h / 1000)) for p in raw_pts]

    loop = asyncio.get_running_loop()
    playable_mp4, events = await loop.run_in_executor(
        None, _run_indexing_pipeline, str(raw_path), video_dir, roi_polygon
    )

    VIDEO_STORE[video_id] = {
        "playable_mp4": playable_mp4,
        "events": events,
    }

    return {
        "video_id": video_id,
        "playable_url": f"http://localhost:8000/api/video/{video_id}",
        "event_count": len(events),
        "events": [
            {
                "id": ev.event_id,
                "startSec": round(ev.start_sec, 1),
                "timestampStr": ev.start_str,
                "endStr": ev.end_str,
                "confidence": 1.0,
                "summary": f"Motion activity detected ({ev.frame_count} active frames).",
                "tags": ["motion"],
                "matched": True,
            }
            for ev in events
        ],
    }

@app.post("/api/search")
async def search_events(req: SearchRequest):
    record = VIDEO_STORE.get(req.video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video ID not found. Index a video first.")

    if req.provider == "gemini" and not req.api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required for cloud BYOK mode.")

    events: list[VideoEvent] = record["events"]
    semaphore = asyncio.Semaphore(4 if req.provider == "ollama" else 10)

    async def evaluate_event(ev: VideoEvent):
        async with semaphore:
            try:
                if req.provider == "gemini":
                    res: EventSearchResult = await analyze_contact_sheet_gemini(
                        ev.contact_sheet_path, req.query, req.api_key
                    )
                else:
                    res: EventSearchResult = await analyze_contact_sheet_ollama(
                        ev.contact_sheet_path, req.query, model=req.ollama_model
                    )
                return {
                    "id": ev.event_id,
                    "startSec": round(ev.start_sec, 1),
                    "timestampStr": res.best_timestamp or ev.start_str,
                    "endStr": ev.end_str,
                    "confidence": res.confidence,
                    "summary": res.summary,
                    "tags": res.tags,
                    "matched": res.matched,
                }
            except Exception as exc:
                return {
                    "id": ev.event_id,
                    "startSec": round(ev.start_sec, 1),
                    "timestampStr": ev.start_str,
                    "endStr": ev.end_str,
                    "confidence": 0.0,
                    "summary": f"Evaluation error: {exc}",
                    "tags": ["error"],
                    "matched": False,
                }

    results = await asyncio.gather(*(evaluate_event(ev) for ev in events))
    results.sort(key=lambda r: (r["matched"], r["confidence"]), reverse=True)
    return {"results": results}

@app.get("/api/video/{video_id}")
async def stream_playable_video(video_id: str):
    record = VIDEO_STORE.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(record["playable_mp4"], media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

### `frontend/src/App.tsx`
```tsx
import React, { useState, useRef } from 'react';
import { Upload, Search, Shield, Crosshair, Play, Clock, CheckCircle2, Trash2, Sparkles } from 'lucide-react';

interface Point {
  x: number;
  y: number;
}

interface SecurityEvent {
  id: number;
  startSec: number;
  timestampStr: string;
  endStr: string;
  confidence: number;
  summary: string;
  tags: string[];
  matched: boolean;
}

export default function App() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [roiMode, setRoiMode] = useState<boolean>(false);
  const [roiPoints, setRoiPoints] = useState<Point[]>([]);
  const [query, setQuery] = useState<string>('');
  const [provider, setProvider] = useState<'ollama' | 'gemini'>('ollama');
  const [apiKey, setApiKey] = useState<string>('');
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [activeEventId, setActiveEventId] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setIsScanning(true);

    const formData = new FormData();
    formData.append('file', file);
    if (roiPoints.length >= 3) {
      formData.append('roi_json', JSON.stringify(roiPoints));
    }

    try {
      const res = await fetch('http://localhost:8000/api/index', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setVideoId(data.video_id);
      setVideoUrl(data.playable_url);
      setEvents(data.events);
    } finally {
      setIsScanning(false);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!roiMode || !overlayRef.current) return;
    const rect = overlayRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 1000);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 1000);
    setRoiPoints((prev) => [...prev, { x, y }]);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoId || !query.trim()) return;
    setIsScanning(true);

    try {
      const res = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          query,
          provider,
          api_key: provider === 'gemini' ? apiKey : null,
        }),
      });
      const data = await res.json();
      setEvents(data.results);
    } finally {
      setIsScanning(false);
    }
  };

  const jumpToTimestamp = (ev: SecurityEvent) => {
    setActiveEventId(ev.id);
    if (videoRef.current) {
      videoRef.current.currentTime = ev.startSec;
      videoRef.current.play();
    }
  };

  const polygonString = roiPoints.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <header className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">LocalSight CCTV Scrubber</h1>
            <p className="text-xs text-slate-400">
              100% Local-First Home Footage Indexing & Semantic Event Search
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value as 'ollama' | 'gemini')}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200"
          >
            <option value="ollama">Offline Ollama (qwen2.5-vl:7b)</option>
            <option value="gemini">Cloud BYOK (Gemini 2.5 Flash)</option>
          </select>

          {provider === 'gemini' && (
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Gemini API Key"
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 w-44"
            />
          )}

          <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-sm font-medium transition">
            <Upload className="w-4 h-4" />
            <span>{fileName ? 'Switch CCTV File' : 'Load Local CCTV Video'}</span>
            <input
              type="file"
              accept="video/*,.dav,.mkv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 pt-6">
        <div className="lg:col-span-7 space-y-4">
          <div className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center">
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  className="w-full h-full object-contain"
                />
                <svg
                  ref={overlayRef}
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  onClick={handleCanvasClick}
                  className={`absolute inset-0 w-full h-full ${
                    roiMode ? 'cursor-crosshair pointer-events-auto' : 'pointer-events-none'
                  }`}
                >
                  {roiPoints.length > 0 && (
                    <polygon
                      points={polygonString}
                      fill="rgba(16, 185, 129, 0.2)"
                      stroke="#10b981"
                      strokeWidth="4"
                    />
                  )}
                  {roiPoints.map((pt, i) => (
                    <circle key={i} cx={pt.x} cy={pt.y} r="10" fill="#10b981" />
                  ))}
                </svg>
              </>
            ) : (
              <div className="text-center p-8 space-y-3">
                <Upload className="w-10 h-10 text-slate-600 mx-auto" />
                <p className="text-sm text-slate-300 font-medium">
                  Drop an exported NVR/SD-card clip (.mp4, .mkv, .dav)
                </p>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Video stays strictly on your machine. Motion filtering strips 90%+ of static frames before AI indexing.
                </p>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/70 border border-slate-800 p-3.5 rounded-xl">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              <Crosshair className="w-4 h-4 text-emerald-400" />
              <span>
                <strong>Focus Zone (ROI):</strong>{' '}
                {roiPoints.length >= 3
                  ? `Active (${roiPoints.length}-point mask ignoring street background)`
                  : 'Full frame (Draw a zone around your porch or parked car)'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setRoiMode(!roiMode)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  roiMode
                    ? 'bg-emerald-500 text-slate-950'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                }`}
              >
                {roiMode ? 'Done Drawing Zone' : 'Draw Detection Zone'}
              </button>
              {roiPoints.length > 0 && (
                <button
                  type="button"
                  onClick={() => setRoiPoints([])}
                  className="p-1.5 rounded-lg bg-slate-800 hover:bg-red-950/60 text-slate-400 hover:text-red-400 transition"
                  title="Clear ROI"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 flex flex-col space-y-4">
          <form onSubmit={handleSearch} className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Ask Your Footage
            </label>
            <div className="relative flex items-center">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder='e.g., "delivery parcel" or "person near car"'
                className="w-full bg-slate-900 border border-slate-700 focus:border-emerald-500 rounded-xl pl-10 pr-24 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none"
              />
              <button
                type="submit"
                disabled={isScanning || !videoId}
                className="absolute right-2 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg text-xs font-semibold transition flex items-center gap-1.5"
              >
                <Sparkles className="w-3.5 h-3.5" />
                {isScanning ? 'Scanning...' : 'Find'}
              </button>
            </div>
          </form>

          <div className="flex-1 space-y-3 overflow-y-auto max-h-[540px] pr-1">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Detected Motion Events ({events.filter((e) => e.matched).length} matching)</span>
              <span>Click card to jump</span>
            </div>

            {events
              .filter((ev) => ev.matched)
              .map((ev) => (
                <div
                  key={ev.id}
                  onClick={() => jumpToTimestamp(ev)}
                  className={`p-4 rounded-xl border cursor-pointer transition space-y-2 ${
                    activeEventId === ev.id
                      ? 'bg-emerald-950/30 border-emerald-500'
                      : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md">
                      <Clock className="w-3.5 h-3.5" />
                      {ev.timestampStr} – {ev.endStr}
                    </span>
                    <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                      {Math.round(ev.confidence * 100)}% match
                    </span>
                  </div>

                  <p className="text-sm text-slate-200 leading-relaxed">{ev.summary}</p>

                  <div className="flex items-center justify-between pt-1">
                    <div className="flex gap-1.5">
                      {ev.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400">
                      <Play className="w-3 h-3 fill-current" /> Jump
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </main>
    </div>
  );
}
```

---

## 5. Testing & Acceptance Criteria for Claude Code
1. **Scaffold & Persistence:** Implement `backend/db.py` with SQLite persistence so indexed videos survive server restarts, and wire `main.py` to read/write from SQLite.
2. **Automated Pytest Suite (`backend/tests/test_pipeline.py`):**
   * Generate a synthetic 10-second `640x360` OpenCV `.mp4` containing static frames (`0–3s`), motion inside `(100, 100)–(250, 250)` (`4–6s`), and motion outside `(500, 50)–(600, 150)` (`7–9s`).
   * Verify `2` events are detected without an ROI mask and **only `1` event** (`4–6s`) is detected when masked to `[(80, 80), (280, 80), (280, 280), (80, 280)]`.
   * Verify `build_contact_sheets` produces a valid `1440x540` JPEG file and FastAPI endpoints (`/api/health`, `/api/index`, `/api/search`) pass with mocked VLM calls.
3. **Frontend Build Check:** Scaffold Vite + React + TypeScript + Tailwind in `frontend/` and verify `npm run build` exits with zero errors.