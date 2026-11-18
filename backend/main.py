"""LocalSight FastAPI server: ingest, index, search and stream local CCTV footage."""
import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db
from config import workspace_dir
from events import VideoEvent, build_contact_sheets, cluster_into_events
from ingest import (
    ensure_browser_mp4,
    extract_motion_frames,
    get_video_dimensions,
    normalize_roi,
    parse_timestamp,
    probe_video,
)
from vlm_client import EventSearchResult, analyze_contact_sheet_gemini, analyze_contact_sheet_ollama

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".dav", ".mov", ".avi", ".m4v", ".h264", ".h265", ".264", ".265"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="LocalSight CCTV API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RoiPoint(BaseModel):
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=1000)


class ReindexRequest(BaseModel):
    roi: list[RoiPoint] | None = None


class SearchRequest(BaseModel):
    video_id: str
    query: str = Field(min_length=1, max_length=500)
    provider: Literal["ollama", "gemini"] = "ollama"
    api_key: str | None = None
    ollama_model: str = "qwen2.5-vl:7b"


# ---------- helpers ----------

def _event_payload(ev: VideoEvent, **overrides) -> dict:
    base = {
        "id": ev.event_id,
        "startSec": round(ev.start_sec, 1),
        "endSec": round(ev.end_sec, 1),
        "timestampStr": ev.start_str,
        "endStr": ev.end_str,
        "confidence": 1.0,
        "summary": f"Motion activity detected ({ev.frame_count} active frames).",
        "tags": ["motion"],
        "matched": True,
    }
    base.update(overrides)
    return base


def _video_payload(video_id: str, events: list[VideoEvent]) -> dict:
    return {
        "video_id": video_id,
        "playable_url": f"/api/video/{video_id}",
        "event_count": len(events),
        "events": [_event_payload(ev) for ev in events],
    }


def _parse_roi(roi_json: str | None) -> list[dict] | None:
    if not roi_json:
        return None
    try:
        pts = [RoiPoint.model_validate(p).model_dump() for p in json.loads(roi_json)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid roi_json: {exc}") from exc
    return pts if len(pts) >= 3 else None


def _scan(playable_mp4: str, video_dir: Path, roi_points: list[dict] | None) -> list[VideoEvent]:
    """CPU-bound: ROI motion extraction + clustering + contact sheets."""
    roi_polygon = None
    if roi_points:
        w, h = get_video_dimensions(playable_mp4)
        roi_polygon = normalize_roi(roi_points, w, h)
    motion_frames = extract_motion_frames(playable_mp4, sample_fps=1.0, roi_polygon=roi_polygon)
    clusters = cluster_into_events(motion_frames, gap_tolerance_sec=4.0)
    events_dir = video_dir / "events"
    shutil.rmtree(events_dir, ignore_errors=True)
    return build_contact_sheets(clusters, output_dir=str(events_dir))


def _run_indexing_pipeline(
    raw_path: str, video_dir: Path, roi_points: list[dict] | None
) -> tuple[str, float | None, list[VideoEvent]]:
    playable_mp4 = ensure_browser_mp4(raw_path, output_dir=str(video_dir))
    duration = probe_video(playable_mp4)["duration"]
    return playable_mp4, duration, _scan(playable_mp4, video_dir, roi_points)


def _save_upload(file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer, length=8 * 1024 * 1024)


def _require_video(video_id: str) -> dict:
    record = db.get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video ID not found. Index a video first.")
    return record


# ---------- endpoints ----------

@app.get("/api/health")
async def health_check():
    ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ffmpeg": ffmpeg_ok, "ollama_reachable": ollama_ok}


@app.post("/api/index")
async def index_video(file: UploadFile = File(...), roi_json: str | None = Form(default=None)):
    filename = Path(file.filename or "upload.mp4").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {Path(filename).suffix or '(none)'}")
    roi_points = _parse_roi(roi_json)

    video_id = uuid.uuid4().hex[:10]
    video_dir = workspace_dir() / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    raw_path = video_dir / filename

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _save_upload, file, raw_path)
        playable_mp4, duration, events = await loop.run_in_executor(
            None, _run_indexing_pipeline, str(raw_path), video_dir, roi_points
        )
    except Exception as exc:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Could not process video: {exc}") from exc

    db.insert_video(video_id, filename, playable_mp4, duration, json.dumps(roi_points) if roi_points else None)
    db.replace_events(video_id, events)
    return _video_payload(video_id, events)


@app.post("/api/videos/{video_id}/reindex")
async def reindex_video(video_id: str, req: ReindexRequest):
    record = _require_video(video_id)
    roi_points = [p.model_dump() for p in req.roi] if req.roi and len(req.roi) >= 3 else None
    playable_mp4 = record["playable_mp4_path"]
    loop = asyncio.get_running_loop()
    try:
        events = await loop.run_in_executor(None, _scan, playable_mp4, Path(playable_mp4).parent, roi_points)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not re-scan video: {exc}") from exc
    db.update_roi(video_id, json.dumps(roi_points) if roi_points else None)
    db.replace_events(video_id, events)
    return _video_payload(video_id, events)


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    record = _require_video(video_id)
    payload = _video_payload(video_id, db.list_events(video_id))
    payload["original_filename"] = record["original_filename"]
    payload["duration_sec"] = record["duration_sec"]
    payload["roi"] = json.loads(record["roi_json"]) if record["roi_json"] else None
    return payload


@app.post("/api/search")
async def search_events(req: SearchRequest):
    _require_video(req.video_id)
    if req.provider == "gemini" and not req.api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required for cloud BYOK mode.")

    events = db.list_events(req.video_id)
    semaphore = asyncio.Semaphore(4 if req.provider == "ollama" else 10)

    def _safe_error(exc: Exception) -> str:
        msg = f"{type(exc).__name__}: {exc}"
        if req.api_key:
            msg = msg.replace(req.api_key, "***")
        return msg[:300]

    async def evaluate_event(ev: VideoEvent) -> dict:
        async with semaphore:
            try:
                if req.provider == "gemini":
                    res: EventSearchResult = await analyze_contact_sheet_gemini(
                        ev.contact_sheet_path, req.query, req.api_key
                    )
                else:
                    res = await analyze_contact_sheet_ollama(
                        ev.contact_sheet_path, req.query, model=req.ollama_model
                    )
            except Exception as exc:  # one bad sheet must never abort the batch
                return _event_payload(
                    ev, confidence=0.0, summary=f"Evaluation error: {_safe_error(exc)}",
                    tags=["error"], matched=False,
                )
            # Seek to the matching tile when the model's timestamp falls inside the event window.
            start_sec = ev.start_sec
            best = parse_timestamp(res.best_timestamp) if res.best_timestamp else None
            if best is not None and ev.start_sec - 1 <= best <= ev.end_sec + 1:
                start_sec = max(ev.start_sec, best)
            return _event_payload(
                ev,
                startSec=round(start_sec, 1),
                timestampStr=res.best_timestamp if best is not None else ev.start_str,
                confidence=res.confidence,
                summary=res.summary,
                tags=res.tags,
                matched=res.matched,
            )

    results = await asyncio.gather(*(evaluate_event(ev) for ev in events))
    results.sort(key=lambda r: (r["matched"], r["confidence"]), reverse=True)
    return {"results": results}


@app.get("/api/video/{video_id}")
async def stream_playable_video(video_id: str):
    record = db.get_video(video_id)
    if not record or not Path(record["playable_mp4_path"]).exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(record["playable_mp4_path"], media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
