"""LocalSight FastAPI server: ingest, index, search and stream local CCTV footage."""
import asyncio
import json
import time
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
from fastapi.staticfiles import StaticFiles

from config import frontend_dist_dir, workspace_dir
from events import VideoEvent, build_contact_sheets, cluster_into_events
from ingest import (
    ensure_browser_mp4,
    extract_motion_frames,
    get_video_dimensions,
    normalize_roi,
    parse_timestamp,
    probe_video,
)
from vlm_client import (
    CLOUDFLARE_MODELS,
    DEFAULT_CLOUDFLARE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    GEMINI_MODELS,
    OLLAMA_TAGS_URL,
    OPENROUTER_MODELS_URL,
    EventSearchResult,
    analyze_contact_sheet_cloudflare,
    analyze_contact_sheet_gemini,
    analyze_contact_sheet_ollama,
    analyze_contact_sheet_openrouter,
    filter_free_vision_models,
)

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


Provider = Literal["ollama", "gemini", "openrouter", "cloudflare"]
CLOUD_PROVIDERS: dict[str, str] = {
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "cloudflare": "Cloudflare Workers AI",
}
# Concurrent contact sheets per provider. Ollama serves one at a time on laptops; free cloud
# tiers rate-limit bursts (OpenRouter free models ~20 req/min).
CONCURRENCY: dict[str, int] = {"ollama": 1, "gemini": 10, "openrouter": 2, "cloudflare": 4}


class SearchRequest(BaseModel):
    video_id: str
    query: str = Field(min_length=1, max_length=500)
    provider: Provider = "ollama"
    api_key: str | None = Field(default=None, max_length=500)
    ollama_model: str = Field(default=DEFAULT_OLLAMA_MODEL, pattern=r"^[A-Za-z0-9._:/-]{1,100}$")
    # Interpolated into the Gemini URL path, so restrict to model-ID characters.
    gemini_model: str = Field(default=DEFAULT_GEMINI_MODEL, pattern=r"^gemini-[a-z0-9.-]{1,60}$")
    openrouter_model: str = Field(default=DEFAULT_OPENROUTER_MODEL, pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._:-]{1,120}$")
    cloudflare_model: str = Field(default=DEFAULT_CLOUDFLARE_MODEL, pattern=r"^@cf/[A-Za-z0-9._-]+/[A-Za-z0-9._-]{1,120}$")
    # Interpolated into the Cloudflare URL path: Cloudflare account IDs are 32 hex chars.
    cloudflare_account_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


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


async def check_ollama_ready(model: str) -> None:
    """Fail fast with an actionable 503 instead of N identical per-event connection errors."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(OLLAMA_TAGS_URL)
            r.raise_for_status()
            installed = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running at localhost:11434. Install it (https://ollama.com or "
                f"`brew install ollama`), start it with `ollama serve`, run `ollama pull {model}`, "
                "or switch to Cloud BYOK (Gemini)."
            ),
        )
    if model not in installed and f"{model}:latest" not in installed:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is running but model '{model}' is not installed. Run `ollama pull {model}`.",
        )


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
    ollama_models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(OLLAMA_TAGS_URL)
            ollama_ok = r.status_code == 200
            if ollama_ok:
                ollama_models = sorted(m.get("name", "") for m in r.json().get("models", []) if m.get("name"))
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "ollama_reachable": ollama_ok,
        "ollama_models": ollama_models,
        "default_ollama_model": DEFAULT_OLLAMA_MODEL,
        "gemini_models": list(GEMINI_MODELS),
        "default_gemini_model": DEFAULT_GEMINI_MODEL,
        "default_openrouter_model": DEFAULT_OPENROUTER_MODEL,
        "cloudflare_models": list(CLOUDFLARE_MODELS),
        "default_cloudflare_model": DEFAULT_CLOUDFLARE_MODEL,
    }


_OPENROUTER_CACHE: dict = {"at": 0.0, "models": []}
OPENROUTER_CACHE_TTL_SEC = 600


@app.get("/api/openrouter/models")
async def openrouter_free_vision_models():
    """Live list of OpenRouter's free, image-capable models (public endpoint, no key or user data sent)."""
    now = time.monotonic()
    if _OPENROUTER_CACHE["models"] and now - _OPENROUTER_CACHE["at"] < OPENROUTER_CACHE_TTL_SEC:
        return {"models": _OPENROUTER_CACHE["models"], "default": DEFAULT_OPENROUTER_MODEL}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(OPENROUTER_MODELS_URL)
            r.raise_for_status()
            models = filter_free_vision_models(r.json().get("data", []))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load OpenRouter model list: {type(exc).__name__}") from exc
    _OPENROUTER_CACHE.update(at=now, models=models)
    return {"models": models, "default": DEFAULT_OPENROUTER_MODEL}


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
    if req.provider in CLOUD_PROVIDERS and not req.api_key:
        label = CLOUD_PROVIDERS[req.provider]
        what = "API token" if req.provider == "cloudflare" else "API key"
        raise HTTPException(status_code=400, detail=f"{label} {what} is required for cloud BYOK mode.")
    if req.provider == "cloudflare" and not req.cloudflare_account_id:
        raise HTTPException(status_code=400, detail="Cloudflare account ID is required (32-character hex ID from the dashboard).")

    events = db.list_events(req.video_id)
    if not events:
        return {"results": []}
    if req.provider == "ollama":
        await check_ollama_ready(req.ollama_model)
    semaphore = asyncio.Semaphore(CONCURRENCY[req.provider])

    def _safe_error(exc: Exception) -> str:
        msg = f"{type(exc).__name__}: {exc}"
        if req.api_key:
            msg = msg.replace(req.api_key, "***")
        return msg[:300]

    async def evaluate_event(ev: VideoEvent) -> dict:
        async with semaphore:
            try:
                res: EventSearchResult
                if req.provider == "gemini":
                    res = await analyze_contact_sheet_gemini(
                        ev.contact_sheet_path, req.query, req.api_key, model=req.gemini_model
                    )
                elif req.provider == "openrouter":
                    res = await analyze_contact_sheet_openrouter(
                        ev.contact_sheet_path, req.query, req.api_key, model=req.openrouter_model
                    )
                elif req.provider == "cloudflare":
                    res = await analyze_contact_sheet_cloudflare(
                        ev.contact_sheet_path, req.query, req.api_key, req.cloudflare_account_id,
                        model=req.cloudflare_model,
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


def mount_frontend(target: FastAPI, dist: Path) -> bool:
    """Serve the built React app at / (after all /api routes) so a release runs from one command."""
    if not (dist / "index.html").is_file():
        return False
    target.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return True


mount_frontend(app, frontend_dist_dir())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
