import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import db
import ingest
import main
from conftest import HAS_FFMPEG, ROI_NORMALIZED, ROI_PIXELS, requires_ffmpeg
from events import build_contact_sheets, cluster_into_events
from ingest import (
    MotionFrame,
    ensure_browser_mp4,
    extract_motion_frames,
    format_timestamp,
    normalize_roi,
    parse_timestamp,
    probe_video,
)
from vlm_client import (
    EventSearchResult,
    analyze_contact_sheet_gemini,
    analyze_contact_sheet_ollama,
    parse_result,
)


# ---------------------------------------------------------------- timestamps

@pytest.mark.parametrize("sec,expected", [(0, "00:00:00"), (59.9, "00:00:59"), (3661, "01:01:01")])
def test_format_timestamp(sec, expected):
    assert format_timestamp(sec) == expected


def test_parse_timestamp():
    assert parse_timestamp("01:01:01") == 3661.0
    assert parse_timestamp("02:05") == 125.0
    assert parse_timestamp("garbage") is None
    assert parse_timestamp("") is None


def test_normalize_roi_clamps_and_scales():
    pts = normalize_roi([{"x": 0, "y": 0}, {"x": 1000, "y": 0}, {"x": 1500, "y": 1000}], 640, 360)
    assert pts == [(0, 0), (639, 0), (639, 359)]
    assert normalize_roi([{"x": 1, "y": 1}], 640, 360) is None


# ---------------------------------------------------------------- motion + ROI

def test_motion_without_roi_yields_two_events(synthetic_video):
    frames = extract_motion_frames(str(synthetic_video))
    clusters = cluster_into_events(frames, gap_tolerance_sec=4.0)
    assert len(clusters) == 2
    assert 4 <= clusters[0][0].timestamp_sec <= 7
    assert 14 <= clusters[1][0].timestamp_sec <= 17
    # Noise reduction: most of the 20 sampled frames are dropped
    assert len(frames) <= 10


def test_motion_with_roi_yields_one_event(synthetic_video):
    frames = extract_motion_frames(str(synthetic_video), roi_polygon=ROI_PIXELS)
    clusters = cluster_into_events(frames, gap_tolerance_sec=4.0)
    assert len(clusters) == 1
    assert 4 <= clusters[0][0].timestamp_sec <= 7
    assert clusters[0][-1].timestamp_sec <= 8


def test_static_video_has_no_events(tmp_path):
    import cv2

    path = tmp_path / "static.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for _ in range(50):
        w.write(np.full((180, 320, 3), 50, dtype=np.uint8))
    w.release()
    assert extract_motion_frames(str(path)) == []


# ---------------------------------------------------------------- contact sheets

def test_contact_sheet_is_1440x540_jpeg(synthetic_video, tmp_path):
    clusters = cluster_into_events(extract_motion_frames(str(synthetic_video)))
    events = build_contact_sheets(clusters, output_dir=str(tmp_path / "events"))
    assert len(events) == 2
    for ev in events:
        with Image.open(ev.contact_sheet_path) as img:
            assert img.format == "JPEG"
            assert img.size == (1440, 540)
        assert ev.frame_count >= 1


def test_contact_sheet_pads_short_clusters(tmp_path):
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    cluster = [MotionFrame(1.0, "00:00:01", frame, 0.1), MotionFrame(2.0, "00:00:02", frame, 0.1)]
    [ev] = build_contact_sheets([cluster], output_dir=str(tmp_path))
    with Image.open(ev.contact_sheet_path) as img:
        arr = np.asarray(img.convert("L"))
    assert img.size == (1440, 540)
    # Bottom-right tile is a black pad; top-left tile is bright
    assert arr[300:, 1000:].mean() < 10
    assert arr[100:250, 200:400].mean() > 150


# ---------------------------------------------------------------- ffmpeg remux

@requires_ffmpeg
def test_remux_mp4v_transcodes_to_h264(synthetic_video, tmp_path):
    assert probe_video(str(synthetic_video))["codec"] == "mpeg4"
    out = ensure_browser_mp4(str(synthetic_video), output_dir=str(tmp_path))
    info = probe_video(out)
    assert info["codec"] == "h264"
    assert info["duration"] == pytest.approx(20, abs=0.5)


@requires_ffmpeg
def test_remux_h264_uses_stream_copy(h264_video, tmp_path, monkeypatch):
    seen: list[list[str]] = []
    real_run = ingest._run
    monkeypatch.setattr(ingest, "_run", lambda cmd: (seen.append(cmd), real_run(cmd))[1])
    out = ensure_browser_mp4(str(h264_video), output_dir=str(tmp_path))
    ffmpeg_cmds = [c for c in seen if c[0] == "ffmpeg"]
    assert ffmpeg_cmds and "copy" in ffmpeg_cmds[0]
    assert probe_video(out)["codec"] == "h264"


@requires_ffmpeg
def test_remux_hevc_mkv_to_h264(tmp_path):
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx265" not in encoders:
        pytest.skip("libx265 encoder not available")
    src = tmp_path / "camera.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
         "-c:v", "libx265", "-x265-params", "log-level=error", str(src)],
        check=True, capture_output=True,
    )
    assert probe_video(str(src))["codec"] == "hevc"
    out = ensure_browser_mp4(str(src), output_dir=str(tmp_path / "out"))
    assert out.endswith(".mp4")
    assert probe_video(out)["codec"] == "h264"


@requires_ffmpeg
def test_remux_failure_raises_clear_error(tmp_path):
    bad = tmp_path / "broken.dav"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(RuntimeError):
        ensure_browser_mp4(str(bad), output_dir=str(tmp_path / "out"))
    assert not (tmp_path / "out" / "broken_playable.mp4").exists()


# ---------------------------------------------------------------- API

async def _ollama_ready(model: str) -> None:
    return None


def _index(client: TestClient, video: Path, roi=None):
    data = {"roi_json": json.dumps(roi)} if roi else {}
    with video.open("rb") as fh:
        return client.post("/api/index", files={"file": (video.name, fh, "video/mp4")}, data=data)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "ffmpeg", "ollama_reachable"}
    assert isinstance(body["ollama_reachable"], bool)
    assert body["ffmpeg"] is HAS_FFMPEG


def test_index_rejects_unsupported_extension(client, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with f.open("rb") as fh:
        r = client.post("/api/index", files={"file": ("notes.txt", fh, "text/plain")})
    assert r.status_code == 400


@requires_ffmpeg
def test_index_without_and_with_roi(client, synthetic_video, workspace):
    r = _index(client, synthetic_video)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_count"] == 2
    assert body["playable_url"] == f"/api/video/{body['video_id']}"
    ev = body["events"][0]
    assert set(ev) >= {"id", "startSec", "timestampStr", "endStr", "confidence", "summary", "tags", "matched"}

    stored = db.list_events(body["video_id"])
    assert len(stored) == 2
    assert all(Path(e.contact_sheet_path).exists() for e in stored)
    assert Path(stored[0].contact_sheet_path).is_relative_to(workspace)

    r2 = _index(client, synthetic_video, roi=ROI_NORMALIZED)
    assert r2.status_code == 200, r2.text
    assert r2.json()["event_count"] == 1
    assert json.loads(db.get_video(r2.json()["video_id"])["roi_json"])[0]["x"] == ROI_NORMALIZED[0]["x"]


def test_index_rejects_bad_roi_json(client, synthetic_video):
    with synthetic_video.open("rb") as fh:
        r = client.post("/api/index", files={"file": ("a.mp4", fh, "video/mp4")}, data={"roi_json": "{nope"})
    assert r.status_code == 400


@requires_ffmpeg
def test_reindex_with_roi_replaces_events(client, synthetic_video):
    vid = _index(client, synthetic_video).json()["video_id"]
    r = client.post(f"/api/videos/{vid}/reindex", json={"roi": ROI_NORMALIZED})
    assert r.status_code == 200, r.text
    assert r.json()["event_count"] == 1
    assert len(db.list_events(vid)) == 1
    events_dir = Path(db.get_video(vid)["playable_mp4_path"]).parent / "events"
    assert len(list(events_dir.glob("*.jpg"))) == 1

    # Clearing the ROI restores full-frame detection
    r = client.post(f"/api/videos/{vid}/reindex", json={"roi": None})
    assert r.json()["event_count"] == 2
    assert db.get_video(vid)["roi_json"] is None

    assert client.post("/api/videos/missing/reindex", json={"roi": None}).status_code == 404


@requires_ffmpeg
def test_search_ranks_and_isolates_failures(client, synthetic_video, monkeypatch):
    vid = _index(client, synthetic_video).json()["video_id"]
    calls: list[str] = []

    async def fake_ollama(image_path, user_query, model="qwen2.5vl:7b", **_):
        calls.append(image_path)
        if image_path.endswith("event_001.jpg"):
            raise ValueError("malformed JSON from model")
        return EventSearchResult(
            matched=True, confidence=0.91, best_timestamp="00:00:15",
            summary="A white box moves near the gate.", tags=["box", "gate"],
        )

    monkeypatch.setattr(main, "analyze_contact_sheet_ollama", fake_ollama)
    monkeypatch.setattr(main, "check_ollama_ready", _ollama_ready)
    r = client.post("/api/search", json={"video_id": vid, "query": "white box", "provider": "ollama"})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 2 and len(calls) == 2
    assert results[0]["matched"] is True and results[0]["confidence"] == 0.91
    assert results[0]["timestampStr"] == "00:00:15"
    assert results[0]["startSec"] == pytest.approx(15, abs=1)
    assert results[1]["matched"] is False and results[1]["tags"] == ["error"]
    assert "malformed" in results[1]["summary"]


@requires_ffmpeg
def test_search_gemini_requires_key_and_redacts_it(client, synthetic_video, monkeypatch):
    vid = _index(client, synthetic_video).json()["video_id"]
    r = client.post("/api/search", json={"video_id": vid, "query": "car", "provider": "gemini"})
    assert r.status_code == 400

    async def failing_gemini(image_path, user_query, api_key, **_):
        raise RuntimeError(f"upstream rejected key {api_key}")

    monkeypatch.setattr(main, "analyze_contact_sheet_gemini", failing_gemini)
    r = client.post("/api/search", json={"video_id": vid, "query": "car", "provider": "gemini", "api_key": "SECRET123"})
    assert r.status_code == 200
    assert "SECRET123" not in r.text
    assert all(res["matched"] is False for res in r.json()["results"])


@requires_ffmpeg
def test_search_ollama_unreachable_returns_actionable_503(client, synthetic_video, monkeypatch):
    vid = _index(client, synthetic_video).json()["video_id"]
    monkeypatch.setattr(main, "OLLAMA_TAGS_URL", "http://127.0.0.1:9/api/tags")  # nothing listens here
    r = client.post("/api/search", json={"video_id": vid, "query": "person", "provider": "ollama"})
    assert r.status_code == 503
    assert "ollama serve" in r.json()["detail"] and "qwen2.5vl:7b" in r.json()["detail"]


@requires_ffmpeg
def test_search_ollama_model_missing_returns_503(client, synthetic_video, monkeypatch):
    vid = _index(client, synthetic_video).json()["video_id"]
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        )
        return real_client(*args, **kwargs)

    monkeypatch.setattr(main.httpx, "AsyncClient", fake_client)
    r = client.post("/api/search", json={"video_id": vid, "query": "person", "provider": "ollama"})
    assert r.status_code == 503
    assert "ollama pull qwen2.5vl:7b" in r.json()["detail"]


def test_default_ollama_model_is_valid_library_tag():
    assert main.SearchRequest(video_id="v", query="q").ollama_model == "qwen2.5vl:7b"


def test_search_unknown_video_404(client):
    r = client.post("/api/search", json={"video_id": "nope", "query": "x", "provider": "ollama"})
    assert r.status_code == 404


@requires_ffmpeg
def test_persistence_across_restart(workspace, synthetic_video, monkeypatch):
    with TestClient(main.app) as c1:
        vid = _index(c1, synthetic_video).json()["video_id"]

    async def fake_ollama(image_path, user_query, **_):
        return EventSearchResult(matched=False, confidence=0.1, best_timestamp="", summary="nothing", tags=[])

    monkeypatch.setattr(main, "analyze_contact_sheet_ollama", fake_ollama)
    monkeypatch.setattr(main, "check_ollama_ready", _ollama_ready)
    with TestClient(main.app) as c2:  # fresh app lifespan, same on-disk SQLite
        r = c2.get(f"/api/videos/{vid}")
        assert r.status_code == 200 and r.json()["event_count"] == 2
        assert c2.post("/api/search", json={"video_id": vid, "query": "x"}).status_code == 200
        assert c2.get(f"/api/video/{vid}").status_code == 200


@requires_ffmpeg
def test_video_supports_range_requests(client, synthetic_video):
    vid = _index(client, synthetic_video).json()["video_id"]
    r = client.get(f"/api/video/{vid}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"
    assert client.get("/api/video/missing").status_code == 404


def test_foreign_keys_cascade(workspace):
    db.init_db()
    db.insert_video("v1", "a.mp4", "/tmp/a.mp4", 1.0, None)
    from events import VideoEvent

    db.replace_events("v1", [VideoEvent(1, 0.0, 1.0, "00:00:00", "00:00:01", "/tmp/e.jpg", 2)])
    with db.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.execute("DELETE FROM videos WHERE video_id = ?", ("v1",))
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


# ---------------------------------------------------------------- VLM adapters

VALID_JSON = '{"matched": true, "confidence": 1.7, "best_timestamp": "00:00:05", "summary": "A person.", "tags": ["Person", "Gate"]}'


def test_parse_result_strips_fences_and_clamps():
    res = parse_result(f"```json\n{VALID_JSON}\n```")
    assert res.matched and res.confidence == 1.0 and res.tags == ["person", "gate"]
    with pytest.raises(Exception):
        parse_result("I think there's a person")


def test_ollama_adapter(tmp_path):
    img = tmp_path / "s.jpg"
    Image.new("RGB", (1440, 540)).save(img)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": VALID_JSON}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    res = asyncio.run(analyze_contact_sheet_ollama(str(img), "person", client=client))
    assert res.matched and res.best_timestamp == "00:00:05"
    assert seen["url"].startswith("http://localhost:11434/api/chat")
    assert seen["body"]["format"] == "json" and seen["body"]["messages"][1]["images"]


def test_gemini_adapter_sends_key_in_header(tmp_path):
    img = tmp_path / "s.jpg"
    Image.new("RGB", (1440, 540)).save(img)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": VALID_JSON}]}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    res = asyncio.run(analyze_contact_sheet_gemini(str(img), "person", "KEY-abc", client=client))
    assert res.confidence == 1.0
    assert seen["key"] == "KEY-abc" and "KEY-abc" not in seen["url"]
    assert "gemini-2.5-flash:generateContent" in seen["url"]
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"


# ---------------------------------------------------------------- bundled frontend

def test_mount_frontend_serves_spa_without_shadowing_api(tmp_path):
    from fastapi import FastAPI

    dist = tmp_path / "dist"
    assert main.mount_frontend(FastAPI(), dist) is False  # no build -> nothing mounted

    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>LocalSight</title>")
    app = FastAPI()

    @app.get("/api/ping")
    def ping():
        return {"pong": True}

    assert main.mount_frontend(app, dist) is True
    with TestClient(app) as c:
        assert "LocalSight" in c.get("/").text
        assert c.get("/api/ping").json() == {"pong": True}
