"""FFmpeg remuxing to browser-playable H.264 MP4 and OpenCV ROI motion extraction."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def parse_timestamp(value: str) -> float | None:
    """Parse HH:MM:SS (or MM:SS) to seconds; None if malformed."""
    try:
        parts = [int(p) for p in value.strip().split(":")]
    except (ValueError, AttributeError):
        return None
    if not 2 <= len(parts) <= 3 or any(p < 0 for p in parts):
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return float(h * 3600 + m * 60 + s)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {tail}")
    return proc


def probe_video(path: str) -> dict:
    """Returns {'codec': str | None, 'duration': float | None} via ffprobe."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name:format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    codec = streams[0].get("codec_name") if streams else None
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {"codec": codec, "duration": duration}


def ensure_browser_mp4(input_path: str, output_dir: str = "./workspace") -> str:
    """Stream-copies H.264 MP4/MOV; transcodes .dav/.h265/.mkv/etc. into H.264 .mp4."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    in_file = Path(input_path)
    out_file = Path(output_dir) / f"{in_file.stem}_playable.mp4"

    if out_file.exists() and out_file.stat().st_size > 0:
        return str(out_file)

    codec = probe_video(str(in_file))["codec"]
    if codec is None:
        raise RuntimeError(f"No video stream found in {in_file.name}")

    common_tail = ["-an", "-movflags", "+faststart", str(out_file)]
    if codec == "h264" and in_file.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        cmd = ["ffmpeg", "-y", "-i", str(in_file), "-c:v", "copy", *common_tail]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(in_file),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
            "-pix_fmt", "yuv420p", *common_tail,
        ]
    try:
        _run(cmd)
    except Exception:
        out_file.unlink(missing_ok=True)
        raise
    return str(out_file)


def get_video_dimensions(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Could not read video dimensions for {path}")
    return w, h


def normalize_roi(
    points: list[dict], width: int, height: int
) -> list[tuple[int, int]] | None:
    """Convert UI 0–1000 normalized points to native pixel coords (clamped)."""
    if not points or len(points) < 3:
        return None
    out: list[tuple[int, int]] = []
    for p in points:
        x = min(1000.0, max(0.0, float(p["x"])))
        y = min(1000.0, max(0.0, float(p["y"])))
        out.append((int(round(x * (width - 1) / 1000)), int(round(y * (height - 1) / 1000))))
    return out


def extract_motion_frames(
    video_path: str,
    sample_fps: float = 1.0,
    roi_polygon: list[tuple[int, int]] | None = None,
    pixel_diff_threshold: int = 25,
    min_changed_ratio: float = 0.008,
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
    active_area = max(1, width * height)
    if roi_polygon and len(roi_polygon) >= 3:
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(roi_polygon, dtype=np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)
        active_area = max(1, int(np.count_nonzero(roi_mask)))

    prev_gray = None
    motion_frames: list[MotionFrame] = []
    frame_idx = 0

    try:
        while True:
            # grab() skips decoding-to-BGR for frames we don't sample
            if frame_idx % frame_interval != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if roi_mask is not None:
                gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(delta, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
                motion_ratio = int(np.count_nonzero(thresh)) / active_area

                if motion_ratio >= min_changed_ratio:
                    ts_sec = frame_idx / native_fps
                    motion_frames.append(
                        MotionFrame(
                            timestamp_sec=ts_sec,
                            timestamp_str=format_timestamp(ts_sec),
                            frame=frame.copy(),
                            motion_score=round(motion_ratio, 4),
                        )
                    )

            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()
    return motion_frames
