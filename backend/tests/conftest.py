import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WIDTH, HEIGHT, FPS, DURATION = 640, 360, 10, 20
ROI_PIXELS = [(80, 80), (280, 80), (280, 280), (80, 280)]
# Same ROI in the UI's 0–1000 normalized space (relative to 640x360).
ROI_NORMALIZED = [{"x": round(x * 1000 / (WIDTH - 1), 1), "y": round(y * 1000 / (HEIGHT - 1), 1)} for x, y in ROI_PIXELS]

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _write_synthetic(path: Path) -> None:
    """
    20s 640x360 @10fps on a static grey background:
      * 0–4s static
      * 4–7s moving 80px square inside (100,100)-(250,250)   -> inside ROI
      * 7–14s static
      * 14–17s moving 70px square inside (500,50)-(600,150)  -> outside ROI
      * 17–20s static
    The 7s gap between bursts exceeds the 4.0s clustering tolerance.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "cv2.VideoWriter failed to open"
    for i in range(DURATION * FPS):
        t = i / FPS
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        if 4 <= t < 7:
            x = 100 + int((t - 4) / 3 * 70)          # 100..170, square stays within x<=250
            cv2.rectangle(frame, (x, 120), (x + 80, 200), (255, 255, 255), -1)
        if 14 <= t < 17:
            x = 500 + int((t - 14) / 3 * 30)          # 500..530, square stays within x<=600
            cv2.rectangle(frame, (x, 65), (x + 70, 135), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fixtures") / "synthetic.mp4"
    _write_synthetic(path)
    return path


@pytest.fixture(scope="session")
def h264_video(tmp_path_factory, synthetic_video) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("fixtures_h264") / "already_h264.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(synthetic_video), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    ws = tmp_path / "workspace"
    monkeypatch.setenv("LOCALSIGHT_WORKSPACE", str(ws))
    return ws


@pytest.fixture
def client(workspace):
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        yield c
