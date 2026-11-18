"""SQLite persistence for indexed videos and their motion events."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import db_path
from events import VideoEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    playable_mp4_path TEXT NOT NULL,
    duration_sec REAL,
    roi_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    event_index INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    start_str TEXT NOT NULL,
    end_str TEXT NOT NULL,
    frame_count INTEGER NOT NULL,
    contact_sheet_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_video ON events(video_id, event_index);
"""


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path or db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def insert_video(
    video_id: str,
    original_filename: str,
    playable_mp4_path: str,
    duration_sec: float | None,
    roi_json: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO videos (video_id, original_filename, playable_mp4_path, duration_sec, roi_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (video_id, original_filename, playable_mp4_path, duration_sec, roi_json),
        )


def get_video(video_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None


def update_roi(video_id: str, roi_json: str | None) -> None:
    with connect() as conn:
        conn.execute("UPDATE videos SET roi_json = ? WHERE video_id = ?", (roi_json, video_id))


def replace_events(video_id: str, events: list[VideoEvent]) -> None:
    """Atomically replace all events for a video (used by index and reindex)."""
    with connect() as conn:
        conn.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO events (video_id, event_index, start_sec, end_sec, start_str, end_str, "
            "frame_count, contact_sheet_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (video_id, ev.event_id, ev.start_sec, ev.end_sec, ev.start_str, ev.end_str,
                 ev.frame_count, ev.contact_sheet_path)
                for ev in events
            ],
        )


def list_events(video_id: str) -> list[VideoEvent]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE video_id = ? ORDER BY event_index", (video_id,)
        ).fetchall()
    return [
        VideoEvent(
            event_id=r["event_index"],
            start_sec=r["start_sec"],
            end_sec=r["end_sec"],
            start_str=r["start_str"],
            end_str=r["end_str"],
            contact_sheet_path=r["contact_sheet_path"],
            frame_count=r["frame_count"],
        )
        for r in rows
    ]
