"""Temporal motion clustering and 2x3 timestamp-burned contact-sheet builder."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
    gap_tolerance_sec: float = 4.0,
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
    tile_size: tuple[int, int] = (480, 270),
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2,
            )
            tiles.append(tile)

        while len(tiles) < 6:
            tiles.append(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))

        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])

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
                frame_count=len(cluster),
            )
        )

    return events
