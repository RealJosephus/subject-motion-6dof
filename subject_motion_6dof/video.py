from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch


class VideoTimingError(RuntimeError):
    """Raised when video frame timestamps cannot be mapped safely to label time."""


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        rate = float(value)
        return rate if rate > 0 else None
    num, den = value.split("/", 1)
    den_f = float(den)
    if den_f == 0:
        return None
    rate = float(num) / den_f
    return rate if rate > 0 else None


def _run_ffprobe(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        raise VideoTimingError(
            "ffprobe is required to verify constant-frame-rate video timing. "
            "Install FFmpeg/ffprobe or convert the video to CFR before training/generation."
        )
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration:"
            "frame=best_effort_timestamp_time,pkt_pts_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise VideoTimingError(
            f"ffprobe failed for {path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoTimingError(f"ffprobe returned invalid JSON for {path}") from exc


def _frame_timestamps(data: dict) -> list[float]:
    timestamps = []
    for frame in data.get("frames", []) or []:
        raw = frame.get("best_effort_timestamp_time", frame.get("pkt_pts_time"))
        if raw is None or raw == "N/A":
            continue
        timestamps.append(float(raw))
    return timestamps


def _validate_cfr_timestamps(
    path: Path,
    timestamps: list[float],
    tolerance_ms: float,
) -> float | None:
    if len(timestamps) < 3:
        return None
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    if np.any(deltas <= 0):
        raise VideoTimingError(
            f"Non-monotonic frame timestamps in {path}; refusing to map label milliseconds by frame index."
        )
    median_delta = float(np.median(deltas))
    max_jitter_ms = float(np.max(np.abs(deltas - median_delta)) * 1000.0)
    if max_jitter_ms > tolerance_ms:
        raise VideoTimingError(
            f"Variable-frame-rate video detected: {path}. "
            f"max frame interval jitter is {max_jitter_ms:.3f} ms "
            f"(tolerance {tolerance_ms:.3f} ms). Convert to CFR before use."
        )
    return 1.0 / median_delta


def _probe_video_timing(path: Path, require_cfr: bool, cfr_tolerance_ms: float) -> tuple[int, float, int, int]:
    data = _run_ffprobe(path)
    streams = data.get("streams", []) or []
    if not streams:
        raise VideoTimingError(f"No video stream found in {path}")
    stream = streams[0]
    timestamps = _frame_timestamps(data)
    timestamp_fps = _validate_cfr_timestamps(path, timestamps, cfr_tolerance_ms)

    avg_fps = _parse_rate(stream.get("avg_frame_rate"))
    nominal_fps = _parse_rate(stream.get("r_frame_rate"))
    if require_cfr and timestamp_fps is None:
        if avg_fps is not None and nominal_fps is not None:
            rate_delta_ms = abs((1.0 / avg_fps) - (1.0 / nominal_fps)) * 1000.0
            if rate_delta_ms > cfr_tolerance_ms:
                raise VideoTimingError(
                    f"Variable-frame-rate video detected from stream rates: {path}. "
                    f"avg_frame_rate={stream.get('avg_frame_rate')}, "
                    f"r_frame_rate={stream.get('r_frame_rate')}."
                )
        raise VideoTimingError(
            f"Could not verify constant-frame-rate timestamps for {path}; refusing to continue silently."
        )

    fps = avg_fps or nominal_fps or timestamp_fps
    if fps is None:
        raise VideoTimingError(f"Could not determine FPS for {path}")

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if timestamps:
        frames = len(timestamps)
    elif stream.get("nb_frames") not in (None, "N/A"):
        frames = int(stream["nb_frames"])
    else:
        frames = 0
    return frames, fps, width, height


def probe_video(
    path: str | Path,
    require_cfr: bool = True,
    cfr_tolerance_ms: float = 0.5,
) -> tuple[int, float, int, int]:
    path = Path(path)
    if require_cfr:
        frames, fps, width, height = _probe_video_timing(path, require_cfr, cfr_tolerance_ms)
        if frames > 0 and width > 0 and height > 0:
            return frames, fps, width, height

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, fps, width, height


def _letterbox_frame(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize with preserved aspect ratio and center-pad to [height, width]."""
    target_width, target_height = image_size
    src_height, src_width = frame.shape[:2]
    if src_width <= 0 or src_height <= 0:
        return np.zeros((target_height, target_width, 3), dtype=frame.dtype)

    scale = min(target_width / src_width, target_height / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    padded = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    padded[top : top + new_height, left : left + new_width] = resized
    return padded


def read_video_window(
    path: str | Path,
    start: int,
    length: int,
    image_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    width, height = image_size
    frames = []
    last = None
    for _ in range(length):
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _letterbox_frame(frame, image_size)
            last = frame
        elif last is None:
            last = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(last)
    cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
