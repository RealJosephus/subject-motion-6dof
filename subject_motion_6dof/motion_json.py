from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHANNELS = ("x", "y", "z", "roll", "pitch", "yaw")


def frame_times_ms(num_frames: int, fps: float) -> np.ndarray:
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return np.arange(num_frames, dtype=np.float64) * 1000.0 / fps


def frame_index_to_ms(frame_idx: int, fps: float) -> int:
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return int(np.floor(float(frame_idx) * 1000.0 / fps + 0.5 + 1e-9))


def load_motion_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _actions_for_channel(annotation: dict, channel: str) -> list[dict]:
    channels = annotation.get("channels", {}) or {}
    return (channels.get(channel, {}) or {}).get("actions", []) or []


def _action_arrays(actions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for item in actions or []:
        if "at" not in item or "pos" not in item:
            continue
        at = float(item["at"])
        pos = np.clip(float(item["pos"]), 0.0, 100.0)
        points.append((at, pos))
    if not points:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)

    points.sort(key=lambda item: item[0])
    times = []
    values = []
    for at, pos in points:
        if times and at == times[-1]:
            values[-1] = pos
        else:
            times.append(at)
            values.append(pos)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def motion_json_to_dense_targets(
    annotation: dict,
    num_frames: int,
    fps: float,
    value_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((num_frames, len(CHANNELS)), dtype=np.float32)
    mask = np.zeros((num_frames, len(CHANNELS)), dtype=bool)
    if num_frames <= 0:
        return target, mask

    frame_times = frame_times_ms(num_frames, fps)
    for idx, name in enumerate(CHANNELS):
        times, values = _action_arrays(_actions_for_channel(annotation, name))
        if len(times) == 0:
            continue
        elif len(times) == 1:
            dense = np.full(num_frames, values[0], dtype=np.float32)
        else:
            dense = np.interp(frame_times, times, values).astype(np.float32)
        target[:, idx] = dense
        mask[:, idx] = True

    return target / value_scale * 100.0, mask


def predictions_to_motion_json(
    pred: np.ndarray,
    fps: float,
    threshold_delta: float = 0.0,
) -> dict:
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.clip(np.rint(pred), 0, 100).astype(np.int32)

    def actions_for_channel(values: np.ndarray) -> list[dict]:
        actions = []
        last_pos = None
        for frame_idx, pos in enumerate(values.tolist()):
            if last_pos is not None and abs(pos - last_pos) < threshold_delta:
                continue
            actions.append({"at": frame_index_to_ms(frame_idx, fps), "pos": int(pos)})
            last_pos = pos
        return actions

    out = {"channels": {}}
    for idx, name in enumerate(CHANNELS):
        out["channels"][name] = {"actions": actions_for_channel(pred[:, idx])}
    return out


def save_motion_json(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
