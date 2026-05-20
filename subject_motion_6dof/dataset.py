from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .motion_json import motion_json_to_dense_targets, load_motion_json
from .video import probe_video, read_video_window


class SubjectMotion6DoFVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        chunk_len: int = 128,
        image_size: tuple[int, int] = (512, 512),
        require_cfr: bool = True,
        cfr_tolerance_ms: float = 0.5,
    ):
        self.data_root = Path(data_root)
        self.chunk_len = int(chunk_len)
        if self.chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        self.image_size = image_size
        self.require_cfr = bool(require_cfr)
        self.cfr_tolerance_ms = float(cfr_tolerance_ms)
        self.items = []
        self._build_index()

    def _build_index(self) -> None:
        videos = sorted(
            list(self.data_root.rglob("*.mp4"))
            + list(self.data_root.rglob("*.mov"))
            + list(self.data_root.rglob("*.mkv"))
        )
        for video_path in videos:
            ann_path = video_path.with_suffix(".json")
            if not ann_path.exists():
                continue
            num_frames, fps, _, _ = probe_video(
                video_path,
                require_cfr=self.require_cfr,
                cfr_tolerance_ms=self.cfr_tolerance_ms,
            )
            if num_frames <= 0:
                continue
            annotation = load_motion_json(ann_path)
            target, mask = motion_json_to_dense_targets(annotation, num_frames, fps)
            if not mask.any():
                continue

            starts = list(range(0, num_frames, self.chunk_len))
            for start in starts:
                end = min(start + self.chunk_len, num_frames)
                self.items.append(
                    {
                        "video": video_path,
                        "fps": fps,
                        "num_frames": num_frames,
                        "start": int(start),
                        "length": int(end - start),
                        "target": target,
                        "mask": mask,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        start = item["start"]
        length = item["length"]
        width, height = self.image_size
        frames = torch.zeros((self.chunk_len, 3, height, width), dtype=torch.float32)
        if length > 0:
            frames[:length] = read_video_window(
                item["video"], start, length, self.image_size
            )
        target = np.zeros((self.chunk_len, 6), dtype=np.float32)
        mask = np.zeros((self.chunk_len, 6), dtype=bool)
        target[:length] = item["target"][start : start + length]
        mask[:length] = item["mask"][start : start + length]
        return {
            "frames": frames,
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "length": torch.tensor(length, dtype=torch.long),
            "meta": {
                "video": str(item["video"]),
                "start": start,
                "num_frames": item["num_frames"],
                "fps": item["fps"],
            },
        }
