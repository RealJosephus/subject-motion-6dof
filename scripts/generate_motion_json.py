from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subject_motion_6dof.checkpoint import load_subject_motion_6dof_checkpoint
from subject_motion_6dof.config import load_config
from subject_motion_6dof.motion_json import predictions_to_motion_json, save_motion_json
from subject_motion_6dof.model import build_model
from subject_motion_6dof.video import probe_video, read_video_window


def autocast_dtype(cfg) -> torch.dtype:
    fp16_type = str(cfg.TRAIN.get("FP16_TYPE", "bfloat16")).lower()
    if fp16_type in ("bf16", "bfloat16"):
        return torch.bfloat16
    if fp16_type in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subject_motion_6dof.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=None,
        help="Number of new frames to read and SAM-3D encode per batch.",
    )
    args = parser.parse_args()

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    load_subject_motion_6dof_checkpoint(model, args.ckpt, strict=False)
    amp_dtype = autocast_dtype(cfg)

    total_frames, fps, _, _ = probe_video(
        args.video,
        require_cfr=bool(cfg.DATA.get("REQUIRE_CFR", True)),
        cfr_tolerance_ms=float(cfg.DATA.get("CFR_TOLERANCE_MS", 0.5)),
    )
    encode_batch_size = max(args.encode_batch_size or int(cfg.DATA.CHUNK_LEN), 1)
    pred = np.zeros((total_frames, 6), dtype=np.float32)

    write_start = 0
    prev_actions = None
    action_state = None
    pose_token_buffer = None
    pbar = tqdm(total=total_frames)
    while write_start < total_frames:
        valid_len = min(encode_batch_size, total_frames - write_start)
        frames = read_video_window(
            args.video,
            write_start,
            valid_len,
            image_size=tuple(cfg.MODEL.IMAGE_SIZE),
        )[None].to(device)
        amp_enabled = bool(cfg.TRAIN.get("USE_FP16", True)) and device.type == "cuda"
        chunk_pred = []
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pose_tokens = model.encode_frames(frames)
            for local_idx in range(valid_len):
                next_pred, action_state, pose_token_buffer = model.predict_actions_step(
                    pose_tokens[:, local_idx],
                    prev_actions=prev_actions,
                    action_state=action_state,
                    pose_token_buffer=pose_token_buffer,
                )
                chunk_pred.append(next_pred[0].float())
                prev_actions = next_pred
        pred[write_start : write_start + valid_len] = torch.stack(chunk_pred).cpu().numpy()
        pbar.update(valid_len)
        write_start += valid_len
    pbar.close()

    data = predictions_to_motion_json(pred, fps=fps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_motion_json(data, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
