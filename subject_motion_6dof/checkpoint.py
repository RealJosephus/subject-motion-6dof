from __future__ import annotations

from pathlib import Path

import torch

SKIP_PREFIXES = (
    "head_pose",
    "head_pose_hand",
    "head_camera",
    "head_camera_hand",
    "decoder_hand",
    "init_pose_hand",
    "init_camera_hand",
    "init_to_token_mhr_hand",
    "prev_to_token_mhr_hand",
    "ray_cond_emb_hand",
    "prompt_to_token_hand",
    "keypoint_embedding_hand",
    "keypoint3d_embedding_hand",
    "keypoint_posemb_linear",
    "keypoint_posemb_linear_hand",
    "keypoint_feat_linear",
    "keypoint_feat_linear_hand",
    "keypoint3d_posemb_linear",
    "keypoint3d_posemb_linear_hand",
    "hand_",
    "bbox_embed",
)


def normalize_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    state = raw.get("state_dict", raw)
    out = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        for prefix in ("module.", "model.", "sam3d_body.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def filter_sam3d_weights(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    kept = {}
    skipped = {"prefix": [], "missing": [], "shape": []}
    for key, value in source_state.items():
        if key.startswith(SKIP_PREFIXES):
            skipped["prefix"].append(key)
            continue
        if key not in target_state:
            skipped["missing"].append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped["shape"].append(key)
            continue
        kept[key] = value
    return kept, skipped


def load_subject_motion_6dof_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = False) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "weights.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = normalize_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}
