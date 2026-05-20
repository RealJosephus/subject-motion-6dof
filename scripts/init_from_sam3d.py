from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subject_motion_6dof.checkpoint import filter_sam3d_weights, normalize_state_dict
from subject_motion_6dof.config import load_config
from subject_motion_6dof.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subject_motion_6dof.yaml")
    parser.add_argument("--source_ckpt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    target_state = model.state_dict()

    raw = torch.load(args.source_ckpt, map_location="cpu", weights_only=False)
    source_state = normalize_state_dict(raw)
    filtered, skipped = filter_sam3d_weights(source_state, target_state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    output = {
        "state_dict": model.state_dict(),
        "meta": {
            "source_ckpt": str(Path(args.source_ckpt).resolve()),
            "loaded_keys": len(filtered),
            "missing_after_load": list(missing),
            "unexpected_after_load": list(unexpected),
            "skipped_counts": {k: len(v) for k, v in skipped.items()},
            "note": "MHR/camera/hand decoder/head weights are intentionally dropped.",
        },
    }
    Path(args.output_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_ckpt)
    print(f"Saved initialized checkpoint: {args.output_ckpt}")
    print(f"Loaded keys: {len(filtered)}")
    print(f"Skipped counts: {output['meta']['skipped_counts']}")
    print(f"Missing target keys: {len(missing)}")


if __name__ == "__main__":
    main()
