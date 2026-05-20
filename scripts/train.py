from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subject_motion_6dof.checkpoint import load_subject_motion_6dof_checkpoint
from subject_motion_6dof.config import load_config
from subject_motion_6dof.dataset import SubjectMotion6DoFVideoDataset
from subject_motion_6dof.losses import masked_motion_loss
from subject_motion_6dof.model import build_model

WEIGHTS_FILENAME = "weights.pt"
STATE_FILENAME = "state.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_dtype(cfg) -> torch.dtype:
    fp16_type = str(cfg.TRAIN.get("FP16_TYPE", "bfloat16")).lower()
    if fp16_type in ("bf16", "bfloat16"):
        return torch.bfloat16
    if fp16_type in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def make_optimizer(model, cfg):
    groups = []
    if not cfg.TRAINING.get("FREEZE_BACKBONE", True):
        groups.append(
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": float(cfg.OPTIM.BACKBONE_LR),
                "name": "backbone",
            }
        )
    groups.append(
        {
            "params": [p for p in model.decoder.parameters() if p.requires_grad],
            "lr": float(cfg.OPTIM.SAM_DECODER_LR),
            "name": "sam_decoder",
        }
    )
    head_params = list(model.temporal_head.parameters())
    groups.append(
        {
            "params": head_params,
            "lr": float(cfg.OPTIM.TEMPORAL_HEAD_LR),
            "name": "temporal_head",
        }
    )
    extra = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("backbone.", "decoder.", "temporal_head.")):
            continue
        extra.append(param)
    if extra:
        groups.append(
            {"params": extra, "lr": float(cfg.OPTIM.SAM_DECODER_LR), "name": "sam_aux"}
        )
    return torch.optim.AdamW(groups, weight_decay=float(cfg.OPTIM.WEIGHT_DECAY))


def make_scheduler(optimizer: torch.optim.Optimizer, cfg, total_steps: int):
    optim_cfg = cfg.OPTIM
    scheduler_type = str(optim_cfg.get("SCHEDULER", "cosine")).lower()
    warmup_steps = int(optim_cfg.get("WARMUP_STEPS", 0))
    min_lr_ratio = float(optim_cfg.get("MIN_LR_RATIO", 0.0))

    if scheduler_type in ("none", "constant"):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
        if scheduler_type == "linear":
            factor = 1.0 - progress
        elif scheduler_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loader(dataset, cfg, shuffle: bool) -> DataLoader:
    num_workers = int(cfg.TRAINING.NUM_WORKERS)
    return DataLoader(
        dataset,
        batch_size=int(cfg.TRAINING.BATCH_SIZE),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def make_dataset(data_root: str | Path, cfg) -> SubjectMotion6DoFVideoDataset:
    return SubjectMotion6DoFVideoDataset(
        data_root,
        chunk_len=cfg.DATA.CHUNK_LEN,
        image_size=tuple(cfg.MODEL.IMAGE_SIZE),
        require_cfr=bool(cfg.DATA.get("REQUIRE_CFR", True)),
        cfr_tolerance_ms=float(cfg.DATA.get("CFR_TOLERANCE_MS", 0.5)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    frames = batch["frames"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    lengths = batch["length"].to(device, non_blocking=True)
    return frames, target, mask, lengths


def meta_at(meta: dict[str, Any], key: str, idx: int) -> Any:
    value = meta[key]
    if torch.is_tensor(value):
        return value[idx].item()
    return value[idx]


def detach_action_state(
    action_state: list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    if action_state is None:
        return None
    return [item.detach() for item in action_state]


def cached_stream_state(
    stream_states: dict[str, dict[str, Any]],
    video: str,
    start: int,
) -> dict[str, Any] | None:
    stream_state = stream_states.get(video)
    if start == 0 or stream_state is None:
        return None
    if int(stream_state.get("next_start", -1)) != start:
        return None
    return stream_state


def store_stream_state(
    stream_states: dict[str, dict[str, Any]],
    video: str,
    next_start: int,
    num_frames: int,
    out: dict,
) -> None:
    if next_start >= num_frames:
        stream_states.pop(video, None)
        return
    stream_states[video] = {
        "next_start": int(next_start),
        "prev_actions": out["prev_actions"].detach(),
        "action_state": detach_action_state(out["action_state"]),
        "pose_token_buffer": out["pose_token_buffer"].detach(),
    }


def train_step(
    model,
    batch,
    cfg,
    device: torch.device,
    amp_dtype: torch.dtype,
    stream_states: dict[str, dict[str, Any]] | None = None,
) -> torch.Tensor:
    frames, target, mask, lengths = move_batch(batch, device)
    amp_enabled = bool(cfg.TRAIN.get("USE_FP16", True)) and device.type == "cuda"
    stream_states = stream_states if stream_states is not None else {}
    preds = []

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        for sample_idx in range(frames.shape[0]):
            video = str(meta_at(batch["meta"], "video", sample_idx))
            start = int(meta_at(batch["meta"], "start", sample_idx))
            num_frames = int(meta_at(batch["meta"], "num_frames", sample_idx))
            cached = cached_stream_state(stream_states, video, start)
            out = model(
                frames[sample_idx : sample_idx + 1],
                lengths=lengths[sample_idx : sample_idx + 1],
                initial_actions=None if cached is None else cached["prev_actions"],
                action_state=None if cached is None else cached["action_state"],
                pose_token_buffer=None if cached is None else cached["pose_token_buffer"],
            )
            preds.append(out["pred_actions"])
            next_start = start + int(lengths[sample_idx].detach().cpu())
            store_stream_state(stream_states, video, next_start, num_frames, out)

        pred_actions = torch.cat(preds, dim=0)
        return masked_motion_loss(
            pred_actions,
            target,
            mask,
            loss_type=cfg.TRAINING.LOSS_TYPE,
        )


@torch.no_grad()
def evaluate(model, loader, cfg, device: torch.device, amp_dtype: torch.dtype, max_batches: int) -> float:
    model.eval()
    losses = []
    stream_states: dict[str, dict[str, Any]] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        loss = train_step(model, batch, cfg, device, amp_dtype, stream_states)
        losses.append(float(loss.detach().cpu()))
    model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def init_wandb(args, cfg, output_dir: Path):
    wandb_cfg = cfg.get("WANDB", {})
    enabled = bool(args.wandb or wandb_cfg.get("ENABLE", False))
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but not installed. Install it with `pip install wandb`.") from exc

    project = args.wandb_project or wandb_cfg.get("PROJECT", "subject-motion-6dof")
    name = args.wandb_name or wandb_cfg.get("NAME", None)
    mode = args.wandb_mode or wandb_cfg.get("MODE", None)
    return wandb.init(
        project=project,
        name=name,
        mode=mode,
        dir=str(output_dir),
        config=cfg.to_dict(),
    )


def checkpoint_root(output_dir: Path) -> Path:
    return output_dir


def checkpoint_dir_for_step(output_dir: Path, step: int) -> Path:
    return checkpoint_root(output_dir) / f"checkpoint-{step}"


def step_from_checkpoint_dir(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return None


def find_latest_checkpoint_dir(output_dir: Path) -> Path | None:
    root = checkpoint_root(output_dir)
    if not root.exists():
        return None
    candidates = []
    for path in root.iterdir():
        step = step_from_checkpoint_dir(path)
        if step is not None and (path / STATE_FILENAME).exists() and (path / WEIGHTS_FILENAME).exists():
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_resume_path(args, output_dir: Path) -> Path | None:
    if args.resume:
        return Path(args.resume)
    return find_latest_checkpoint_dir(output_dir)


def checkpoint_files(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / WEIGHTS_FILENAME, path / STATE_FILENAME
    if path.name == WEIGHTS_FILENAME:
        return path, path.with_name(STATE_FILENAME)
    if path.name == STATE_FILENAME:
        return path.with_name(WEIGHTS_FILENAME), path
    return path, path


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    weights_path, state_path = checkpoint_files(path)
    weights = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(weights.get("state_dict", weights), strict=False)

    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state.get("rng_state"))
    return int(state.get("step", weights.get("step", 0)))


def prune_old_checkpoints(output_dir: Path, keep: int) -> None:
    keep = max(int(keep), 1)
    root = checkpoint_root(output_dir)
    if not root.exists():
        return
    checkpoints = []
    for path in root.iterdir():
        step = step_from_checkpoint_dir(path)
        if step is not None:
            checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    for _, path in checkpoints[keep:]:
        shutil.rmtree(path)


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg,
    step: int,
) -> Path:
    root = checkpoint_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_dir_for_step(output_dir, step)
    tmp_dir = root / f".tmp_checkpoint-{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "checkpoint_format": "subject_motion_6dof_weights_v1",
        },
        tmp_dir / WEIGHTS_FILENAME,
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "rng_state": capture_rng_state(),
            "weights_file": WEIGHTS_FILENAME,
            "checkpoint_format": "subject_motion_6dof_train_state_v1",
        },
        tmp_dir / STATE_FILENAME,
    )

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    prune_old_checkpoints(output_dir, keep=int(cfg.TRAINING.get("KEEP_CHECKPOINTS", 1)))
    return final_dir


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    out = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        out[f"lr/{name}"] = float(group["lr"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subject_motion_6dof.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--val_data_root", default=None)
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        help="Model weights to load before training, usually produced by scripts/init_from_sam3d.py.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional explicit checkpoint-{step} dir/state.pt/weights.pt. By default, the latest checkpoint under output_dir is resumed automatically.",
    )
    parser.add_argument("--output_dir", default="outputs/subject_motion_6dof")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    args = parser.parse_args()

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    cfg = load_config(args.config)
    set_seed(int(cfg.TRAINING.get("SEED", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(cfg.TRAINING.get("EPOCHS", 1))
    epochs = max(epochs, 1)
    grad_accum_steps = args.gradient_accumulation_steps or int(
        cfg.TRAINING.get("GRADIENT_ACCUMULATION_STEPS", 1)
    )
    grad_accum_steps = max(grad_accum_steps, 1)
    save_every = args.save_every or int(cfg.TRAINING.get("SAVE_EVERY", 2000))
    log_every = args.log_every or int(cfg.TRAINING.get("LOG_EVERY", 20))
    amp_dtype = autocast_dtype(cfg)
    resume_path = resolve_resume_path(args, output_dir)
    pretrained_ckpt = args.pretrained_ckpt
    step = 0

    dataset = make_dataset(args.data_root, cfg)
    if len(dataset) == 0:
        raise RuntimeError("No labeled chunks found.")
    loader = make_loader(dataset, cfg, shuffle=False)
    micro_batches_per_epoch = len(loader)
    if micro_batches_per_epoch == 0:
        raise RuntimeError(
            "No training batches produced. Reduce TRAINING.BATCH_SIZE or add more labeled chunks."
        )
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    scheduler = make_scheduler(optimizer, cfg, total_steps)

    if resume_path is not None and resume_path.exists():
        step = load_training_checkpoint(resume_path, model, optimizer, scheduler, device)
        print(f"Resumed checkpoint: {resume_path} at step {step}")
    elif args.resume:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif pretrained_ckpt:
        info = load_subject_motion_6dof_checkpoint(model, pretrained_ckpt, strict=False)
        print(
            "Loaded pretrained checkpoint. "
            f"Missing={len(info['missing'])}, unexpected={len(info['unexpected'])}"
        )
    else:
        print(f"No checkpoint found under {output_dir}; starting from scratch.")

    print(
        "Training batch setup: "
        f"micro_batch={int(cfg.TRAINING.BATCH_SIZE)}, "
        f"grad_accum={grad_accum_steps}, "
        f"effective_batch={int(cfg.TRAINING.BATCH_SIZE) * grad_accum_steps} chunks, "
        f"effective_frames={int(cfg.TRAINING.BATCH_SIZE) * grad_accum_steps * int(cfg.DATA.CHUNK_LEN)}, "
        f"chunk_len={int(cfg.DATA.CHUNK_LEN)} new frames, "
        f"visual_max_len={int(cfg.DATA.MAX_LEN)} tokens, "
        "shuffle=False, "
        f"epochs={epochs}, "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch}, "
        f"total_optimizer_steps={total_steps}"
    )

    val_loader = None
    validate_every = int(cfg.TRAINING.get("VALIDATE_EVERY", 0))
    if args.val_data_root and validate_every > 0:
        val_dataset = make_dataset(args.val_data_root, cfg)
        if len(val_dataset) == 0:
            raise RuntimeError("No labeled validation chunks found.")
        val_loader = make_loader(val_dataset, cfg, shuffle=False)

    wandb_run = init_wandb(args, cfg, output_dir)

    model.train()
    pbar = tqdm(total=total_steps, initial=min(step, total_steps))
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    accum_target = grad_accum_steps
    accum_loss = 0.0
    start_epoch = min(step // optimizer_steps_per_epoch, epochs)
    for epoch in range(start_epoch, epochs):
        stream_states: dict[str, dict[str, Any]] = {}
        if step >= total_steps:
            break
        for batch_idx, batch in enumerate(loader):
            if step >= total_steps:
                break
            if accum_count == 0:
                remaining_micro_batches = micro_batches_per_epoch - batch_idx
                accum_target = min(grad_accum_steps, remaining_micro_batches)

            loss = train_step(model, batch, cfg, device, amp_dtype, stream_states)
            (loss / accum_target).backward()
            accum_count += 1
            accum_loss += float(loss.detach().cpu())

            if accum_count < accum_target:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.TRAINING.GRAD_CLIP)
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            loss_value = accum_loss / accum_count
            accum_count = 0
            accum_target = grad_accum_steps
            accum_loss = 0.0
            pbar.set_postfix(
                epoch=epoch + 1,
                loss=loss_value,
                lr=optimizer.param_groups[-1]["lr"],
            )

            if step % log_every == 0:
                logs = {
                    "train/loss": loss_value,
                    "train/grad_norm": float(grad_norm.detach().cpu()),
                    "train/epoch": epoch + 1,
                    "train/gradient_accumulation_steps": grad_accum_steps,
                    "step": step,
                }
                logs.update(current_lrs(optimizer))
                if wandb_run is not None:
                    wandb_run.log(logs, step=step)

            if val_loader is not None and step % validate_every == 0:
                val_loss = evaluate(
                    model,
                    val_loader,
                    cfg,
                    device,
                    amp_dtype,
                    max_batches=int(cfg.TRAINING.get("VALIDATION_MAX_BATCHES", 0)),
                )
                print(f"validation step={step} loss={val_loss:.6f}")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss, "step": step}, step=step)

            if save_every > 0 and step % save_every == 0:
                save_checkpoint(output_dir, model, optimizer, scheduler, cfg, step)
    pbar.close()
    save_checkpoint(output_dir, model, optimizer, scheduler, cfg, step)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
