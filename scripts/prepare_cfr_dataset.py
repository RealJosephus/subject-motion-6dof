from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def find_videos(input_root: Path, recursive: bool) -> list[Path]:
    paths = input_root.rglob("*") if recursive else input_root.iterdir()
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    fps: float,
    size: int,
    encoder: str,
    cq: int,
    preset: str,
    hwaccel: str,
    keep_audio: bool,
    overwrite: bool,
) -> list[str]:
    vf = (
        f"fps=fps={fps},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    if hwaccel != "none":
        cmd.extend(["-hwaccel", hwaccel])
    cmd.extend(["-i", str(src), "-vf", vf, "-c:v", encoder])

    if encoder.endswith("_nvenc"):
        cmd.extend(["-preset", preset, "-cq:v", str(cq), "-b:v", "0"])
    elif encoder == "libx264":
        cmd.extend(["-preset", preset, "-crf", str(cq)])

    cmd.extend(["-pix_fmt", "yuv420p"])
    if keep_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(dst)])
    return cmd


def run_cmd(cmd: list[str], dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        return True, ""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr.strip() or proc.stdout.strip())


def output_paths(video: Path, input_root: Path, output_root: Path) -> tuple[Path, Path]:
    rel = video.relative_to(input_root)
    out_video = (output_root / rel).with_suffix(".mp4")
    out_json = out_video.with_suffix(".json")
    return out_video, out_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch convert videos to 512x512 aspect-preserving padded CFR MP4s "
            "and copy same-name .json files."
        )
    )
    parser.add_argument("--input_root", required=True, help="Source dataset directory.")
    parser.add_argument("--output_root", required=True, help="Destination dataset directory.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target constant frame rate.")
    parser.add_argument("--size", type=int, default=512, help="Square output size.")
    parser.add_argument(
        "--encoder",
        default="h264_nvenc",
        choices=["h264_nvenc", "hevc_nvenc", "libx264"],
        help="Video encoder. h264_nvenc uses NVIDIA hardware encoding.",
    )
    parser.add_argument(
        "--hwaccel",
        default="none",
        choices=["none", "auto", "cuda"],
        help=(
            "Optional hardware decoder acceleration. Encoding uses --encoder; "
            "leave this as none if a CPU filter graph is more stable."
        ),
    )
    parser.add_argument("--preset", default="p4", help="Encoder preset, e.g. p4 for NVENC.")
    parser.add_argument("--cq", type=int, default=19, help="NVENC CQ or libx264 CRF value.")
    parser.add_argument("--keep_audio", action="store_true", help="Keep/transcode audio.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print ffmpeg commands only.")
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only scan input_root directly instead of recursively.",
    )
    parser.add_argument(
        "--allow_missing_json",
        action="store_true",
        help="Convert videos even when the same-name .json is missing.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("input_root and output_root must be different directories.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH.")

    videos = find_videos(input_root, recursive=not args.no_recursive)
    if not videos:
        raise RuntimeError(f"No videos found under {input_root}")

    converted = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for video in tqdm(videos, desc="Preparing videos"):
        annotation = video.with_suffix(".json")
        if not annotation.exists() and not args.allow_missing_json:
            skipped += 1
            continue

        out_video, out_json = output_paths(video, input_root, output_root)
        if (
            not args.overwrite
            and out_video.exists()
            and (out_json.exists() or not annotation.exists())
        ):
            skipped += 1
            continue

        out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(
            src=video,
            dst=out_video,
            fps=args.fps,
            size=args.size,
            encoder=args.encoder,
            cq=args.cq,
            preset=args.preset,
            hwaccel=args.hwaccel,
            keep_audio=args.keep_audio,
            overwrite=args.overwrite,
        )
        ok, message = run_cmd(cmd, dry_run=args.dry_run)
        if not ok:
            failures.append((video, message))
            continue

        if annotation.exists() and not args.dry_run:
            shutil.copy2(annotation, out_json)
        converted += 1

    print(
        f"Done. converted={converted}, skipped={skipped}, "
        f"failed={len(failures)}, output_root={output_root}"
    )
    if failures:
        for path, message in failures[:20]:
            print(f"\nFAILED: {path}\n{message}", file=sys.stderr)
        if len(failures) > 20:
            print(f"\n... {len(failures) - 20} more failures omitted", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
