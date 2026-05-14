import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def extract_keyframes(
    video_path: str,
    output_dir: str,
    interval_sec: float = 30.0,
    video_duration: Optional[float] = None,
) -> list[dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import warnings
        warnings.warn(
            "ffmpeg not found. Install ffmpeg to enable keyframe extraction. "
            "Skipping keyframes."
        )
        return []

    if not Path(video_path).exists():
        import warnings
        warnings.warn(f"Video file not found: {video_path}")
        return []

    duration = video_duration
    if duration is None or duration <= 0:
        duration = _get_duration_ffprobe(video_path, ffmpeg)
    if duration is None or duration <= 0:
        import warnings
        warnings.warn(
            "Could not determine video duration. Skipping keyframe extraction."
        )
        return []

    timestamps = _compute_timestamps(duration, interval_sec)
    keyframes = []

    for ts in timestamps:
        filename = f"frame_{int(ts):06d}.jpg"
        output_path = out / filename
        if output_path.exists():
            keyframes.append({
                "timestamp_sec": ts,
                "path": str(output_path),
            })
            continue

        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and output_path.exists():
            keyframes.append({
                "timestamp_sec": ts,
                "path": str(output_path),
            })

    return keyframes


def _compute_timestamps(duration: float, interval_sec: float) -> list[float]:
    if duration <= 0 or interval_sec <= 0:
        return []
    count = max(1, int(math.ceil(duration / interval_sec)))
    return [min(i * interval_sec, duration) for i in range(count)]


def _get_duration_ffprobe(
    video_path: str, ffmpeg_bin: str
) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffprobe_dir = str(Path(ffmpeg_bin).parent)
        ffprobe = str(Path(ffprobe_dir) / "ffprobe")
        if not Path(ffprobe).exists():
            return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        pass
    return None
