import json
import re
from pathlib import Path
from typing import Optional


def load_sidecar_transcript(video_path: str) -> Optional[list[dict]]:
    base = Path(video_path)
    for ext in [".vtt", ".srt", ".txt", ".json"]:
        transcript_path = base.with_suffix(ext)
        if transcript_path.exists():
            return _parse_transcript_file(str(transcript_path), ext)
    return None


def _parse_transcript_file(path: str, ext: str) -> list[dict]:
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        return []
    if ext == ".srt":
        return _parse_srt(text)
    if ext == ".vtt":
        return _parse_vtt(text)
    if ext == ".txt":
        return _parse_txt(text)
    return []


def _parse_srt(text: str) -> list[dict]:
    entries = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        time_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line.strip()
            elif re.match(r"^\d+$", line.strip()):
                continue
            else:
                text_lines.append(line.strip())
        if time_line is None or not text_lines:
            continue
        start_str, end_str = time_line.split("-->")
        start_sec = _parse_srt_timestamp(start_str.strip())
        end_sec = _parse_srt_timestamp(end_str.strip())
        entries.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": " ".join(text_lines),
        })
    return entries


def _parse_vtt(text: str) -> list[dict]:
    entries = []
    if text.startswith("\ufeff"):
        text = text[1:]
    text = re.sub(r"^WEBVTT.*?(?=\n\S)", "", text, flags=re.DOTALL).strip()
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        time_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line.strip()
            elif line.startswith("NOTE ") or line.startswith("NOTE\n"):
                text_lines = []
                break
            elif line.startswith("Kind:") or line.startswith("Language:"):
                continue
            else:
                text_lines.append(line.strip())
        if time_line is None or not text_lines:
            continue
        start_str, end_str = time_line.split("-->")
        start_sec = _parse_vtt_timestamp(start_str.strip())
        end_sec = _parse_vtt_timestamp(end_str.strip())
        entries.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": " ".join(text_lines),
        })
    return entries


def _parse_txt(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    import math
    total_chars = sum(len(p) for p in paragraphs)
    estimated_duration = max(total_chars / 15.0, 30.0)
    sec_per_char = estimated_duration / total_chars if total_chars > 0 else 0.1
    entries = []
    current_time = 0.0
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            duration = max(len(sentence) * sec_per_char, 1.0)
            entries.append({
                "start_sec": current_time,
                "end_sec": current_time + duration,
                "text": sentence,
            })
            current_time += duration
    return entries


def _parse_srt_timestamp(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def _parse_vtt_timestamp(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def transcribe_audio(audio_path: str, model_name: str = "base") -> list[dict]:
    import shutil
    whisper_bin = shutil.which("whisper")
    if not whisper_bin:
        import warnings
        warnings.warn(
            "Whisper not installed. Install with: pip install openai-whisper. "
            "Skipping ASR transcription."
        )
        return []
    try:
        import whisper
    except ImportError:
        import warnings
        warnings.warn(
            "openai-whisper is not installed. "
            "Install with: pip install openai-whisper. "
            "Skipping ASR transcription."
        )
        return []

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start_sec": seg["start"],
            "end_sec": seg["end"],
            "text": seg["text"].strip(),
        })
    return segments
