import re


def chunk_transcript(
    transcript_entries: list[dict],
    min_duration: float = 30.0,
    max_duration: float = 120.0,
) -> list[dict]:
    """Split transcript into non-overlapping semantic chunks of 30–120 seconds.

    Each entry is assigned to exactly one chunk (no overlap, no gaps).
    Splits prefer sentence-ending boundaries once min_duration is reached;
    hard-breaks at max_duration when no sentence boundary is available.
    """
    if not transcript_entries:
        return []

    entries = _normalize_entries(transcript_entries)
    if not entries:
        return []

    chunks = []
    i = 0
    n = len(entries)

    while i < n:
        chunk_start = entries[i]["start_sec"]
        last_sentence_j = None

        j = i
        while j < n:
            duration = entries[j]["end_sec"] - chunk_start
            if _is_sentence_end(entries[j]["text"]) and duration >= min_duration:
                last_sentence_j = j
            if duration > max_duration:
                if last_sentence_j is not None:
                    end_j = last_sentence_j
                else:
                    # No sentence boundary — split just before this entry, or
                    # force-include if it's the very first entry in the chunk.
                    end_j = j - 1 if j > i else j
                break
            j += 1
        else:
            end_j = n - 1

        texts = [entries[k]["text"] for k in range(i, end_j + 1)]
        text = re.sub(r"\s+", " ", " ".join(texts)).strip()
        if text:
            chunks.append({
                "start_sec": chunk_start,
                "end_sec": entries[end_j]["end_sec"],
                "transcript_text": text,
            })

        i = end_j + 1

    return _absorb_short_chunks(chunks, min_duration, max_duration)


def _normalize_entries(entries: list[dict]) -> list[dict]:
    normalized = []
    for e in entries:
        text = re.sub(r"\s+", " ", e.get("text", "")).strip()
        if not text:
            continue
        start = float(e.get("start_sec", 0.0))
        end = float(e.get("end_sec", 0.0))
        if end < start:
            end = start  # clamp: a broken parser must not produce inverted timestamps
        normalized.append({
            "start_sec": start,
            "end_sec": end,
            "text": text,
        })
    normalized.sort(key=lambda x: x["start_sec"])
    return normalized


def _is_sentence_end(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text.strip()))


def _absorb_short_chunks(
    chunks: list[dict], min_duration: float, max_duration: float
) -> list[dict]:
    """Merge trailing short chunks (< min_duration) into their predecessor."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for chunk in chunks[1:]:
        duration = chunk["end_sec"] - chunk["start_sec"]
        if duration < min_duration:
            last = result[-1]
            combined = chunk["end_sec"] - last["start_sec"]
            if combined <= max_duration:
                last["end_sec"] = chunk["end_sec"]
                last["transcript_text"] = re.sub(
                    r"\s+",
                    " ",
                    last["transcript_text"] + " " + chunk["transcript_text"],
                ).strip()
                continue
        result.append(chunk)
    return result
