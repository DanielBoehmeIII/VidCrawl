import re


def chunk_transcript(
    transcript_entries: list[dict],
    max_duration: float = 60.0,
    overlap_sec: float = 5.0,
) -> list[dict]:
    if not transcript_entries:
        return []

    entries = _normalize_entries(transcript_entries)
    chunks = []
    i = 0
    n = len(entries)

    while i < n:
        chunk_start = entries[i]["start_sec"]
        chunk_end = chunk_start
        accumulated_text = []
        j = i

        while j < n:
            candidate_end = entries[j]["end_sec"]
            candidate_text = entries[j]["text"]

            if candidate_end - chunk_start > max_duration:
                chunk_duration_candidate = candidate_end - chunk_start
                if chunk_end == chunk_start:
                    chunk_end = candidate_end
                    accumulated_text.append(candidate_text)
                    j += 1
                break
            else:
                chunk_end = candidate_end
                accumulated_text.append(candidate_text)
                j += 1

        text = " ".join(accumulated_text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            chunks.append({
                "start_sec": chunk_start,
                "end_sec": chunk_end,
                "transcript_text": text,
            })

        if j >= n:
            break

        overlap_start = chunk_end - overlap_sec
        i = j - 1
        while i >= 0 and entries[i]["start_sec"] > overlap_start:
            i -= 1
        if i < 0:
            i = 0
        while i < n and entries[i]["start_sec"] < overlap_start:
            i += 1

    if not chunks:
        return []

    merged = _merge_adjacent_chunks(chunks, max_duration)
    return merged


def _normalize_entries(entries: list[dict]) -> list[dict]:
    normalized = []
    for e in entries:
        text = re.sub(r"\s+", " ", e.get("text", "")).strip()
        if not text:
            continue
        normalized.append({
            "start_sec": float(e.get("start_sec", 0.0)),
            "end_sec": float(e.get("end_sec", 0.0)),
            "text": text,
        })
    normalized.sort(key=lambda x: x["start_sec"])
    return normalized


def _merge_adjacent_chunks(
    chunks: list[dict], max_duration: float
) -> list[dict]:
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        last = merged[-1]
        gap = chunk["start_sec"] - last["end_sec"]
        new_duration = chunk["end_sec"] - last["start_sec"]

        if gap < 1.0 and new_duration <= max_duration * 1.5:
            last["end_sec"] = chunk["end_sec"]
            last["transcript_text"] = (
                last["transcript_text"] + " " + chunk["transcript_text"]
            )
            last["transcript_text"] = re.sub(r"\s+", " ", last["transcript_text"]).strip()
        else:
            merged.append(chunk)

    return merged
