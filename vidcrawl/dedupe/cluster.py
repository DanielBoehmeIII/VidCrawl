import json
import uuid

from vidcrawl.dedupe.normalize import content_hash, get_combined_text, normalize_text
from vidcrawl.dedupe.novelty import score_novelty
from vidcrawl.dedupe.similarity import combined_similarity
from vidcrawl.models import Duplicate


def run_dedupe(conn, threshold: float = 0.75, dry_run: bool = False) -> dict:
    stats = {
        "total_before": 0,
        "total_after": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "same_idea": 0,
        "variants": 0,
        "unique_moments": 0,
    }

    moments = conn.execute(
        """SELECT m.rowid, m.moment_id, m.video_id, m.transcript_text,
                  m.ocr_text, m.ideas, m.content_hash, m.created_at,
                  v.title as video_title
           FROM moments m
           JOIN videos v ON m.video_id = v.video_id
           ORDER BY m.created_at"""
    ).fetchall()
    stats["total_before"] = len(moments)

    if len(moments) < 2:
        stats["total_after"] = len(moments)
        return stats

    hash_groups = {}
    for m in moments:
        ideas_text = _get_ideas_text(m)
        combined = get_combined_text(
            m["transcript_text"], m["ocr_text"], ideas_text
        )
        h = content_hash(combined)
        hash_groups.setdefault(h, []).append(m)

    inserted_ids = set()

    for h, group in hash_groups.items():
        if len(group) > 1:
            canonical = _select_canonical(group)
            for m in group:
                if m["moment_id"] != canonical["moment_id"]:
                    if not dry_run:
                        dup = Duplicate(
                            dup_id=f"dup:{uuid.uuid4().hex[:12]}",
                            moment_id=m["moment_id"],
                            canonical_moment_id=canonical["moment_id"],
                            similarity_score=1.0,
                            novelty_score=0.0,
                            method="exact_hash",
                            duplicate_type="exact",
                            item_type="moment",
                            reason="Exact text match after normalization",
                        )
                        conn.execute(
                            """INSERT INTO duplicates
                               (dup_id, moment_id, canonical_moment_id,
                                similarity_score, novelty_score, method,
                                duplicate_type, item_type, reason)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                dup.dup_id,
                                dup.moment_id,
                                dup.canonical_moment_id,
                                dup.similarity_score,
                                dup.novelty_score,
                                dup.method,
                                dup.duplicate_type,
                                dup.item_type,
                                dup.reason,
                            ),
                        )
                    inserted_ids.add(m["moment_id"])
                    stats["exact_duplicates"] += 1

    canonicals = []
    for h, group in hash_groups.items():
        canonicals.append(_select_canonical(group))

    for i in range(len(canonicals)):
        a = canonicals[i]
        a_ideas_text = _get_ideas_text(a)
        a_ideas = json.loads(a["ideas"]) if a["ideas"] else []
        a_idea_types = [idea.get("type", "") for idea in a_ideas]

        for j in range(i + 1, len(canonicals)):
            b = canonicals[j]
            b_ideas_text = _get_ideas_text(b)
            b_ideas = json.loads(b["ideas"]) if b["ideas"] else []
            b_idea_types = [idea.get("type", "") for idea in b_ideas]

            text_sim = combined_similarity(
                a["transcript_text"], b["transcript_text"]
            )
            if text_sim < threshold:
                continue

            novelty = score_novelty(
                candidate_text=b["transcript_text"],
                canonical_text=a["transcript_text"],
                candidate_ocr=b["ocr_text"],
                canonical_ocr=a["ocr_text"],
                candidate_idea_types=b_idea_types,
                canonical_idea_types=a_idea_types,
                candidate_ideas_text=b_ideas_text,
                canonical_ideas_text=a_ideas_text,
            )

            if novelty["novelty_score"] < 0.2:
                dup_type = "near_text"
            elif novelty["novelty_score"] < 0.5:
                dup_type = "same_idea"
            else:
                dup_type = "variant"

            qual_a = _canonical_quality(a)
            qual_b = _canonical_quality(b)
            canon = a if qual_a >= qual_b else b
            dup = b if canon["moment_id"] == a["moment_id"] else a

            if dup["moment_id"] not in inserted_ids:
                if not dry_run:
                    dup_rec = Duplicate(
                        dup_id=f"dup:{uuid.uuid4().hex[:12]}",
                        moment_id=dup["moment_id"],
                        canonical_moment_id=canon["moment_id"],
                        similarity_score=round(text_sim, 4),
                        novelty_score=novelty["novelty_score"],
                        method="near_text",
                        duplicate_type=dup_type,
                        item_type="moment",
                        reason=novelty["reason"],
                    )
                    conn.execute(
                        """INSERT INTO duplicates
                           (dup_id, moment_id, canonical_moment_id,
                            similarity_score, novelty_score, method,
                            duplicate_type, item_type, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            dup_rec.dup_id,
                            dup_rec.moment_id,
                            dup_rec.canonical_moment_id,
                            dup_rec.similarity_score,
                            dup_rec.novelty_score,
                            dup_rec.method,
                            dup_rec.duplicate_type,
                            dup_rec.item_type,
                            dup_rec.reason,
                        ),
                    )
                inserted_ids.add(dup["moment_id"])
                if dup_type == "near_text":
                    stats["near_duplicates"] += 1
                elif dup_type == "same_idea":
                    stats["same_idea"] += 1
                else:
                    stats["variants"] += 1

    if not dry_run:
        conn.commit()

    stats["total_after"] = stats["total_before"] - stats["exact_duplicates"]
    stats["unique_moments"] = stats["total_before"] - len(inserted_ids)
    return stats


def _get_ideas_text(m) -> str:
    if not m["ideas"]:
        return ""
    ideas_list = json.loads(m["ideas"])
    parts = []
    for idea in ideas_list:
        parts.append(f"{idea.get('type', '')}: {idea.get('text', '')}")
    return " ".join(parts)


def _select_canonical(group: list) -> dict:
    return max(group, key=_canonical_quality)


def _canonical_quality(m) -> float:
    score = 0.0
    if m["transcript_text"] and m["transcript_text"].strip():
        score += 2.0
    if m["ocr_text"] and m["ocr_text"].strip():
        score += 1.5
    ideas = json.loads(m["ideas"]) if m["ideas"] else []
    if ideas:
        score += 1.0
    text_len = len((m["transcript_text"] or "").strip())
    if 20 <= text_len <= 500:
        score += 0.5
    elif text_len > 500:
        score += 0.2
    return score
