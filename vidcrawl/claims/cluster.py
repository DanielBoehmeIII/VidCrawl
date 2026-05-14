import json
from typing import Any

from vidcrawl.db import get_db
from vidcrawl.claims.normalize import normalize_claim


def detect_contradictions(db_path: str) -> list[dict]:
    conn = get_db(db_path)
    try:
        claims = conn.execute(
            "SELECT claim_id, claim_text, claim_type, source_moment_id FROM claims"
        ).fetchall()
        contradictions: list[dict] = []
        neg_prefixes = ["do not ", "dont ", "never ", "avoid ", "stop "]
        pos_prefixes = ["do ", "always ", "use ", "recommend "]

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a = claims[i]
                b = claims[j]
                ta = a["claim_text"].lower()
                tb = b["claim_text"].lower()
                is_neg_a = any(ta.startswith(p) for p in neg_prefixes)
                is_neg_b = any(tb.startswith(p) for p in neg_prefixes)
                is_pos_a = any(ta.startswith(p) for p in pos_prefixes)
                is_pos_b = any(tb.startswith(p) for p in pos_prefixes)
                if is_neg_a and is_pos_b:
                    contradictions.append({
                        "type": "contradiction",
                        "claim_a": a["claim_id"],
                        "claim_b": b["claim_id"],
                        "reason": "negative vs positive directive",
                        "text_a": a["claim_text"][:100],
                        "text_b": b["claim_text"][:100],
                    })
                elif is_pos_a and is_neg_b:
                    contradictions.append({
                        "type": "contradiction",
                        "claim_a": a["claim_id"],
                        "claim_b": b["claim_id"],
                        "reason": "positive vs negative directive",
                        "text_a": a["claim_text"][:100],
                        "text_b": b["claim_text"][:100],
                    })

        conn.commit()
        return contradictions
    finally:
        conn.close()
