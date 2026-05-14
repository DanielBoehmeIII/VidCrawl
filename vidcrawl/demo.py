import json
import uuid
from pathlib import Path
from typing import Optional

from vidcrawl.db import (
    get_db,
    init_db,
    insert_evidence,
    insert_idea,
    insert_keyframe,
    insert_moment,
    insert_video,
    make_idea_id,
    make_moment_id,
    make_video_id,
    rebuild_fts,
)
from vidcrawl.models import Evidence, Idea, Keyframe, Moment, Video


def create_demo_corpus(db_path: Path, data_dir: Optional[Path] = None) -> list[dict]:
    conn = get_db(str(db_path))
    init_db(conn)

    videos_data = _build_demo_videos()
    moments_data = _build_demo_moments(videos_data)

    video_records = []
    for v in videos_data:
        video = Video(**v)
        insert_video(conn, video)
        video_records.append(video)

    inserted_moments = []
    for md in moments_data:
        moment = Moment(
            moment_id=md["moment_id"],
            video_id=md["video_id"],
            start_sec=md["start_sec"],
            end_sec=md["end_sec"],
            transcript_text=md["transcript_text"],
            ocr_text=md.get("ocr_text", ""),
            ideas=md.get("ideas", []),
            keyframe_paths=md.get("keyframe_paths", []),
        )
        insert_moment(conn, moment)

        for idea in moment.ideas:
            insert_idea(conn, idea)

        insert_evidence(
            conn,
            Evidence(
                evidence_id=f"ev:{uuid.uuid4().hex[:12]}",
                moment_id=moment.moment_id,
                modality="transcript",
                content=moment.transcript_text,
            ),
        )
        if moment.ocr_text:
            insert_evidence(
                conn,
                Evidence(
                    evidence_id=f"ev:{uuid.uuid4().hex[:12]}",
                    moment_id=moment.moment_id,
                    modality="ocr",
                    content=moment.ocr_text,
                ),
            )
        for idea in moment.ideas:
            insert_evidence(
                conn,
                Evidence(
                    evidence_id=f"ev:{uuid.uuid4().hex[:12]}",
                    moment_id=moment.moment_id,
                    modality="idea",
                    content=f"[{idea.type}] {idea.text}",
                ),
            )

        inserted_moments.append(
            {
                "moment_id": moment.moment_id,
                "video_id": moment.video_id,
                "start_sec": moment.start_sec,
                "end_sec": moment.end_sec,
            }
        )

    conn.commit()
    rebuild_fts(conn)
    conn.commit()
    conn.close()

    return inserted_moments


def _build_demo_videos() -> list[dict]:
    return [
        {
            "video_id": "demo_coding",
            "title": "Building a Playwright MCP Server for Browser Automation",
            "source": "local",
            "url": "https://youtube.com/watch?v=demo_coding_001",
            "duration_sec": 600.0,
            "status": "ready",
            "metadata": {"description": "A tutorial on building a Playwright MCP server"},
        },
        {
            "video_id": "demo_ml",
            "title": "Transformer Model Architecture Deep Dive",
            "source": "local",
            "url": "https://youtube.com/watch?v=demo_ml_002",
            "duration_sec": 1200.0,
            "status": "ready",
            "metadata": {"description": "Conference talk on transformer architecture"},
        },
        {
            "video_id": "demo_ux",
            "title": "User Research Methods for Product Teams",
            "source": "local",
            "url": "https://youtube.com/watch?v=demo_ux_003",
            "duration_sec": 900.0,
            "status": "ready",
            "metadata": {"description": "Podcast on user research interview methods"},
        },
    ]


def _build_demo_moments(videos: list[dict]) -> list[dict]:
    vid_map = {v["video_id"]: v for v in videos}
    moments = []

    moments.extend(_coding_moments(vid_map["demo_coding"]))
    moments.extend(_ml_moments(vid_map["demo_ml"]))
    moments.extend(_ux_moments(vid_map["demo_ux"]))
    moments.extend(_duplicate_moments(vid_map["demo_coding"]))

    return moments


def _duplicate_moments(video: dict) -> list[dict]:
    vid = video["video_id"]
    return [
        {
            "moment_id": make_moment_id(vid, 70.0, 85.0),
            "video_id": vid,
            "start_sec": 70.0,
            "end_sec": 85.0,
            "transcript_text": "Welcome to this tutorial on building a Playwright MCP server. "
                "MCP stands for Model Context Protocol and it allows AI assistants to interact with browsers.",
            "ocr_text": "Playwright MCP Server Tutorial Introduction",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 70.0, 85.0), 0),
                    moment_id=make_moment_id(vid, 70.0, 85.0),
                    type="definition",
                    text="MCP is a protocol for AI to interact with browsers",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 85.0, 100.0),
            "video_id": vid,
            "start_sec": 85.0,
            "end_sec": 100.0,
            "transcript_text": "Welcome to this tutorial on building a Playwright MCP server. "
                "MCP stands for Model Context Protocol and it lets AI assistants interact with browsers directly.",
            "ocr_text": "Playwright MCP Server Introduction",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 85.0, 100.0), 0),
                    moment_id=make_moment_id(vid, 85.0, 100.0),
                    type="definition",
                    text="MCP lets AI interact with browsers",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 100.0, 115.0),
            "video_id": vid,
            "start_sec": 100.0,
            "end_sec": 115.0,
            "transcript_text": "Be careful when running headless browser tests in automated mode. "
                "Some websites can detect headless Chrome and block access. "
                "You should use proper browser contexts and user agents to avoid getting blocked.",
            "ocr_text": "Warning: Headless browser detection bypass",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 100.0, 115.0), 0),
                    moment_id=make_moment_id(vid, 100.0, 115.0),
                    type="warning",
                    text="Avoid headless browser detection with proper contexts",
                ),
            ],
            "keyframe_paths": [],
        },
    ]


def _coding_moments(video: dict) -> list[dict]:
    vid = video["video_id"]
    return [
        {
            "moment_id": make_moment_id(vid, 0.0, 12.0),
            "video_id": vid,
            "start_sec": 0.0,
            "end_sec": 12.0,
            "transcript_text": "Welcome to this tutorial on building a Playwright MCP server. "
                "MCP stands for Model Context Protocol and it allows AI assistants to interact with browsers.",
            "ocr_text": "Playwright MCP Server Tutorial Introduction",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 0.0, 12.0), 0),
                    moment_id=make_moment_id(vid, 0.0, 12.0),
                    type="definition",
                    text="MCP is a protocol for AI to interact with browsers",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 12.0, 25.0),
            "video_id": vid,
            "start_sec": 12.0,
            "end_sec": 25.0,
            "transcript_text": "First, install the Playwright package using npm. "
                "Run npm init and then npm install playwright. "
                "This will download the browser binaries automatically.",
            "ocr_text": "npm install playwright\nnpx playwright install",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 12.0, 25.0), 0),
                    moment_id=make_moment_id(vid, 12.0, 25.0),
                    type="step",
                    text="Install Playwright with npm install",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 25.0, 40.0),
            "video_id": vid,
            "start_sec": 25.0,
            "end_sec": 40.0,
            "transcript_text": "Be careful when running browser tests in headless mode. "
                "Some websites detect automated browsers. "
                "You should use proper browser contexts to avoid detection.",
            "ocr_text": "Warning: Headless detection",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 25.0, 40.0), 0),
                    moment_id=make_moment_id(vid, 25.0, 40.0),
                    type="warning",
                    text="Be careful with headless browser detection",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 40.0, 55.0),
            "video_id": vid,
            "start_sec": 40.0,
            "end_sec": 55.0,
            "transcript_text": "For example, you can use Playwright to navigate to a page, "
                "click elements, and extract text. "
                "This is useful for testing web applications automatically.",
            "ocr_text": "page.goto() page.click() page.textContent()",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 40.0, 55.0), 0),
                    moment_id=make_moment_id(vid, 40.0, 55.0),
                    type="example",
                    text="Use Playwright to navigate and extract text",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 55.0, 70.0),
            "video_id": vid,
            "start_sec": 55.0,
            "end_sec": 70.0,
            "transcript_text": "Let's compare Playwright with Selenium. "
                "Playwright is faster and has better auto-waiting. "
                "Selenium has broader language support.",
            "ocr_text": "Playwright vs Selenium Comparison",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 55.0, 70.0), 0),
                    moment_id=make_moment_id(vid, 55.0, 70.0),
                    type="comparison",
                    text="Playwright is faster than Selenium with better auto-waiting",
                ),
            ],
            "keyframe_paths": [],
        },
    ]


def _ml_moments(video: dict) -> list[dict]:
    vid = video["video_id"]
    return [
        {
            "moment_id": make_moment_id(vid, 0.0, 15.0),
            "video_id": vid,
            "start_sec": 0.0,
            "end_sec": 15.0,
            "transcript_text": "Today we will explore the transformer model architecture. "
                "Transformers revolutionized natural language processing with their attention mechanism. "
                "The key innovation is self-attention.",
            "ocr_text": "Transformer Architecture Attention Is All You Need",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 0.0, 15.0), 0),
                    moment_id=make_moment_id(vid, 0.0, 15.0),
                    type="definition",
                    text="Transformer is a model architecture based on self-attention",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 15.0, 30.0),
            "video_id": vid,
            "start_sec": 15.0,
            "end_sec": 30.0,
            "transcript_text": "The encoder processes input sequences using multi-head attention. "
                "Each head learns different relationships between words. "
                "The decoder generates output sequences autoregressively.",
            "ocr_text": "Multi-Head Attention Encoder-Decoder",
            "ideas": [],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 30.0, 45.0),
            "video_id": vid,
            "start_sec": 30.0,
            "end_sec": 45.0,
            "transcript_text": "First, tokens are embedded into vectors. "
                "Then positional encoding adds information about word order. "
                "Finally the attention layers compute context-aware representations.",
            "ocr_text": "Token Embedding + Positional Encoding → Attention",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 30.0, 45.0), 0),
                    moment_id=make_moment_id(vid, 30.0, 45.0),
                    type="step",
                    text="Token embedding, positional encoding, then attention layers",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 45.0, 60.0),
            "video_id": vid,
            "start_sec": 45.0,
            "end_sec": 60.0,
            "transcript_text": "Compared to recurrent neural networks, transformers process all tokens in parallel. "
                "This makes training much faster. "
                "However, the attention mechanism has quadratic complexity.",
            "ocr_text": "Transformers vs RNN: Parallel vs Sequential",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 45.0, 60.0), 0),
                    moment_id=make_moment_id(vid, 45.0, 60.0),
                    type="comparison",
                    text="Transformers process tokens in parallel unlike RNNs",
                ),
            ],
            "keyframe_paths": [],
        },
    ]


def _ux_moments(video: dict) -> list[dict]:
    vid = video["video_id"]
    return [
        {
            "moment_id": make_moment_id(vid, 0.0, 14.0),
            "video_id": vid,
            "start_sec": 0.0,
            "end_sec": 14.0,
            "transcript_text": "Welcome to this discussion on user research methods. "
                "Understanding your users is the foundation of good product design. "
                "We will cover interviews, surveys, and usability testing.",
            "ocr_text": "User Research Methods: Interviews, Surveys, Usability Testing",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 0.0, 14.0), 0),
                    moment_id=make_moment_id(vid, 0.0, 14.0),
                    type="definition",
                    text="User research helps understand user needs and behaviors",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 14.0, 28.0),
            "video_id": vid,
            "start_sec": 14.0,
            "end_sec": 28.0,
            "transcript_text": "First, prepare your interview guide with open-ended questions. "
                "Avoid leading questions that bias the responses. "
                "Record sessions with participant permission.",
            "ocr_text": "Interview Guide: Open-Ended Questions",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 14.0, 28.0), 0),
                    moment_id=make_moment_id(vid, 14.0, 28.0),
                    type="step",
                    text="Prepare interview guide with open-ended questions",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 28.0, 42.0),
            "video_id": vid,
            "start_sec": 28.0,
            "end_sec": 42.0,
            "transcript_text": "Be careful not to interrupt participants during interviews. "
                "Let them finish their thoughts completely. "
                "Silence can be productive, wait for them to elaborate.",
            "ocr_text": "",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 28.0, 42.0), 0),
                    moment_id=make_moment_id(vid, 28.0, 42.0),
                    type="warning",
                    text="Do not interrupt participants during interviews",
                ),
            ],
            "keyframe_paths": [],
        },
        {
            "moment_id": make_moment_id(vid, 42.0, 56.0),
            "video_id": vid,
            "start_sec": 42.0,
            "end_sec": 56.0,
            "transcript_text": "For example, you might ask about a user's daily workflow. "
                "Ask them to walk through a typical task step by step. "
                "This reveals pain points and opportunities for improvement.",
            "ocr_text": "Example: Walk through daily workflow",
            "ideas": [
                Idea(
                    idea_id=make_idea_id(make_moment_id(vid, 42.0, 56.0), 0),
                    moment_id=make_moment_id(vid, 42.0, 56.0),
                    type="example",
                    text="Ask users to walk through their daily workflow",
                ),
            ],
            "keyframe_paths": [],
        },
    ]
