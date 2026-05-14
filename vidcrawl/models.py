from typing import Any, Optional

from pydantic import BaseModel, Field


class Video(BaseModel):
    video_id: str
    title: str
    source: str
    url: Optional[str] = None
    duration_sec: float
    status: str = "pending"
    transcript_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class Idea(BaseModel):
    idea_id: str
    moment_id: str
    type: str
    text: str
    confidence: float = 0.7
    source: str = "rule"


class Moment(BaseModel):
    moment_id: str
    video_id: str
    start_sec: float
    end_sec: float
    transcript_text: str = ""
    ocr_text: str = ""
    ideas: list[Idea] = Field(default_factory=list)
    keyframe_paths: list[str] = Field(default_factory=list)
    content_hash: Optional[str] = None
    parent_moment_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class Evidence(BaseModel):
    evidence_id: str
    moment_id: str
    modality: str
    content: str
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class Keyframe(BaseModel):
    keyframe_id: str
    moment_id: Optional[str] = None
    video_id: str
    timestamp_sec: float
    file_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    ocr_text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class Duplicate(BaseModel):
    dup_id: str
    moment_id: str
    canonical_moment_id: str
    similarity_score: float = 1.0
    novelty_score: float = 0.0
    method: str = "exact_hash"
    duplicate_type: str = "exact"
    item_type: str = "moment"
    reason: str = ""
    created_at: Optional[str] = None


class IngestionRun(BaseModel):
    run_id: str
    video_id: str
    status: str = "running"
    pipeline_steps: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class SearchResult(BaseModel):
    moment: Moment
    relevance_score: float
    matched_on: list[str]
    video_title: str
    source_url: str
    moment_url: str
