from pathlib import Path


class Config:
    data_dir: Path
    db_path: Path
    videos_dir: Path
    frames_dir: Path
    transcripts_dir: Path

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir).resolve()
        self.db_path = self.data_dir / "vidcrawl.db"
        self.videos_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames"
        self.transcripts_dir = self.data_dir / "transcripts"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)


def get_config(data_dir: str | Path | None = None) -> Config:
    if data_dir is None:
        return Config()
    return Config(data_dir)
