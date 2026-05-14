from vidcrawl.utils.hashing import content_hash, sha256_prefix
from vidcrawl.utils.time import format_timestamp, parse_timestamp


class TestHashing:
    def test_sha256_prefix(self):
        h = sha256_prefix("hello world")
        assert len(h) == 16
        assert isinstance(h, str)

    def test_sha256_prefix_variable_length(self):
        h = sha256_prefix("hello world", length=8)
        assert len(h) == 8

    def test_content_hash_stable(self):
        h1 = content_hash("hello world", "extra text")
        h2 = content_hash("hello world", "extra text")
        assert h1 == h2

    def test_content_hash_different(self):
        h1 = content_hash("hello world")
        h2 = content_hash("goodbye world")
        assert h1 != h2

    def test_content_hash_empty(self):
        h = content_hash("", "")
        assert len(h) == 64
        assert isinstance(h, str)

    def test_content_hash_truncates(self):
        long_text = "a" * 1000
        h = content_hash(long_text)
        assert len(h) == 64


class TestTimeUtils:
    def test_format_timestamp_seconds(self):
        assert format_timestamp(65) == "1:05"

    def test_format_timestamp_hours(self):
        assert format_timestamp(3661) == "1:01:01"

    def test_format_timestamp_zero(self):
        assert format_timestamp(0) == "0:00"

    def test_parse_timestamp_mmss(self):
        assert parse_timestamp("1:05") == 65.0

    def test_parse_timestamp_hhmmss(self):
        assert parse_timestamp("1:01:01") == 3661.0

    def test_parse_timestamp_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_timestamp("invalid")
