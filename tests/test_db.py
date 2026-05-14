from vidcrawl.db import (
    generate_run_id,
    get_db,
    get_moment_count_by_video,
    get_moments_by_video,
    get_video,
    init_db,
    insert_ingestion_run,
    insert_moment,
    insert_video,
    list_videos,
    make_moment_id,
    rebuild_fts,
)
from vidcrawl.models import IngestionRun, Moment, Video


class TestDatabaseInit:
    def test_init_creates_tables(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        assert "videos" in table_names
        assert "moments" in table_names
        assert "modal_evidence" in table_names
        assert "ideas" in table_names
        assert "keyframes" in table_names
        assert "duplicates" in table_names
        assert "ingestion_runs" in table_names
        conn.close()

    def test_init_creates_fts_table(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        assert "moments_fts" in table_names
        conn.close()

    def test_init_is_idempotent(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        init_db(conn)
        conn.close()


class TestVideoCRUD:
    def test_insert_and_get(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v = Video(
            video_id="test123", title="Test Video", source="local",
            duration_sec=120.0, status="pending",
        )
        insert_video(conn, v)
        conn.commit()

        retrieved = get_video(conn, "test123")
        assert retrieved is not None
        assert retrieved.video_id == "test123"
        assert retrieved.title == "Test Video"
        assert retrieved.source == "local"
        assert retrieved.duration_sec == 120.0
        conn.close()

    def test_list_empty(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        assert list_videos(conn) == []
        conn.close()

    def test_list_with_data(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v1 = Video(video_id="v1", title="First", source="local", duration_sec=10.0)
        v2 = Video(
            video_id="v2", title="Second", source="youtube",
            duration_sec=20.0, url="https://youtube.com/watch?v=v2",
        )
        insert_video(conn, v1)
        insert_video(conn, v2)
        conn.commit()

        videos = list_videos(conn)
        assert len(videos) == 2
        conn.close()

    def test_get_nonexistent(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        assert get_video(conn, "nonexistent") is None
        conn.close()


class TestMomentCRUD:
    def test_insert_and_get(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v = Video(video_id="v1", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)

        m = Moment(
            moment_id=make_moment_id("v1", 0.0, 10.0),
            video_id="v1",
            start_sec=0.0,
            end_sec=10.0,
            transcript_text="Hello world",
        )
        insert_moment(conn, m)
        conn.commit()

        moments = get_moments_by_video(conn, "v1")
        assert len(moments) == 1
        assert moments[0].transcript_text == "Hello world"
        assert moments[0].start_sec == 0.0
        assert moments[0].end_sec == 10.0
        conn.close()

    def test_moment_count(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v = Video(video_id="v1", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)
        for start, end, text in [
            (0.0, 10.0, "First"),
            (10.0, 20.0, "Second"),
            (20.0, 30.0, "Third"),
        ]:
            insert_moment(
                conn,
                Moment(
                    moment_id=make_moment_id("v1", start, end),
                    video_id="v1",
                    start_sec=start,
                    end_sec=end,
                    transcript_text=text,
                ),
            )
        conn.commit()
        assert get_moment_count_by_video(conn, "v1") == 3
        conn.close()


class TestIngestionRun:
    def test_insert_run(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v = Video(video_id="v1", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)
        run = IngestionRun(
            run_id=generate_run_id(),
            video_id="v1",
            status="running",
            pipeline_steps=["register_metadata"],
        )
        insert_ingestion_run(conn, run)
        conn.commit()

        rows = conn.execute("SELECT * FROM ingestion_runs").fetchall()
        assert len(rows) == 1
        assert rows[0]["video_id"] == "v1"
        conn.close()


class TestFTS:
    def test_rebuild_fts(self, tmp_db):
        conn = get_db(tmp_db)
        init_db(conn)
        v = Video(video_id="v1", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)
        insert_moment(
            conn,
            Moment(
                moment_id=make_moment_id("v1", 0.0, 10.0),
                video_id="v1",
                start_sec=0.0,
                end_sec=10.0,
                transcript_text="Hello world from test",
            ),
        )
        conn.commit()
        rebuild_fts(conn)
        conn.commit()

        results = conn.execute(
            "SELECT moment_id FROM moments_fts WHERE moments_fts MATCH 'hello'"
        ).fetchall()
        assert len(results) == 1
        assert results[0]["moment_id"] == "v1:0.00:10.00"
        conn.close()
