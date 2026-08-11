-- Schema for the semantic-search corpus. GIVEN — you do not need to change this,
-- except the one embedding-dimension number noted below when you pick a model.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS segments;
CREATE TABLE segments (
    id        TEXT PRIMARY KEY,   -- e.g. day1_0007
    video     TEXT NOT NULL,      -- day1, day2, ...
    seq       INT  NOT NULL,      -- order within the video
    start_s   REAL NOT NULL,      -- segment start, seconds into the recording
    end_s     REAL NOT NULL,      -- segment end, seconds
    speaker   TEXT,               -- diarized speaker label, if known
    text      TEXT NOT NULL,      -- transcript text for this segment
    ocr_text  TEXT,               -- on-screen slide text overlapping this span (may be NULL)

    -- The meaning vector. You fill this from embed.py.
    -- CHANGE 384 to match your model's output dimension:
    --   all-MiniLM-L6-v2 -> 384,  bge-base -> 768,  e5-large -> 1024, ...
    embedding vector(384),

    -- Full-text search vector over transcript + slide text. This powers the
    -- keyword baseline you must beat (and, on Day 10, the keyword half of hybrid).
    fts tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(text, '') || ' ' || coalesce(ocr_text, ''))) STORED
);

CREATE INDEX segments_fts_idx   ON segments USING gin (fts);
CREATE INDEX segments_video_idx ON segments (video);

-- The vector (HNSW) index is intentionally NOT created here.
-- Creating and tuning it is your Day 8 task. When you are ready:
--
--   CREATE INDEX ON segments USING hnsw (embedding vector_cosine_ops)
--       WITH (m = 16, ef_construction = 64);
--
-- and tune ef_search per query with:  SET hnsw.ef_search = 40;
