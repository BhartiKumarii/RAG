"""Retrieval. search_vector() IS YOUR WORK (Day 7-8). The keyword baseline is
given, so you always have something to beat and, on Day 10, a second half to
fuse with.

Each search function takes a query string and k, and returns a ranked list of
segment ids (best first), most k long.
"""
from db import get_conn
from embed import embed_query


# ---------------------------------------------------------------------------
# YOUR CODE HERE (Day 7 = make it work, Day 8 = make it fast with an index)
# ---------------------------------------------------------------------------


def search_vector(query: str, k: int = 5) -> list[str]:
    """Return the k nearest segment ids (best first). Embed the query, then
    ORDER BY embedding <=> qv LIMIT k."""
    qv = embed_query(query)
    rows = get_conn().execute(
        """SELECT id
             FROM segments
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        (qv, k),
    ).fetchall()
    return [r[0] for r in rows]
# ---------------------------------------------------------------------------
# GIVEN — keyword baseline (Postgres full-text search). This is the bar.
# ---------------------------------------------------------------------------


def search_keyword(query: str, k: int = 5) -> list[str]:
    """Rank segments by Postgres full-text relevance over transcript + slide text."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT id
             FROM segments
            WHERE fts @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank(fts, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s""",
        (query, query, k),
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# OPTIONAL (Day 10) — hybrid: fuse vector + keyword with Reciprocal Rank Fusion.
# ---------------------------------------------------------------------------


def search_hybrid(query: str, k: int = 5, rrf_k: int = 60) -> list[str]:
    """Fuse search_vector and search_keyword by rank (Reciprocal Rank Fusion).

    Pull a wide list (k*4) from each searcher, then score every id that appears
    in either list by the sum over lists of 1 / (rrf_k + rank), where rank is
    1-based. RRF fuses by RANK, not raw score, so the vector cosine distance and
    the FTS ts_rank never have to be put on the same scale. Return the top k.
    """
    wide = k * 4
    ranked = [search_vector(query, wide), search_keyword(query, wide)]
    scores: dict[str, float] = {}
    for ids in ranked:
        for rank, sid in enumerate(ids, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]
