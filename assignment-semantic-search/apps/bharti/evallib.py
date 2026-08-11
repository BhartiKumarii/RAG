"""Shared toolkit for the Day 7-10 experiments.

Everything the experiment scripts used to copy-paste lives here once: the DB
connection, the embedder, the golden set, and the scoring rule (identical to
starter/eval.py). Import it; don't repeat it.

    from evallib import get_conn, embed, load_golden, score, is_hit, spans_db
"""
import json
import os
import statistics
import time

# Model is in the local HF cache; stay offline so a stray HF token doesn't
# trigger a failing online auth check.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STARTER = os.path.join(ROOT, "starter")
CORPUS = os.path.join(ROOT, "corpus")
GOLDEN = os.path.join(STARTER, "golden.jsonl")


# --- database --------------------------------------------------------------
def get_conn():
    """One connection, PG* from env (defaults match the local Docker setup)."""
    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "semantic_search_bharti"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        autocommit=True,
    )
    register_vector(conn)
    return conn


def spans_db(conn, table="segments"):
    """id -> (video, start_s, end_s), the map the scorer needs for overlap."""
    rows = conn.execute(f"SELECT id, video, start_s, end_s FROM {table}").fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


# --- embedder (cached per model name) --------------------------------------
_cache = {}


def model(name="all-MiniLM-L6-v2"):
    if name not in _cache:
        _cache[name] = SentenceTransformer(name)
    return _cache[name]


def embed(texts, name="all-MiniLM-L6-v2", prefix=""):
    """Normalized float32 vectors (cosine == dot). `prefix` is for instruction
    models like bge that want 'Represent this sentence...: ' on the query."""
    return model(name).encode([prefix + t for t in texts],
                              normalize_embeddings=True).astype(np.float32)


def embed_query(text, name="all-MiniLM-L6-v2", prefix=""):
    return embed([text], name, prefix)[0]


# --- golden set + scoring (the exact rule starter/eval.py uses) ------------
def load_golden():
    return [json.loads(l) for l in open(GOLDEN) if l.strip()]


def is_hit(seg_id, q, spans):
    """A hit = same recording AND its time span overlaps the answer window."""
    s = spans.get(seg_id)
    return bool(s) and s[0] == q["video"] and s[1] <= q["answer_end"] and s[2] >= q["answer_start"]


def first_rank(ids, q, spans):
    """1-based rank of the first correct id, or None if none of them hit."""
    return next((i + 1 for i, sid in enumerate(ids) if is_hit(sid, q, spans)), None)


def score(searcher, golden, spans, k=5):
    """Run `searcher(question, k)` over the golden set -> recall / mrr / p50 ms.
    `searcher` returns a ranked list of segment ids (best first)."""
    hits, rr, lats = 0, 0.0, []
    for q in golden:
        t0 = time.perf_counter()
        ids = searcher(q["question"], k)
        lats.append((time.perf_counter() - t0) * 1000)
        rank = first_rank(ids, q, spans)
        if rank:
            hits += 1
            rr += 1.0 / rank
    n = len(golden)
    return {"recall": hits / n, "mrr": rr / n, "p50": statistics.median(lats)}


def row(label, r, width=18):
    """Format one result line for the little tables the experiments print."""
    return f"  {label:<{width}}{r['recall']:>8.2f}{r['mrr']:>8.2f}{r['p50']:>9.1f}"
