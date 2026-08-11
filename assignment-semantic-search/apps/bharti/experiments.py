"""Day 7-10 experiments — one entry point, one shared toolkit (evallib.py).

Every experiment changes ONE thing and reports the number on the golden set, so
each delta is attributable. Run one by name:

    python experiments.py explore     # Day 7  look at nearest neighbours
    python experiments.py pgvector    # Day 8  HNSW index + ef_search sweep
    python experiments.py models      # Day 8  MiniLM vs bge-small (needs network for bge)
    python experiments.py chunk       # Day 9  chunk size + OCR fusion
    python experiments.py hybrid      # Day 10 hybrid (vector+keyword) scorecard

HEADLINE RESULTS (both recordings, 17 golden queries, MiniLM, k=5):
    keyword  0.18 / 0.15      vector  0.94 / 0.65      hybrid  0.82 / 0.64   (recall / mrr)
    -> the simplest setup wins: cue-level chunks, transcript only, pure vector.
       Every fancier lever (bigger chunks, OCR fusion, hybrid) was measured and REGRESSED.
"""
import sys

import evallib as E

sys.path.insert(0, E.STARTER)                       # reuse the given plumbing
from load_corpus import parse_srt, find_srt, load_ocr, ocr_for_span  # noqa: E402
import os                                            # noqa: E402


# ===========================================================================
# Day 7 — explore: embed one recording, show the top-5 for a few golden Qs.
# No DB, no index — just in-memory cosine so you can SEE what embeddings do.
# ===========================================================================
def explore():
    q_ids = ["g01", "g02", "g06"]
    cues = parse_srt(find_srt(os.path.join(E.CORPUS, "day1")))
    segs = [c for c in cues if c["text"]]
    vecs = E.embed([s["text"] for s in segs])       # every segment
    golden = {q["id"]: q for q in E.load_golden()}

    for qid in q_ids:
        q = golden[qid]
        sims = vecs @ E.embed_query(q["question"])
        top = sims.argsort()[::-1][:5]
        print(f"\n[{qid}] {q['question']}")
        print(f"      right moment {q['answer_start']}-{q['answer_end']}s · note: {q['note']}")
        for rank, i in enumerate(top, 1):
            s = segs[i]
            hit = "HIT " if (s["start"] < q["answer_end"] and s["end"] > q["answer_start"]) else "    "
            print(f"  {rank}. [{hit}] {int(s['start'])//60:02d}:{int(s['start'])%60:02d} "
                  f"sim={sims[i]:.3f}  {s['text'][:64]}")
    # FINDING: g01/g02 land the right moment at rank ~2 (a restated question or a
    # later recap out-ranks the actual answer); g06 (flavored-water analogy) is
    # missed — a metaphor with no lexical/semantic overlap defeats the encoder.


# ===========================================================================
# Shared helper for the DB experiments: build a throwaway table from rows,
# embed `texts`, add an HNSW index. Rows are dicts with the segment fields.
# ===========================================================================
def build_table(conn, name, rows, texts, model_name="all-MiniLM-L6-v2", dims=384):
    vecs = E.embed(texts, model_name)
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"""CREATE TABLE {name} (
        id text PRIMARY KEY, video text, start_s real, end_s real,
        text text, ocr_text text, embedding vector({dims}),
        fts tsvector GENERATED ALWAYS AS
            (to_tsvector('english', coalesce(text,'')||' '||coalesce(ocr_text,''))) STORED)""")
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {name}(id,video,start_s,end_s,text,ocr_text,embedding)"
            f" VALUES(%s,%s,%s,%s,%s,%s,%s)",
            [(r["id"], r["video"], r["start"], r["end"], r["text"], r["ocr_text"], v)
             for r, v in zip(rows, vecs)])
    conn.execute(f"CREATE INDEX {name}_fts ON {name} USING gin(fts)")
    conn.execute(f"CREATE INDEX {name}_hnsw ON {name} "
                 f"USING hnsw(embedding vector_cosine_ops) WITH (m=16, ef_construction=64)")
    conn.execute(f"ANALYZE {name}")
    return {r["id"]: (r["video"], r["start"], r["end"]) for r in rows}


def corpus_rows(chunker):
    """Parse every recording, chunk it, attach overlapping OCR. -> list of dicts."""
    out = []
    for video in ("day1", "day2"):
        folder = os.path.join(E.CORPUS, video)
        cues, ocr = parse_srt(find_srt(folder)), load_ocr(folder)
        for i, seg in enumerate(chunker(cues)):
            if seg["text"]:
                out.append({"id": f"{video}_{i:04d}", "video": video, "start": seg["start"],
                            "end": seg["end"], "text": seg["text"],
                            "ocr_text": ocr_for_span(ocr, seg["start"], seg["end"])})
    return out


def one_cue(cues):
    return list(cues)


def merged(cues, target=350):
    """Merge consecutive same-speaker cues until ~target chars (Day 9 lever)."""
    out, buf = [], None
    for c in cues:
        if buf and c["speaker"] == buf["speaker"] and len(buf["text"]) < target:
            buf["text"] = (buf["text"] + " " + c["text"]).strip(); buf["end"] = c["end"]
        else:
            if buf:
                out.append(buf)
            buf = dict(c)
    if buf:
        out.append(buf)
    return out


def vec_searcher(conn, table, model_name="all-MiniLM-L6-v2", prefix=""):
    """A searcher(question, k) -> ids, over `table` with vector distance."""
    def search(query, k):
        qv = E.embed_query(query, model_name, prefix)
        return [r[0] for r in conn.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (qv, k)).fetchall()]
    return search


# ===========================================================================
# Day 8 — pgvector: build both recordings into a table, score vs the keyword
# baseline, then sweep hnsw.ef_search with everything else held fixed.
# ===========================================================================
def pgvector():
    conn = E.get_conn()
    rows = corpus_rows(one_cue)
    spans = build_table(conn, "seg_tmp", rows, [r["text"] for r in rows])
    golden = E.load_golden()

    def keyword(query, k):
        return [r[0] for r in conn.execute(
            "SELECT id FROM seg_tmp WHERE fts @@ websearch_to_tsquery('english',%s) "
            "ORDER BY ts_rank(fts, websearch_to_tsquery('english',%s)) DESC LIMIT %s",
            (query, query, k)).fetchall()]

    print(f"  {'method':<18}{'recall':>8}{'mrr':>8}{'p50 ms':>9}")
    print("  " + "-" * 43)
    print(E.row("keyword", E.score(keyword, golden, spans)))
    print(E.row("vector", E.score(vec_searcher(conn, "seg_tmp"), golden, spans)))

    print("\n  ef_search sweep (recall/mrr flat = index already exact at this size):")
    conn.execute("SET enable_seqscan = off")               # force the HNSW path
    for ef in (10, 40, 100):
        conn.execute(f"SET hnsw.ef_search = {ef}")
        print(E.row(f"ef_search={ef}", E.score(vec_searcher(conn, "seg_tmp"), golden, spans)))
    conn.execute("DROP TABLE IF EXISTS seg_tmp")
    conn.close()
    # FINDING: vector 0.94 vs keyword 0.18; ef_search 10/40/100 all identical —
    # at ~450 vectors HNSW returns the exact neighbours, so nothing to tune.


# ===========================================================================
# Day 8 — models: change ONLY the embedding model. MiniLM vs bge-small (both
# 384d). bge is pulled from the Hub, so this one needs network / an HF token.
# ===========================================================================
def models():
    conn = E.get_conn()
    rows = corpus_rows(one_cue)
    golden = E.load_golden()
    specs = [("minilm", "all-MiniLM-L6-v2", ""),
             ("bge", "BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ")]

    print(f"  {'model':<18}{'recall':>8}{'mrr':>8}{'p50 ms':>9}")
    print("  " + "-" * 43)
    for key, hf, qprefix in specs:
        try:
            passage_prefix = "" if key == "minilm" else ""     # bge: instruction on query only
            spans = build_table(conn, f"seg_{key}", rows,
                                [passage_prefix + r["text"] for r in rows], model_name=hf)
            print(E.row(key, E.score(vec_searcher(conn, f"seg_{key}", hf, qprefix), golden, spans)))
            conn.execute(f"DROP TABLE IF EXISTS seg_{key}")
        except Exception as e:                                 # bge needs network/token
            print(f"  {key:<18}  skipped ({type(e).__name__}: run online with HF_TOKEN set)")
    conn.close()
    # FINDING: recall TIE (both 0.94, both miss g06); bge has better MRR (top-1
    # ordering) but ~2.5x query latency. Ship MiniLM unless top-1 quality is the goal.


# ===========================================================================
# Day 9 — chunk: two levers, one at a time. V0 baseline (cue-level, transcript)
# vs +OCR, vs +merge, vs both.
# ===========================================================================
def chunk():
    conn = E.get_conn()
    golden = E.load_golden()
    cue, mrg = corpus_rows(one_cue), corpus_rows(merged)
    print(f"  chunks: cue-level={len(cue)}  merged(~350c)={len(mrg)}\n")

    def txt(rows, fuse):
        return [(r["text"] + "\n" + r["ocr_text"]) if (fuse and r["ocr_text"]) else r["text"] for r in rows]

    variants = [("V0 baseline", cue, False), ("V1 +OCR", cue, True),
                ("V2 +merge", mrg, False), ("V3 +merge+OCR", mrg, True)]
    print(f"  {'variant':<18}{'recall':>8}{'mrr':>8}{'p50 ms':>9}")
    print("  " + "-" * 43)
    for label, rows, fuse in variants:
        spans = build_table(conn, "seg_v", rows, txt(rows, fuse))
        print(E.row(label, E.score(vec_searcher(conn, "seg_v"), golden, spans)))
    conn.execute("DROP TABLE IF EXISTS seg_v")
    conn.close()
    # FINDING: both levers REGRESS. V0 is already at the recall ceiling (16/17);
    # its one miss is a metaphor neither lever fixes. OCR adds slide-chrome noise
    # (-0.18); merging averages several ideas into one vector (-0.23). Keep V0.


# ===========================================================================
# Day 10 — hybrid: score the starter's RRF fusion vs vector/keyword, then sweep
# a weight on the keyword list to test whether any blend beats pure vector.
# ===========================================================================
def hybrid():
    import search                                    # the real starter searchers
    conn = E.get_conn()
    spans = E.spans_db(conn, "segments")
    golden = E.load_golden()

    print(f"  {'method':<18}{'recall':>8}{'mrr':>8}{'p50 ms':>9}")
    print("  " + "-" * 43)
    for name, fn in [("keyword", search.search_keyword), ("vector", search.search_vector),
                     ("hybrid", search.search_hybrid)]:
        print(E.row(name, E.score(fn, golden, spans)))

    def weighted(w):
        def search_fn(query, k, rrf_k=60):
            lists = [(1.0, search.search_vector(query, k * 4)), (w, search.search_keyword(query, k * 4))]
            sc = {}
            for weight, ids in lists:
                for rank, sid in enumerate(ids, 1):
                    sc[sid] = sc.get(sid, 0.0) + weight / (rrf_k + rank)
            return sorted(sc, key=sc.get, reverse=True)[:k]
        return search_fn

    print("\n  weight on keyword list (0.0 = pure vector):")
    for w in (0.0, 0.25, 0.5, 1.0):
        print(E.row(f"w_kw={w}", E.score(weighted(w), golden, spans)))
    conn.close()
    # FINDING: hybrid LOSES to vector (0.82 vs 0.94). Keyword is too weak here
    # (0.18); RRF gives its confident-wrong ranks weight that displaces good
    # vector hits. Every w_kw>0 loses recall. Ship pure vector.


COMMANDS = {"explore": explore, "pgvector": pgvector, "models": models,
            "chunk": chunk, "hybrid": hybrid}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in COMMANDS:
        print("usage: python experiments.py [%s]" % " | ".join(COMMANDS))
        raise SystemExit(1)
    COMMANDS[cmd]()
