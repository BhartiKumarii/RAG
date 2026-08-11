"""Score retrieval against the golden set. GIVEN — this is how you grade yourself.

A retrieved segment counts as CORRECT for a query if it comes from the same
recording and its time span overlaps the query's answer window. That makes the
score robust to how you chunk (your Day 9 lever): move the boundaries however
you like, a hit on the right moment still counts.

Runs every searcher that is implemented (keyword always; vector and hybrid once
you fill their stubs) and prints Recall@k, MRR, and median latency side by side.

  python eval.py            # k = 5
  python eval.py 10         # k = 10
"""
import json
import os
import statistics
import sys
import time

from db import get_conn
import search

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in open(GOLDEN) if line.strip()]


def segment_spans() -> dict[str, tuple[str, float, float]]:
    """id -> (video, start_s, end_s) for overlap scoring."""
    conn = get_conn()
    rows = conn.execute("SELECT id, video, start_s, end_s FROM segments").fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def is_hit(seg_id: str, q: dict, spans: dict) -> bool:
    span = spans.get(seg_id)
    if not span:
        return False
    video, start, end = span
    return video == q["video"] and start <= q["answer_end"] and end >= q["answer_start"]


def score(searcher, golden: list[dict], spans: dict) -> dict:
    recall_hits, rr_sum, lats = 0, 0.0, []
    per_query = []
    for q in golden:
        t0 = time.perf_counter()
        ids = searcher(q["question"], K)
        lats.append((time.perf_counter() - t0) * 1000)
        rank = next((i + 1 for i, sid in enumerate(ids) if is_hit(sid, q, spans)), None)
        if rank:
            recall_hits += 1
            rr_sum += 1.0 / rank
        per_query.append((q["id"], q["keyword_would_miss"], rank))
    n = len(golden)
    return {
        "recall": recall_hits / n,
        "mrr": rr_sum / n,
        "p50_ms": statistics.median(lats),
        "per_query": per_query,
    }


def run(name: str, searcher, golden, spans):
    try:
        return name, score(searcher, golden, spans)
    except NotImplementedError:
        return name, None
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  {name}: error — {e}", file=sys.stderr)
        return name, None


def main() -> None:
    golden = load_golden()
    spans = segment_spans()
    if not spans:
        raise SystemExit("no segments in the database — run load_corpus.py and embed.py first")

    methods = [
        ("keyword", search.search_keyword),
        ("vector", search.search_vector),
        ("hybrid", search.search_hybrid),
    ]
    results = [run(n, s, golden, spans) for n, s in methods]

    print(f"\n  Golden set: {len(golden)} queries · Recall@{K} / MRR / median latency\n")
    print(f"  {'method':<10} {'recall':>8} {'mrr':>8} {'p50 ms':>9}")
    print("  " + "-" * 37)
    for name, r in results:
        if r is None:
            print(f"  {name:<10} {'(not implemented)':>27}")
        else:
            print(f"  {name:<10} {r['recall']:>8.2f} {r['mrr']:>8.2f} {r['p50_ms']:>9.1f}")

    # Where does meaning-search earn its keep? Break out the queries we tagged
    # as ones plain keyword search should miss.
    vec = next((r for n, r in results if n == "vector" and r), None)
    if vec:
        miss = [p for p in vec["per_query"] if p[1]]
        hit = sum(1 for _, _, rank in miss if rank)
        print(f"\n  On the {len(miss)} 'keyword-would-miss' queries, vector found {hit}.")


if __name__ == "__main__":
    main()
