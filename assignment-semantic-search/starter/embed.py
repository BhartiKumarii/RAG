"""Embeddings. THIS IS YOUR WORK (Day 7).

Fill in embed_texts(). The main() below is given: it reads every segment that
has no embedding yet, calls your embed_texts() in batches, and writes the
vectors back to the database. You should not need to change main().

Run it once after load_corpus.py:
  python embed.py

Then re-run whenever you change the model or re-load the corpus.
"""
import sys

from db import get_conn
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# YOUR CODE HERE
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of vectors (one per text).

    Same-length list out as in. Every vector must have the SAME dimension, and
    that dimension must match the `vector(N)` in schema.sql.

    Suggested starting point (Day 7): the sentence-transformers library, e.g.

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dims
        return model.encode(texts, normalize_embeddings=True).tolist()

    Load the model ONCE at module import, not per call, or this will crawl.
    Comparing models is your Day 8 lever; get one working first.
    """
    return _model.encode(texts, normalize_embeddings=True).tolist()



def embed_query(text: str) -> list[float]:
    """Embed a single query. Uses the same model as the corpus, by design."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# GIVEN — batch-embed every segment missing a vector, write it back.
# ---------------------------------------------------------------------------

BATCH = 64


def main() -> None:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, text, coalesce(ocr_text, '') FROM segments WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        print("nothing to embed (all segments already have vectors)")
        return

    print(f"embedding {len(rows)} segments...")
    done = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        # What you embed matters (Day 9). By default we embed the transcript
        # text; join in the slide text here if you want to try that lever.
        texts = [t for _, t, _ in chunk]
        vecs = embed_texts(texts)
        for (seg_id, _, _), vec in zip(chunk, vecs):
            conn.execute("UPDATE segments SET embedding = %s WHERE id = %s", (vec, seg_id))
        done += len(chunk)
        print(f"  {done}/{len(rows)}", file=sys.stderr)
    print(f"done — {done} segments embedded")


if __name__ == "__main__":
    main()
