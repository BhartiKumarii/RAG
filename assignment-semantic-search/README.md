# Semantic search over our own training recordings

Build search that finds the **moment** in a recording where something was said,
by **meaning**, not by matching words. Your corpus is our own AI/ML training
sessions. Your job is the embeddings-and-vectors half; the video-to-text
extraction is done for you.

## The two halves

| Given (you run it) | Yours (you build it) |
|---|---|
| `tools/` — video → diarized transcript, slide-text (OCR), objects | the **embedding** function |
| `corpus/` — the extracted text for each session | the **vector search** query |
| `starter/schema.sql`, `load_corpus.py` — the database and loader | index tuning, what-to-embed, chunking |
| `starter/eval.py`, `golden.jsonl` — the scorer and question set | on Day 10, **hybrid** (vector + keyword) |
| `starter/search.py` `search_keyword` — the baseline to beat | |

You edit exactly two functions: `embed.py:embed_texts` and
`search.py:search_vector` (plus `search_hybrid` on Day 10). Everything else you
consume.

## The bar

1. **Beat keyword search** on the golden set. `eval.py` prints both; this is the floor.
2. **Approach a hosted embedder.** We run a strong cloud model (Vertex or
   OpenAI) once over the same corpus and questions to set the target. Getting
   your free, local, open stack close to it is the real result, and if you win
   on cost or on the Arabic-style hard queries, say so.

Every claim of "better" is a number on `golden.jsonl`, same as the last assignment.

## The week

| Day | Focus |
|---|---|
| 6 | video, semantic search, and the problem (why keyword search misses meaning) |
| 7 | embeddings: meaning as geometry — fill `embed_texts`, look at the neighbours |
| 8 | store in pgvector, add and tune the HNSW index, first Recall@5 vs keyword |
| 9 | what you embed: chunk size, and fusing in the slide (OCR) text |
| 10 | vectors are not enough: hybrid retrieval, and the head-to-head scorecard |

## Start

Read `tools/README.md` (how the corpus was made) and `starter/README.md` (how
to run it). Then, for Day 6, read one recording's diarized transcript and its
`ocr.json`, and bring five questions you would ask it with the timestamps of
their answers.

Everything runs offline. No cloud account needed for your half.
