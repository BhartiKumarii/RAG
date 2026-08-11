# Starter

Everything here is given except two functions, which are your whole job:

- `embed.py` → **`embed_texts()`**  (Day 7)
- `search.py` → **`search_vector()`**  (Day 7-8), then **`search_hybrid()`** (Day 10)

The rest, the schema, the corpus loader, the keyword baseline, and the scorer,
is plumbing you run but do not rewrite.

## Setup (once)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Postgres + pgvector (Docker is easiest):
docker run -d --name pgv -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
export PGPASSWORD=postgres PGDATABASE=semantic_search
psql "postgresql://postgres:postgres@localhost/postgres" -c "CREATE DATABASE semantic_search;"
psql "postgresql://postgres:postgres@localhost/semantic_search" -f schema.sql
```

## The loop

```bash
python load_corpus.py ../corpus/day1 day1     # given: transcript + OCR -> rows
python load_corpus.py ../corpus/day2 day2
python embed.py                               # YOUR embed_texts() fills vectors
python eval.py                                # scores every method vs the golden set
```

`eval.py` prints Recall@5, MRR, and latency for `keyword` (given), `vector`
(yours), and `hybrid` (yours, Day 10). Beat `keyword` first.

## What "correct" means

A result counts if it lands on the **right moment**: same recording, and its
time span overlaps the answer window in `golden.jsonl`. So you are free to
re-chunk however you like (your Day 9 lever) without breaking the score.

## The rule that carries over from the last assignment

Change **one thing at a time** and report the number before and after on the
same golden set. One proven delta beats five techniques you can only name.
