# Reference app

A complete, working search service. Copy its shape into your own `apps/<name>/`.

## What it does

- `GET /` — a minimal search UI (box + results with speaker and timestamp).
- `POST /api/search` `{query, k, method}` — `method` is `vector`, `keyword`, or `hybrid`.
- `GET /healthz` — liveness for the deploy.

It embeds the query with `all-MiniLM-L6-v2` (384 dims, matching the schema),
runs pgvector / full-text / RRF search against `segments`, and returns hits.

## Run it

```bash
pip install -r requirements.txt
export PGDATABASE=semantic_search PGUSER=postgres PORT=8091
uvicorn main:app --host 127.0.0.1 --port $PORT
# open http://127.0.0.1:8091
```

The corpus must already be loaded and embedded into your database first
(see `../../starter`: schema, `load_corpus.py`, `embed.py`). This app only
serves search.

## Make it yours

This reference uses one fixed model and a plain query. The assignment is to do
better: your `embed.py`/`search.py` choices (model, chunking, hybrid weighting)
are what move the numbers. Swap the reference search here for your own and prove
the gain on the golden set before you ship.

## The contract (keep these)

- `main.py` exposes an ASGI `app`, started as `uvicorn main:app --host 127.0.0.1 --port $PORT`.
- Bind `127.0.0.1`, read `PORT` and `PG*` from the environment.
- Provide `requirements.txt` and `GET /healthz`.
