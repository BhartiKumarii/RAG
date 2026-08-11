"""Your search app. Deploys to the box behind your own Cloudflare tunnel.

It already RUNS as-is (/, /healthz work). Your job is the two stubs below,
embed_query() and search_vector(). Until you fill them, /api/search returns a
501 telling you so. Model the implementations on ../_reference.

Contract (do not change): ASGI `app`, bind 127.0.0.1 on $PORT, read PG* from
env, keep /healthz. Deploy = push to main.
"""
import os

from sentence_transformers import SentenceTransformer

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pgvector.psycopg import register_vector
from pydantic import BaseModel

app = FastAPI(title="Semantic Search — bharti")

# Load the embedder once at import — must match the model/dims used to embed
# the corpus (all-MiniLM-L6-v2 = 384 dims, matches starter/schema.sql).
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
_model = SentenceTransformer(MODEL_NAME)

def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "semantic_search_bharti"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        autocommit=True,
    )
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# YOUR WORK — implement these two. See ../_reference for a working version.
# ---------------------------------------------------------------------------
def embed_query(text: str) -> list[float]:
    """Embed the query with the SAME model you embedded the corpus with.
    all-MiniLM-L6-v2 (384 dims, matches schema). Model is loaded once above."""
    return _model.encode([text], normalize_embeddings=True)[0].tolist()


def _rows_to_hits(rows: list) -> list[dict]:
    return [
        {
            "id": r[0], "video": r[1],
            "start_s": round(r[2], 1), "end_s": round(r[3], 1),
            "speaker": r[4], "text": r[5],
            "score": round(float(r[6]), 4) if r[6] is not None else None,
        }
        for r in rows
    ]


def search_vector(conn, query: str, k: int) -> list[dict]:
    """Return the k nearest segments as dicts with id, video, start_s, end_s,
    speaker, text, score. Embed the query, then ORDER BY embedding <=> qv LIMIT k."""
    qv = embed_query(query)
    rows = conn.execute(
        """SELECT id, video, start_s, end_s, speaker, text,
                  1 - (embedding <=> %s::vector) AS score
             FROM segments
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        (qv, qv, k),
    ).fetchall()
    return _rows_to_hits(rows)


# ---------------------------------------------------------------------------
# Compare-mode: keyword and hybrid, exposed alongside vector for exploration.
# experiments.py's "hybrid" scorecard found pure vector wins on this corpus
# (0.94 vs 0.82 recall — keyword alone is too weak at 0.18 and drags hybrid
# down), so vector stays the default here. These two let a visitor pick a
# different method in the UI and see that finding live, on their own query.
# ---------------------------------------------------------------------------
def search_keyword(conn, query: str, k: int) -> list[dict]:
    rows = conn.execute(
        """SELECT id, video, start_s, end_s, speaker, text,
                  ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
             FROM segments
            WHERE fts @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s""",
        (query, query, k),
    ).fetchall()
    return _rows_to_hits(rows)


def search_hybrid(conn, query: str, k: int, rrf_k: int = 60) -> list[dict]:
    """Fuse search_vector and search_keyword by Reciprocal Rank Fusion."""
    wide = max(k * 4, 20)
    vec, kw = search_vector(conn, query, wide), search_keyword(conn, query, wide)
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for ranked in (vec, kw):
        for rank, hit in enumerate(ranked, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank)
            meta[hit["id"]] = hit
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for seg_id, score in top:
        hit = dict(meta[seg_id])
        hit["score"] = round(score, 5)
        out.append(hit)
    return out


SEARCHERS = {"vector": search_vector, "keyword": search_keyword, "hybrid": search_hybrid}


# ---------------------------------------------------------------------------
# Given — the serving layer. You should not need to change below here.
# ---------------------------------------------------------------------------

class SearchReq(BaseModel):
    query: str
    k: int = 5
    method: str = "vector"  # vector (default — see finding above) | keyword | hybrid


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/search")
def api_search(req: SearchReq):
    searcher = SEARCHERS.get(req.method, search_vector)
    conn = get_conn()
    try:
        hits = searcher(conn, req.query, req.k)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"{e}. Fill the stubs in main.py.")
    finally:
        conn.close()
    return {"query": req.query, "method": req.method, "hits": hits}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset=utf-8><title>Semantic Search</title>
<body style="font:15px system-ui;max-width:720px;margin:40px auto;padding:0 16px">
<h1>Search the training recordings</h1>
<form id=f style="display:flex;gap:8px">
<input id=q style="flex:1;padding:9px" placeholder="ask something">
<select id=m style="padding:9px">
<option value=vector>vector (default)</option>
<option value=keyword>keyword</option>
<option value=hybrid>hybrid</option>
</select>
<button style="padding:9px">Search</button></form>
<p style="color:#666;font-size:13px">vector is the shipped default — it beat keyword and
hybrid on the golden set (0.94 vs 0.18 vs 0.82 recall@5). Pick another method to see why.</p>
<div id=o></div>
<script>f.onsubmit=async e=>{e.preventDefault();o.innerHTML='...';
const r=await fetch('/api/search',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({query:q.value,method:m.value,k:8})});const d=await r.json();
if(d.detail){o.innerHTML='<p style=color:#a00>'+d.detail+'</p>';return;}
if(!(d.hits||[]).length){o.innerHTML='<p style=color:#666>no results</p>';return;}
o.innerHTML=(d.hits||[]).map(h=>{const t=Math.floor(h.start_s||0);
return '<div style="border:1px solid #eee;border-radius:8px;padding:10px;margin:8px 0">'
+'<small style=color:#666>'+(h.score??'')+' · '+h.video+' · '+String(Math.floor(t/60)).padStart(2,'0')+':'
+String(t%60).padStart(2,'0')+' · '+(h.speaker||'?')+'</small><br>'+h.text+'</div>';}).join('');};
</script></body>"""
