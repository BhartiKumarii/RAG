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
    return [
        {
            "id": r[0], "video": r[1],
            "start_s": round(r[2], 1), "end_s": round(r[3], 1),
            "speaker": r[4], "text": r[5],
            "score": round(float(r[6]), 4) if r[6] is not None else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Given — the serving layer. You should not need to change below here.
# ---------------------------------------------------------------------------

class SearchReq(BaseModel):
    query: str
    k: int = 5


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/search")
def api_search(req: SearchReq):
    conn = get_conn()
    try:
        hits = search_vector(conn, req.query, req.k)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"{e}. Fill the stubs in main.py.")
    finally:
        conn.close()
    return {"query": req.query, "hits": hits}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset=utf-8><title>Semantic Search</title>
<body style="font:15px system-ui;max-width:720px;margin:40px auto;padding:0 16px">
<h1>Search the training recordings</h1>
<form id=f><input id=q style="width:78%;padding:9px" placeholder="ask something">
<button style="padding:9px">Search</button></form><div id=o></div>
<script>f.onsubmit=async e=>{e.preventDefault();o.innerHTML='...';
const r=await fetch('/api/search',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({query:q.value,k:8})});const d=await r.json();
if(d.detail){o.innerHTML='<p style=color:#a00>'+d.detail+'</p>';return;}
o.innerHTML=(d.hits||[]).map(h=>{const t=Math.floor(h.start_s||0);
return '<div style="border:1px solid #eee;border-radius:8px;padding:10px;margin:8px 0">'
+'<small style=color:#666>'+h.video+' · '+String(Math.floor(t/60)).padStart(2,'0')+':'
+String(t%60).padStart(2,'0')+' · '+(h.speaker||'?')+'</small><br>'+h.text+'</div>';}).join('');};
</script></body>"""
