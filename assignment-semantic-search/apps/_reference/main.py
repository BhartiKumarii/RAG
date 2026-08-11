"""Reference search app for the semantic-search assignment.

A small FastAPI service that searches the corpus in your own Postgres and
returns the matching moments (speaker + timestamp + text). This is the shape
infra deploys: bind 127.0.0.1, read PORT and PG* from the environment, expose
/healthz. Model your own apps/<name>/ on this.

Run locally:
    pip install -r requirements.txt
    export PGDATABASE=semantic_search PORT=8091
    uvicorn main:app --host 127.0.0.1 --port $PORT

The corpus must already be loaded and embedded (see ../../starter). This app
only serves search; it does not ingest.
"""
import os

import psycopg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pgvector.psycopg import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load the embedder once at process start. Must match the model (and the
# vector dimension) used to embed the corpus. all-MiniLM-L6-v2 = 384 dims,
# which matches starter/schema.sql.
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
_model = SentenceTransformer(MODEL_NAME)

app = FastAPI(title="Semantic Search (reference)")


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "semantic_search"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        autocommit=True,
    )
    register_vector(conn)
    return conn


def embed_query(text: str) -> list[float]:
    return _model.encode([text], normalize_embeddings=True)[0].tolist()


def _rows_to_hits(rows: list) -> list[dict]:
    hits = []
    for r in rows:
        hits.append({
            "id": r[0], "video": r[1],
            "start_s": round(r[2], 1), "end_s": round(r[3], 1),
            "speaker": r[4], "text": r[5],
            "score": round(float(r[6]), 4) if r[6] is not None else None,
        })
    return hits


def search_vector(conn, query: str, k: int) -> list[dict]:
    qv = embed_query(query)
    rows = conn.execute(
        """SELECT id, video, start_s, end_s, speaker, text,
                  1 - (embedding <=> %s) AS score
             FROM segments
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s""",
        (qv, qv, k),
    ).fetchall()
    return _rows_to_hits(rows)


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
    wide = max(k * 4, 20)
    vec = search_vector(conn, query, wide)
    kw = search_keyword(conn, query, wide)
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


class SearchReq(BaseModel):
    query: str
    k: int = 5
    method: str = "hybrid"  # vector | keyword | hybrid


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/api/search")
def api_search(req: SearchReq):
    fn = {"vector": search_vector, "keyword": search_keyword, "hybrid": search_hybrid}
    searcher = fn.get(req.method, search_hybrid)
    conn = get_conn()
    try:
        hits = searcher(conn, req.query, req.k)
    finally:
        conn.close()
    return {"query": req.query, "method": req.method, "hits": hits}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Search</title>
<style>
  body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#111}
  h1{font-size:20px}
  form{display:flex;gap:8px;margin:16px 0}
  input[type=text]{flex:1;padding:10px;border:1px solid #ccc;border-radius:6px}
  select,button{padding:10px;border:1px solid #ccc;border-radius:6px;background:#fff}
  button{background:#111;color:#fff;cursor:pointer}
  .hit{border:1px solid #eee;border-radius:8px;padding:12px;margin:10px 0}
  .meta{color:#666;font-size:13px;margin-bottom:4px}
  .score{float:right;color:#999}
</style></head>
<body>
  <h1>Search the training recordings</h1>
  <form id="f">
    <input type="text" id="q" placeholder="e.g. what does cosine similarity measure?" autofocus>
    <select id="m"><option value="hybrid">hybrid</option><option value="vector">vector</option><option value="keyword">keyword</option></select>
    <button>Search</button>
  </form>
  <div id="out"></div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onsubmit=async e=>{
  e.preventDefault();
  out.innerHTML='searching...';
  const r=await fetch('/api/search',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({query:document.getElementById('q').value,method:document.getElementById('m').value,k:8})});
  const d=await r.json();
  if(!d.hits.length){out.innerHTML='no results';return;}
  out.innerHTML=d.hits.map(h=>{
    const t=Math.floor(h.start_s), mm=String(Math.floor(t/60)).padStart(2,'0'), ss=String(t%60).padStart(2,'0');
    return `<div class="hit"><div class="meta"><span class="score">${h.score??''}</span>
      ${h.video} · ${mm}:${ss} · ${h.speaker||'?'}</div>${h.text}</div>`;
  }).join('');
};
</script>
</body></html>"""
