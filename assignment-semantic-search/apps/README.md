# apps

Each intern ships one **served app** here, in their own folder:

```
apps/akshay/   apps/ananya/   apps/bharti/
```

The app is a small HTTP service that searches your corpus and returns the
matching moments. It runs on the shared EC2 box next to Postgres, and a
Cloudflare tunnel points your own domain at it. You build the app; infra
(Siva/Hemanth) does the deploy and tunnel wiring.

`apps/_reference/` is a complete, working example. Model your app on it.

## The contract (what infra expects from your folder)

Keep these so the deploy script and the tunnel work without per-person tweaks:

- **`requirements.txt`** at the root of your folder.
- **`main.py`** exposing an ASGI `app` object (FastAPI/Starlette). It is started as
  `uvicorn main:app --host 127.0.0.1 --port $PORT`.
- **Bind to `127.0.0.1` and read `PORT` from the environment.** Never bind `0.0.0.0`;
  the tunnel reaches you on localhost.
- **Read Postgres from `PG*` env vars** (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`,
  `PGPASSWORD`). Do not hardcode a connection string.
- A **`GET /healthz`** returning 200 so the deploy can verify you came up.

Anything else (extra routes, a UI, background warmup) is yours.

## Your own database

You each get your **own** Postgres database so your embeddings never collide:

| intern | database            | app folder     |
|--------|---------------------|----------------|
| akshay | `semantic_search_akshay` | `apps/akshay/` |
| ananya | `semantic_search_ananya` | `apps/ananya/` |
| bharti | `semantic_search_bharti` | `apps/bharti/` |

Your `PGDATABASE` env var is set to yours at deploy. It has `pgvector` and the
`segments` table already (from `starter/schema.sql`). Load and embed your corpus
into it with the `starter/` scripts.

## How it goes live (infra does this)

- Your app runs as a systemd service on `ess-yuviworks`, on a free `127.0.0.1` port.
- The `yuviworks-matrix` tunnel exposes it as `<you>-origin.yuviworks.com`.
- A thin reverse-proxy Worker sits in front on your public domain and injects an
  Access service token, so only the Worker can reach your origin. People reach the
  Worker through the shared `@yuviworks.com` login.
- Deploy is push-to-main → GitHub Actions → `wrangler deploy` + a remote deploy on
  the box.

You do not write the Worker or touch the tunnel. Just keep the contract above.
