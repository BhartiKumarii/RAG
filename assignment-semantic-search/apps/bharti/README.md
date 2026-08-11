# bharti

Your search app goes in this folder.

- **Your database:** `semantic_search_bharti` (set as `PGDATABASE` at deploy).
- **Model your app on** [`../_reference`](../_reference).
- **Keep the contract** (see [`../README.md`](../README.md)): `main.py` with an ASGI
  `app`, bind `127.0.0.1` on `$PORT`, read `PG*` from env, provide `requirements.txt`
  and `GET /healthz`.

Push to `main` and infra deploys you to the box and points your domain at it.
