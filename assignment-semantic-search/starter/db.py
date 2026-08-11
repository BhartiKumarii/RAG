"""Database connection helper. GIVEN — you should not need to touch this.

Reads standard PG* environment variables:
  PGHOST (default localhost), PGPORT (5432), PGDATABASE (semantic_search),
  PGUSER (postgres), PGPASSWORD ("").

Every module opens its connection through get_conn() so the pgvector type
adapter is registered consistently.
"""
import os

import psycopg
from pgvector.psycopg import register_vector


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "semantic_search"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        autocommit=True,
    )
    # Teaches psycopg to send/receive `vector` columns as Python lists / numpy
    # arrays instead of raw strings. Requires the vector extension to exist,
    # which schema.sql creates.
    try:
        register_vector(conn)
    except Exception:
        # The extension may not be created yet on the very first connection
        # (e.g. right before applying schema.sql). Safe to ignore then.
        pass
    return conn
