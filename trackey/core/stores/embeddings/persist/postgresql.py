import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import UUID

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

from trackey.core.interfaces.store import EmbeddingRepository
from trackey.data.schemas.identity import Identity


logger = logging.getLogger(__name__)


class PostgresEmbeddingRepository(EmbeddingRepository):
    def __init__(self, dsn: str, table: str = "embeddings"):
        if psycopg2 is None:
            raise ImportError("PostgreSQL backend requires 'psycopg2-binary' to be installed.")
        self.table = table
        self._conn = psycopg2.connect(dsn)
        self._initialize()

    def _initialize(self):
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id          INTEGER NOT NULL,
                    global_id   UUID NOT NULL,
                    modality    TEXT NOT NULL,
                    vector      BYTEA NOT NULL,
                    PRIMARY KEY (id, modality),
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
            """).format(sql.Identifier(self.table)))
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} (global_id)"
            ).format(sql.Identifier(f"{self.table}_global_id_idx"), sql.Identifier(self.table)))
        self._conn.commit()

    def save(self, id: int, global_id: UUID, modality: str, vector: np.ndarray) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                INSERT INTO {} (id, global_id, modality, vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id, modality) DO NOTHING
            """).format(sql.Identifier(self.table)),
            (id, str(global_id), modality, vector.astype(np.float32).tobytes()))
        self._conn.commit()

    def delete(self, id: int, modality: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {} WHERE id = %s AND modality = %s").format(sql.Identifier(self.table)), (id,modality))
        self._conn.commit()

    def load_all(self) -> List[Tuple[int, UUID, str, np.ndarray]]:
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT id, global_id, modality, vector FROM {}").format(sql.Identifier(self.table)))
            rows = cur.fetchall()
        return [
            (r[0], UUID(r[1]) if isinstance(r[1], str) else r[1], r[2], np.frombuffer(r[3], dtype=np.float32))
            for r in rows
        ]


if __name__ == '__main__':
    from uuid import uuid4
    con = "postgresql://trackey:password@localhost:5432/trackey"
    db = PostgresEmbeddingRepository(dsn=con, table="people_body_embedding")
    d_body = 128
    d_face = 512
    nb = 100
    nq = 10

    xb_body = np.random.random((nb, d_body)).astype('float32')
    xq_body = np.random.random((nq, d_body)).astype('float32')

    xb_face = np.random.random((nb, d_face)).astype('float32')
    xq_face = np.random.random((nq, d_face)).astype('float32')

    current_body = uuid4()
    # print(db.load_all())
    # db.save(1, current_body, modality='face', vector=xb_face[0])
    # db.save(1, current_body, modality='body', vector=xb_body[0])
    print(db.load_all())
    
