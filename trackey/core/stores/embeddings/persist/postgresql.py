import logging
import numpy as np
from typing import Optional, List, Tuple
from uuid import UUID

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

from trackey.core.interfaces.store import EmbeddingRepository


logger = logging.getLogger(__name__)


class PostgresEmbeddingRepository(EmbeddingRepository):
    def __init__(self, dsn: str, table: str = "embeddings", retention_days: Optional[int] = None):
        if psycopg2 is None:
            raise ImportError("PostgreSQL backend requires 'psycopg2-binary' to be installed.")
        self.table = table
        self.retention_days = retention_days
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
                    created_at  TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (id, modality)
                )
            """).format(sql.Identifier(self.table)))
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} (global_id)"
            ).format(sql.Identifier(f"{self.table}_global_id_idx"), sql.Identifier(self.table)))
        self._conn.commit()
        if self.retention_days is not None:
            self.purge_expired()

    def purge_expired(self) -> None:
        """Delete rows whose created_at is older than retention_days."""
        if self.retention_days is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE created_at < NOW() - INTERVAL %s")
                .format(sql.Identifier(self.table)),
                (f"{self.retention_days} days",),
            )
            deleted = cur.rowcount
        self._conn.commit()
        if deleted > 0:
            logger.info(
                f"[PostgresEmbeddingRepository] purge_expired: removed {deleted} rows older than {self.retention_days} days"
            )

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
    import os
    from dotenv import load_dotenv
    load_dotenv()
    password = os.getenv("DATABASE_PASSWORD")
    con = f"postgresql://trackey:{password}@localhost:5432/trackey"
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

