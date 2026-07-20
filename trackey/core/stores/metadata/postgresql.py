import json
import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

from trackey.core.interfaces.store import IdentityRepository
from trackey.data.schemas.identity import Identity

logger = logging.getLogger(__name__)


class PostgresIdentityRepository(IdentityRepository):
    def __init__(self, repo_name: str, dsn: str):
        if psycopg2 is None:
            raise ImportError("PostgreSQL backend requires 'psycopg2-binary' to be installed.")
        self.repo_name = repo_name

        try:
            self._conn = psycopg2.connect(dsn)
        except psycopg2.OperationalError as e:
            logger.error(
                f"[PostgresIdentityRepository] Could not connect to database"
            )
            raise ConnectionError(
                f"Could not connect to PostgreSQL database."
            ) from e

        self._initialize()

    # -------------------------------------------------------
    # Schema
    # -------------------------------------------------------

    def _initialize(self):
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        global_id   UUID PRIMARY KEY,
                        label       TEXT,
                        metadata    JSONB DEFAULT '{{}}'::jsonb,
                        first_seen  TIMESTAMPTZ,
                        last_seen   TIMESTAMPTZ
                    )
                """).format(
                    sql.Identifier(self.repo_name)
                )
            )

            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} ON {} (label)
                """).format(
                    sql.Identifier(f"{self.repo_name}_label_idx"),
                    sql.Identifier(self.repo_name)
                )
            )

        self._conn.commit()

    # -------------------------------------------------------
    # CRUD
    # -------------------------------------------------------

    def save(self, identity: Identity) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (
                        global_id,
                        label,
                        metadata,
                        first_seen,
                        last_seen
                    )
                    VALUES (%s, %s, %s, %s, %s)

                    ON CONFLICT (global_id)
                    DO UPDATE SET
                        label = EXCLUDED.label,
                        metadata = EXCLUDED.metadata,
                        last_seen = EXCLUDED.last_seen
                """).format(
                    sql.Identifier(self.repo_name)
                ),
                (
                    str(identity.global_id),
                    identity.label,
                    json.dumps(identity.metadata),
                    identity.first_seen,
                    identity.last_seen,
                ),
            )

        self._conn.commit()

    def load(self, global_id: UUID) -> Optional[Identity]:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT
                        global_id,
                        label,
                        metadata,
                        first_seen,
                        last_seen
                    FROM {}
                    WHERE global_id = %s
                """).format(
                    sql.Identifier(self.repo_name)
                ),
                (str(global_id),),
            )

            row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_identity(row)

    def load_all(self) -> List[Identity]:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT
                        global_id,
                        label,
                        metadata,
                        first_seen,
                        last_seen
                    FROM {}
                """).format(
                    sql.Identifier(self.repo_name)
                )
            )

            rows = cur.fetchall()

        return [self._row_to_identity(r) for r in rows]

    def find_by_label(self, label: str) -> Optional[Identity]:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT
                        global_id,
                        label,
                        metadata,
                        first_seen,
                        last_seen
                    FROM {}
                    WHERE label = %s
                """).format(
                    sql.Identifier(self.repo_name)
                ),
                (label,),
            )

            row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_identity(row)

    def touch(self, global_id: UUID, last_seen: datetime) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    UPDATE {}
                    SET last_seen = %s
                    WHERE global_id = %s
                """).format(
                    sql.Identifier(self.repo_name)
                ),
                (
                    last_seen,
                    str(global_id),
                ),
            )

        self._conn.commit()

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    @staticmethod
    def _row_to_identity(row) -> Identity:
        return Identity(
            global_id=row[0] if isinstance(row[0], UUID) else UUID(row[0]),
            label=row[1],
            metadata=row[2] or {},
            first_seen=row[3],
            last_seen=row[4],
        )

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def close(self):
        self._conn.close()


if __name__ == '__main__':
    from uuid import uuid4
    con = "postgresql://trackey:password@localhost:5432/trackey"
    db = PostgresIdentityRepository(repo_name="people" , dsn=con)
    print(db.load_all())
    print(db.save(Identity(global_id="bf3d9301-36ad-487d-aeb6-8eb4120b434b", label="Mohamed")))
    print(db.save(Identity(global_id="6ccacb97-4a40-42d7-84c9-66c198db9dd7", label="Emad")))

    print(db.load_all())
