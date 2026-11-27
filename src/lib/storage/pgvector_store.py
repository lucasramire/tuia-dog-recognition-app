from __future__ import annotations

import json
import logging
import re

from pgvector.psycopg import register_vector
from psycopg import connect, sql

from lib.schemas import EmbeddingRecord

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "embeddings"
_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def validate_table_name(table: str) -> str:
    """Nombre de tabla en minusculas, sin comillas ni espacios (se usa como identificador SQL)."""
    if not _TABLE_NAME.match(table):
        raise ValueError(f"Nombre de tabla invalido para pgvector: '{table}'")
    return table


class PgVectorEmbeddingStore:
    """Base vectorial sobre PostgreSQL + pgvector (provista por la catedra).

    `table` permite una tabla por modelo de embeddings (mejora opcional 7.3): cada modelo tiene
    su dimension y una columna vector(dim) no acepta otra, asi que el selector de modelo de la
    Etapa 1 funciona sin reindexar. Por defecto, la tabla original `embeddings`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        embedding_dim: int = 512,
        table: str = DEFAULT_TABLE,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.table = validate_table_name(table)
        self.conn = connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            autocommit=True,
        )
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        expected_type = f"vector({self.embedding_dim})"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                  AND c.relname = %s
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """,
                (self.table,),
            )
            row = cur.fetchone()
            if row is not None and row[0] != expected_type:
                logger.warning(
                    "%s.embedding is %s but %s is required; dropping table (data removed)",
                    self.table,
                    row[0],
                    expected_type,
                )
                cur.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(self.table)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id_imagen TEXT PRIMARY KEY,
                        embedding vector({}) NOT NULL,
                        path TEXT NOT NULL,
                        breed TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                ).format(sql.Identifier(self.table), sql.Literal(int(self.embedding_dim)))
            )

    @staticmethod
    def _row_to_record(row: tuple) -> EmbeddingRecord:
        return EmbeddingRecord(
            id_imagen=row[0],
            embedding=list(row[1]),
            path=row[2],
            breed=row[3],
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
        )

    def all(self) -> list[EmbeddingRecord]:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id_imagen, embedding, path, breed, metadata FROM {}").format(
                    sql.Identifier(self.table)
                )
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Cantidad de embeddings indexados (sin traerlos a Python)."""
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self.table)))
            return int(cur.fetchone()[0])

    def append(self, record: EmbeddingRecord) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (id_imagen, embedding, path, breed, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id_imagen) DO NOTHING
                    """
                ).format(sql.Identifier(self.table)),
                (
                    record.id_imagen,
                    record.embedding,
                    record.path,
                    record.breed,
                    json.dumps(record.metadata, ensure_ascii=True),
                ),
            )

    def search(self, query: list[float], k: int = 10) -> list[EmbeddingRecord]:
        """Top-k vecinos por distancia coseno (operador <=> de pgvector)."""
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT id_imagen, embedding, path, breed, metadata
                    FROM {}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ).format(sql.Identifier(self.table)),
                (query, k),
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]
