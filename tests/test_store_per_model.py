"""Tests de una base por modelo de embeddings (mejora opcional 7.3) y del control de dimension (R-02)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.bootstrap import build_services, build_store, store_json_for, store_size, store_table_for
from lib.config import Settings
from lib.schemas import EmbeddingRecord


def _settings(tmp_path: Path, **kwargs) -> Settings:
    values = {"use_pgvector": False, "embeddings_path": tmp_path / "embeddings.json"}
    values.update(kwargs)
    return Settings(_env_file=None, **values)


def test_table_per_model_keeps_original_for_baseline():
    assert store_table_for("baseline") == "embeddings"
    assert store_table_for("resnet18_finetuned") == "embeddings_resnet18_finetuned"
    assert store_table_for("cnn_custom") == "embeddings_cnn_custom"


def test_json_per_model(tmp_path):
    s = _settings(tmp_path)
    assert store_json_for(s, "baseline") == tmp_path / "embeddings.json"
    assert store_json_for(s, "cnn_custom") == tmp_path / "embeddings_cnn_custom.json"


def test_embedding_dim_for_each_model(tmp_path):
    s = _settings(tmp_path)
    assert [s.embedding_dim_for(m) for m in ("baseline", "resnet18_finetuned", "cnn_custom")] == [2048, 512, 256]
    with pytest.raises(ValueError):
        s.embedding_dim_for("otro")


def test_build_store_separates_models_and_counts(tmp_path):
    s = _settings(tmp_path)
    baseline, custom = build_store(s, "baseline"), build_store(s, "cnn_custom")
    custom.append(EmbeddingRecord(id_imagen="a", embedding=[0.0] * 256, path="x.jpg", breed="Pug", metadata={}))
    assert store_size(baseline) == 0 and store_size(custom) == 1
    assert (tmp_path / "embeddings_cnn_custom.json").exists()


def test_store_size_prefers_count():
    class Counted:
        def count(self) -> int:
            return 7

        def all(self):
            raise AssertionError("no deberia leer todo")

    assert store_size(Counted()) == 7


def test_build_services_checks_embedding_dim_at_startup(tmp_path):
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        build_services(_settings(tmp_path, embedding_model="baseline", embedding_dim=512))


def test_table_name_validation():
    pytest.importorskip("psycopg")
    from lib.storage.pgvector_store import validate_table_name

    assert validate_table_name("embeddings_cnn_custom") == "embeddings_cnn_custom"
    for bad in ("Embeddings", "emb; DROP TABLE x", "1tabla", "con espacio", ""):
        with pytest.raises(ValueError):
            validate_table_name(bad)


def test_pgvector_table_per_model_roundtrip():
    """Contra la base real (docker compose up postgres -d); se saltea si no hay base."""
    pytest.importorskip("psycopg")
    from lib.config import settings
    from lib.storage.pgvector_store import PgVectorEmbeddingStore

    try:
        store = PgVectorEmbeddingStore(
            host=settings.postgres_host, port=settings.postgres_port, dbname=settings.postgres_db,
            user=settings.postgres_user, password=settings.postgres_password,
            embedding_dim=3, table="embeddings_pytest_tmp",
        )
    except Exception as exc:  # sin base corriendo
        pytest.skip(f"PostgreSQL no disponible: {exc}")
    try:
        store.append(EmbeddingRecord(id_imagen="a", embedding=[1.0, 0.0, 0.0], path="a.jpg", breed="Pug", metadata={}))
        store.append(EmbeddingRecord(id_imagen="b", embedding=[0.0, 1.0, 0.0], path="b.jpg", breed="Beagle", metadata={}))
        assert store.count() == 2
        assert [r.breed for r in store.search([0.9, 0.1, 0.0], k=2)] == ["Pug", "Beagle"]
    finally:
        with store.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS embeddings_pytest_tmp")
