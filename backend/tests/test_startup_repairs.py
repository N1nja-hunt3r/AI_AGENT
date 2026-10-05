from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integration.app import _load_allowed_origins
from app.scripts import init_vector_db


def test_load_allowed_origins_defaults_to_explicit_local_origins(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert _load_allowed_origins() == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_load_allowed_origins_rejects_wildcard(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "*,http://localhost:3000")
    assert _load_allowed_origins() == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_init_vector_db_creates_chroma_path(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PATH", str(target))
    assert init_vector_db.main() == 0
    assert target.is_dir()
