"""
paths.py

Centralized, typed resolution of project filesystem paths: data, logs,
artifacts, config, and cache directories, with creation helpers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final


def _detect_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a project marker file."""
    markers = (".git", "pyproject.toml", "setup.py", "requirements.txt")
    current = start.resolve()
    for parent in (current, *current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current


class ProjectPaths:
    """Resolves and exposes canonical project directories."""

    def __init__(self, root: Path | str | None = None) -> None:
        env_root = os.environ.get("APP_ROOT_DIR")
        root_path: Path
        if root is not None:
            root_path = Path(root).resolve()
        elif env_root:
            root_path = Path(env_root).resolve()
        else:
            root_path = _detect_project_root(Path(__file__).parent)
        self.root: Final[Path] = root_path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def memory_store_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def vectordb_dir(self) -> Path:
        return self.data_dir / "vectordb"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.artifacts_dir / "exports"

    def all_managed_dirs(self) -> list[Path]:
        return [
            self.data_dir,
            self.logs_dir,
            self.config_dir,
            self.artifacts_dir,
            self.cache_dir,
            self.tmp_dir,
            self.models_dir,
            self.memory_store_dir,
            self.vectordb_dir,
            self.uploads_dir,
            self.exports_dir,
        ]

    def ensure_dirs(self) -> None:
        """Create all managed directories if they do not already exist."""
        for directory in self.all_managed_dirs():
            directory.mkdir(parents=True, exist_ok=True)

    def resolve(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.root.joinpath(*parts)

    def log_file(self, name: str) -> Path:
        """Resolve a path for a named log file within logs_dir."""
        return self.logs_dir / name

    def config_file(self, name: str) -> Path:
        """Resolve a path for a named config file within config_dir."""
        return self.config_dir / name

    def artifact_file(self, name: str) -> Path:
        """Resolve a path for a named artifact within artifacts_dir."""
        return self.artifacts_dir / name

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "config_dir": str(self.config_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "cache_dir": str(self.cache_dir),
            "tmp_dir": str(self.tmp_dir),
        }


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    """Return the process-wide cached ProjectPaths singleton."""
    return ProjectPaths()


def reset_project_paths_cache() -> None:
    """Clear the cached ProjectPaths singleton (primarily for testing)."""
    get_project_paths.cache_clear()
