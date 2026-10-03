"""Database backup and restore utilities for SQLite and PostgreSQL with compression."""
from __future__ import annotations

import asyncio
import gzip
import logging
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    driver: str = os.getenv("DB_DRIVER", "sqlite")
    sqlite_path: str = os.getenv("DB_NAME", "app.db")
    pg_host: str = os.getenv("DB_HOST", "localhost")
    pg_port: str = os.getenv("DB_PORT", "5432")
    pg_name: str = os.getenv("DB_NAME", "app")
    pg_user: str = os.getenv("DB_USER", "postgres")
    pg_password: str = os.getenv("DB_PASSWORD", "")
    backup_dir: str = os.getenv("DB_BACKUP_DIR", "./backups")
    interval_seconds: int = int(os.getenv("DB_BACKUP_INTERVAL_SECONDS", "86400"))


class BackupManager:
    """Creates, compresses, restores, and schedules database backups."""

    def __init__(self, config: Optional[BackupConfig] = None) -> None:
        self.config = config or BackupConfig()
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        self._scheduler_task: Optional[asyncio.Task] = None

    def _timestamped_path(self, suffix: str) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return Path(self.config.backup_dir) / f"backup_{ts}.{suffix}"

    def backup_sqlite(self, compress: bool = True) -> Path:
        """Create a consistent SQLite backup using the native backup API."""
        src = sqlite3.connect(self.config.sqlite_path)
        dest_path = self._timestamped_path("db")
        dest = sqlite3.connect(str(dest_path))
        try:
            src.backup(dest)
        finally:
            dest.close()
            src.close()
        logger.info("SQLite backup written to %s", dest_path)
        return self._compress(dest_path) if compress else dest_path

    def backup_postgresql(self, compress: bool = True) -> Path:
        """Create a PostgreSQL backup using pg_dump."""
        dest_path = self._timestamped_path("sql")
        env = os.environ.copy()
        env["PGPASSWORD"] = self.config.pg_password
        cmd = [
            "pg_dump",
            "-h", self.config.pg_host,
            "-p", str(self.config.pg_port),
            "-U", self.config.pg_user,
            "-F", "p",
            "-f", str(dest_path),
            self.config.pg_name,
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {result.stderr}")
        logger.info("PostgreSQL backup written to %s", dest_path)
        return self._compress(dest_path) if compress else dest_path

    def backup(self, compress: bool = True) -> Path:
        """Dispatch to the correct backup method based on configured driver."""
        if self.config.driver == "sqlite":
            return self.backup_sqlite(compress=compress)
        if self.config.driver == "postgresql":
            return self.backup_postgresql(compress=compress)
        raise ValueError(f"Unsupported driver: {self.config.driver}")

    def _compress(self, path: Path) -> Path:
        gz_path = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        path.unlink(missing_ok=True)
        logger.info("Compressed backup to %s", gz_path)
        return gz_path

    def _decompress(self, path: Path) -> Path:
        out_path = path.with_suffix("")
        with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return out_path

    def restore_sqlite(self, backup_path: Path) -> None:
        """Restore a SQLite database from a backup file (optionally gzipped)."""
        source = self._decompress(backup_path) if backup_path.suffix == ".gz" else backup_path
        shutil.copyfile(source, self.config.sqlite_path)
        logger.info("Restored SQLite database from %s", source)

    def restore_postgresql(self, backup_path: Path) -> None:
        """Restore a PostgreSQL database from a SQL dump (optionally gzipped)."""
        source = self._decompress(backup_path) if backup_path.suffix == ".gz" else backup_path
        env = os.environ.copy()
        env["PGPASSWORD"] = self.config.pg_password
        cmd = [
            "psql",
            "-h", self.config.pg_host,
            "-p", str(self.config.pg_port),
            "-U", self.config.pg_user,
            "-d", self.config.pg_name,
            "-f", str(source),
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"psql restore failed: {result.stderr}")
        logger.info("Restored PostgreSQL database from %s", source)

    def restore(self, backup_path: Path) -> None:
        """Dispatch to the correct restore method based on configured driver."""
        if self.config.driver == "sqlite":
            self.restore_sqlite(backup_path)
        elif self.config.driver == "postgresql":
            self.restore_postgresql(backup_path)
        else:
            raise ValueError(f"Unsupported driver: {self.config.driver}")

    async def start_scheduler(
        self, on_complete: Optional[Callable[[Path], Awaitable[None]]] = None
    ) -> None:
        """Run periodic backups in the background at the configured interval."""

        async def _loop() -> None:
            while True:
                try:
                    path = await asyncio.to_thread(self.backup)
                    if on_complete:
                        await on_complete(path)
                except Exception:  # noqa: BLE001
                    logger.exception("Scheduled backup failed")
                await asyncio.sleep(self.config.interval_seconds)

        self._scheduler_task = asyncio.create_task(_loop())
        logger.info("Backup scheduler started, interval=%ss", self.config.interval_seconds)

    async def stop_scheduler(self) -> None:
        """Cancel the running backup scheduler task, if any."""
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            self._scheduler_task = None
            logger.info("Backup scheduler stopped")

    def health_check(self) -> bool:
        """Verify the backup directory is writable and backup tools are reachable."""
        try:
            test_file = Path(self.config.backup_dir) / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            if self.config.driver == "postgresql":
                return shutil.which("pg_dump") is not None and shutil.which("psql") is not None
            return Path(self.config.sqlite_path).exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("Backup health check failed: %s", exc)
            return False
