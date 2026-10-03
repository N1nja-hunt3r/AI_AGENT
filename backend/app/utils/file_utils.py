"""
file_utils.py

File system utilities: safe read/write, move, delete, compression
(gzip/zip), and metadata extraction.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class FileMetadata:
    path: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    is_dir: bool
    extension: str
    checksum_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "is_dir": self.is_dir,
            "extension": self.extension,
            "checksum_sha256": self.checksum_sha256,
        }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents."""
    return Path(path).read_text(encoding=encoding)


def write_text(path: PathLike, content: str, *, encoding: str = "utf-8", atomic: bool = True) -> None:
    """Write text to a file, optionally using an atomic write-and-rename."""
    target = Path(path)
    _ensure_parent(target)
    if atomic:
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(content, encoding=encoding)
        tmp_path.replace(target)
    else:
        target.write_text(content, encoding=encoding)


def read_bytes(path: PathLike) -> bytes:
    """Read a binary file and return its contents."""
    return Path(path).read_bytes()


def write_bytes(path: PathLike, data: bytes, *, atomic: bool = True) -> None:
    """Write binary data to a file, optionally using an atomic write-and-rename."""
    target = Path(path)
    _ensure_parent(target)
    if atomic:
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(target)
    else:
        target.write_bytes(data)


def read_json(path: PathLike) -> Any:
    """Read and parse a JSON file."""
    return json.loads(read_text(path))


def write_json(path: PathLike, data: Any, *, indent: int = 2, atomic: bool = True) -> None:
    """Serialize data as JSON and write it to a file."""
    write_text(path, json.dumps(data, indent=indent, default=str), atomic=atomic)


def append_text(path: PathLike, content: str, *, encoding: str = "utf-8") -> None:
    """Append text to a file, creating it if necessary."""
    target = Path(path)
    _ensure_parent(target)
    with target.open("a", encoding=encoding) as handle:
        handle.write(content)


def move_file(src: PathLike, dst: PathLike, *, overwrite: bool = False) -> Path:
    """Move a file from `src` to `dst`."""
    source = Path(src)
    destination = Path(dst)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    _ensure_parent(destination)
    shutil.move(str(source), str(destination))
    return destination


def copy_file(src: PathLike, dst: PathLike, *, overwrite: bool = False) -> Path:
    """Copy a file from `src` to `dst`, preserving metadata."""
    source = Path(src)
    destination = Path(dst)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    _ensure_parent(destination)
    shutil.copy2(str(source), str(destination))
    return destination


def delete_file(path: PathLike, *, missing_ok: bool = True) -> bool:
    """Delete a file; returns True if a file was removed."""
    target = Path(path)
    try:
        target.unlink(missing_ok=missing_ok)
        return True
    except FileNotFoundError:
        if missing_ok:
            return False
        raise


def delete_dir(path: PathLike, *, missing_ok: bool = True) -> bool:
    """Recursively delete a directory; returns True if it was removed."""
    target = Path(path)
    if not target.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(str(target))
    shutil.rmtree(target)
    return True


def list_files(directory: PathLike, *, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """List files in a directory matching a glob pattern."""
    base = Path(directory)
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return sorted(p for p in iterator if p.is_file())


def compute_checksum(path: PathLike, *, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Compute a hex digest checksum for a file using the given algorithm."""
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_metadata(path: PathLike, *, with_checksum: bool = False) -> FileMetadata:
    """Build a FileMetadata snapshot for a given path."""
    target = Path(path)
    stat = target.stat()
    return FileMetadata(
        path=str(target),
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        is_dir=target.is_dir(),
        extension=target.suffix.lower(),
        checksum_sha256=compute_checksum(target) if with_checksum and target.is_file() else None,
    )


def gzip_file(src: PathLike, dst: Optional[PathLike] = None) -> Path:
    """Compress a file using gzip, returning the output path."""
    source = Path(src)
    destination = Path(dst) if dst else source.with_suffix(source.suffix + ".gz")
    _ensure_parent(destination)
    with source.open("rb") as f_in, gzip.open(destination, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return destination


def gunzip_file(src: PathLike, dst: Optional[PathLike] = None) -> Path:
    """Decompress a gzip file, returning the output path."""
    source = Path(src)
    destination = Path(dst) if dst else source.with_suffix("")
    _ensure_parent(destination)
    with gzip.open(source, "rb") as f_in, destination.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return destination


def zip_paths(paths: Iterable[PathLike], dst: PathLike, *, base_dir: Optional[PathLike] = None) -> Path:
    """Create a zip archive containing the given paths."""
    destination = Path(dst)
    _ensure_parent(destination)
    root = Path(base_dir) if base_dir else None
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in paths:
            item_path = Path(item)
            arcname = item_path.relative_to(root) if root else item_path.name
            archive.write(item_path, arcname=str(arcname))
    return destination


def unzip_archive(src: PathLike, dst_dir: PathLike) -> List[Path]:
    """Extract a zip archive into a target directory, returning extracted paths."""
    destination_dir = Path(dst_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as archive:
        archive.extractall(destination_dir)
        return [destination_dir / name for name in archive.namelist()]


def safe_join(base: PathLike, *parts: str) -> Path:
    """Join path parts under `base`, raising if traversal escapes the base."""
    base_resolved = Path(base).resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        raise ValueError(f"Path traversal detected: {candidate} escapes {base_resolved}")
    return candidate


def ensure_directory(path: PathLike) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
