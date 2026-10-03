from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityType,
    ExecutionContext,
)


class FileAction:
    LIST = "list"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"
    INFO = "info"
    MKDIR = "mkdir"
    EXISTS = "exists"


class FileCapability(Capability[dict[str, Any]]):
    METADATA = CapabilityMetadata(
        name="file_agent",
        version="1.0.0",
        capability_type=CapabilityType.FILES,
        description="File system operations: read, write, list, search, delete files and directories",
        supported_actions=(
            FileAction.LIST,
            FileAction.READ,
            FileAction.WRITE,
            FileAction.DELETE,
            FileAction.SEARCH,
            FileAction.INFO,
            FileAction.MKDIR,
            FileAction.EXISTS,
        ),
        tags=("files", "storage", "file-system"),
    )

    def __init__(self, allowed_base_paths: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._allowed_base_paths = allowed_base_paths or [os.getcwd()]
        self._allowed_base_paths = [os.path.abspath(p) for p in self._allowed_base_paths]

    def _resolve_path(self, path: str) -> str:
        resolved = os.path.abspath(os.path.expanduser(path))
        for base in self._allowed_base_paths:
            if resolved.startswith(base):
                return resolved
        base_str = ", ".join(self._allowed_base_paths)
        raise PermissionError(f"Access denied: path must be under allowed directories: {base_str}")

    async def _do_initialize(self) -> None:
        self._logger.info(f"File capability initialized with base paths: {self._allowed_base_paths}")

    async def _do_shutdown(self) -> None:
        self._logger.info("File capability shut down")

    async def _do_execute(self, context: ExecutionContext) -> dict[str, Any]:
        action = context.action
        params = context.parameters

        if action == FileAction.LIST:
            return await self._handle_list(params)
        elif action == FileAction.READ:
            return await self._handle_read(params)
        elif action == FileAction.WRITE:
            return await self._handle_write(params)
        elif action == FileAction.DELETE:
            return await self._handle_delete(params)
        elif action == FileAction.SEARCH:
            return await self._handle_search(params)
        elif action == FileAction.INFO:
            return await self._handle_info(params)
        elif action == FileAction.MKDIR:
            return await self._handle_mkdir(params)
        elif action == FileAction.EXISTS:
            return await self._handle_exists(params)
        else:
            raise ValueError(f"Unsupported file action: {action}")

    async def _handle_list(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", "."))
        pattern = params.get("pattern", "*")
        include_hidden = params.get("include_hidden", False)

        entries: list[dict[str, Any]] = []
        try:
            for entry in os.scandir(path):
                if not include_hidden and entry.name.startswith("."):
                    continue
                if not self._matches_pattern(entry.name, pattern):
                    continue
                try:
                    stat = entry.stat()
                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": entry.is_dir(),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue
        except PermissionError as e:
            return {"success": False, "error": str(e), "entries": []}

        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return {"success": True, "path": path, "entries": entries, "count": len(entries)}

    async def _handle_read(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))
        encoding = params.get("encoding", "utf-8")
        max_size = params.get("max_size", 10 * 1024 * 1024)

        try:
            size = os.path.getsize(path)
            if size > max_size:
                return {"success": False, "error": f"File too large: {size} bytes (max: {max_size})"}

            with open(path, "r", encoding=encoding) as f:
                content = f.read()

            return {"success": True, "path": path, "content": content, "size": size}
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            return {"success": False, "error": str(e)}

    async def _handle_write(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))
        content = params.get("content", "")
        mode = params.get("mode", "w")
        encoding = params.get("encoding", "utf-8")

        if mode not in ("w", "a"):
            return {"success": False, "error": f"Unsupported write mode: {mode}"}

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return {"success": True, "path": path, "size": len(content)}
        except (OSError, PermissionError) as e:
            return {"success": False, "error": str(e)}

    async def _handle_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))
        recursive = params.get("recursive", False)

        try:
            if os.path.isdir(path):
                if not recursive:
                    return {"success": False, "error": "Cannot delete directory without recursive flag"}
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)
            return {"success": True, "path": path}
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {"success": False, "error": str(e)}

    async def _handle_search(self, params: dict[str, Any]) -> dict[str, Any]:
        root_path = self._resolve_path(params.get("path", "."))
        query = params.get("query", "").lower()
        pattern = params.get("pattern", "*")
        max_results = params.get("max_results", 50)
        content_search = params.get("content_search", False)

        results: list[dict[str, Any]] = []

        try:
            for entry in os.scandir(root_path):
                if len(results) >= max_results:
                    break
                try:
                    if not self._matches_pattern(entry.name, pattern):
                        continue
                    if content_search and not entry.is_dir():
                        try:
                            with open(entry.path, "r", encoding="utf-8", errors="ignore") as f:
                                head = f.read(4096)
                            if query not in head.lower():
                                continue
                        except OSError:
                            continue

                    stat = entry.stat()
                    results.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": entry.is_dir(),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue
        except PermissionError as e:
            return {"success": False, "error": str(e), "results": results}

        return {"success": True, "query": query, "results": results, "count": len(results)}

    async def _handle_info(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))

        try:
            stat = os.stat(path)
            return {
                "success": True,
                "path": path,
                "exists": True,
                "is_dir": os.path.isdir(path),
                "is_file": os.path.isfile(path),
                "is_link": os.path.islink(path),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "permissions": oct(stat.st_mode)[-3:],
            }
        except FileNotFoundError:
            return {"success": True, "path": path, "exists": False}

    async def _handle_mkdir(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))
        parents = params.get("parents", True)

        try:
            if parents:
                os.makedirs(path, exist_ok=True)
            else:
                os.mkdir(path)
            return {"success": True, "path": path}
        except (OSError, PermissionError) as e:
            return {"success": False, "error": str(e)}

    async def _handle_exists(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(params.get("path", ""))
        return {"success": True, "path": path, "exists": os.path.exists(path)}

    @staticmethod
    def _matches_pattern(name: str, pattern: str) -> bool:
        if pattern in ("*", "*.*"):
            return True
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in name
        if pattern.startswith("*"):
            return name.endswith(pattern[1:])
        if pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        return name == pattern
