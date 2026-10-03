"""
Kernel Capability Base — core abstractions for the agent kernel.

This module provides the base classes that all capabilities
in the AI OS platform extend.  Capabilities are pluggable
components that give the agent system specific functionality
(mouse, keyboard, browser, terminal, file operations, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CapabilityStatus(str, Enum):
    """Operational status of a capability."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    INITIALIZING = "initializing"


@dataclass
class CapabilityMetadata:
    """
    Describes a capability — its identity, version, and contract.
    """

    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    supported_actions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = "AI OS Team"


class Capability(ABC):
    """
    Abstract base class that every kernel capability must implement.

    Lifecycle
    ---------
    __init__  →  initialize()  →  execute()  →  shutdown()
                   ↑                                  │
                   └──── health_check() ◄──────────────┘
    """

    metadata: CapabilityMetadata
    status: CapabilityStatus = CapabilityStatus.INITIALIZING

    async def initialize(self) -> None:
        """
        Prepare the capability for first use.
        Override for any one-time setup (connection pools, warm caches, …).
        """
        logger.info("[%s] Initialised.", self.__class__.__name__)

    async def shutdown(self) -> None:
        """
        Gracefully tear down the capability.
        Override to release resources (close sockets, flush buffers, …).
        """
        logger.info("[%s] Shut down.", self.__class__.__name__)

    @abstractmethod
    async def execute(
        self,
        action: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform *action* with the given *params*.

        Parameters
        ----------
        action : str
            The action identifier (e.g. ``"mouse_click"``).
        params : dict
            Action-specific keyword arguments.
        context : dict or None
            Runtime context from the kernel (session, user, …).

        Returns
        -------
        dict
            A serialisable result (always a dict, even on failure).
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Return a summary of the capability's health."""
        return {"status": self.status.value}


__all__ = [
    "Capability",
    "CapabilityMetadata",
    "CapabilityStatus",
]
