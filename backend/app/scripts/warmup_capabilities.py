"""Eagerly discover and initialize capabilities."""

from __future__ import annotations

import asyncio
import logging

from app.integration.capability_loader import get_capability_loader

logger = logging.getLogger(__name__)


async def _run() -> int:
    loader = await get_capability_loader()
    loaded = await loader.load_all()
    logger.info("Capability warmup loaded %d capabilities", len(loaded))
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
