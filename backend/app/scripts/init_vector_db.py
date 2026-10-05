"""Initialize filesystem prerequisites for vector DB backends."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    chroma_path = Path(os.getenv("CHROMA_PATH", "/app/data/chroma"))
    chroma_path.mkdir(parents=True, exist_ok=True)
    logger.info("Vector DB path ready at %s", chroma_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
