"""Preload prompt templates into the prompt registry."""

from __future__ import annotations

import logging
from pathlib import Path

from app.prompts.registry import get_default_registry

logger = logging.getLogger(__name__)


def main() -> int:
    template_dir = Path(__file__).resolve().parents[1] / "prompts"
    registry = get_default_registry(template_dir=template_dir)
    loaded = registry.register_directory(directory=template_dir)
    logger.info("Prompt warmup loaded %d templates", len(loaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
