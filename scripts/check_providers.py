"""Provider Validation Script — checks all LLM providers are healthy."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.providers import ProviderRegistry
from backend.app.utils.config import NVIDIASettings
from backend.app.utils.env_manager import parse_env_file

DISPLAY_NAMES = {
    "deepseek": "DeepSeek V4 Pro",
    "flash": "DeepSeek V4 Flash",
    "llama": "Llama 3.3 70B",
    "qwen": "Qwen 3.5 397B",
    "vision": "Llama 3.2 90B Vision",
    "embed": "NV-Embed-v1",
    "whisper": "Whisper Large v3",
    "tts": "Magpie TTS",
}


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
    if env_path.exists():
        for key, value in parse_env_file(env_path).items():
            if key not in os.environ:
                os.environ[key] = value


async def main():
    _load_env()
    settings = NVIDIASettings()
    registry = ProviderRegistry()
    registry.initialize_defaults(settings)

    results = await registry.health_check_all()

    passed = 0
    total = len(results)

    for name in ["deepseek", "flash", "llama", "qwen", "vision", "embed", "whisper", "tts"]:
        display = DISPLAY_NAMES.get(name, name)
        healthy = results.get(name, False)
        status = "Healthy" if healthy else "Failed"
        dots = "." * (15 - len(display))
        print(f"{display}{dots}{status}")
        if healthy:
            passed += 1

    print(f"\n{passed}/{total} Passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
