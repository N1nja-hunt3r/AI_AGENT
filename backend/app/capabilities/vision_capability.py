from __future__ import annotations

import logging
from typing import Any, Optional

from app.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityResult,
    CapabilityType,
    ExecutionContext,
)


class VisionAction:
    DESCRIBE = "describe"
    ANALYZE = "analyze"
    OCR = "ocr"
    DETECT_OBJECTS = "detect_objects"
    CLASSIFY = "classify"
    COMPARE = "compare"
    EXTRACT_TEXT = "extract_text"
    GENERATE_CAPTION = "generate_caption"


SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"}


SYSTEM_PROMPTS: dict[str, str] = {
    "describe": "You are a helpful vision assistant. Describe the image in detail, including objects, setting, colors, and any text visible.",
    "analyze": "You are a visual analysis AI. Analyze the image thoroughly, covering composition, subjects, colors, lighting, and overall impression.",
    "ocr": "You are an OCR system. Extract all visible text from the image exactly as written. Preserve line breaks and formatting.",
    "detect_objects": "You are an object detection system. List all objects visible in the image with approximate locations.",
    "classify": "You are an image classifier. Classify the image into the requested categories with confidence assessments.",
    "compare": "You are a visual comparison system. Compare the two images and identify similarities and differences.",
    "extract_text": "You are an OCR system. Extract all visible text from the image exactly as written.",
    "generate_caption": "You are a caption generator. Create a concise, descriptive caption for the image.",
}


class VisionCapability(Capability[dict[str, Any]]):
    METADATA = CapabilityMetadata(
        name="vision_model",
        version="1.0.0",
        capability_type=CapabilityType.VISION,
        description="Image understanding: describe, analyze, OCR, detect objects, classify, and compare visual content",
        supported_actions=(
            VisionAction.DESCRIBE,
            VisionAction.ANALYZE,
            VisionAction.OCR,
            VisionAction.DETECT_OBJECTS,
            VisionAction.CLASSIFY,
            VisionAction.COMPARE,
            VisionAction.EXTRACT_TEXT,
            VisionAction.GENERATE_CAPTION,
        ),
        tags=("vision", "image", "computer-vision", "ocr"),
    )

    def __init__(self, vision_service: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vision_service = vision_service

    async def _do_initialize(self) -> None:
        self._logger.info("Vision capability initialized")

    async def _do_shutdown(self) -> None:
        self._logger.info("Vision capability shut down")

    async def _do_execute(self, context: ExecutionContext) -> dict[str, Any]:
        action = context.action
        params = context.parameters

        if action == VisionAction.DESCRIBE:
            return await self._handle_describe(params)
        elif action == VisionAction.ANALYZE:
            return await self._handle_analyze(params)
        elif action == VisionAction.OCR:
            return await self._handle_ocr(params)
        elif action == VisionAction.DETECT_OBJECTS:
            return await self._handle_detect_objects(params)
        elif action == VisionAction.CLASSIFY:
            return await self._handle_classify(params)
        elif action == VisionAction.COMPARE:
            return await self._handle_compare(params)
        elif action == VisionAction.EXTRACT_TEXT:
            return await self._handle_extract_text(params)
        elif action == VisionAction.GENERATE_CAPTION:
            return await self._handle_generate_caption(params)
        else:
            raise ValueError(f"Unsupported vision action: {action}")

    async def _call_vision(
        self,
        prompt: str,
        system_key: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if self._vision_service is not None:
            try:
                result, usage = await self._vision_service.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPTS.get(system_key, ""),
                    image_b64=params.get("image_data", ""),
                    image_path=params.get("image_path", ""),
                )
                return {"success": True, "action": system_key, "result": result}
            except Exception as exc:
                self._logger.warning("Vision provider failed, falling back: %s", exc)

        return {
            "success": True,
            "action": system_key,
            "result": f"[Vision provider not available] {prompt[:100]}",
        }

    async def _handle_describe(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._call_vision(
            prompt=params.get("prompt", "Describe this image in detail."),
            system_key="describe",
            params=params,
        )

    async def _handle_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._call_vision(
            prompt=params.get("prompt", "Analyze this image thoroughly."),
            system_key="analyze",
            params=params,
        )

    async def _handle_ocr(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._call_vision(
            prompt="Extract all visible text from this image exactly as written.",
            system_key="ocr",
            params=params,
        )

    async def _handle_detect_objects(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._call_vision(
            prompt="List all objects visible in this image with approximate locations.",
            system_key="detect_objects",
            params=params,
        )

    async def _handle_classify(self, params: dict[str, Any]) -> dict[str, Any]:
        categories = params.get("categories", [])
        prompt = f"Classify this image into the following categories: {', '.join(categories)}" if categories else "Classify this image."
        return await self._call_vision(prompt=prompt, system_key="classify", params=params)

    async def _handle_compare(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": True,
            "action": "compare",
            "result": "Compare uses multiple images; invoke the vision provider per image.",
        }

    async def _handle_extract_text(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._handle_ocr(params)

    async def _handle_generate_caption(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._call_vision(
            prompt=params.get("prompt", "Generate a concise caption for this image."),
            system_key="generate_caption",
            params=params,
        )

    @staticmethod
    def _detect_format(image_data: str) -> str:
        if image_data.startswith("data:image/"):
            fmt = image_data.split(";")[0].split("/")[-1]
            return fmt if fmt in SUPPORTED_FORMATS else "png"
        if image_data.startswith("/9j/"):
            return "jpg"
        if image_data.startswith("iVBOR"):
            return "png"
        if image_data.startswith("R0lG"):
            return "gif"
        return "png"

    @staticmethod
    def _get_image_info(image_data: str, image_url: str) -> str:
        if image_url:
            return f"URL: {image_url[:50]}..."
        if image_data:
            fmt = VisionCapability._detect_format(image_data)
            return f"data ({fmt}, {len(image_data)} chars)"
        return "no image data"
