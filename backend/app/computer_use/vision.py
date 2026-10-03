from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from app.computer_use.base_computer import (
    CapabilityMetadata,
    CapabilityPriority,
    ComputerCapability,
    ExecutionRequest,
    HealthCheckResult,
    HealthStatus,
)
from app.providers.provider_registry import ProviderRegistry


class VisionError(Exception):
    pass


class ImageDecodeError(VisionError):
    pass


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TextRegion:
    text: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class OcrResult:
    full_text: str
    regions: tuple[TextRegion, ...]
    duration_seconds: float


class DetectedObjectClass(Enum):
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    ICON = "icon"
    IMAGE = "image"
    CHECKBOX = "checkbox"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedObject:
    object_class: DetectedObjectClass
    bounding_box: BoundingBox
    confidence: float
    label: str = ""


@dataclass(frozen=True)
class ImageAnalysis:
    width: int
    height: int
    channels: int
    mean_brightness: float
    dominant_colors: tuple[tuple[int, int, int], ...]
    edge_density: float
    duration_seconds: float


@dataclass(frozen=True)
class ScreenUnderstanding:
    ocr_result: OcrResult
    detected_objects: tuple[DetectedObject, ...]
    image_analysis: ImageAnalysis


class OcrEngine:
    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        raise NotImplementedError


class TesseractOcrEngine(OcrEngine):
    def __init__(self, language: str = "eng") -> None:
        self._language = language

    def _run_sync(self, image_bytes: bytes) -> OcrResult:
        import pytesseract
        from PIL import Image

        start = time.monotonic()
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ImageDecodeError(f"failed to decode image: {exc}") from exc

        data = pytesseract.image_to_data(
            image, lang=self._language, output_type=pytesseract.Output.DICT
        )
        regions: list[TextRegion] = []
        texts: list[str] = []

        for index in range(len(data.get("text", []))):
            text = data["text"][index].strip()
            if not text:
                continue
            confidence_raw = data["conf"][index]
            try:
                confidence = float(confidence_raw) / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            regions.append(
                TextRegion(
                    text=text,
                    confidence=max(0.0, confidence),
                    bounding_box=BoundingBox(
                        x=int(data["left"][index]),
                        y=int(data["top"][index]),
                        width=int(data["width"][index]),
                        height=int(data["height"][index]),
                    ),
                )
            )
            texts.append(text)

        return OcrResult(
            full_text=" ".join(texts),
            regions=tuple(regions),
            duration_seconds=time.monotonic() - start,
        )

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        return await asyncio.to_thread(self._run_sync, image_bytes)


class ObjectDetector:
    async def detect(self, image_bytes: bytes) -> tuple[DetectedObject, ...]:
        raise NotImplementedError


class OpenCvContourDetector(ObjectDetector):
    def __init__(
        self,
        min_area: int = 100,
        max_area_ratio: float = 0.9,
    ) -> None:
        self._min_area = min_area
        self._max_area_ratio = max_area_ratio

    def _detect_sync(self, image_bytes: bytes) -> tuple[DetectedObject, ...]:
        import cv2
        import numpy as np

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError("failed to decode image for object detection")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = width * height * self._max_area_ratio
        detections: list[DetectedObject] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0.0

            if 0.8 <= aspect_ratio <= 6.0 and h < height * 0.15:
                object_class = DetectedObjectClass.BUTTON
            elif aspect_ratio > 4.0:
                object_class = DetectedObjectClass.TEXT_FIELD
            elif 0.9 <= aspect_ratio <= 1.1 and w < width * 0.05:
                object_class = DetectedObjectClass.ICON
            else:
                object_class = DetectedObjectClass.UNKNOWN

            confidence = min(1.0, area / max_area + 0.3)
            detections.append(
                DetectedObject(
                    object_class=object_class,
                    bounding_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    confidence=confidence,
                )
            )

        return tuple(detections)

    async def detect(self, image_bytes: bytes) -> tuple[DetectedObject, ...]:
        return await asyncio.to_thread(self._detect_sync, image_bytes)


class ImageAnalyzer:
    def _analyze_sync(self, image_bytes: bytes) -> ImageAnalysis:
        import cv2
        import numpy as np

        start = time.monotonic()
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError("failed to decode image for analysis")

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)
        k = min(5, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        dominant_colors = tuple(
            (int(b), int(g), int(r)) for b, g, r in centers.astype(int)
        )

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(width * height)

        return ImageAnalysis(
            width=width,
            height=height,
            channels=channels,
            mean_brightness=mean_brightness,
            dominant_colors=dominant_colors,
            edge_density=edge_density,
            duration_seconds=time.monotonic() - start,
        )

    async def analyze(self, image_bytes: bytes) -> ImageAnalysis:
        return await asyncio.to_thread(self._analyze_sync, image_bytes)


class NvidiaVisionEngine:
    """Vision engine that uses the NVIDIA vision API for image understanding."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._provider = registry.get("vision")

    async def describe_image(self, image_bytes: bytes, prompt: str = "Describe this image in detail.") -> str:
        import base64
        image_b64 = base64.b64encode(image_bytes).decode()
        content, _ = await self._provider.generate(
            prompt=prompt,
            image_b64=image_b64,
        )
        return content


class VisionCapability(ComputerCapability):
    def __init__(
        self,
        ocr_engine: Optional[OcrEngine] = None,
        object_detector: Optional[ObjectDetector] = None,
        image_analyzer: Optional[ImageAnalyzer] = None,
        nvidia_vision: Optional[NvidiaVisionEngine] = None,
        metadata: Optional[CapabilityMetadata] = None,
    ) -> None:
        super().__init__(
            metadata
            or CapabilityMetadata(
                name="vision",
                version="1.0.0",
                description="OCR, object detection, and image analysis for screen understanding",
                priority=CapabilityPriority.NORMAL,
                timeout_seconds=20.0,
                approval_required=False,
            )
        )
        self._ocr_engine = ocr_engine or TesseractOcrEngine()
        self._object_detector = object_detector or OpenCvContourDetector()
        self._image_analyzer = image_analyzer or ImageAnalyzer()
        self._nvidia_vision = nvidia_vision

    async def _on_initialize(self) -> None:
        pass

    async def _on_shutdown(self) -> None:
        pass

    async def _on_health_check(self) -> HealthCheckResult:
        try:
            import cv2  # noqa: F401
            import pytesseract  # noqa: F401

            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.HEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
            )
        except ImportError as exc:
            return HealthCheckResult(
                capability_name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_seconds=None,
                checked_at_epoch=time.time(),
                detail=f"missing dependency: {exc}",
            )

    async def _on_execute(self, request: ExecutionRequest) -> Any:
        action = request.action
        params = request.parameters
        image_bytes = bytes(params["image_bytes"]) if "image_bytes" in params else b""

        if action == "ocr" or action == "extract_text":
            return await self.extract_text(image_bytes)
        if action == "detect_objects":
            return await self.detect_objects(image_bytes)
        if action == "analyze_image":
            return await self.analyze_image(image_bytes)
        if action == "understand_screen":
            return await self.understand_screen(image_bytes)
        if action == "describe_image" and self._nvidia_vision:
            prompt = params.get("prompt", "Describe this image in detail.")
            return await self._nvidia_vision.describe_image(image_bytes, prompt=prompt)

        raise ValueError(f"unknown vision action: {action}")

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        if not image_bytes:
            raise VisionError("image_bytes is required for OCR")
        return await self._ocr_engine.extract_text(image_bytes)

    async def detect_objects(self, image_bytes: bytes) -> tuple[DetectedObject, ...]:
        if not image_bytes:
            raise VisionError("image_bytes is required for object detection")
        return await self._object_detector.detect(image_bytes)

    async def analyze_image(self, image_bytes: bytes) -> ImageAnalysis:
        if not image_bytes:
            raise VisionError("image_bytes is required for image analysis")
        return await self._image_analyzer.analyze(image_bytes)

    async def understand_screen(self, image_bytes: bytes) -> ScreenUnderstanding:
        if not image_bytes:
            raise VisionError("image_bytes is required for screen understanding")

        ocr_result, detected_objects, image_analysis = await asyncio.gather(
            self._ocr_engine.extract_text(image_bytes),
            self._object_detector.detect(image_bytes),
            self._image_analyzer.analyze(image_bytes),
        )
        return ScreenUnderstanding(
            ocr_result=ocr_result,
            detected_objects=detected_objects,
            image_analysis=image_analysis,
        )
