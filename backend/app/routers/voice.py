"""
voice.py

FastAPI router for text-to-speech (TTS), speech-to-text (STT),
voice listing, voice session management, wake word detection,
assistant state, and greeting.

All endpoints mounted under /api/v1/voice.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.services.greeting import GreetingService, Personality, get_time_of_day
from app.services.assistant_state import AssistantState, StateConfig, state_machine
from app.services.wake_word import WakeWordState, wake_word_service

logger = logging.getLogger(__name__)

# Router loader uses this to set the prefix instead of the default
PREFIX = "/api/v1/voice"

router = APIRouter(tags=["voice"])

_greeting_service: Optional[GreetingService] = None


def get_greeting_service() -> GreetingService:
    global _greeting_service
    if _greeting_service is None:
        _greeting_service = GreetingService()
    return _greeting_service


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
_voice_sessions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class VoiceItem(BaseModel):
    id: str
    name: str
    gender: Optional[str] = None
    language: str = "en"
    category: str = "standard"


class VoicesResponse(BaseModel):
    voices: List[VoiceItem]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    voice: Optional[str] = None
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = "mp3"
    options: Optional[Dict[str, Any]] = None


class TTSResponse(BaseModel):
    audio_url: str
    format: str
    duration_seconds: float = 0.0
    model: str = ""
    latency_ms: float = 0.0


class STTResponse(BaseModel):
    text: str
    model: str = ""
    language: Optional[str] = None
    latency_ms: float = 0.0
    confidence: float = 0.0


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    voice: Optional[str] = None
    language: str = "en"
    settings: Optional[Dict[str, Any]] = None


class VoiceSession(BaseModel):
    id: str
    title: str
    voice: Optional[str] = None
    language: str = "en"
    settings: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    created_at: str
    updated_at: str


class CreateSessionResponse(BaseModel):
    session: VoiceSession


class GetSessionResponse(BaseModel):
    session: VoiceSession


class DeleteSessionResponse(BaseModel):
    success: bool
    id: str
    deleted_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Auth dependency (optional — not required for health or voice listing)
# ---------------------------------------------------------------------------
async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token",
        )
    return user


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    return getattr(request.state, "user", None)


# ---------------------------------------------------------------------------
# Service dependency helpers
# ---------------------------------------------------------------------------
def get_tts_service(request: Request) -> Any:
    return getattr(request.app.state, "tts_service", None)


def get_stt_service(request: Request) -> Any:
    return getattr(request.app.state, "stt_service", None)


# ---------------------------------------------------------------------------
# Wake Word
# ---------------------------------------------------------------------------
class WakeWordStatus(BaseModel):
    enabled: bool
    state: str
    wake_word: str
    activation_count: int
    cooldown_seconds: float


class WakeWordToggleRequest(BaseModel):
    enabled: bool


class WakeWordActivateRequest(BaseModel):
    confidence: float = 1.0


class WakeWordResponse(BaseModel):
    success: bool
    activated: bool
    state: str
    message: str = ""


# ---------------------------------------------------------------------------
# Assistant State
# ---------------------------------------------------------------------------
class AssistantStateResponse(BaseModel):
    state: str
    is_active: bool
    is_available: bool
    idle_timeout_seconds: int


class AssistantStateTransitionRequest(BaseModel):
    target: str
    reason: Optional[str] = None


class AssistantStateHistoryItem(BaseModel):
    from_state: str
    to_state: str
    reason: Optional[str] = None
    timestamp: float


class AssistantStateHistoryResponse(BaseModel):
    history: list[AssistantStateHistoryItem]


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------
class GreetingRequest(BaseModel):
    personality: str = "professional"
    user_name: Optional[str] = None
    classic_mode: bool = False
    greeting_type: str = "greeting"  # "greeting", "wake", "welcome_back"


class GreetingResponse(BaseModel):
    greeting: str
    personality: str
    time_of_day: str
    tone: str


class PersonalityListItem(BaseModel):
    id: str
    name: str
    description: str


# ---------------------------------------------------------------------------
# Mock voice data
# ---------------------------------------------------------------------------
_MOCK_VOICES: List[Dict[str, Any]] = [
    {"id": "en-US-Wavenet-D", "name": "Wavenet D (Male)", "gender": "male", "language": "en", "category": "wavenet"},
    {"id": "en-US-Wavenet-F", "name": "Wavenet F (Female)", "gender": "female", "language": "en", "category": "wavenet"},
    {"id": "en-US-Neural2-A", "name": "Neural2 A (Male)", "gender": "male", "language": "en", "category": "neural"},
    {"id": "en-US-Neural2-C", "name": "Neural2 C (Female)", "gender": "female", "language": "en", "category": "neural"},
    {"id": "en-GB-Neural2-B", "name": "British Neural2 B (Male)", "gender": "male", "language": "en-GB", "category": "neural"},
    {"id": "en-GB-Neural2-A", "name": "British Neural2 A (Female)", "gender": "female", "language": "en-GB", "category": "neural"},
    {"id": "es-ES-Neural2-B", "name": "Spanish Neural2 B (Male)", "gender": "male", "language": "es", "category": "neural"},
    {"id": "fr-FR-Neural2-A", "name": "French Neural2 A (Female)", "gender": "female", "language": "fr", "category": "neural"},
    {"id": "de-DE-Neural2-B", "name": "German Neural2 B (Male)", "gender": "male", "language": "de", "category": "neural"},
    {"id": "ja-JP-Neural2-B", "name": "Japanese Neural2 B (Male)", "gender": "male", "language": "ja", "category": "neural"},
    {"id": "zh-CN-Wavenet-A", "name": "Chinese Wavenet A (Female)", "gender": "female", "language": "zh", "category": "wavenet"},
    {"id": "hi-IN-Neural2-A", "name": "Hindi Neural2 A (Female)", "gender": "female", "language": "hi", "category": "neural"},
]


# ---------------------------------------------------------------------------
# GET /voices — list available voices
# ---------------------------------------------------------------------------
@router.get("/voices", response_model=VoicesResponse)
async def list_voices(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> VoicesResponse:
    """Return the list of available TTS voices."""
    return VoicesResponse(voices=[VoiceItem(**v) for v in _MOCK_VOICES])


# ---------------------------------------------------------------------------
# POST /tts — text-to-speech
# ---------------------------------------------------------------------------
@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    payload: TTSRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> TTSResponse:
    """Synthesize speech from text using the configured TTS provider."""
    tts_service = get_tts_service(request)

    if tts_service is not None:
        try:
            result = await tts_service.synthesize(
                text=payload.text,
                voice=payload.voice,
                language=payload.language,
                speed=payload.speed,
                **({"format": payload.format} if payload.format else {}),
                **(payload.options or {}),
            )
            return TTSResponse(
                audio_url=f"data:audio/{payload.format};base64,{result.audio_data.hex()}",
                format=payload.format,
                duration_seconds=result.duration_seconds,
                model=result.model,
                latency_ms=result.latency_ms,
            )
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    logger.warning("TTS service not available; returning mock data")
    return TTSResponse(
        audio_url="data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA",
        format=payload.format,
        duration_seconds=len(payload.text) * 0.06,
        model="mock-tts",
        latency_ms=120.0,
    )


# ---------------------------------------------------------------------------
# POST /stt — speech-to-text (multipart file upload)
# ---------------------------------------------------------------------------
@router.post("/stt", response_model=STTResponse)
async def transcribe_speech(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> STTResponse:
    """Transcribe an uploaded audio file using the configured STT provider."""
    stt_service = get_stt_service(request)

    try:
        audio_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if stt_service is not None:
        try:
            result = await stt_service.transcribe(
                audio_file=file.filename,
                audio_bytes=audio_bytes,
                language=language,
            )
            return STTResponse(
                text=result.text,
                model=result.model,
                language=result.language or language,
                latency_ms=result.latency_ms,
                confidence=1.0,
            )
        except Exception as exc:
            logger.exception("STT transcription failed")
            raise HTTPException(status_code=500, detail=f"STT transcription failed: {exc}") from exc

    logger.warning("STT service not available; returning mock transcription")
    return STTResponse(
        text="This is a mock transcription of the uploaded audio file.",
        model="mock-stt",
        language=language or "en",
        latency_ms=85.0,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# POST /sessions — create a voice session
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_voice_session(
    payload: CreateSessionRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> CreateSessionResponse:
    """Create a new voice session for tracking ongoing TTS/STT interactions."""
    session_id = str(uuid.uuid4())
    now = _now_iso()
    session: Dict[str, Any] = {
        "id": session_id,
        "title": payload.title or "Voice Session",
        "voice": payload.voice,
        "language": payload.language,
        "settings": payload.settings or {},
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    _voice_sessions[session_id] = session
    logger.info("Created voice session: %s", session_id)
    return CreateSessionResponse(session=VoiceSession(**session))


# ---------------------------------------------------------------------------
# GET /sessions/{id} — get a voice session
# ---------------------------------------------------------------------------
@router.get("/sessions/{id}", response_model=GetSessionResponse)
async def get_voice_session(
    id: str,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> GetSessionResponse:
    """Retrieve a voice session by its ID."""
    session = _voice_sessions.get(id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Voice session not found: {id}")
    return GetSessionResponse(session=VoiceSession(**session))


# ---------------------------------------------------------------------------
# DELETE /sessions/{id} — delete a voice session
# ---------------------------------------------------------------------------
@router.delete("/sessions/{id}", response_model=DeleteSessionResponse)
async def delete_voice_session(
    id: str,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> DeleteSessionResponse:
    """Delete a voice session by its ID."""
    if id not in _voice_sessions:
        raise HTTPException(status_code=404, detail=f"Voice session not found: {id}")
    del _voice_sessions[id]
    logger.info("Deleted voice session: %s", id)
    return DeleteSessionResponse(success=True, id=id, deleted_at=_now_iso())


# ---------------------------------------------------------------------------
# GET /wake-word — get wake word status
# ---------------------------------------------------------------------------
@router.get("/wake-word", response_model=WakeWordStatus)
async def get_wake_word_status(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> WakeWordStatus:
    """Get current wake word detection status."""
    stats = wake_word_service.get_stats()
    return WakeWordStatus(
        enabled=stats["enabled"],
        state=stats["state"],
        wake_word=stats["wake_word"],
        activation_count=stats["activation_count"],
        cooldown_seconds=stats["cooldown_seconds"],
    )


# ---------------------------------------------------------------------------
# PUT /wake-word — toggle wake word
# ---------------------------------------------------------------------------
@router.put("/wake-word", response_model=WakeWordStatus)
async def toggle_wake_word(
    payload: WakeWordToggleRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> WakeWordStatus:
    """Enable or disable wake word detection."""
    await wake_word_service.set_enabled(payload.enabled)
    stats = wake_word_service.get_stats()
    return WakeWordStatus(
        enabled=stats["enabled"],
        state=stats["state"],
        wake_word=stats["wake_word"],
        activation_count=stats["activation_count"],
        cooldown_seconds=stats["cooldown_seconds"],
    )


# ---------------------------------------------------------------------------
# POST /wake-word/activate — trigger wake word activation
# ---------------------------------------------------------------------------
@router.post("/wake-word/activate", response_model=WakeWordResponse)
async def trigger_wake_word(
    payload: WakeWordActivateRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> WakeWordResponse:
    """Trigger wake word activation (called by client-side detection)."""
    activated = await wake_word_service.activate(confidence=payload.confidence)
    stats = wake_word_service.get_stats()
    return WakeWordResponse(
        success=True,
        activated=activated,
        state=stats["state"],
        message="Wake word activated" if activated else "On cooldown",
    )


# ---------------------------------------------------------------------------
# POST /wake-word/deactivate — deactivate from wake word
# ---------------------------------------------------------------------------
@router.post("/wake-word/deactivate", response_model=WakeWordResponse)
async def deactivate_wake_word(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> WakeWordResponse:
    """Deactivate wake word after handling request."""
    await wake_word_service.deactivate()
    return WakeWordResponse(
        success=True,
        activated=False,
        state=wake_word_service.state.value,
        message="Deactivated",
    )


# ---------------------------------------------------------------------------
# GET /state — get assistant state
# ---------------------------------------------------------------------------
@router.get("/state", response_model=AssistantStateResponse)
async def get_assistant_state(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> AssistantStateResponse:
    """Get current assistant state."""
    return AssistantStateResponse(
        state=state_machine.state_name,
        is_active=state_machine.is_active(),
        is_available=state_machine.is_available(),
        idle_timeout_seconds=state_machine.get_config().idle_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# POST /state/transition — transition assistant state
# ---------------------------------------------------------------------------
@router.post("/state/transition", response_model=AssistantStateResponse)
async def transition_assistant_state(
    payload: AssistantStateTransitionRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> AssistantStateResponse:
    """Transition assistant to a new state."""
    try:
        target = AssistantState(payload.target)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {payload.target}")

    success = await state_machine.transition(target, reason=payload.reason)
    if not success:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot transition from {state_machine.state_name} to {payload.target}",
        )

    return AssistantStateResponse(
        state=state_machine.state_name,
        is_active=state_machine.is_active(),
        is_available=state_machine.is_available(),
        idle_timeout_seconds=state_machine.get_config().idle_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# GET /state/history — get state transition history
# ---------------------------------------------------------------------------
@router.get("/state/history", response_model=AssistantStateHistoryResponse)
async def get_state_history(
    request: Request,
    limit: int = 10,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> AssistantStateHistoryResponse:
    """Get recent state transition history."""
    history = state_machine.get_history(limit=limit)
    return AssistantStateHistoryResponse(
        history=[
            AssistantStateHistoryItem(
                from_state=e.from_state.value,
                to_state=e.to_state.value,
                reason=e.reason,
                timestamp=e.timestamp,
            )
            for e in history
        ]
    )


# ---------------------------------------------------------------------------
# GET /greeting — get a greeting
# ---------------------------------------------------------------------------
@router.get("/greeting", response_model=GreetingResponse)
async def get_greeting(
    request: Request,
    personality: str = "professional",
    user_name: Optional[str] = None,
    classic_mode: bool = False,
    greeting_type: str = "greeting",
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> GreetingResponse:
    """Generate a time-appropriate greeting."""
    service = get_greeting_service()

    try:
        svc_personality = Personality(personality)
    except ValueError:
        svc_personality = Personality.PROFESSIONAL

    service.set_personality(svc_personality)
    if classic_mode:
        service.set_classic_mode(True)
    if user_name:
        service.set_user_name(user_name)

    if greeting_type == "wake":
        text = service.wake_greeting(user_name)
    elif greeting_type == "welcome_back":
        text = service.welcome_back(user_name)
    else:
        text = service.greet(user_name)

    return GreetingResponse(
        greeting=text,
        personality=personality,
        time_of_day=get_time_of_day().value,
        tone=service.get_profile().tone,
    )


# ---------------------------------------------------------------------------
# POST /greeting — generate a greeting with full config
# ---------------------------------------------------------------------------
@router.post("/greeting", response_model=GreetingResponse)
async def custom_greeting(
    payload: GreetingRequest,
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> GreetingResponse:
    """Generate a custom greeting with full personality config."""
    service = get_greeting_service()

    try:
        svc_personality = Personality(payload.personality)
    except ValueError:
        svc_personality = Personality.PROFESSIONAL

    service.set_personality(svc_personality)
    service.set_classic_mode(payload.classic_mode)
    if payload.user_name:
        service.set_user_name(payload.user_name)

    if payload.greeting_type == "wake":
        text = service.wake_greeting(payload.user_name)
    elif payload.greeting_type == "welcome_back":
        text = service.welcome_back(payload.user_name)
    else:
        text = service.greet(payload.user_name)

    return GreetingResponse(
        greeting=text,
        personality=payload.personality,
        time_of_day=get_time_of_day().value,
        tone=service.get_profile().tone,
    )


# ---------------------------------------------------------------------------
# GET /personalities — list available personalities
# ---------------------------------------------------------------------------
_PERSONALITIES = [
    PersonalityListItem(id="professional", name="Professional", description="Polite and efficient, suitable for work environments."),
    PersonalityListItem(id="friendly", name="Friendly", description="Warm and casual, like talking to a friend."),
    PersonalityListItem(id="jarvis", name="JARVIS", description="Sophisticated assistant, inspired by Iron Man's AI."),
    PersonalityListItem(id="minimal", name="Minimal", description="Short and to the point. No fluff."),
    PersonalityListItem(id="custom", name="Custom", description="Fully customizable greetings and responses."),
]


@router.get("/personalities", response_model=list[PersonalityListItem])
async def list_personalities(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> list[PersonalityListItem]:
    """List available voice personalities."""
    return _PERSONALITIES
