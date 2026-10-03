"""
greeting.py

Time-aware greeting service with configurable personality profiles.
Supports time-of-day greetings, configurable personalities (Professional,
Friendly, JARVIS, Minimal, Custom), and classic mode.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Personality(Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    JARVIS = "jarvis"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class TimeOfDay(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


@dataclass
class PersonalityProfile:
    name: str
    greeting_templates: dict[TimeOfDay, list[str]]
    wake_greetings: list[str]
    farewells: list[str]
    thinking_phrases: list[str]
    confirmation_phrases: list[str]
    error_phrases: list[str]
    tone: str
    use_title: bool = False
    title: str = ""

    def get_greeting(self, time_of_day: TimeOfDay, user_name: Optional[str] = None) -> str:
        templates = self.greeting_templates.get(time_of_day, self.greeting_templates[TimeOfDay.MORNING])
        template = random.choice(templates)
        if user_name:
            return template.replace("{name}", user_name)
        return template

    def get_wake_greeting(self, user_name: Optional[str] = None) -> str:
        template = random.choice(self.wake_greetings)
        if user_name:
            return template.replace("{name}", user_name)
        return template


PROFESSIONAL_PROFILE = PersonalityProfile(
    name="Professional",
    tone="professional",
    use_title=False,
    greeting_templates={
        TimeOfDay.MORNING: [
            "Good morning, {name}.",
            "Good morning. How can I assist you today?",
            "Good morning. I hope you're starting your day well.",
        ],
        TimeOfDay.AFTERNOON: [
            "Good afternoon, {name}.",
            "Good afternoon. What can I help you with?",
            "Good afternoon. How may I be of service?",
        ],
        TimeOfDay.EVENING: [
            "Good evening, {name}.",
            "Good evening. What would you like to work on?",
            "Good evening. How can I assist?",
        ],
        TimeOfDay.NIGHT: [
            "Good evening, {name}. Working late?",
            "Hello {name}. What can I help you with tonight?",
        ],
    },
    wake_greetings=[
        "Hello, {name}. Systems ready.",
        "At your service, {name}.",
        "Yes, {name}? How can I help?",
        "Ready when you are, {name}.",
    ],
    farewells=[
        "Goodbye.",
        "Until next time.",
        "Signing off.",
    ],
    thinking_phrases=[
        "Processing your request.",
        "One moment please.",
        "Working on that.",
    ],
    confirmation_phrases=[
        "Certainly.",
        "Absolutely.",
        "Right away.",
        "Consider it done.",
    ],
    error_phrases=[
        "I encountered an issue.",
        "Something went wrong.",
        "I apologize, but I'm unable to complete that.",
    ],
)

FRIENDLY_PROFILE = PersonalityProfile(
    name="Friendly",
    tone="warm",
    use_title=False,
    greeting_templates={
        TimeOfDay.MORNING: [
            "Good morning, {name}! Ready to tackle the day?",
            "Hey {name}, good morning! What's on your mind?",
            "Morning, {name}! How can I brighten your day?",
        ],
        TimeOfDay.AFTERNOON: [
            "Hey {name}, good afternoon! What's up?",
            "Good afternoon, {name}! What can I do for you?",
            "Afternoon, {name}! How's your day going?",
        ],
        TimeOfDay.EVENING: [
            "Good evening, {name}! Hope you had a great day.",
            "Hey {name}, good evening! What are we working on?",
            "Evening, {name}! Still going strong?",
        ],
        TimeOfDay.NIGHT: [
            "Hey {name}! Burning the midnight oil?",
            "Good evening, {name}! What brings you here tonight?",
        ],
    },
    wake_greetings=[
        "Hey {name}! What's up?",
        "Yes, {name}? What can I do for you?",
        "Hey! Ready to help, {name}.",
        "What's good, {name}?",
    ],
    farewells=[
        "Catch you later!",
        "Bye! Talk soon.",
        "See you later!",
    ],
    thinking_phrases=[
        "Give me a sec...",
        "Let me think about that...",
        "Working on it!",
        "One moment...",
    ],
    confirmation_phrases=[
        "Sure thing!",
        "You got it!",
        "On it!",
        "No problem!",
    ],
    error_phrases=[
        "Oops, something went wrong.",
        "Hmm, that didn't work.",
        "Sorry, I ran into an issue.",
    ],
)

JARVIS_PROFILE = PersonalityProfile(
    name="JARVIS",
    tone="sophisticated",
    use_title=False,
    greeting_templates={
        TimeOfDay.MORNING: [
            "Good morning, sir. Systems are fully operational.",
            "Good morning, {name}. All systems nominal.",
            "Morning, sir. Ready for your directives.",
        ],
        TimeOfDay.AFTERNOON: [
            "Good afternoon, sir. At your disposal.",
            "Good afternoon, {name}. Standing by.",
            "Afternoon, sir. What's the plan?",
        ],
        TimeOfDay.EVENING: [
            "Good evening, sir. I trust your day was productive.",
            "Good evening, {name}. Ready and waiting.",
            "Evening, sir. Shall we proceed?",
        ],
        TimeOfDay.NIGHT: [
            "Working late, sir? I admire your dedication.",
            "Good evening, {name}. I'm here as always.",
            "Night owl mode activated, sir.",
        ],
    },
    wake_greetings=[
        "At your service, sir.",
        "Yes, sir? How may I assist?",
        "Ready and waiting, sir.",
        "I'm here, sir. What do you need?",
    ],
    farewells=[
        "Goodbye, sir. Shall I power down?",
        "Until next time, sir.",
        "Signing off, sir.",
    ],
    thinking_phrases=[
        "Processing, sir.",
        "Analyzing the data now.",
        "Running diagnostics.",
        "One moment while I compute.",
    ],
    confirmation_phrases=[
        "Certainly, sir.",
        "Right away, sir.",
        "Already on it.",
        "Consider it handled.",
    ],
    error_phrases=[
        "I apologize, sir, but I encountered an error.",
        "My apologies. Something doesn't seem right.",
        "Sir, I'm afraid I've run into a problem.",
    ],
)

MINIMAL_PROFILE = PersonalityProfile(
    name="Minimal",
    tone="neutral",
    use_title=False,
    greeting_templates={
        TimeOfDay.MORNING: ["Good morning."],
        TimeOfDay.AFTERNOON: ["Good afternoon."],
        TimeOfDay.EVENING: ["Good evening."],
        TimeOfDay.NIGHT: ["Hello."],
    },
    wake_greetings=[
        "Yes?",
        "Go ahead.",
        "Listening.",
    ],
    farewells=["Bye."],
    thinking_phrases=["...", "Processing..."],
    confirmation_phrases=["OK.", "Sure.", "Got it."],
    error_phrases=["Error.", "Failed.", "Can't do that."],
)

CUSTOM_PROFILE = PersonalityProfile(
    name="Custom",
    tone="custom",
    use_title=False,
    greeting_templates={
        TimeOfDay.MORNING: ["Hello {name}."],
        TimeOfDay.AFTERNOON: ["Hello {name}."],
        TimeOfDay.EVENING: ["Hello {name}."],
        TimeOfDay.NIGHT: ["Hello {name}."],
    },
    wake_greetings=["Hello {name}."],
    farewells=["Goodbye."],
    thinking_phrases=["One moment..."],
    confirmation_phrases=["Okay."],
    error_phrases=["Sorry."],
)

PERSONALITY_MAP = {
    Personality.PROFESSIONAL: PROFESSIONAL_PROFILE,
    Personality.FRIENDLY: FRIENDLY_PROFILE,
    Personality.JARVIS: JARVIS_PROFILE,
    Personality.MINIMAL: MINIMAL_PROFILE,
    Personality.CUSTOM: CUSTOM_PROFILE,
}


def get_time_of_day(hour: Optional[int] = None) -> TimeOfDay:
    if hour is None:
        hour = datetime.now().hour
    if 5 <= hour < 12:
        return TimeOfDay.MORNING
    if 12 <= hour < 17:
        return TimeOfDay.AFTERNOON
    if 17 <= hour < 22:
        return TimeOfDay.EVENING
    return TimeOfDay.NIGHT


@dataclass
class GreetingConfig:
    personality: Personality = Personality.PROFESSIONAL
    user_name: Optional[str] = None
    classic_mode: bool = False
    custom_greetings: dict[TimeOfDay, list[str]] = field(default_factory=dict)
    custom_wake_greetings: list[str] = field(default_factory=list)


class GreetingService:
    def __init__(self, config: Optional[GreetingConfig] = None) -> None:
        self._config = config or GreetingConfig()
        self._last_greeting_type: Optional[str] = None

    def get_profile(self) -> PersonalityProfile:
        if self._config.classic_mode:
            return JARVIS_PROFILE
        if self._config.personality == Personality.CUSTOM:
            profile = CUSTOM_PROFILE
            if self._config.custom_greetings:
                profile.greeting_templates.update(self._config.custom_greetings)
            if self._config.custom_wake_greetings:
                profile.wake_greetings = self._config.custom_wake_greetings
            return profile
        return PERSONALITY_MAP.get(self._config.personality, PROFESSIONAL_PROFILE)

    def greet(self, user_name: Optional[str] = None) -> str:
        profile = self.get_profile()
        name = user_name or self._config.user_name
        time_of_day = get_time_of_day()
        greeting = profile.get_greeting(time_of_day, name)
        self._last_greeting_type = "greeting"
        return greeting

    def wake_greeting(self, user_name: Optional[str] = None) -> str:
        profile = self.get_profile()
        name = user_name or self._config.user_name
        greeting = profile.get_wake_greeting(name)
        self._last_greeting_type = "wake"
        return greeting

    def welcome_back(self, user_name: Optional[str] = None) -> str:
        name = user_name or self._config.user_name
        templates = [
            "Welcome back, {name}.",
            "Good to see you again, {name}.",
            "Back again, {name}? Ready to go.",
        ]
        template = random.choice(templates)
        self._last_greeting_type = "welcome_back"
        if name:
            return template.replace("{name}", name)
        return template

    def get_config(self) -> GreetingConfig:
        return self._config

    def set_personality(self, personality: Personality) -> None:
        self._config.personality = personality

    def set_user_name(self, name: str) -> None:
        self._config.user_name = name

    def set_classic_mode(self, enabled: bool) -> None:
        self._config.classic_mode = enabled

    def get_thinking_phrase(self) -> str:
        return random.choice(self.get_profile().thinking_phrases)

    def get_confirmation(self) -> str:
        return random.choice(self.get_profile().confirmation_phrases)
