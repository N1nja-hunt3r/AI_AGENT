"""SQLAlchemy 2.0 ORM models for the application domain."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import CHAR, TypeDecorator


class GUID(TypeDecorator):
    """Platform-independent UUID type: native UUID on PostgreSQL, CHAR(36) elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        return str(value) if isinstance(value, uuid.UUID) else value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


class Base(DeclarativeBase):
    pass


class UUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


agent_capability_association = Table(
    "agent_capabilities",
    Base.metadata,
    Column("agent_id", GUID, ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
    Column("capability_id", GUID, ForeignKey("capabilities.id", ondelete="CASCADE"), primary_key=True),
)


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    conversations: Mapped[List["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    agents: Mapped[List["Agent"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    approvals: Mapped[List["Approval"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    budgets: Mapped[List["Budget"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    settings: Mapped[Optional["Settings"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_users_username", "username"),
        Index("ix_users_email", "email"),
    )


class Conversation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Conversation")
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="conversations")
    tasks: Mapped[List["Task"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    artifacts: Mapped[List["Artifact"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_user_id_created_at", "user_id", "created_at"),
    )


class Memory(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "memories"

    user_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    user: Mapped["User"] = relationship(back_populates="memories")
    conversation: Mapped[Optional["Conversation"]] = relationship(back_populates="memories")

    __table_args__ = (
        Index("ix_memories_user_id", "user_id"),
        Index("ix_memories_conversation_id", "conversation_id"),
    )


class Capability(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "capabilities"

    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    schema: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    agents: Mapped[List["Agent"]] = relationship(
        secondary=agent_capability_association, back_populates="capabilities"
    )

    __table_args__ = (Index("ix_capabilities_name", "name"),)


class Agent(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "agents"

    owner_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False, default="generalist")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    owner: Mapped["User"] = relationship(back_populates="agents")
    capabilities: Mapped[List["Capability"]] = relationship(
        secondary=agent_capability_association, back_populates="agents"
    )
    tasks: Mapped[List["Task"]] = relationship(back_populates="agent")

    __table_args__ = (
        Index("ix_agents_owner_id", "owner_id"),
        UniqueConstraint("owner_id", "name", name="uq_agents_owner_name"),
    )


class Task(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tasks"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    conversation: Mapped["Conversation"] = relationship(back_populates="tasks")
    agent: Mapped[Optional["Agent"]] = relationship(back_populates="tasks")
    execution_plans: Mapped[List["ExecutionPlan"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    artifacts: Mapped[List["Artifact"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    budget: Mapped[Optional["Budget"]] = relationship(
        back_populates="task", uselist=False, cascade="all, delete-orphan"
    )
    approvals: Mapped[List["Approval"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    checkpoints: Mapped[List["Checkpoint"]] = relationship(back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_tasks_conversation_id", "conversation_id"),
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_agent_id", "agent_id"),
    )


class ExecutionPlan(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "execution_plans"

    task_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    steps: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(50), default="draft", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    task: Mapped["Task"] = relationship(back_populates="execution_plans")

    __table_args__ = (Index("ix_execution_plans_task_id", "task_id"),)


class Artifact(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "artifacts"

    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True
    )
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)

    task: Mapped[Optional["Task"]] = relationship(back_populates="artifacts")
    conversation: Mapped[Optional["Conversation"]] = relationship(back_populates="artifacts")

    __table_args__ = (
        Index("ix_artifacts_task_id", "task_id"),
        Index("ix_artifacts_conversation_id", "conversation_id"),
    )


class Budget(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "budgets"

    user_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID, ForeignKey("tasks.id", ondelete="CASCADE"), unique=True, nullable=True
    )
    limit_amount: Mapped[float] = mapped_column(Float, nullable=False)
    spent_amount: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)

    user: Mapped["User"] = relationship(back_populates="budgets")
    task: Mapped[Optional["Task"]] = relationship(back_populates="budget")

    __table_args__ = (Index("ix_budgets_user_id", "user_id"),)


class Approval(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "approvals"

    task_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped["Task"] = relationship(back_populates="approvals")
    user: Mapped["User"] = relationship(back_populates="approvals")

    __table_args__ = (
        Index("ix_approvals_task_id", "task_id"),
        Index("ix_approvals_user_id", "user_id"),
    )


class Checkpoint(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "checkpoints"

    task_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    task: Mapped["Task"] = relationship(back_populates="checkpoints")

    __table_args__ = (
        Index("ix_checkpoints_task_id", "task_id"),
        UniqueConstraint("task_id", "sequence", name="uq_checkpoints_task_sequence"),
    )


class Settings(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "settings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    preferences: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    theme: Mapped[str] = mapped_column(String(50), default="system", nullable=False)
    locale: Mapped[str] = mapped_column(String(20), default="en-US", nullable=False)

    user: Mapped["User"] = relationship(back_populates="settings")

    __table_args__ = (Index("ix_settings_user_id", "user_id"),)
