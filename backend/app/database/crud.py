"""Generic async CRUD operations with pagination, filtering, search, and bulk support."""
from __future__ import annotations

import uuid
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

ModelType = TypeVar("ModelType", bound=DeclarativeBase)


class CRUDBase(Generic[ModelType]):
    """Generic async CRUD repository for a single SQLAlchemy model."""

    def __init__(self, model: Type[ModelType]) -> None:
        self.model = model

    async def create(self, session: AsyncSession, *, data: dict[str, Any]) -> ModelType:
        obj = self.model(**data)
        session.add(obj)
        await session.flush()
        await session.refresh(obj)
        return obj

    async def get(self, session: AsyncSession, id_: uuid.UUID) -> Optional[ModelType]:
        return await session.get(self.model, id_)

    async def get_or_404(self, session: AsyncSession, id_: uuid.UUID) -> ModelType:
        obj = await self.get(session, id_)
        if obj is None:
            raise ValueError(f"{self.model.__name__} with id={id_} not found")
        return obj

    async def list(
        self,
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[dict[str, Any]] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
    ) -> Sequence[ModelType]:
        stmt = select(self.model)
        if filters:
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            stmt = stmt.order_by(column.desc() if descending else column.asc())
        stmt = stmt.offset(skip).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def count(self, session: AsyncSession, *, filters: Optional[dict[str, Any]] = None) -> int:
        stmt = select(func.count()).select_from(self.model)
        if filters:
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
        result = await session.execute(stmt)
        return result.scalar_one()

    async def search(
        self,
        session: AsyncSession,
        *,
        query: str,
        fields: Sequence[str],
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[ModelType]:
        conditions = [
            getattr(self.model, field).ilike(f"%{query}%")
            for field in fields
            if hasattr(self.model, field)
        ]
        if not conditions:
            return []
        stmt = select(self.model).where(or_(*conditions)).offset(skip).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def update(
        self, session: AsyncSession, *, id_: uuid.UUID, data: dict[str, Any]
    ) -> Optional[ModelType]:
        obj = await self.get(session, id_)
        if obj is None:
            return None
        for field, value in data.items():
            if hasattr(obj, field):
                setattr(obj, field, value)
        await session.flush()
        await session.refresh(obj)
        return obj

    async def delete(self, session: AsyncSession, *, id_: uuid.UUID) -> bool:
        obj = await self.get(session, id_)
        if obj is None:
            return False
        await session.delete(obj)
        await session.flush()
        return True

    async def bulk_create(self, session: AsyncSession, *, items: Sequence[dict[str, Any]]) -> Sequence[ModelType]:
        objs = [self.model(**item) for item in items]
        session.add_all(objs)
        await session.flush()
        return objs

    async def bulk_update(
        self, session: AsyncSession, *, filters: dict[str, Any], data: dict[str, Any]
    ) -> int:
        stmt = update(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        stmt = stmt.values(**data)
        result = await session.execute(stmt)
        await session.flush()
        return result.rowcount or 0

    async def bulk_delete(self, session: AsyncSession, *, ids: Sequence[uuid.UUID]) -> int:
        stmt = delete(self.model).where(self.model.id.in_(ids))  # type: ignore[attr-defined]
        result = await session.execute(stmt)
        await session.flush()
        return result.rowcount or 0
