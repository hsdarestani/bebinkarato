from datetime import datetime, timezone
from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    language_code: Mapped[str] = mapped_column(String(16), default="en")
    timezone: Mapped[str] = mapped_column(String(64), default="Europe/Berlin")
    plan: Mapped[str] = mapped_column(String(16), default="free")
    usage_month: Mapped[str] = mapped_column(String(7), default="")
    ai_requests_month: Mapped[int] = mapped_column(Integer, default=0)
    voice_seconds_month: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)
    last_seen_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)

    tasks: Mapped[list["Task"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    batch_id: Mapped[str] = mapped_column(String(40), index=True)

    title: Mapped[str] = mapped_column(String(500))
    notes: Mapped[str] = mapped_column(Text, default="")
    project: Mapped[str] = mapped_column(String(200), default="")
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    estimated_minutes: Mapped[int] = mapped_column(Integer, default=30)

    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    source: Mapped[str] = mapped_column(String(16), default="text")
    original_text: Mapped[str] = mapped_column(Text, default="")

    due_at: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    due_source: Mapped[str] = mapped_column(String(16), default="none")
    scheduled_at: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    reminder_at: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    reminder_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)

    user: Mapped[User] = relationship(back_populates="tasks")
