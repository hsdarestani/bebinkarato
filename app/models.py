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
    reports: Mapped[list["WorkReport"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserMode(Base):
    __tablename__ = "user_modes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    mode: Mapped[str] = mapped_column(String(16), default="tasks")
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)


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


class WorkReport(Base):
    __tablename__ = "work_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id"), nullable=True, index=True)

    title: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str] = mapped_column(Text, default="")
    project: Mapped[str] = mapped_column(String(200), default="")
    category: Mapped[str] = mapped_column(String(100), default="work")
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    work_date: Mapped[str] = mapped_column(String(10), index=True)

    source: Mapped[str] = mapped_column(String(24), default="text")
    original_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="submitted", index=True)

    created_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)

    user: Mapped[User] = relationship(back_populates="reports")
    attachments: Mapped[list["ReportAttachment"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class ReportAttachment(Base):
    __tablename__ = "report_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("work_reports.id"), index=True)
    telegram_file_id: Mapped[str] = mapped_column(Text)
    telegram_file_unique_id: Mapped[str] = mapped_column(String(255), default="")
    file_name: Mapped[str] = mapped_column(String(500), default="")
    mime_type: Mapped[str] = mapped_column(String(255), default="")
    file_type: Mapped[str] = mapped_column(String(32), default="document")
    local_path: Mapped[str] = mapped_column(Text, default="")
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)

    report: Mapped[WorkReport] = relationship(back_populates="attachments")


class PendingAction(Base):
    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="")
    ref: Mapped[str] = mapped_column(String(120), default="")
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)


class AgentDraft(Base):
    __tablename__ = "agent_drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now_iso)
