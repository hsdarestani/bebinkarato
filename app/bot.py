from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import jdatetime
from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.ai import AIError, CloudflareAI
from app.config import get_settings
from app.db import SessionLocal
from app.models import AgentDraft, PendingAction, ReportAttachment, Task, User, WorkReport, utc_now_iso

settings = get_settings()
ai = CloudflareAI()
UPLOAD_ROOT = Path("/data/uploads")
IRAN_TZ = "Asia/Tehran"

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(value) -> str:
    return str(value).translate(FA_DIGITS)


def jalali_date(gregorian_date: str) -> str:
    try:
        d = datetime.strptime(gregorian_date[:10], "%Y-%m-%d").date()
        j = jdatetime.date.fromgregorian(date=d)
        return fa_num(f"{j.year:04d}/{j.month:02d}/{j.day:02d}")
    except Exception:
        return gregorian_date


def local_datetime(dt_iso: str | None) -> str:
    if not dt_iso:
        return "زمان مشخص نشده"
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(ZoneInfo(IRAN_TZ))
        j = jdatetime.datetime.fromgregorian(datetime=dt)
        return fa_num(f"{j.year:04d}/{j.month:02d}/{j.day:02d} ساعت {dt:%H:%M}")
    except Exception:
        return "زمان مشخص نشده"


def today_gregorian() -> str:
    return datetime.now(ZoneInfo(IRAN_TZ)).date().isoformat()


def _upsert_user(update: Update) -> User:
    tg = update.effective_user
    if tg is None:
        raise RuntimeError("Telegram user is missing")
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.telegram_id == tg.id))
        if user is None:
            user = User(
                telegram_id=tg.id,
                username=tg.username,
                first_name=tg.first_name,
                last_name=tg.last_name,
                language_code="fa",
                timezone=IRAN_TZ,
                usage_month=month,
            )
            db.add(user)
        else:
            user.username = tg.username
            user.first_name = tg.first_name
            user.last_name = tg.last_name
            user.language_code = "fa"
            user.timezone = IRAN_TZ
            user.last_seen_at = utc_now_iso()
            if user.usage_month != month:
                user.usage_month = month
                user.ai_requests_month = 0
                user.voice_seconds_month = 0
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user


def _quota_ok(user: User, voice_seconds: int = 0) -> bool:
    if user.plan != "free":
        return True
    if user.ai_requests_month >= settings.free_ai_requests:
        return False
    if voice_seconds and user.voice_seconds_month + voice_seconds > settings.free_voice_minutes * 60:
        return False
    return True


def _charge_usage(user_id: int, voice_seconds: int = 0) -> None:
    with SessionLocal() as db:
        user = db.get(User, user_id)
        if user:
            user.ai_requests_month += 1
            user.voice_seconds_month += max(0, voice_seconds)
            db.commit()


def _set_pending(user_id: int, kind: str, ref: str) -> None:
    with SessionLocal() as db:
        item = db.scalar(select(PendingAction).where(PendingAction.user_id == user_id))
        if item is None:
            item = PendingAction(user_id=user_id, kind=kind, ref=ref)
            db.add(item)
        else:
            item.kind = kind
            item.ref = ref
            item.updated_at = utc_now_iso()
        db.commit()


def _get_pending(user_id: int) -> tuple[str, str] | None:
    with SessionLocal() as db:
        item = db.scalar(select(PendingAction).where(PendingAction.user_id == user_id))
        return (item.kind, item.ref) if item else None


def _clear_pending(user_id: int) -> None:
    with SessionLocal() as db:
        item = db.scalar(select(PendingAction).where(PendingAction.user_id == user_id))
        if item:
            db.delete(item)
            db.commit()


def _default_reminder(scheduled_at: str | None, due_at: str | None) -> str | None:
    target = scheduled_at or due_at
    if not target:
        return None
    try:
        dt = datetime.fromisoformat(target)
        reminder = dt - timedelta(minutes=30)
        now = datetime.now(timezone.utc)
        if reminder <= now < dt:
            reminder = now + timedelta(minutes=1)
        return reminder.astimezone(timezone.utc).isoformat() if reminder > now else None
    except Exception:
        return None


def task_preview_keyboard(batch_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ ثبتش کن", callback_data=f"task_confirm|{batch_id}"),
            InlineKeyboardButton("✏️ ویرایشش کن", callback_data=f"task_edit|{batch_id}"),
        ],
        [InlineKeyboardButton("🗑 بیخیال", callback_data=f"task_discard|{batch_id}")],
    ])


def report_preview_keyboard(report_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ ثبتش کن", callback_data=f"report_confirm|{report_id}"),
            InlineKeyboardButton("✏️ ویرایشش کن", callback_data=f"report_edit|{report_id}"),
        ],
        [InlineKeyboardButton("🗑 بیخیال", callback_data=f"report_discard|{report_id}")],
    ])


def done_match_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ آره، انجام شدن", callback_data="done_match_confirm|pending"),
            InlineKeyboardButton("✏️ نه، اصلاحش کنیم", callback_data="done_match_edit|pending"),
        ],
        [InlineKeyboardButton("🗑 بیخیال", callback_data="done_match_discard|pending")],
    ])


def agent_preview_keyboard(draft_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ انجامش بده", callback_data=f"agent_confirm|{draft_id}"),
            InlineKeyboardButton("✏️ یه چیزیشو عوض کن", callback_data=f"agent_edit|{draft_id}"),
        ],
        [InlineKeyboardButton("🗑 بیخیال", callback_data=f"agent_discard|{draft_id}")],
    ])


def _planning_task_payload(task: Task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "notes": task.notes,
        "project": task.project,
        "priority": task.priority,
        "estimated_minutes": task.estimated_minutes,
        "status": task.status,
        "scheduled_at": task.scheduled_at,
        "due_at": task.due_at,
        "due_source": task.due_source,
        "original_text": task.original_text,
    }


def _agent_open_tasks(user_id: int) -> list[dict]:
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task)
            .where(Task.user_id == user_id, Task.status.in_(["todo", "doing"]))
            .order_by(Task.created_at.asc())
            .limit(100)
        ).all()
        return [_planning_task_payload(task) for task in tasks]


def _priority_fa(value: str) -> str:
    return {
        "low": "کم",
        "medium": "معمولی",
        "high": "زیاد",
        "urgent": "فوری",
    }.get(value, value)


def _render_agent_preview(user_id: int, payload: dict) -> str:
    operations = payload.get("operations") or []
    if not operations:
        return "تغییری برای انجام دادن پیدا نکردم."

    with SessionLocal() as db:
        ids = [int(op["task_id"]) for op in operations if op.get("task_id") is not None]
        tasks = db.scalars(
            select(Task).where(Task.user_id == user_id, Task.id.in_(ids))
        ).all() if ids else []
        by_id = {task.id: task for task in tasks}

    lines = ["این تغییرات رو روی برنامه‌ت انجام بدم؟ 👇", ""]
    for op in operations:
        kind = op.get("type")
        if kind == "create":
            task = op.get("task") or {}
            line = f"➕ {task.get('title') or 'کار جدید'}"
            if task.get("scheduled_at"):
                line += f"\n   🗓 {local_datetime(task['scheduled_at'])}"
            if task.get("due_at"):
                line += f"\n   ⏳ ددلاین: {local_datetime(task['due_at'])}"
            if task.get("priority") and task.get("priority") != "medium":
                line += f"\n   ⚡ اولویت: {_priority_fa(task['priority'])}"
            lines.append(line)
            continue

        task_id = int(op.get("task_id") or 0)
        current = by_id.get(task_id)
        title = current.title if current else f"کار شماره {fa_num(task_id)}"

        if kind == "delete":
            lines.append(f"🗑 حذف: {title}")
        elif kind == "complete":
            lines.append(f"✅ انجام‌شده: {title}")
        elif kind == "update":
            changes = op.get("changes") or {}
            line = f"✏️ {title}"
            if changes.get("title") and changes["title"] != title:
                line += f"\n   عنوان جدید: {changes['title']}"
            if "scheduled_at" in changes:
                line += (
                    f"\n   🗓 زمان جدید: {local_datetime(changes.get('scheduled_at'))}"
                    if changes.get("scheduled_at")
                    else "\n   🗓 زمان‌بندی برداشته بشه"
                )
            if "due_at" in changes:
                line += (
                    f"\n   ⏳ ددلاین جدید: {local_datetime(changes.get('due_at'))}"
                    if changes.get("due_at")
                    else "\n   ⏳ ددلاین برداشته بشه"
                )
            if changes.get("priority"):
                line += f"\n   ⚡ اولویت: {_priority_fa(changes['priority'])}"
            if changes.get("estimated_minutes"):
                line += f"\n   ⏱ تخمین: {fa_num(changes['estimated_minutes'])} دقیقه"
            if changes.get("project"):
                line += f"\n   📁 پروژه: {changes['project']}"
            lines.append(line)

    lines.append("")
    lines.append("تا تأیید نکنی هیچ‌کدوم اعمال نمی‌شن.")
    return "\n".join(lines)


async def _save_agent_preview(
    update: Update,
    user: User,
    source_text: str,
    payload: dict,
    existing_draft_id: int | None = None,
) -> None:
    with SessionLocal() as db:
        if existing_draft_id:
            draft = db.get(AgentDraft, existing_draft_id)
            if not draft or draft.user_id != user.id:
                draft = None
        else:
            draft = None

        if draft is None:
            draft = AgentDraft(
                user_id=user.id,
                source_text=source_text,
                payload_json=json.dumps(payload, ensure_ascii=False),
                status="draft",
            )
            db.add(draft)
        else:
            draft.source_text = source_text
            draft.payload_json = json.dumps(payload, ensure_ascii=False)
            draft.status = "draft"
            draft.updated_at = utc_now_iso()
        db.commit()
        db.refresh(draft)
        draft_id = draft.id

    _set_pending(user.id, "agent_preview", str(draft_id))
    await update.effective_message.reply_text(
        _render_agent_preview(user.id, payload),
        reply_markup=agent_preview_keyboard(draft_id),
    )


async def _run_planning_agent(
    update: Update,
    user: User,
    text: str,
    voice_seconds: int = 0,
    current_draft: dict | None = None,
    existing_draft_id: int | None = None,
) -> bool:
    if not _quota_ok(user, voice_seconds):
        return False

    tasks = _agent_open_tasks(user.id)
    try:
        result = await ai.planning_agent(
            text,
            tasks,
            IRAN_TZ,
            current_draft=current_draft,
        )
        _charge_usage(user.id, voice_seconds)
    except Exception as exc:
        print(f"Planning agent error: {exc}")
        return False

    mode = result.get("mode")
    if mode == "today":
        await today(update, None)
        return True
    if mode == "upcoming":
        await upcoming(update, None)
        return True
    if mode == "today_reports":
        await today_reports(update, None)
        return True
    if mode == "reports":
        await my_reports(update, None)
        return True
    if mode == "mutate":
        await _save_agent_preview(
            update,
            user,
            text,
            result,
            existing_draft_id=existing_draft_id,
        )
        return True
    if mode == "report":
        return False
    return False


async def _revise_agent_preview(
    update: Update,
    user: User,
    text: str,
    pending: tuple[str, str],
    voice_seconds: int = 0,
) -> bool:
    if pending[0] != "agent_edit":
        return False
    try:
        draft_id = int(pending[1])
    except Exception:
        _clear_pending(user.id)
        return False

    with SessionLocal() as db:
        draft = db.get(AgentDraft, draft_id)
        if not draft or draft.user_id != user.id or draft.status != "draft":
            _clear_pending(user.id)
            return False
        try:
            current = json.loads(draft.payload_json)
        except Exception:
            current = {}
        original = draft.source_text

    combined = f"دستور قبلی کاربر: {original}\nاصلاح جدید: {text}"
    handled = await _run_planning_agent(
        update,
        user,
        combined,
        voice_seconds=voice_seconds,
        current_draft=current,
        existing_draft_id=draft_id,
    )
    if handled:
        return True

    await update.effective_message.reply_text(
        "نتونستم این اصلاح رو با اطمینان روی برنامه اعمال کنم. یه کم ساده‌تر بگو چی عوض شه."
    )
    return True


def _apply_agent_draft(user_id: int, draft_id: int) -> tuple[int, int, int, int]:
    created = updated = deleted = completed = 0
    with SessionLocal() as db:
        draft = db.get(AgentDraft, draft_id)
        if not draft or draft.user_id != user_id or draft.status != "draft":
            return created, updated, deleted, completed

        try:
            payload = json.loads(draft.payload_json)
        except Exception:
            return created, updated, deleted, completed

        for op in payload.get("operations") or []:
            kind = op.get("type")
            if kind == "create":
                item = op.get("task") or {}
                due_at = item.get("due_at") if item.get("due_source") == "explicit" else None
                scheduled_at = item.get("scheduled_at")
                reminder_at = item.get("reminder_at") or _default_reminder(scheduled_at, due_at)
                db.add(Task(
                    user_id=user_id,
                    batch_id=str(uuid4()),
                    title=str(item.get("title") or "کار جدید"),
                    notes=str(item.get("notes") or ""),
                    project=str(item.get("project") or ""),
                    priority=item.get("priority") if item.get("priority") in {"low","medium","high","urgent"} else "medium",
                    estimated_minutes=int(item.get("estimated_minutes") or 30),
                    status="todo",
                    source="agent",
                    original_text=draft.source_text,
                    due_at=due_at,
                    due_source="explicit" if due_at else "none",
                    scheduled_at=scheduled_at,
                    reminder_at=reminder_at,
                ))
                created += 1
                continue

            try:
                task_id = int(op.get("task_id"))
            except Exception:
                continue
            task = db.get(Task, task_id)
            if not task or task.user_id != user_id or task.status not in {"todo", "doing"}:
                continue

            if kind == "delete":
                db.delete(task)
                deleted += 1
            elif kind == "complete":
                task.status = "done"
                task.updated_at = utc_now_iso()
                existing = db.scalar(
                    select(WorkReport).where(
                        WorkReport.user_id == user_id,
                        WorkReport.task_id == task.id,
                        WorkReport.status == "submitted",
                    )
                )
                if existing is None:
                    db.add(WorkReport(
                        user_id=user_id,
                        task_id=task.id,
                        title=task.title,
                        summary=task.notes or task.title,
                        project=task.project or "",
                        category="کار انجام‌شده",
                        duration_minutes=None,
                        work_date=today_gregorian(),
                        source="Agent برنامه‌ریزی",
                        original_text=draft.source_text,
                        status="submitted",
                    ))
                completed += 1
            elif kind == "update":
                changes = op.get("changes") or {}
                for field in ("title", "notes", "project", "priority", "estimated_minutes", "scheduled_at", "due_at", "due_source", "reminder_at"):
                    if field in changes:
                        setattr(task, field, changes[field])
                if "scheduled_at" in changes or "due_at" in changes:
                    if "reminder_at" not in changes:
                        task.reminder_at = _default_reminder(task.scheduled_at, task.due_at)
                    task.reminder_sent = False
                task.updated_at = utc_now_iso()
                updated += 1

        draft.status = "applied"
        draft.updated_at = utc_now_iso()
        db.commit()

    return created, updated, deleted, completed


def _task_payload(task: Task) -> dict:
    return {
        "title": task.title,
        "notes": task.notes,
        "project": task.project,
        "priority": task.priority,
        "estimated_minutes": task.estimated_minutes,
        "due_at": task.due_at,
        "due_source": task.due_source,
        "scheduled_at": task.scheduled_at,
        "reminder_at": task.reminder_at,
    }


def _report_payload(report: WorkReport) -> dict:
    return {
        "title": report.title,
        "summary": report.summary,
        "project": report.project,
        "category": report.category,
        "duration_minutes": report.duration_minutes,
        "work_date": report.work_date,
    }


def render_task_preview(tasks: list[Task]) -> str:
    if not tasks:
        return "چیزی برای ثبت پیدا نکردم."
    lines = []
    first = tasks[0]
    if first.source == "voice" and first.original_text:
        heard = first.original_text.strip().replace("\n", " ")
        if len(heard) > 450:
            heard = heard[:447] + "..."
        lines.extend(["از ویست اینو شنیدم 🎙", f"«{heard}»", ""])
    lines.append("اینم چیزیه که ازش فهمیدم 👇")
    for i, task in enumerate(tasks, start=1):
        line = f"\n{fa_num(i)}. {task.title}"
        if task.project:
            line += f"\n   📁 {task.project}"
        if task.scheduled_at:
            line += f"\n   🗓 پیشنهاد من: {local_datetime(task.scheduled_at)}"
        if task.due_at:
            line += f"\n   ⏳ ددلاین خودت: {local_datetime(task.due_at)}"
        if task.estimated_minutes:
            line += f"\n   ⏱ تخمین من: حدود {fa_num(task.estimated_minutes)} دقیقه"
        lines.append(line)
    lines.append("\nاگه حتی یه کلمه‌شم غلطه، «ویرایشش کن» رو بزن و خیلی عادی بگو چی عوض شه.")
    return "\n".join(lines)


def render_report_preview(report: WorkReport) -> str:
    lines = []
    if report.source == "voice" and report.original_text:
        heard = report.original_text.strip().replace("\n", " ")
        if len(heard) > 450:
            heard = heard[:447] + "..."
        lines.extend(["از ویست اینو شنیدم 🎙", f"«{heard}»", ""])
    lines.extend([
        "این گزارشیه که از حرفت فهمیدم 👇",
        "",
        f"✅ {report.title}",
    ])
    if report.summary and report.summary != report.title:
        lines.append(f"📝 {report.summary}")
    if report.project:
        lines.append(f"📁 {report.project}")
    if report.duration_minutes:
        lines.append(f"⏱ {fa_num(report.duration_minutes)} دقیقه")
    lines.append(f"📅 {jalali_date(report.work_date)}")
    if report.attachments:
        lines.append(f"📎 {fa_num(len(report.attachments))} فایل پیوست")
    lines.append("\nدرسته؟")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    _clear_pending(user.id)
    text = (
        "سلام رفیق 👋\n\n"
        "منو و فرم نداریم. هرچی می‌خوای عادی بهم بگو یا ویس بده.\n\n"
        "مثلاً:\n"
        "«فردا باید قرارداد رو بفرستم و تا پنجشنبه سایت رو جمع کنم»\n"
        "«امروز باگ پرداخت رو حل کردم و دو ساعت روش بودم»\n"
        "«امروز چی دارم؟»\n"
        "«گزارش کارامو نشون بده»\n\n"
        "خودم می‌فهمم منظورت چیه، بعد قبل از ثبت نشونت می‌دم چی فهمیدم."
    )
    await update.effective_message.reply_text(text, reply_markup=ReplyKeyboardRemove())


async def _create_task_preview(update: Update, user: User, text: str, voice_seconds: int = 0) -> None:
    if not _quota_ok(user, voice_seconds):
        await update.effective_message.reply_text("سهمیه هوش مصنوعی این ماهت پر شده 😅")
        return
    msg = await update.effective_message.reply_text("بذار ببینم ازش چی درمیاد…")
    try:
        parsed = await ai.parse_tasks(text, IRAN_TZ, "fa")
        _charge_usage(user.id, voice_seconds)
    except AIError as exc:
        print(f"Task AI error: {exc}")
        await msg.edit_text("یه گیری پیش اومد و نتونستم درست بفهممش 😅 یه بار دیگه بفرست.")
        return
    if not parsed:
        await msg.edit_text("از این پیام کار مشخصی درنیاوردم. یه کم واضح‌تر بگو چی باید انجام بدی.")
        return

    batch_id = str(uuid4())
    with SessionLocal() as db:
        deadline_is_explicit = _deadline_was_explicit(text)
        for item in parsed:
            due_at = item.get("due_at") if deadline_is_explicit else None
            due_source = item.get("due_source", "none") if due_at else "none"
            db.add(Task(
                user_id=user.id,
                batch_id=batch_id,
                title=item["title"],
                notes=item.get("notes", ""),
                project=item.get("project", ""),
                priority=item.get("priority", "medium"),
                estimated_minutes=item.get("estimated_minutes", 30),
                status="draft",
                source="voice" if voice_seconds else "text",
                original_text=text,
                due_at=due_at,
                due_source=due_source,
                scheduled_at=item.get("scheduled_at"),
                reminder_at=item.get("reminder_at") or _default_reminder(item.get("scheduled_at"), due_at),
            ))
        db.commit()
        tasks = db.scalars(select(Task).where(Task.batch_id == batch_id).order_by(Task.id)).all()
        preview = render_task_preview(tasks)

    _set_pending(user.id, "task_preview", batch_id)
    await msg.edit_text(preview, reply_markup=task_preview_keyboard(batch_id))


def _normalize_fa(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
    value = re.sub(r"[ـ‌\u200c]", " ", value)
    value = re.sub(r"[^\w\sآ-ی]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _completion_refers_to_existing_tasks(text: str) -> bool:
    n = _normalize_fa(text)
    return bool(re.search(
        r"(تسک|کار(?:ا|ها)?(?:ی)?\s*(?:قبلی|بالا|همون)|همون\s+(?:سه|دو|چند|کار)|"
        r"اونا\s+رو\s+انجام\s+دادم|این(?:ا|ها)\s+رو\s+انجام\s+دادم|"
        r"همه(?:شون|ش)\s+رو\s+انجام\s+دادم)",
        n,
    ))


def _meaningful_tokens(text: str) -> set[str]:
    stop = {
        "را","رو","به","با","از","در","برای","یه","یک","این","اون","همون","و","که","هم",
        "کردم","کرده","انجام","دادم","رفتم","خریدم","گرفتم","تموم","تمام","شد","شده",
        "باید","کنم","کنید","کن","برم","بخرم","برگزار","راست","درست","کار","گزارش",
    }
    return {
        token for token in _normalize_fa(text).split()
        if len(token) >= 3 and token not in stop
    }


def _texts_semantically_overlap(a: str, b: str) -> bool:
    left = _meaningful_tokens(a)
    right = _meaningful_tokens(b)
    if not left or not right:
        return False
    for x in left:
        for y in right:
            if x == y:
                return True
            if len(x) >= 3 and len(y) >= 3 and (x in y or y in x):
                return True
    return False


def _filter_unmatched_reports(
    user_text: str,
    unmatched: list[dict],
    matched_tasks: list[Task],
) -> list[dict]:
    clean = []
    matched_texts = [
        " ".join(filter(None, [task.title, task.notes, task.project]))
        for task in matched_tasks
    ]
    for item in unmatched[:5]:
        candidate_text = " ".join(filter(None, [
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            str(item.get("project") or ""),
        ]))

        # گزارش جدید باید واقعاً از همین پیام فعلی کاربر آمده باشد.
        if not _texts_semantically_overlap(candidate_text, user_text):
            continue

        # اگر همان کاری است که همین الان به یک Task باز Match شده، دوباره گزارش مستقل نساز.
        if any(_texts_semantically_overlap(candidate_text, task_text) for task_text in matched_texts):
            continue

        clean.append(item)
    return clean


def _local_completed_task_matches(text: str, tasks: list[dict]) -> list[int]:
    n = _normalize_fa(text)
    stop = {
        "را","رو","به","با","از","در","برای","یه","یک","این","اون","همون","و","که",
        "کردم","کرده","انجام","دادم","رفتم","خریدم","گرفتم","تموم","تمام","شد","شده",
        "باید","کنم","کنید","کن","برم","بخرم","برگزار","راست","درست",
    }
    input_tokens = {t for t in n.split() if len(t) >= 2 and t not in stop}
    if not input_tokens:
        return []

    prepared = []
    token_frequency: dict[str, int] = {}
    for task in tasks:
        task_text = " ".join([
            str(task.get("title") or ""),
            str(task.get("project") or ""),
            str(task.get("notes") or ""),
        ])
        tn = _normalize_fa(task_text)
        task_tokens = {t for t in tn.split() if len(t) >= 2 and t not in stop}
        prepared.append((task, task_tokens))
        for token in task_tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    matches = []
    for task, task_tokens in prepared:
        overlap = input_tokens & task_tokens
        if not overlap:
            continue

        # واژه بلند، دو واژه مشترک، یا یک اسم کوتاه ولی یکتا مثل «موز».
        strong = any(len(t) >= 4 for t in overlap)
        unique_short = any(len(t) >= 3 and token_frequency.get(t) == 1 for t in overlap)
        if strong or unique_short or len(overlap) >= 2:
            try:
                matches.append(int(task["id"]))
            except Exception:
                pass
    return matches[:10]


async def _create_completion_match_preview(
    update: Update,
    user: User,
    text: str,
    source: str,
    voice_seconds: int = 0,
) -> bool:
    with SessionLocal() as db:
        open_tasks = db.scalars(
            select(Task)
            .where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))
            .order_by(Task.created_at.desc())
            .limit(30)
        ).all()
        task_payloads = [
            {
                "id": task.id,
                "title": task.title,
                "notes": task.notes,
                "project": task.project,
                "original_text": task.original_text,
            }
            for task in open_tasks
        ]

    if not task_payloads:
        return False

    force_existing = _completion_refers_to_existing_tasks(text)
    matched = {"matched_task_ids": [], "unmatched_reports": []}

    if _quota_ok(user, voice_seconds):
        try:
            matched = await ai.match_completed_tasks(text, task_payloads, IRAN_TZ)
            _charge_usage(user.id, voice_seconds)
        except Exception as exc:
            print(f"Completion matching error: {exc}")

    matched_ids = [int(x) for x in (matched.get("matched_task_ids") or [])]
    unmatched = matched.get("unmatched_reports") or []

    if not matched_ids:
        matched_ids = _local_completed_task_matches(text, task_payloads)

    if not matched_ids:
        if force_existing:
            lines = [
                "فهمیدم داری درباره کارای قبلیت حرف می‌زنی، ولی دقیق نتونستم تشخیص بدم کدوما رو می‌گی.",
                "",
                "کارای بازت ایناست 👇",
            ]
            for i, task in enumerate(task_payloads[:12], start=1):
                lines.append(f"{fa_num(i)}. {task['title']}")
            lines.append("")
            lines.append("مثلاً بگو «۲ و ۳ رو انجام دادم» یا اسم کارها رو بگو.")
            await update.effective_message.reply_text("\n".join(lines))
            return True
        return False

    with SessionLocal() as db:
        matched_tasks = db.scalars(
            select(Task)
            .where(
                Task.user_id == user.id,
                Task.id.in_(matched_ids),
                Task.status.in_(["todo", "doing"]),
            )
            .order_by(Task.id)
        ).all()
        valid_ids = [task.id for task in matched_tasks]
        if not valid_ids:
            return False

        unmatched = _filter_unmatched_reports(text, unmatched, matched_tasks)

        unmatched_report_ids = []
        for item in unmatched[:5]:
            report = WorkReport(
                user_id=user.id,
                title=item.get("title") or "گزارش کار",
                summary=item.get("summary") or item.get("title") or "",
                project=item.get("project") or "",
                category=item.get("category") or "کار",
                duration_minutes=item.get("duration_minutes"),
                work_date=item.get("work_date") or today_gregorian(),
                source=source,
                original_text=text,
                status="draft_match",
            )
            db.add(report)
            db.flush()
            unmatched_report_ids.append(report.id)
        db.commit()

        lines = ["از حرفت فهمیدم این کارای قبلیت انجام شدن 👇", ""]
        for task in matched_tasks:
            lines.append(f"✅ {task.title}")
        if unmatched_report_ids:
            reports = db.scalars(
                select(WorkReport).where(WorkReport.id.in_(unmatched_report_ids)).order_by(WorkReport.id)
            ).all()
            lines.append("")
            lines.append("این‌ها هم کار جدید بودن و جدا گزارششون می‌کنم:")
            for report in reports:
                lines.append(f"➕ {report.title}")

    ref = json.dumps(
        {"t": valid_ids, "r": unmatched_report_ids},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if len(ref) > 120:
        with SessionLocal() as db:
            drafts = db.scalars(
                select(WorkReport).where(
                    WorkReport.id.in_(unmatched_report_ids),
                    WorkReport.user_id == user.id,
                    WorkReport.status == "draft_match",
                )
            ).all()
            for report in drafts:
                db.delete(report)
            db.commit()
        return False

    _set_pending(user.id, "done_match_preview", ref)
    lines.append("")
    lines.append("همین‌ها رو انجام‌شده بزنم؟")
    await update.effective_message.reply_text(
        "\n".join(lines),
        reply_markup=done_match_keyboard(),
    )
    return True


async def _create_report_preview(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user: User,
    text: str,
    source: str,
    voice_seconds: int = 0,
    attachment: dict | None = None,
) -> None:
    if not text.strip():
        text = "گزارش کار همراه با فایل پیوست"
    try:
        parsed = await ai.parse_report(text, IRAN_TZ, "fa") if _quota_ok(user, voice_seconds) else {
            "title": text[:180],
            "summary": text,
            "project": "",
            "category": "کار",
            "duration_minutes": None,
            "work_date": today_gregorian(),
        }
        if _quota_ok(user, voice_seconds):
            _charge_usage(user.id, voice_seconds)
    except Exception as exc:
        print(f"Report parse fallback: {exc}")
        parsed = {
            "title": text[:180],
            "summary": text,
            "project": "",
            "category": "کار",
            "duration_minutes": None,
            "work_date": today_gregorian(),
        }

    with SessionLocal() as db:
        report = WorkReport(
            user_id=user.id,
            title=parsed["title"],
            summary=parsed.get("summary", ""),
            project=parsed.get("project", ""),
            category=parsed.get("category", "کار"),
            duration_minutes=parsed.get("duration_minutes"),
            work_date=parsed.get("work_date") or today_gregorian(),
            source=source,
            original_text=text,
            status="draft",
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        report_id = report.id

    if attachment:
        await _save_attachment(context, report_id=report_id, **attachment)

    with SessionLocal() as db:
        report = db.get(WorkReport, report_id)
        _ = list(report.attachments)
        preview = render_report_preview(report)

    _set_pending(user.id, "report_preview", str(report_id))
    await update.effective_message.reply_text(preview, reply_markup=report_preview_keyboard(report_id))


async def _save_attachment(
    context: ContextTypes.DEFAULT_TYPE,
    report_id: int,
    file_id: str,
    unique_id: str,
    file_name: str,
    mime_type: str,
    file_type: str,
    size_bytes: int | None,
    caption: str,
    raw_bytes: bytes | None = None,
) -> int:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name or file_type)[:180] or file_type
    target_dir = UPLOAD_ROOT / str(report_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}_{safe_name}"
    local_path = ""
    try:
        if raw_bytes is not None:
            target.write_bytes(raw_bytes)
        else:
            tg_file = await context.bot.get_file(file_id)
            await tg_file.download_to_drive(custom_path=target)
        local_path = str(target)
    except Exception as exc:
        print(f"Attachment download failed: {exc}")

    with SessionLocal() as db:
        item = ReportAttachment(
            report_id=report_id,
            telegram_file_id=file_id,
            telegram_file_unique_id=unique_id or "",
            file_name=file_name or safe_name,
            mime_type=mime_type or "",
            file_type=file_type,
            local_path=local_path,
            size_bytes=size_bytes,
            caption=caption or "",
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item.id


def _deadline_was_explicit(text: str) -> bool:
    normalized = (text or "").replace("ي", "ی").replace("ك", "ک")
    return bool(re.search(
        r"(ددلاین|مهلت|تا\s+(?:امروز|فردا|پس\s*فردا|آخر|پایان|ساعت|قبل)|"
        r"قبل\s+از|نهایت(?:ا|اً)?|حداکثر|باید\s+تا)",
        normalized,
        flags=re.I,
    ))


def _clean_fake_deadlines(tasks: list[Task]) -> None:
    for task in tasks:
        if task.due_at and not _deadline_was_explicit(task.original_text or ""):
            task.due_at = None
            task.due_source = "none"
            if task.reminder_at and not task.scheduled_at:
                task.reminder_at = None


def _simple_replacement(instruction: str) -> tuple[str, str] | None:
    text = (instruction or "").strip()
    text = re.sub(r"^(?:میگم|می‌گم|منظورم|یعنی)\s+", "", text).strip()
    text = text.strip(" .،,!؟?")

    patterns = [
        r"^(.+?)\s+(?:نه|نیست)\s*[,،]?\s*(.+?)$",
        r"^(.+?)\s+(?:رو|را)\s+(?:بکن|کن|تبدیل\s+کن\s+به)\s+(.+?)$",
        r"^(.+?)\s+(?:بشه|بشود)\s+(.+?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.I)
        if not match:
            continue
        old = match.group(1).strip(" «»\"'")
        new = match.group(2).strip(" «»\"'")
        if old and new and old != new and len(old) <= 80 and len(new) <= 80:
            return old, new
    return None


def _simple_delete_index(instruction: str) -> int | None:
    text = (instruction or "").strip()
    words = {
        "اولی": 0, "اول": 0,
        "دومی": 1, "دوم": 1,
        "سومی": 2, "سوم": 2,
        "چهارمی": 3, "چهارم": 3,
        "پنجمی": 4, "پنجم": 4,
        "ششمی": 5, "ششم": 5,
    }
    if not re.search(r"(حذف|پاک|بردار|بیخیال)", text):
        return None
    for word, index in words.items():
        if word in text:
            return index
    match = re.search(r"(?:شماره|مورد)\s*([۰-۹0-9]+)", text)
    if match:
        raw = match.group(1).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        try:
            return max(0, int(raw) - 1)
        except Exception:
            return None
    return None


async def _try_simple_task_edit(update: Update, user: User, ref: str, instruction: str) -> bool:
    replacement = _simple_replacement(instruction)
    delete_index = _simple_delete_index(instruction)

    if replacement is None and delete_index is None:
        return False

    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task)
            .where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
            .order_by(Task.id)
        ).all()
        if not tasks:
            _clear_pending(user.id)
            return False

        changed = False
        if replacement is not None:
            old, new = replacement
            for task in tasks:
                for field in ("title", "notes", "project"):
                    value = getattr(task, field) or ""
                    if old in value:
                        setattr(task, field, value.replace(old, new))
                        changed = True

            if not changed:
                await update.effective_message.reply_text(
                    f"«{old}» رو توی چیزایی که فهمیده بودم پیدا نکردم. یه کم دقیق‌تر بگو کدوم مورد رو می‌گی."
                )
                return True

        elif delete_index is not None:
            if delete_index >= len(tasks):
                await update.effective_message.reply_text(
                    "اون شماره‌ای که گفتی توی لیست نیست 😅"
                )
                return True
            db.delete(tasks[delete_index])
            changed = True

        remaining = [task for i, task in enumerate(tasks) if delete_index is None or i != delete_index]
        _clean_fake_deadlines(remaining)
        for task in remaining:
            task.updated_at = utc_now_iso()
        db.commit()

        tasks = db.scalars(
            select(Task)
            .where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
            .order_by(Task.id)
        ).all()
        if not tasks:
            _clear_pending(user.id)
            await update.effective_message.reply_text("اوکی، دیگه چیزی از این لیست نموند.")
            return True
        preview = render_task_preview(tasks)

    _set_pending(user.id, "task_preview", ref)
    if replacement is not None:
        await update.effective_message.reply_text(
            f"آره گرفتم 👌 «{replacement[0]}» رو کردم «{replacement[1]}»."
        )
    else:
        await update.effective_message.reply_text("اوکی، حذفش کردم 👌")
    await update.effective_message.reply_text(preview, reply_markup=task_preview_keyboard(ref))
    return True


async def _handle_edit_text(update: Update, user: User, instruction: str, pending: tuple[str, str]) -> bool:
    kind, ref = pending
    if kind == "task_edit":
        with SessionLocal() as db:
            tasks = db.scalars(
                select(Task)
                .where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
                .order_by(Task.id)
            ).all()
            current = [_task_payload(t) for t in tasks]
            original_text = tasks[0].original_text if tasks else ""
            original_source = tasks[0].source if tasks else "edit"

        try:
            revised = await ai.revise_tasks(
                current,
                instruction,
                IRAN_TZ,
                original_text=original_text,
            )
        except Exception as exc:
            print(f"Task semantic revision error: {exc}")
            if await _try_simple_task_edit(update, user, ref, instruction):
                return True
            await update.effective_message.reply_text(
                "منظورت رو گرفتم ولی نتونستم با اطمینان اصلاحش کنم 😅 یه بار خیلی کوتاه‌تر بگو چی باید عوض شه."
            )
            return True
        with SessionLocal() as db:
            old = db.scalars(
                select(Task).where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
            ).all()
            for task in old:
                db.delete(task)
            allow_deadline = (
                _deadline_was_explicit(original_text)
                or _deadline_was_explicit(instruction)
                or any(t.get("due_at") for t in current)
            )
            for item in revised:
                due_at = item.get("due_at") if allow_deadline else None
                due_source = item.get("due_source", "none") if due_at else "none"
                db.add(Task(
                    user_id=user.id, batch_id=ref, title=item["title"], notes=item.get("notes", ""),
                    project=item.get("project", ""), priority=item.get("priority", "medium"),
                    estimated_minutes=item.get("estimated_minutes", 30), status="draft", source=original_source,
                    original_text=original_text, due_at=due_at, due_source=due_source,
                    scheduled_at=item.get("scheduled_at"),
                    reminder_at=item.get("reminder_at") or _default_reminder(item.get("scheduled_at"), due_at),
                ))
            db.commit()
            tasks = db.scalars(
                select(Task)
                .where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
                .order_by(Task.id)
            ).all()
            preview = render_task_preview(tasks)
        _set_pending(user.id, "task_preview", ref)
        await update.effective_message.reply_text(
            "آره، منظورت رو گرفتم و اصلاحش کردم 👌"
        )
        await update.effective_message.reply_text(preview, reply_markup=task_preview_keyboard(ref))
        return True

    if kind == "report_edit":
        report_id = int(ref)
        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            if not report or report.user_id != user.id:
                _clear_pending(user.id)
                return False
            current = _report_payload(report)
        try:
            revised = await ai.revise_report(current, instruction, IRAN_TZ)
        except Exception as exc:
            print(f"Report revision error: {exc}")
            await update.effective_message.reply_text("نتونستم اصلاحش کنم 😅 یه بار دیگه بگو چی رو عوض کنم.")
            return True
        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            report.title = revised["title"]
            report.summary = revised.get("summary", "")
            report.project = revised.get("project", "")
            report.category = revised.get("category", "کار")
            report.duration_minutes = revised.get("duration_minutes")
            report.work_date = revised.get("work_date") or report.work_date
            report.updated_at = utc_now_iso()
            db.commit()
            db.refresh(report)
            _ = list(report.attachments)
            preview = render_report_preview(report)
        _set_pending(user.id, "report_preview", str(report_id))
        await update.effective_message.reply_text(preview, reply_markup=report_preview_keyboard(report_id))
        return True
    return False


async def _replace_edit_from_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user: User,
    transcript: str,
    voice_seconds: int,
    pending: tuple[str, str],
) -> bool:
    kind, ref = pending

    if kind == "task_edit":
        if not _quota_ok(user, voice_seconds):
            await update.effective_message.reply_text("سهمیه هوش مصنوعی این ماهت پر شده 😅")
            return True
        try:
            fresh = await ai.parse_tasks(transcript, IRAN_TZ, "fa")
            if not fresh:
                raise AIError("No tasks parsed from replacement voice.")
            _charge_usage(user.id, voice_seconds)
        except Exception as exc:
            print(f"Voice task replacement error: {exc}")
            await update.effective_message.reply_text(
                "ویستو شنیدم، ولی نتونستم از نو به برنامه تبدیلش کنم 😅 دوباره بفرست یا خیلی کوتاه بگو چی باید عوض شه."
            )
            return True

        with SessionLocal() as db:
            old = db.scalars(
                select(Task).where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
            ).all()
            if not old:
                _clear_pending(user.id)
                return False
            for task in old:
                db.delete(task)
            deadline_is_explicit = _deadline_was_explicit(transcript)
            for item in fresh:
                due_at = item.get("due_at") if deadline_is_explicit else None
                due_source = item.get("due_source", "none") if due_at else "none"
                db.add(Task(
                    user_id=user.id,
                    batch_id=ref,
                    title=item["title"],
                    notes=item.get("notes", ""),
                    project=item.get("project", ""),
                    priority=item.get("priority", "medium"),
                    estimated_minutes=item.get("estimated_minutes", 30),
                    status="draft",
                    source="voice",
                    original_text=transcript,
                    due_at=due_at,
                    due_source=due_source,
                    scheduled_at=item.get("scheduled_at"),
                    reminder_at=item.get("reminder_at") or _default_reminder(
                        item.get("scheduled_at"), due_at
                    ),
                ))
            db.commit()
            tasks = db.scalars(
                select(Task).where(Task.batch_id == ref, Task.user_id == user.id).order_by(Task.id)
            ).all()
            preview = render_task_preview(tasks)

        _set_pending(user.id, "task_preview", ref)
        await update.effective_message.reply_text(
            "اوکی، این ویس رو نسخه جدید حرفت گرفتم و از اول چیدمش 👇"
        )
        await update.effective_message.reply_text(preview, reply_markup=task_preview_keyboard(ref))
        return True

    if kind == "report_edit":
        report_id = int(ref)
        try:
            fresh = await ai.parse_report(transcript, IRAN_TZ, "fa")
            if _quota_ok(user, voice_seconds):
                _charge_usage(user.id, voice_seconds)
        except Exception as exc:
            print(f"Voice report replacement error: {exc}")
            await update.effective_message.reply_text(
                "ویستو شنیدم، ولی نتونستم گزارش رو از نو بسازم 😅 دوباره بفرست یا کوتاه بگو چی باید عوض شه."
            )
            return True

        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            if not report or report.user_id != user.id or report.status != "draft":
                _clear_pending(user.id)
                return False
            report.title = fresh["title"]
            report.summary = fresh.get("summary", "")
            report.project = fresh.get("project", "")
            report.category = fresh.get("category", "کار")
            report.duration_minutes = fresh.get("duration_minutes")
            report.work_date = fresh.get("work_date") or report.work_date
            report.source = "voice"
            report.original_text = transcript
            report.updated_at = utc_now_iso()
            db.commit()
            db.refresh(report)
            _ = list(report.attachments)
            preview = render_report_preview(report)

        _set_pending(user.id, "report_preview", str(report_id))
        await update.effective_message.reply_text(
            "اوکی، این ویس رو نسخه جدید گزارش گرفتم و از اول ساختمش 👇"
        )
        await update.effective_message.reply_text(preview, reply_markup=report_preview_keyboard(report_id))
        return True

    return False


async def _refine_done_match_preview(
    update: Update,
    user: User,
    text: str,
    pending: tuple[str, str],
) -> bool:
    kind, ref = pending
    if kind != "done_match_preview":
        return False

    try:
        payload = json.loads(ref)
        current_ids = [int(x) for x in payload.get("t", [])]
        report_ids = [int(x) for x in payload.get("r", [])]
    except Exception:
        _clear_pending(user.id)
        return False

    with SessionLocal() as db:
        open_tasks = db.scalars(
            select(Task)
            .where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))
            .order_by(Task.id)
        ).all()
        task_payloads = [
            {
                "id": task.id,
                "title": task.title,
                "notes": task.notes,
                "project": task.project,
                "original_text": task.original_text,
            }
            for task in open_tasks
        ]

    if not task_payloads:
        _clear_pending(user.id)
        return False

    selected_ids = None
    if _quota_ok(user):
        try:
            selected_ids = await ai.refine_completed_task_selection(
                text,
                current_ids,
                task_payloads,
                IRAN_TZ,
            )
            _charge_usage(user.id)
        except Exception as exc:
            print(f"Completion refinement AI error: {exc}")

    if selected_ids is None:
        selected_ids = list(current_ids)

    # fallback محلی برای اضافه/حذف‌های خیلی روشن، فقط اگر AI نتیجه معنادار نداد
    if not selected_ids and current_ids:
        n = _normalize_fa(text)
        local = _local_completed_task_matches(text, task_payloads)
        if re.search(r"(انجام\s+ندادم|نکردم|نه|بردار|حذف)", n) and local:
            selected_ids = [task_id for task_id in current_ids if task_id not in local]
        elif "فقط" in n and local:
            selected_ids = local
        elif local:
            selected_ids = list(dict.fromkeys(current_ids + local))

    valid_ids = {int(task["id"]) for task in task_payloads}
    selected_ids = [task_id for task_id in selected_ids if task_id in valid_ids]

    # اگر کاربر گفت «هم» و AI به اشتباه قبلی‌ها را انداخت، union محافظه‌کارانه
    n = _normalize_fa(text)
    if "هم" in n:
        selected_ids = list(dict.fromkeys(current_ids + selected_ids))

    ref_new = json.dumps(
        {"t": selected_ids, "r": report_ids},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    _set_pending(user.id, "done_match_preview", ref_new)

    with SessionLocal() as db:
        selected_tasks = db.scalars(
            select(Task)
            .where(Task.user_id == user.id, Task.id.in_(selected_ids))
            .order_by(Task.id)
        ).all() if selected_ids else []

        lines = ["اوکی، پیش‌نمایش رو اصلاح کردم 👌", ""]
        if selected_tasks:
            lines.append("این کارا انجام‌شده حساب می‌شن:")
            for task in selected_tasks:
                lines.append(f"✅ {task.title}")
        else:
            lines.append("فعلاً هیچ Taskی رو انجام‌شده نگه نداشتم.")

        if report_ids:
            reports = db.scalars(
                select(WorkReport)
                .where(
                    WorkReport.user_id == user.id,
                    WorkReport.id.in_(report_ids),
                    WorkReport.status == "draft_match",
                )
                .order_by(WorkReport.id)
            ).all()
            if reports:
                lines.append("")
                lines.append("این گزارش‌های جدید هم جدا می‌مونن:")
                for report in reports:
                    lines.append(f"➕ {report.title}")

    lines.append("")
    lines.append("همین‌ها رو ثبت کنم؟")
    await update.effective_message.reply_text(
        "\n".join(lines),
        reply_markup=done_match_keyboard(),
    )
    return True


async def route_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    voice_seconds: int = 0,
    from_voice: bool = False,
) -> None:
    user = _upsert_user(update)
    pending = _get_pending(user.id)

    if pending and pending[0] == "agent_edit":
        handled = await _revise_agent_preview(
            update,
            user,
            text,
            pending,
            voice_seconds=voice_seconds,
        )
        if handled:
            return

    if pending and pending[0] == "agent_preview":
        correction = _normalize_fa(text)
        if re.match(r"^(نه|نخیر|منظورم|میگم|می گم|اصلاح|عوض|فقط|اون|این)", correction):
            handled = await _revise_agent_preview(
                update,
                user,
                text,
                ("agent_edit", pending[1]),
                voice_seconds=voice_seconds,
            )
            if handled:
                return

    # کاربر لازم نیست برای رد کردن پیش‌نمایش حتماً دکمه بزند.
    # «نه...»، «منظورم...»، «میگم...» یعنی پیش‌نمایش قبلی را کنار بگذار و حرف جدید را بفهم.
    # اگر روی پیش‌نمایش «کارهای انجام‌شده» هستیم، هر اصلاح طبیعی را روی همان مجموعه اعمال کن.
    if pending and pending[0] == "done_match_preview":
        handled = await _refine_done_match_preview(update, user, text, pending)
        if handled:
            return

    correction_text = _normalize_fa(text)
    natural_reject = bool(re.match(
        r"^(نه|نخیر|منظورم|میگم|می گم|اشتباه|درست نیست|اصلاح)",
        correction_text,
    ))
    if pending and natural_reject and pending[0] in {"task_preview", "report_preview", "done_match_preview"}:
        kind, ref = pending
        with SessionLocal() as db:
            if kind == "task_preview":
                drafts = db.scalars(
                    select(Task).where(
                        Task.batch_id == ref,
                        Task.user_id == user.id,
                        Task.status == "draft",
                    )
                ).all()
                for item in drafts:
                    db.delete(item)
            elif kind == "report_preview":
                try:
                    report_id = int(ref)
                except Exception:
                    report_id = 0
                report = db.get(WorkReport, report_id) if report_id else None
                if report and report.user_id == user.id and report.status == "draft":
                    db.delete(report)
            elif kind == "done_match_preview":
                try:
                    payload = json.loads(ref)
                    report_ids = [int(x) for x in payload.get("r", [])]
                except Exception:
                    report_ids = []
                if report_ids:
                    drafts = db.scalars(
                        select(WorkReport).where(
                            WorkReport.user_id == user.id,
                            WorkReport.id.in_(report_ids),
                            WorkReport.status == "draft_match",
                        )
                    ).all()
                    for item in drafts:
                        db.delete(item)
            db.commit()
        _clear_pending(user.id)
        pending = None
    if pending and pending[0] in {"task_edit", "report_edit"}:
        if from_voice:
            handled = await _replace_edit_from_voice(
                update, context, user, text, voice_seconds, pending
            )
        else:
            handled = await _handle_edit_text(update, user, text, pending)
        if handled:
            return

    handled_by_agent = await _run_planning_agent(
        update,
        user,
        text,
        voice_seconds=voice_seconds,
    )
    if handled_by_agent:
        return

    try:
        intent = await ai.classify_intent(text, IRAN_TZ)
    except Exception as exc:
        print(f"Intent error: {exc}")
        intent = "unknown"

    if intent == "today":
        await today(update, context)
    elif intent == "today_reports":
        await today_reports(update, context)
    elif intent == "upcoming":
        await upcoming(update, context)
    elif intent == "reports":
        await my_reports(update, context)
    elif intent == "report":
        source = "voice" if voice_seconds else "text"
        matched_existing = await _create_completion_match_preview(
            update,
            user,
            text,
            source,
            voice_seconds,
        )
        if not matched_existing:
            await _create_report_preview(update, context, user, text, source, voice_seconds)
    elif intent == "plan":
        await _create_task_preview(update, user, text, voice_seconds)
    else:
        await update.effective_message.reply_text(
            "دقیق نفهمیدم چی می‌خوای 😄 عادی بگو؛ مثلاً بگو چه کاری داری، چی انجام دادی، یا بپرس امروز چی داری."
        )


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.effective_message.text or "").strip()
    if text:
        await route_text(update, context, text)


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    voice = update.effective_message.voice
    if not voice:
        return
    msg = await update.effective_message.reply_text("دارم ویستو گوش می‌دم… 🎙")
    try:
        tg_file = await context.bot.get_file(voice.file_id)
        audio = bytes(await tg_file.download_as_bytearray())
        transcript = await ai.transcribe(audio)
        try:
            await msg.delete()
        except Exception:
            pass
        await route_text(
            update,
            context,
            transcript,
            voice.duration or 0,
            from_voice=True,
        )
    except Exception as exc:
        print(f"Voice error: {exc}")
        await msg.edit_text("این ویسه رو نتونستم درست بخونم 😅 یه بار دیگه بفرست.")


async def media_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    message = update.effective_message
    caption = (message.caption or "").strip()
    attachment = None
    source = "فایل"

    if message.document:
        f = message.document
        attachment = {
            "file_id": f.file_id, "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"file-{f.file_unique_id}",
            "mime_type": f.mime_type or "application/octet-stream",
            "file_type": "document", "size_bytes": f.file_size, "caption": caption,
        }
        source = "فایل"
    elif message.photo:
        f = message.photo[-1]
        attachment = {
            "file_id": f.file_id, "unique_id": f.file_unique_id,
            "file_name": f"photo-{f.file_unique_id}.jpg", "mime_type": "image/jpeg",
            "file_type": "photo", "size_bytes": f.file_size, "caption": caption,
        }
        source = "عکس"
    elif message.video:
        f = message.video
        attachment = {
            "file_id": f.file_id, "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"video-{f.file_unique_id}.mp4", "mime_type": f.mime_type or "video/mp4",
            "file_type": "video", "size_bytes": f.file_size, "caption": caption,
        }
        source = "ویدیو"
    elif message.audio:
        f = message.audio
        attachment = {
            "file_id": f.file_id, "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"audio-{f.file_unique_id}.mp3", "mime_type": f.mime_type or "audio/mpeg",
            "file_type": "audio", "size_bytes": f.file_size, "caption": caption,
        }
        source = "صوت"

    if not attachment:
        return

    pending = _get_pending(user.id)
    if pending and pending[0] == "report_preview":
        report_id = int(pending[1])
        await _save_attachment(context, report_id=report_id, **attachment)
        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            _ = list(report.attachments)
            preview = render_report_preview(report)
        await update.effective_message.reply_text(preview, reply_markup=report_preview_keyboard(report_id))
        return

    text = caption or f"{source} برای گزارش کار"
    await _create_report_preview(update, context, user, text, source, attachment=attachment)


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = _upsert_user(update)
    action, _, value = (query.data or "").partition("|")

    if action in {"agent_confirm", "agent_edit", "agent_discard"}:
        try:
            draft_id = int(value)
        except Exception:
            await query.message.reply_text("این پیش‌نمایش دیگه معتبر نیست.")
            return

        with SessionLocal() as db:
            draft = db.get(AgentDraft, draft_id)
            valid = bool(draft and draft.user_id == user.id and draft.status == "draft")

        if not valid:
            _clear_pending(user.id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("این پیش‌نمایش دیگه معتبر نیست. دستور رو دوباره بگو.")
            return

        if action == "agent_edit":
            _set_pending(user.id, "agent_edit", str(draft_id))
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "بگو چی رو عوض کنم؛ خیلی عادی بگو. مثلاً «اولی رو بذار ساعت ۵»، «اون یکی رو حذف نکن»، یا «فردا رو سبک‌تر بچین»."
            )
            return

        if action == "agent_discard":
            with SessionLocal() as db:
                draft = db.get(AgentDraft, draft_id)
                if draft and draft.user_id == user.id:
                    draft.status = "discarded"
                    draft.updated_at = utc_now_iso()
                    db.commit()
            _clear_pending(user.id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("اوکی، هیچ تغییری ندادم 👌")
            return

        created, updated, deleted, completed = _apply_agent_draft(user.id, draft_id)
        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        bits = []
        if created:
            bits.append(f"{fa_num(created)} کار جدید")
        if updated:
            bits.append(f"{fa_num(updated)} تغییر")
        if completed:
            bits.append(f"{fa_num(completed)} انجام‌شده")
        if deleted:
            bits.append(f"{fa_num(deleted)} حذف")
        summary = "، ".join(bits) if bits else "تغییری"
        await query.message.reply_text(f"اوکی شد 👌 {summary} روی برنامه‌ت اعمال شد.")
        return

    if action in {"done_match_confirm", "done_match_edit", "done_match_discard"}:
        pending = _get_pending(user.id)
        if not pending or pending[0] != "done_match_preview":
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("این پیشنهاد دیگه معتبر نیست. دوباره عادی بگو کدوما رو انجام دادی.")
            return

        try:
            payload = json.loads(pending[1])
            task_ids = [int(x) for x in payload.get("t", [])]
            report_ids = [int(x) for x in payload.get("r", [])]
        except Exception:
            _clear_pending(user.id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("یه مشکلی توی این پیشنهاد پیش اومد. دوباره بگو کدوما رو انجام دادی.")
            return

        if action == "done_match_edit":
            with SessionLocal() as db:
                drafts = db.scalars(
                    select(WorkReport).where(
                        WorkReport.user_id == user.id,
                        WorkReport.id.in_(report_ids),
                        WorkReport.status == "draft_match",
                    )
                ).all() if report_ids else []
                for report in drafts:
                    db.delete(report)
                db.commit()
            _clear_pending(user.id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "اوکی، از نو بگو دقیقاً کدوما رو انجام دادی؛ خیلی عادی بگو، خودم با کارای بازت تطبیق می‌دم."
            )
            return

        if action == "done_match_discard":
            with SessionLocal() as db:
                drafts = db.scalars(
                    select(WorkReport).where(
                        WorkReport.user_id == user.id,
                        WorkReport.id.in_(report_ids),
                        WorkReport.status == "draft_match",
                    )
                ).all() if report_ids else []
                for report in drafts:
                    db.delete(report)
                db.commit()
            _clear_pending(user.id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("اوکی، هیچ‌کدوم رو ثبت نکردم.")
            return

        completed_titles = []
        with SessionLocal() as db:
            tasks = db.scalars(
                select(Task).where(
                    Task.user_id == user.id,
                    Task.id.in_(task_ids),
                    Task.status.in_(["todo", "doing"]),
                )
            ).all() if task_ids else []

            for task in tasks:
                task.status = "done"
                task.updated_at = utc_now_iso()
                completed_titles.append(task.title)

                existing = db.scalar(
                    select(WorkReport).where(
                        WorkReport.user_id == user.id,
                        WorkReport.task_id == task.id,
                        WorkReport.status == "submitted",
                    )
                )
                if existing is None:
                    db.add(WorkReport(
                        user_id=user.id,
                        task_id=task.id,
                        title=task.title,
                        summary=task.notes or task.title,
                        project=task.project or "",
                        category="کار انجام‌شده",
                        duration_minutes=None,
                        work_date=today_gregorian(),
                        source="تطبیق با کار قبلی",
                        original_text=task.original_text or task.title,
                        status="submitted",
                    ))

            if report_ids:
                drafts = db.scalars(
                    select(WorkReport).where(
                        WorkReport.user_id == user.id,
                        WorkReport.id.in_(report_ids),
                        WorkReport.status == "draft_match",
                    )
                ).all()
                for report in drafts:
                    report.status = "submitted"
                    report.updated_at = utc_now_iso()

            db.commit()

        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        count = len(completed_titles)
        if count:
            await query.message.reply_text(
                f"اوکی شد 👌 {fa_num(count)} تا کار رو انجام‌شده زدم و هرکدوم جدا رفت تو گزارش کارت."
            )
        else:
            await query.message.reply_text("اوکی، گزارش‌های جدیدت ثبت شدن.")
        return

    if action == "task_confirm":
        with SessionLocal() as db:
            tasks = db.scalars(select(Task).where(Task.batch_id == value, Task.user_id == user.id, Task.status == "draft")).all()
            for task in tasks:
                task.status = "todo"
                task.updated_at = utc_now_iso()
            db.commit()
        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("ثبت شد 👌 از اینجا به بعد یادآوریاشم حواسم هست.")
        return

    if action == "task_edit":
        _set_pending(user.id, "task_edit", value)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("بگو چی رو عوض کنم؟ هرجوری راحتی بگو. مثلاً «بذارش فردا ساعت ۱۰» یا «اون دومی رو حذف کن».")
        return

    if action == "task_discard":
        with SessionLocal() as db:
            tasks = db.scalars(select(Task).where(Task.batch_id == value, Task.user_id == user.id, Task.status == "draft")).all()
            for task in tasks:
                db.delete(task)
            db.commit()
        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("اوکی، بیخیالش شدیم 👌")
        return

    if action == "report_confirm":
        report_id = int(value)
        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            if report and report.user_id == user.id and report.status == "draft":
                report.status = "submitted"
                report.updated_at = utc_now_iso()
                db.commit()
        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("دمت گرم، گزارش ثبت شد ✅")
        return

    if action == "report_edit":
        _set_pending(user.id, "report_edit", value)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("بگو کجاش رو عوض کنم؟ مثلاً «دو ساعت بود» یا «اسم پروژه رو بذار فلان».")
        return

    if action == "report_discard":
        report_id = int(value)
        with SessionLocal() as db:
            report = db.get(WorkReport, report_id)
            if report and report.user_id == user.id and report.status == "draft":
                for att in list(report.attachments):
                    if att.local_path:
                        try:
                            Path(att.local_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                db.delete(report)
                db.commit()
        _clear_pending(user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("اوکی، ثبتش نکردم.")
        return

    if action == "done":
        task_id = int(value)
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            if not task or task.user_id != user.id:
                return
            task.status = "done"
            task.updated_at = utc_now_iso()
            report = WorkReport(
                user_id=user.id,
                task_id=task.id,
                title=task.title,
                summary=task.notes or task.title,
                project=task.project or "",
                category="کار انجام‌شده",
                duration_minutes=None,
                work_date=today_gregorian(),
                source="انجام کار",
                original_text=task.title,
                status="draft",
            )
            db.add(report)
            db.commit()
            db.refresh(report)
            report_id = report.id
            preview = render_report_preview(report)
        _set_pending(user.id, "report_preview", str(report_id))
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "این کار رو انجام‌شده زدم ✅ برای گزارش کار هم اینو ساختم؛ یه نگاه بنداز:",
        )
        await query.message.reply_text(preview, reply_markup=report_preview_keyboard(report_id))


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    tz = ZoneInfo(IRAN_TZ)
    today_date = datetime.now(tz).date()
    with SessionLocal() as db:
        tasks = db.scalars(select(Task).where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))).all()
    selected = []
    for task in tasks:
        target = task.scheduled_at or task.due_at
        if target:
            try:
                if datetime.fromisoformat(target).astimezone(tz).date() == today_date:
                    selected.append(task)
            except Exception:
                pass
    selected.sort(key=lambda t: t.scheduled_at or t.due_at or "")
    today_label = jalali_date(today_gregorian())
    if not selected:
        await update.effective_message.reply_text(f"برای امروز {today_label} چیزی ثبت نکردی.")
        return
    await update.effective_message.reply_text(f"کارای امروزت، {today_label} 👇")
    for task in selected[:30]:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ انجامش دادم", callback_data=f"done|{task.id}")]])
        when = local_datetime(task.scheduled_at or task.due_at)
        await update.effective_message.reply_text(f"• {task.title}\n🗓 {when}", reply_markup=kb)


async def upcoming(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task).where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))
            .order_by(Task.scheduled_at.asc(), Task.due_at.asc())
        ).all()
    if not tasks:
        await update.effective_message.reply_text("فعلاً کار باز ثبت‌شده‌ای نداری.")
        return
    lines = ["کارای بازت ایناست 👇"]
    for task in tasks[:30]:
        line = f"• {task.title}"
        if task.scheduled_at or task.due_at:
            line += f" — {local_datetime(task.scheduled_at or task.due_at)}"
        lines.append(line)
    await update.effective_message.reply_text("\n".join(lines))


async def today_reports(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    today_value = today_gregorian()

    with SessionLocal() as db:
        reports = db.scalars(
            select(WorkReport)
            .where(
                WorkReport.user_id == user.id,
                WorkReport.status == "submitted",
                WorkReport.work_date == today_value,
            )
            .order_by(WorkReport.created_at.asc())
        ).all()

        # داده‌های قدیمیِ ساخته‌شده با باگ قبلی را در خروجی امروز تکراری نشان نده.
        task_linked = [r for r in reports if r.task_id is not None]
        filtered = []
        seen_titles = []
        for report in reports:
            report_text = " ".join(filter(None, [report.title, report.summary, report.project]))

            # گزارش‌های بدون Task اگر حتی به متن اصلی خودشان هم ربطی ندارند، hallucination قدیمی‌اند.
            if report.task_id is None and report.original_text:
                if not _texts_semantically_overlap(report_text, report.original_text):
                    continue

            # اگر یک گزارش مستقل عملاً همان Task انجام‌شده است، فقط نسخه Task-linked را نگه دار.
            if report.task_id is None and any(
                _texts_semantically_overlap(
                    report_text,
                    " ".join(filter(None, [linked.title, linked.summary, linked.project]))
                )
                for linked in task_linked
            ):
                continue

            # تکراری‌های واضح را هم یک بار نشان بده.
            if any(_texts_semantically_overlap(report_text, prev) for prev in seen_titles):
                # فقط وقتی تقریباً یک عنوان‌اند حذف شود؛ گزارش‌های متفاوت با واژه مشترک نگه داشته می‌شوند.
                norm_title = _normalize_fa(report.title)
                if any(norm_title == _normalize_fa(prev_title) for prev_title in seen_titles):
                    continue
            filtered.append(report)
            seen_titles.append(report.title)

        rows = [(r, len(r.attachments)) for r in filtered]

    today_label = jalali_date(today_value)
    if not rows:
        await update.effective_message.reply_text(
            f"برای امروز {today_label} هنوز چیزی به‌عنوان انجام‌شده ثبت نکردی."
        )
        return

    total_minutes = sum((r.duration_minutes or 0) for r, _ in rows)
    parts = [f"امروز تا الان اینا رو جمع کردیم 👇", ""]
    for report, attachment_count in rows:
        line = f"✅ {report.title}"
        if report.project:
            line += f" · {report.project}"
        if report.duration_minutes:
            line += f" · {fa_num(report.duration_minutes)} دقیقه"
        if attachment_count:
            line += f" · 📎 {fa_num(attachment_count)}"
        parts.append(line)

    parts.append("")
    parts.append(f"جمعاً {fa_num(len(rows))} کار ثبت‌شده")
    if total_minutes:
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            parts.append(f"⏱ زمان ثبت‌شده: {fa_num(hours)} ساعت و {fa_num(minutes)} دقیقه")
        elif hours:
            parts.append(f"⏱ زمان ثبت‌شده: {fa_num(hours)} ساعت")
        else:
            parts.append(f"⏱ زمان ثبت‌شده: {fa_num(minutes)} دقیقه")

    await update.effective_message.reply_text("\n".join(parts))


async def my_reports(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    with SessionLocal() as db:
        reports = db.scalars(
            select(WorkReport)
            .where(WorkReport.user_id == user.id, WorkReport.status == "submitted")
            .order_by(WorkReport.created_at.desc())
            .limit(10)
        ).all()
        rows = [(r, len(r.attachments)) for r in reports]
    if not rows:
        await update.effective_message.reply_text("هنوز گزارش کاری ثبت نکردی.")
        return
    parts = ["آخرین گزارش‌هات 👇"]
    for report, attachment_count in rows:
        line = f"• {jalali_date(report.work_date)} · {report.title}"
        if report.project:
            line += f" · {report.project}"
        if report.duration_minutes:
            line += f" · {fa_num(report.duration_minutes)} دقیقه"
        if attachment_count:
            line += f" · 📎 {fa_num(attachment_count)}"
        parts.append(line)
    await update.effective_message.reply_text("\n".join(parts))


def build_application() -> Application | None:
    if not settings.telegram_bot_token:
        return None
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CallbackQueryHandler(callback))
    app.add_handler(MessageHandler(filters.VOICE, voice_message))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO | filters.VIDEO | filters.AUDIO, media_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    return app
