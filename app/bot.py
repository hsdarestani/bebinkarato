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
from app.models import PendingAction, ReportAttachment, Task, User, WorkReport, utc_now_iso

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
        if await _try_simple_task_edit(update, user, ref, instruction):
            return True

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
            revised = await ai.revise_tasks(current, instruction, IRAN_TZ)
        except Exception as exc:
            print(f"Task revision error: {exc}")
            await update.effective_message.reply_text("نتونستم اصلاحش کنم 😅 یه بار دیگه بگو چی رو عوض کنم.")
            return True
        with SessionLocal() as db:
            old = db.scalars(
                select(Task).where(Task.batch_id == ref, Task.user_id == user.id, Task.status == "draft")
            ).all()
            for task in old:
                db.delete(task)
            for item in revised:
                due_at = item.get("due_at") if _deadline_was_explicit(original_text) else None
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


async def route_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    voice_seconds: int = 0,
    from_voice: bool = False,
) -> None:
    user = _upsert_user(update)
    pending = _get_pending(user.id)
    if pending and pending[0] in {"task_edit", "report_edit"}:
        if from_voice:
            handled = await _replace_edit_from_voice(
                update, context, user, text, voice_seconds, pending
            )
        else:
            handled = await _handle_edit_text(update, user, text, pending)
        if handled:
            return

    try:
        intent = await ai.classify_intent(text, IRAN_TZ)
    except Exception as exc:
        print(f"Intent error: {exc}")
        intent = "unknown"

    if intent == "today":
        await today(update, context)
    elif intent == "upcoming":
        await upcoming(update, context)
    elif intent == "reports":
        await my_reports(update, context)
    elif intent == "report":
        await _create_report_preview(update, context, user, text, "voice" if voice_seconds else "text", voice_seconds)
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
