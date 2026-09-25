from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.ai import AIError, CloudflareAI
from app.config import get_settings
from app.db import SessionLocal
from app.models import ReportAttachment, Task, User, UserMode, WorkReport, utc_now_iso

settings = get_settings()
ai = CloudflareAI()
UPLOAD_ROOT = Path("/data/uploads")


def _lang(user: User | None) -> str:
    return "fa" if user and (user.language_code or "").lower().startswith("fa") else "en"


def _t(key: str, lang: str) -> str:
    strings = {
        "thinking": {"fa": "دارم کارهات رو مرتب می‌کنم…", "en": "Turning that into a plan…"},
        "no_tasks": {"fa": "کار مشخصی از این پیام پیدا نکردم.", "en": "I couldn't find a clear action item in that message."},
        "confirmed": {"fa": "✅ برنامه ثبت شد. من یادآوری‌ها رو هم پیگیری می‌کنم.", "en": "✅ Plan saved. I'll handle the reminders too."},
        "discarded": {"fa": "حذف شد.", "en": "Discarded."},
        "quota": {"fa": "سهمیه AI این ماهت تموم شده؛ گزارش کارت بازم ذخیره میشه.", "en": "Your AI allowance is used up; work reports will still be saved."},
        "ai_error": {"fa": "فعلاً نتونستم AI رو اجرا کنم. دوباره امتحان کن.", "en": "I couldn't run the planner right now. Please try again."},
        "report_saved": {"fa": "✅ گزارش کار ثبت شد", "en": "✅ Work report saved"},
    }
    return strings.get(key, {}).get(lang, strings.get(key, {}).get("en", key))


def _keyboard(lang: str) -> ReplyKeyboardMarkup:
    if lang == "fa":
        rows = [
            [KeyboardButton("🗓 برنامه‌ریزی"), KeyboardButton("✅ گزارش کار")],
            [KeyboardButton("📅 امروز"), KeyboardButton("📊 گزارش‌های من")],
        ]
    else:
        rows = [
            [KeyboardButton("🗓 Planning"), KeyboardButton("✅ Work report")],
            [KeyboardButton("📅 Today"), KeyboardButton("📊 My reports")],
        ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, is_persistent=True)


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
                language_code=tg.language_code or "en",
                timezone=settings.default_timezone,
                usage_month=month,
            )
            db.add(user)
            db.flush()
            db.add(UserMode(user_id=user.id, mode="tasks"))
        else:
            user.username = tg.username
            user.first_name = tg.first_name
            user.last_name = tg.last_name
            user.language_code = tg.language_code or user.language_code or "en"
            user.last_seen_at = utc_now_iso()
            if user.usage_month != month:
                user.usage_month = month
                user.ai_requests_month = 0
                user.voice_seconds_month = 0
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user


def _get_mode(user_id: int) -> str:
    with SessionLocal() as db:
        state = db.scalar(select(UserMode).where(UserMode.user_id == user_id))
        return state.mode if state else "tasks"


def _set_mode(user_id: int, mode: str) -> None:
    with SessionLocal() as db:
        state = db.scalar(select(UserMode).where(UserMode.user_id == user_id))
        if state is None:
            state = UserMode(user_id=user_id, mode=mode)
            db.add(state)
        else:
            state.mode = mode
            state.updated_at = utc_now_iso()
        db.commit()


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


def _local(dt_iso: str | None, timezone_name: str) -> str:
    if not dt_iso:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(ZoneInfo(timezone_name))
        return dt.strftime("%d.%m %H:%M")
    except Exception:
        return "—"


def _today_local(user: User) -> str:
    try:
        return datetime.now(ZoneInfo(user.timezone)).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    lang = _lang(user)
    if lang == "fa":
        text = (
            "👋 خیلی ساده کار کن:\n\n"
            "🗓 «برنامه‌ریزی» برای کارهایی که باید انجام بدی\n"
            "✅ «گزارش کار» برای چیزهایی که انجام دادی\n\n"
            "برای هر دو می‌تونی متن یا ویس بفرستی. برای گزارش کار عکس، PDF، فایل، ویدیو و صوت هم می‌تونی پیوست کنی."
        )
    else:
        text = (
            "👋 Keep it simple:\n\n"
            "🗓 Planning is for work you need to do.\n"
            "✅ Work report is for work you've completed.\n\n"
            "Both accept text or voice. Work reports also accept photos, PDFs, files, video, and audio attachments."
        )
    await update.effective_message.reply_text(text, reply_markup=_keyboard(lang))


async def set_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    lang = _lang(user)
    if not context.args:
        await update.effective_message.reply_text(
            ("منطقه زمانی فعلی: " if lang == "fa" else "Current timezone: ") + user.timezone,
            reply_markup=_keyboard(lang),
        )
        return
    value = context.args[0].strip()
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError:
        await update.effective_message.reply_text("Invalid timezone. Example: Europe/Berlin or Asia/Tehran")
        return
    with SessionLocal() as db:
        db_user = db.get(User, user.id)
        if db_user:
            db_user.timezone = value
            db.commit()
    await update.effective_message.reply_text(
        ("✅ تنظیم شد: " if lang == "fa" else "✅ Set to: ") + value,
        reply_markup=_keyboard(lang),
    )


async def task_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    _set_mode(user.id, "tasks")
    lang = _lang(user)
    text = (
        "🗓 حالت برنامه‌ریزی فعاله. هرچی باید انجام بدی با متن یا ویس بفرست."
        if lang == "fa"
        else "🗓 Planning mode is active. Send what you need to do by text or voice."
    )
    await update.effective_message.reply_text(text, reply_markup=_keyboard(lang))


async def report_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    _set_mode(user.id, "reports")
    lang = _lang(user)
    text = (
        "✅ حالت گزارش کار فعاله. هر کاری انجام دادی با متن یا ویس بفرست؛ عکس، PDF و فایل هم می‌تونی مستقیم پیوست کنی."
        if lang == "fa"
        else "✅ Work report mode is active. Send completed work as text or voice, or attach a photo, PDF, or file."
    )
    await update.effective_message.reply_text(text, reply_markup=_keyboard(lang))


async def _create_draft(update: Update, user: User, text: str, source: str, voice_seconds: int = 0) -> None:
    lang = _lang(user)
    if not _quota_ok(user, voice_seconds):
        await update.effective_message.reply_text(_t("quota", lang), reply_markup=_keyboard(lang))
        return

    status_msg = await update.effective_message.reply_text(_t("thinking", lang))
    try:
        parsed = await ai.parse_tasks(text, user.timezone, user.language_code)
        _charge_usage(user.id, voice_seconds)
    except AIError as exc:
        print(f"AI error: {exc}")
        await status_msg.edit_text(_t("ai_error", lang))
        return

    if not parsed:
        await status_msg.edit_text(_t("no_tasks", lang))
        return

    batch_id = str(uuid4())
    with SessionLocal() as db:
        for item in parsed:
            reminder = item.get("reminder_at") or _default_reminder(item.get("scheduled_at"), item.get("due_at"))
            db.add(
                Task(
                    user_id=user.id,
                    batch_id=batch_id,
                    title=item["title"],
                    notes=item.get("notes", ""),
                    project=item.get("project", ""),
                    priority=item.get("priority", "medium"),
                    estimated_minutes=item.get("estimated_minutes", 30),
                    status="draft",
                    source=source,
                    original_text=text,
                    due_at=item.get("due_at"),
                    due_source=item.get("due_source", "none"),
                    scheduled_at=item.get("scheduled_at"),
                    reminder_at=reminder,
                )
            )
        db.commit()

    lines = []
    for i, item in enumerate(parsed, start=1):
        when = _local(item.get("scheduled_at"), user.timezone)
        deadline = _local(item.get("due_at"), user.timezone)
        line = f"{i}. {item['title']}\n   🗓 {when}"
        if item.get("due_at"):
            line += f"  ⏳ {deadline}"
        lines.append(line)

    title = "این برنامه رو پیشنهاد می‌کنم:" if lang == "fa" else "Here's the proposed plan:"
    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ تأیید" if lang == "fa" else "✅ Confirm", callback_data=f"confirm|{batch_id}"),
            InlineKeyboardButton("🗑 حذف" if lang == "fa" else "🗑 Discard", callback_data=f"discard|{batch_id}"),
        ]]
    )
    await status_msg.edit_text(title + "\n\n" + "\n\n".join(lines), reply_markup=keyboard)


def _fallback_report(user: User, text: str, fallback_title: str = "") -> dict:
    clean = (text or "").strip()
    title = clean.splitlines()[0][:180] if clean else fallback_title[:180]
    if not title:
        title = "Work report"
    return {
        "title": title,
        "summary": clean,
        "project": "",
        "category": "work",
        "duration_minutes": None,
        "work_date": _today_local(user),
    }


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


async def _create_report(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user: User,
    text: str,
    source: str,
    voice_seconds: int = 0,
    fallback_title: str = "",
    attachment: dict | None = None,
) -> int:
    lang = _lang(user)
    parsed = _fallback_report(user, text, fallback_title)

    if text.strip() and _quota_ok(user, voice_seconds):
        try:
            parsed = await ai.parse_report(text, user.timezone, user.language_code)
            _charge_usage(user.id, voice_seconds)
        except Exception as exc:
            print(f"Report AI fallback: {exc}")

    with SessionLocal() as db:
        report = WorkReport(
            user_id=user.id,
            title=parsed["title"],
            summary=parsed.get("summary", ""),
            project=parsed.get("project", ""),
            category=parsed.get("category", "work"),
            duration_minutes=parsed.get("duration_minutes"),
            work_date=parsed.get("work_date") or _today_local(user),
            source=source,
            original_text=text,
            status="submitted",
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        report_id = report.id

    attachment_saved = False
    if attachment:
        try:
            await _save_attachment(context, report_id=report_id, **attachment)
            attachment_saved = True
        except Exception as exc:
            print(f"Attachment save failed: {exc}")

    details = [f"{_t('report_saved', lang)}\n{parsed['title']}"]
    if parsed.get("project"):
        details.append(("📁 " if lang == "fa" else "📁 ") + parsed["project"])
    if parsed.get("duration_minutes"):
        details.append(f"⏱ {parsed['duration_minutes']} min")
    details.append(f"📅 {parsed.get('work_date') or _today_local(user)}")
    if attachment_saved:
        details.append("📎 " + ("پیوست ذخیره شد" if lang == "fa" else "Attachment saved"))

    await update.effective_message.reply_text("\n".join(details), reply_markup=_keyboard(lang))
    return report_id


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    text = (update.effective_message.text or "").strip()
    if not text:
        return

    if text in {"🗓 برنامه‌ریزی", "🗓 Planning"}:
        await task_mode(update, context)
        return
    if text in {"✅ گزارش کار", "✅ Work report"}:
        await report_mode(update, context)
        return
    if text in {"📅 امروز", "📅 Today"}:
        await today(update, context)
        return
    if text in {"📊 گزارش‌های من", "📊 My reports"}:
        await my_reports(update, context)
        return

    if _get_mode(user.id) == "reports":
        await _create_report(update, context, user, text, "text")
    else:
        await _create_draft(update, user, text, "text")


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    voice = update.effective_message.voice
    if not voice:
        return

    if _get_mode(user.id) == "reports":
        msg = await update.effective_message.reply_text("🎙️ …")
        try:
            tg_file = await context.bot.get_file(voice.file_id)
            audio = bytes(await tg_file.download_as_bytearray())
            transcript = await ai.transcribe(audio)
            await msg.delete()
            attachment = {
                "file_id": voice.file_id,
                "unique_id": voice.file_unique_id,
                "file_name": f"voice-{voice.file_unique_id}.ogg",
                "mime_type": voice.mime_type or "audio/ogg",
                "file_type": "voice",
                "size_bytes": voice.file_size,
                "caption": transcript,
                "raw_bytes": audio,
            }
            await _create_report(
                update, context, user, transcript, "voice", voice.duration or 0,
                fallback_title="Voice report", attachment=attachment
            )
        except Exception as exc:
            print(f"Voice report error: {exc}")
            try:
                await msg.edit_text(_t("ai_error", _lang(user)))
            except Exception:
                pass
        return

    if not _quota_ok(user, voice.duration or 0):
        await update.effective_message.reply_text(_t("quota", _lang(user)), reply_markup=_keyboard(_lang(user)))
        return
    msg = await update.effective_message.reply_text("🎙️ …")
    try:
        tg_file = await context.bot.get_file(voice.file_id)
        audio = bytes(await tg_file.download_as_bytearray())
        transcript = await ai.transcribe(audio)
        await msg.edit_text("📝 " + transcript[:3500])
        await _create_draft(update, user, transcript, "voice", voice.duration or 0)
    except AIError as exc:
        print(f"Voice AI error: {exc}")
        await msg.edit_text(_t("ai_error", _lang(user)))


async def media_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    message = update.effective_message
    caption = (message.caption or "").strip()
    attachment = None
    fallback_title = ""
    source = "document"

    if message.document:
        f = message.document
        source = "document"
        fallback_title = f.file_name or "Document"
        attachment = {
            "file_id": f.file_id,
            "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"document-{f.file_unique_id}",
            "mime_type": f.mime_type or "application/octet-stream",
            "file_type": "document",
            "size_bytes": f.file_size,
            "caption": caption,
        }
    elif message.photo:
        f = message.photo[-1]
        source = "photo"
        fallback_title = "Photo"
        attachment = {
            "file_id": f.file_id,
            "unique_id": f.file_unique_id,
            "file_name": f"photo-{f.file_unique_id}.jpg",
            "mime_type": "image/jpeg",
            "file_type": "photo",
            "size_bytes": f.file_size,
            "caption": caption,
        }
    elif message.video:
        f = message.video
        source = "video"
        fallback_title = f.file_name or "Video"
        attachment = {
            "file_id": f.file_id,
            "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"video-{f.file_unique_id}.mp4",
            "mime_type": f.mime_type or "video/mp4",
            "file_type": "video",
            "size_bytes": f.file_size,
            "caption": caption,
        }
    elif message.audio:
        f = message.audio
        source = "audio"
        fallback_title = f.file_name or f.title or "Audio"
        attachment = {
            "file_id": f.file_id,
            "unique_id": f.file_unique_id,
            "file_name": f.file_name or f"audio-{f.file_unique_id}.mp3",
            "mime_type": f.mime_type or "audio/mpeg",
            "file_type": "audio",
            "size_bytes": f.file_size,
            "caption": caption,
        }

    if not attachment:
        return

    _set_mode(user.id, "reports")
    await _create_report(
        update,
        context,
        user,
        caption or fallback_title,
        source,
        fallback_title=fallback_title,
        attachment=attachment,
    )


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = _upsert_user(update)
    data = query.data or ""
    action, _, value = data.partition("|")

    if action in {"confirm", "discard"}:
        with SessionLocal() as db:
            tasks = db.scalars(
                select(Task).where(Task.batch_id == value, Task.user_id == user.id, Task.status == "draft")
            ).all()
            if action == "confirm":
                for task in tasks:
                    task.status = "todo"
                    task.updated_at = utc_now_iso()
                db.commit()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(_t("confirmed", _lang(user)), reply_markup=_keyboard(_lang(user)))
            else:
                for task in tasks:
                    db.delete(task)
                db.commit()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(_t("discarded", _lang(user)), reply_markup=_keyboard(_lang(user)))
        return

    if action == "done":
        try:
            task_id = int(value)
        except ValueError:
            return
        created_report = False
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            if task and task.user_id == user.id:
                task.status = "done"
                task.updated_at = utc_now_iso()
                existing = db.scalar(select(WorkReport).where(WorkReport.task_id == task.id))
                if existing is None:
                    db.add(
                        WorkReport(
                            user_id=user.id,
                            task_id=task.id,
                            title=task.title,
                            summary=task.notes or task.title,
                            project=task.project or "",
                            category="task",
                            duration_minutes=task.estimated_minutes or None,
                            work_date=_today_local(user),
                            source="task_done",
                            original_text=task.title,
                            status="submitted",
                        )
                    )
                    created_report = True
                db.commit()
        await query.edit_message_reply_markup(reply_markup=None)
        if created_report:
            await query.message.reply_text(
                "✅ انجام شد و به گزارش کار هم اضافه شد." if _lang(user) == "fa" else "✅ Done and added to your work log.",
                reply_markup=_keyboard(_lang(user)),
            )


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    tz = ZoneInfo(user.timezone)
    today_date = datetime.now(tz).date()
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task).where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))
        ).all()

    selected = []
    for task in tasks:
        target = task.scheduled_at or task.due_at
        if not target:
            continue
        try:
            if datetime.fromisoformat(target).astimezone(tz).date() == today_date:
                selected.append(task)
        except Exception:
            pass
    selected.sort(key=lambda t: t.scheduled_at or t.due_at or "")

    if not selected:
        await update.effective_message.reply_text(
            "امروز کاری برنامه‌ریزی نشده 🎉" if _lang(user) == "fa" else "Nothing scheduled for today 🎉",
            reply_markup=_keyboard(_lang(user)),
        )
        return

    for task in selected[:30]:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Done", callback_data=f"done|{task.id}")]])
        await update.effective_message.reply_text(
            f"• {task.title}\n🗓 {_local(task.scheduled_at or task.due_at, user.timezone)}",
            reply_markup=keyboard,
        )


async def upcoming(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task)
            .where(Task.user_id == user.id, Task.status.in_(["todo", "doing"]))
            .order_by(Task.scheduled_at.asc(), Task.due_at.asc())
        ).all()
    if not tasks:
        await update.effective_message.reply_text(
            "لیستت خالیه." if _lang(user) == "fa" else "Your list is empty.",
            reply_markup=_keyboard(_lang(user)),
        )
        return
    lines = [f"• {task.title} — {_local(task.scheduled_at or task.due_at, user.timezone)}" for task in tasks[:30]]
    await update.effective_message.reply_text("\n".join(lines), reply_markup=_keyboard(_lang(user)))


async def my_reports(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    with SessionLocal() as db:
        reports = db.scalars(
            select(WorkReport)
            .where(WorkReport.user_id == user.id)
            .order_by(WorkReport.created_at.desc())
            .limit(10)
        ).all()
        rows = []
        for report in reports:
            attachment_count = len(report.attachments)
            rows.append((report, attachment_count))

    if not rows:
        await update.effective_message.reply_text(
            "هنوز گزارشی ثبت نکردی." if _lang(user) == "fa" else "You haven't submitted any work reports yet.",
            reply_markup=_keyboard(_lang(user)),
        )
        return

    parts = []
    for report, attachment_count in rows:
        line = f"• {report.work_date} · {report.title}"
        if report.project:
            line += f" · {report.project}"
        if report.duration_minutes:
            line += f" · {report.duration_minutes}m"
        if attachment_count:
            line += f" · 📎 {attachment_count}"
        parts.append(line)

    await update.effective_message.reply_text("\n".join(parts), reply_markup=_keyboard(_lang(user)))


def build_application() -> Application | None:
    if not settings.telegram_bot_token:
        return None
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("upcoming", upcoming))
    app.add_handler(CommandHandler("reports", my_reports))
    app.add_handler(CommandHandler("report", report_mode))
    app.add_handler(CommandHandler("tasks", task_mode))
    app.add_handler(CommandHandler("timezone", set_timezone))
    app.add_handler(CallbackQueryHandler(callback))
    app.add_handler(MessageHandler(filters.VOICE, voice_message))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO | filters.VIDEO | filters.AUDIO, media_report))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    return app
