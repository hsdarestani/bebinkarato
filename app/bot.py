from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.ai import AIError, CloudflareAI
from app.config import get_settings
from app.db import SessionLocal
from app.models import Task, User, utc_now_iso

settings = get_settings()
ai = CloudflareAI()


def _lang(user: User | None) -> str:
    return "fa" if user and (user.language_code or "").lower().startswith("fa") else "en"


def _t(key: str, lang: str) -> str:
    strings = {
        "thinking": {"fa": "دارم کارهات رو مرتب می‌کنم…", "en": "Turning that into a plan…"},
        "no_tasks": {"fa": "کار مشخصی از این پیام پیدا نکردم.", "en": "I couldn't find a clear action item in that message."},
        "confirmed": {"fa": "✅ برنامه ثبت شد. من یادآوری‌ها رو هم پیگیری می‌کنم.", "en": "✅ Plan saved. I'll handle the reminders too."},
        "discarded": {"fa": "حذف شد.", "en": "Discarded."},
        "quota": {"fa": "سهمیه رایگان این ماهت تموم شده. پلن Pro به‌زودی اضافه میشه.", "en": "You've reached this month's free AI allowance. Pro is coming soon."},
        "ai_error": {"fa": "فعلاً نتونستم AI رو اجرا کنم. دوباره امتحان کن.", "en": "I couldn't run the planner right now. Please try again."},
    }
    return strings.get(key, {}).get(lang, strings.get(key, {}).get("en", key))


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
            "👋 هرچی توی ذهنت هست یکجا بفرست؛ متن یا ویس.\n\n"
            "من کارها رو جدا می‌کنم، ددلاین‌های واقعی رو تشخیص میدم، زمان پیشنهادی می‌چینم و یادآوری می‌کنم.\n\n"
            "دستورها:\n/today کارهای امروز\n/upcoming کارهای آینده\n/timezone Europe/Berlin تنظیم منطقه زمانی"
        )
    else:
        text = (
            "👋 Send everything on your mind in one message or voice note.\n\n"
            "I'll turn it into tasks, preserve real deadlines, suggest a schedule, and remind you.\n\n"
            "Commands:\n/today today's tasks\n/upcoming upcoming tasks\n/timezone Europe/Berlin set timezone"
        )
    await update.effective_message.reply_text(text)


async def set_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    lang = _lang(user)
    if not context.args:
        await update.effective_message.reply_text(
            ("منطقه زمانی فعلی: " if lang == "fa" else "Current timezone: ") + user.timezone
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
        db_user.timezone = value
        db.commit()
    await update.effective_message.reply_text(("✅ تنظیم شد: " if lang == "fa" else "✅ Set to: ") + value)


async def _create_draft(update: Update, user: User, text: str, source: str, voice_seconds: int = 0) -> None:
    lang = _lang(user)
    if not _quota_ok(user, voice_seconds):
        await update.effective_message.reply_text(_t("quota", lang))
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


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    text = (update.effective_message.text or "").strip()
    if text:
        await _create_draft(update, user, text, "text")


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = _upsert_user(update)
    voice = update.effective_message.voice
    if not voice:
        return
    if not _quota_ok(user, voice.duration or 0):
        await update.effective_message.reply_text(_t("quota", _lang(user)))
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
                await query.message.reply_text(_t("confirmed", _lang(user)))
            else:
                for task in tasks:
                    db.delete(task)
                db.commit()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(_t("discarded", _lang(user)))
        return

    if action == "done":
        try:
            task_id = int(value)
        except ValueError:
            return
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            if task and task.user_id == user.id:
                task.status = "done"
                task.updated_at = utc_now_iso()
                db.commit()
        await query.edit_message_reply_markup(reply_markup=None)


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
        await update.effective_message.reply_text("امروز کاری برنامه‌ریزی نشده 🎉" if _lang(user) == "fa" else "Nothing scheduled for today 🎉")
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
        await update.effective_message.reply_text("لیستت خالیه." if _lang(user) == "fa" else "Your list is empty.")
        return
    lines = [f"• {task.title} — {_local(task.scheduled_at or task.due_at, user.timezone)}" for task in tasks[:30]]
    await update.effective_message.reply_text("\n".join(lines))


def build_application() -> Application | None:
    if not settings.telegram_bot_token:
        return None
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("upcoming", upcoming))
    app.add_handler(CommandHandler("timezone", set_timezone))
    app.add_handler(CallbackQueryHandler(callback))
    app.add_handler(MessageHandler(filters.VOICE, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    return app
