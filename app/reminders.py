import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Task, User, utc_now_iso


async def reminder_loop(bot, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        now = datetime.now(timezone.utc).isoformat()
        with SessionLocal() as db:
            tasks = db.scalars(
                select(Task).where(
                    Task.status.in_(["todo", "doing"]),
                    Task.reminder_sent.is_(False),
                    Task.reminder_at.is_not(None),
                    Task.reminder_at <= now,
                ).limit(100)
            ).all()
            payloads = []
            for task in tasks:
                user = db.get(User, task.user_id)
                if user:
                    payloads.append((task.id, user.telegram_id, task.title))

        for task_id, telegram_id, title in payloads:
            try:
                await bot.send_message(chat_id=telegram_id, text=f"⏰ {title}")
                with SessionLocal() as db:
                    task = db.get(Task, task_id)
                    if task:
                        task.reminder_sent = True
                        task.updated_at = utc_now_iso()
                        db.commit()
            except Exception as exc:
                print(f"Reminder send failed for task {task_id}: {exc}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
