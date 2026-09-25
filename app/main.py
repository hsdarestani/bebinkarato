import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.admin import router as admin_router
from app.bot import ai, build_application
from app.config import get_settings
from app.db import init_db
from app.reminders import reminder_loop

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    telegram = build_application()
    stop_event = asyncio.Event()
    reminder_task = None

    if telegram is not None:
        await telegram.initialize()
        await telegram.start()
        if telegram.updater is not None:
            await telegram.updater.start_polling(drop_pending_updates=False)
        reminder_task = asyncio.create_task(reminder_loop(telegram.bot, stop_event))
        app.state.telegram_running = True
    else:
        app.state.telegram_running = False

    try:
        yield
    finally:
        stop_event.set()
        if reminder_task is not None:
            try:
                await reminder_task
            except Exception:
                pass
        if telegram is not None:
            if telegram.updater is not None:
                await telegram.updater.stop()
            await telegram.stop()
            await telegram.shutdown()
        await ai.close()


app = FastAPI(title="Bebin Karato", version="0.1.0", lifespan=lifespan)
app.include_router(admin_router)


@app.get("/")
def root() -> dict:
    return {"service": "bebinkarato", "ok": True}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "telegram_configured": bool(settings.telegram_bot_token),
        "cloudflare_configured": bool(settings.cloudflare_api_token),
        "admin_configured": bool(settings.admin_password),
    }
