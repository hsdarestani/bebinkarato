import asyncio
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import CloudflareAI
from app.bot import _simple_replacement, _simple_delete_index, _deadline_was_explicit


async def main() -> None:
    assert _simple_replacement("پاساژ نه باشگاه") == ("پاساژ", "باشگاه")
    assert _simple_replacement("میگم پاساژ نه باشگاه") == ("پاساژ", "باشگاه")
    assert _simple_delete_index("دومی رو حذف کن") == 1
    assert not _deadline_was_explicit("امروز چند تا کار دارم انجام بدم")
    assert _deadline_was_explicit("این کار باید تا فردا تموم بشه")

    ai = CloudflareAI()
    try:
        await ai._resolve_account_id()

        tasks = await ai.parse_tasks(
            "فردا ساعت ده باید به دندانپزشک زنگ بزنم",
            "Asia/Tehran",
            "fa",
        )
        assert tasks and tasks[0].get("title")
        assert not any(x in tasks[0]["title"].lower() for x in ["call", "dentist"])

        report = await ai.parse_report(
            "امروز باگ پرداخت رو حل کردم و دو ساعت روش بودم",
            "Asia/Tehran",
            "fa",
        )
        assert report.get("title")
        assert report.get("work_date")
        assert report.get("duration_minutes") == 120

        assert await ai.classify_intent("رباته رو ساختم کامل", "Asia/Tehran") == "report"
        assert await ai.classify_intent("یه ربات باید بسازم فیچراشو درارم", "Asia/Tehran") == "plan"
        assert await ai.classify_intent("امروز چی دارم؟", "Asia/Tehran") == "today"
        assert await ai.classify_intent("امروز چیکار کردیم؟", "Asia/Tehran") == "today_reports"

        today = datetime.now(ZoneInfo("Asia/Tehran")).date().isoformat()
        draft = {
            "mode": "mutate",
            "operations": [
                {"type": "create", "task": {"title": "بسته بندی چای", "notes": "", "project": "", "priority": "medium", "estimated_minutes": 30, "due_at": None, "due_source": "none", "scheduled_at": f"{today}T15:57:00+03:30", "reminder_at": None}},
                {"type": "create", "task": {"title": "خرید غذا و تشویقی", "notes": "", "project": "", "priority": "medium", "estimated_minutes": 30, "due_at": None, "due_source": "none", "scheduled_at": f"{today}T16:27:00+03:30", "reminder_at": None}},
                {"type": "create", "task": {"title": "مرتب کردن اتاق کار", "notes": "", "project": "", "priority": "medium", "estimated_minutes": 30, "due_at": None, "due_source": "none", "scheduled_at": f"{today}T16:57:00+03:30", "reminder_at": None}},
                {"type": "create", "task": {"title": "تکمیل سایت ریشه و انتشار نسخه جدید", "notes": "", "project": "", "priority": "high", "estimated_minutes": 120, "due_at": None, "due_source": "none", "scheduled_at": f"{today}T17:27:00+03:30", "reminder_at": None}},
            ],
        }
        revised = await ai.revise_planning_draft(
            "ساعتاشو عوض کن. چای رو بذار ساعت ۸ شب، خرید غذا و تشویقی ساعت ۵ عصر، مرتب کردن اتاق ساعت ۱۰ شب، تکمیل سایت هم ساعت ۱۱ شب",
            draft,
            [],
            "Asia/Tehran",
        )
        assert len(revised["operations"]) == 4
        local_hours = [
            datetime.fromisoformat(op["task"]["scheduled_at"]).astimezone(ZoneInfo("Asia/Tehran")).hour
            for op in revised["operations"]
        ]
        assert local_hours == [20, 17, 22, 23], local_hours

    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
