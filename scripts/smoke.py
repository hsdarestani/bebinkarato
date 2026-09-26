import asyncio
import sys
from pathlib import Path

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

        revised = await ai.revise_tasks(
            [
                {
                    "title": "سایت گرویتاس را درست کنید",
                    "notes": "",
                    "project": "سایت گرویتاس",
                    "priority": "medium",
                    "estimated_minutes": 120,
                    "due_at": None,
                    "due_source": "none",
                    "scheduled_at": None,
                    "reminder_at": None,
                },
                {
                    "title": "باشگاه را بردارید",
                    "notes": "",
                    "project": "",
                    "priority": "medium",
                    "estimated_minutes": 30,
                    "due_at": None,
                    "due_source": "none",
                    "scheduled_at": None,
                    "reminder_at": None,
                },
            ],
            "پاساژ رو بردارید نه، برم باشگاه",
            "Asia/Tehran",
            original_text="سایت گرویتاس رو درست کنم، پاساژ رو بردارید، موز بخرم",
        )
        assert len(revised) == 2
        assert "باشگاه" in revised[1]["title"]
        assert "بردارید" not in revised[1]["title"]
    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
