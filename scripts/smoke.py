import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import CloudflareAI


async def main() -> None:
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
    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
