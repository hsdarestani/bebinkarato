import asyncio
import jdatetime
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import CloudflareAI
from app.bot import _simple_replacement, _simple_delete_index, _deadline_was_explicit, _reply_means_completed, _reply_task_title, _target_gregorian_date_from_text, _target_gregorian_date_from_text, _draft_is_delete_only


async def main() -> None:
    assert _simple_replacement("پاساژ نه باشگاه") == ("پاساژ", "باشگاه")
    assert _simple_replacement("میگم پاساژ نه باشگاه") == ("پاساژ", "باشگاه")
    assert _simple_delete_index("دومی رو حذف کن") == 1
    assert not _deadline_was_explicit("امروز چند تا کار دارم انجام بدم")
    assert _deadline_was_explicit("این کار باید تا فردا تموم بشه")
    assert _reply_means_completed("اینو انجام دادم")
    assert _reply_means_completed("تموم شد")
    assert not _reply_means_completed("فردا انجامش میدم")
    assert _reply_task_title("⏰ خرید غذا و تشویقی سرمه") == "خرید غذا و تشویقی سرمه"
    assert _reply_task_title("• جلسه ریشه\n🗓 ۱۴۰۵/۰۷/۰۶ ساعت ۱۵:۳۰") == "جلسه ریشه"

    iran_today = datetime.now(ZoneInfo("Asia/Tehran")).date()
    current_j = jdatetime.date.fromgregorian(date=iran_today)
    expected_7_5 = jdatetime.date(current_j.year, 7, 5).togregorian().isoformat()
    assert _target_gregorian_date_from_text("فقط برای تاریخ ۵/۷") == expected_7_5
    from datetime import timedelta as _td
    assert _target_gregorian_date_from_text("کارای دیروز") == (iran_today - _td(days=1)).isoformat()
    iran_today = datetime.now(ZoneInfo("Asia/Tehran")).date()
    assert _target_gregorian_date_from_text("فقط کارای دیروز رو حذف کن") == (iran_today - timedelta(days=1)).isoformat()
    explicit = _target_gregorian_date_from_text("نه فقط برای تاریخ ۷/۵")
    assert explicit is not None
    assert _draft_is_delete_only({"operations": [{"type": "delete", "task_id": 1}, {"type": "delete", "task_id": 2}]})
    assert not _draft_is_delete_only({"operations": [{"type": "delete", "task_id": 1}, {"type": "update", "task_id": 2}]})

    prepared = CloudflareAI._prepare_daily_brain_dump(
        [
            {"type": "create", "task": {"title": "خرید غذا و تشویقی", "notes": "", "project": "", "priority": "medium", "estimated_minutes": 45, "due_at": None, "due_source": "none", "scheduled_at": None, "reminder_at": None}},
            {"type": "create", "task": {"title": "بسته بندی چای", "notes": "", "project": "", "priority": "medium", "estimated_minutes": 25, "due_at": None, "due_source": "none", "scheduled_at": None, "reminder_at": None}},
            {"type": "create", "task": {"title": "جلسه ریشه ساعت 15:30", "notes": "", "project": "", "priority": "high", "estimated_minutes": 60, "due_at": None, "due_source": "none", "scheduled_at": None, "reminder_at": None}},
            {"type": "create", "task": {"title": "آماده کردن ساختار replication اپلیکیشن سولوشن", "notes": "", "project": "", "priority": "high", "estimated_minutes": 120, "due_at": None, "due_source": "none", "scheduled_at": None, "reminder_at": None}},
        ],
        "خرید غذا و تشویقی\nبسته بندی چای\nجلسه ریشه ساعت 15:30\nآماده کردن ساختار replication اپلیکیشن سولوشن",
        "Asia/Tehran",
    )
    assert prepared[2]["task"]["title"] == "جلسه ریشه"
    meeting_dt = datetime.fromisoformat(prepared[2]["task"]["scheduled_at"]).astimezone(ZoneInfo("Asia/Tehran"))
    assert (meeting_dt.hour, meeting_dt.minute) == (15, 30)
    prepared_starts = [op["task"]["scheduled_at"] for op in prepared]
    assert all(prepared_starts)
    assert len(set(prepared_starts)) == len(prepared_starts)

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

        assert await ai.reply_confirms_completion("اینو رفتیم", "تئاتر بریم", "⏰ تئاتر بریم")
        assert await ai.reply_confirms_completion("جمع شد، بزن انجام شده", "جلسه کتابخوانی", "⏰ جلسه کتابخوانی")
        assert not await ai.reply_confirms_completion("نرسیدیم بریم، بنداز فردا", "تئاتر بریم", "⏰ تئاتر بریم")

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
