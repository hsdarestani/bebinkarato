from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Task, WorkReport

IRAN_TZ = ZoneInfo("Asia/Tehran")


def norm(value: str) -> str:
    value = (value or "").replace("ي", "ی").replace("ك", "ک").replace("‌", " ").lower()
    value = re.sub(r"[^\w\sآ-ی]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def task_day(task: Task) -> str | None:
    raw = task.scheduled_at or task.due_at
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(IRAN_TZ).date().isoformat()
    except Exception:
        return None


def score(task: Task) -> tuple[int, str]:
    return (
        2 if task.scheduled_at else 1 if task.due_at else 0,
        task.updated_at or task.created_at or "",
    )


def main() -> None:
    removed = 0
    with SessionLocal() as db:
        user_ids = db.scalars(select(Task.user_id).distinct()).all()
        for user_id in user_ids:
            tasks = db.scalars(
                select(Task)
                .where(Task.user_id == user_id)
                .order_by(Task.id.asc())
            ).all()

            groups: dict[str, list[Task]] = {}
            for task in tasks:
                key = norm(task.title)
                if key:
                    groups.setdefault(key, []).append(task)

            for group in groups.values():
                if len(group) < 2:
                    continue

                done_tasks = [t for t in group if t.status == "done"]
                open_tasks = [t for t in group if t.status in {"todo", "doing"}]

                # If one exact copy was already completed and reported on the same day,
                # remove open duplicates for that same day (or unscheduled copies).
                done_report_days: set[str] = set()
                for done in done_tasks:
                    reports = db.scalars(
                        select(WorkReport).where(
                            WorkReport.user_id == user_id,
                            WorkReport.task_id == done.id,
                            WorkReport.status == "submitted",
                        )
                    ).all()
                    for report in reports:
                        if report.work_date:
                            done_report_days.add(report.work_date)

                for task in list(open_tasks):
                    day = task_day(task)
                    if done_report_days and (day in done_report_days or day is None):
                        db.delete(task)
                        open_tasks.remove(task)
                        removed += 1

                # Prefer a scheduled copy over an unscheduled exact duplicate.
                scheduled = [t for t in open_tasks if task_day(t) is not None]
                unscheduled = [t for t in open_tasks if task_day(t) is None]
                if scheduled and unscheduled:
                    for task in unscheduled:
                        db.delete(task)
                        removed += 1
                    open_tasks = scheduled

                # Deduplicate exact-title copies on the same scheduled day.
                by_day: dict[str | None, list[Task]] = {}
                for task in open_tasks:
                    by_day.setdefault(task_day(task), []).append(task)

                for day_group in by_day.values():
                    if len(day_group) < 2:
                        continue
                    keep = max(day_group, key=score)
                    for task in day_group:
                        if task.id == keep.id:
                            continue
                        db.delete(task)
                        removed += 1

        db.commit()

    print(f"dedupe complete: removed={removed}")


if __name__ == "__main__":
    main()
