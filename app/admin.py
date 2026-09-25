import hmac
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import ReportAttachment, Task, User, WorkReport, utc_now_iso

settings = get_settings()
router = APIRouter()
security = HTTPBasic(auto_error=False)
HTML_PATH = Path(__file__).parent / "static" / "admin.html"
UPLOAD_ROOT = Path("/data/uploads").resolve()


def require_admin(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin is disabled until ADMIN_PASSWORD is configured.")
    ok = bool(
        credentials
        and hmac.compare_digest(credentials.username, settings.admin_username)
        and hmac.compare_digest(credentials.password, settings.admin_password)
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )


class TaskPatch(BaseModel):
    status: str | None = None
    priority: str | None = None


@router.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
def admin_page() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@router.get("/admin/api/board", dependencies=[Depends(require_admin)])
def board() -> dict:
    with SessionLocal() as db:
        users = db.scalars(select(User).order_by(User.last_seen_at.desc())).all()
        tasks = db.scalars(select(Task).order_by(Task.created_at.desc()).limit(2000)).all()
        reports = db.scalars(select(WorkReport).order_by(WorkReport.created_at.desc()).limit(2000)).all()
        attachments = db.scalars(select(ReportAttachment)).all()

        by_id = {u.id: u for u in users}
        attachments_by_report: dict[int, list[ReportAttachment]] = {}
        for item in attachments:
            attachments_by_report.setdefault(item.report_id, []).append(item)

        task_rows = []
        for task in tasks:
            user = by_id.get(task.user_id)
            task_rows.append(
                {
                    "id": task.id,
                    "title": task.title,
                    "notes": task.notes,
                    "project": task.project,
                    "priority": task.priority,
                    "estimated_minutes": task.estimated_minutes,
                    "status": task.status,
                    "source": task.source,
                    "due_at": task.due_at,
                    "due_source": task.due_source,
                    "scheduled_at": task.scheduled_at,
                    "created_at": task.created_at,
                    "user": _user_payload(user),
                }
            )

        report_rows = []
        for report in reports:
            user = by_id.get(report.user_id)
            files = attachments_by_report.get(report.id, [])
            report_rows.append(
                {
                    "id": report.id,
                    "title": report.title,
                    "summary": report.summary,
                    "project": report.project,
                    "category": report.category,
                    "duration_minutes": report.duration_minutes,
                    "work_date": report.work_date,
                    "source": report.source,
                    "status": report.status,
                    "created_at": report.created_at,
                    "user": _user_payload(user),
                    "attachments": [
                        {
                            "id": item.id,
                            "file_name": item.file_name,
                            "mime_type": item.mime_type,
                            "file_type": item.file_type,
                            "size_bytes": item.size_bytes,
                            "available": bool(item.local_path),
                            "url": f"/admin/files/{item.id}" if item.local_path else None,
                        }
                        for item in files
                    ],
                }
            )

        return {
            "stats": {
                "users": len(users),
                "tasks": len(tasks),
                "reports": len(reports),
                "todo": sum(1 for t in tasks if t.status == "todo"),
                "doing": sum(1 for t in tasks if t.status == "doing"),
                "done": sum(1 for t in tasks if t.status == "done"),
            },
            "tasks": task_rows,
            "reports": report_rows,
        }


def _user_payload(user: User | None) -> dict:
    return {
        "id": user.id if user else None,
        "telegram_id": user.telegram_id if user else None,
        "name": " ".join(
            x for x in [user.first_name if user else None, user.last_name if user else None] if x
        ),
        "username": user.username if user else None,
        "plan": user.plan if user else None,
    }


@router.get("/admin/files/{attachment_id}", dependencies=[Depends(require_admin)])
def admin_file(attachment_id: int):
    with SessionLocal() as db:
        item = db.get(ReportAttachment, attachment_id)
        if not item or not item.local_path:
            raise HTTPException(status_code=404, detail="Attachment not found")
        path = Path(item.local_path).resolve()
        try:
            path.relative_to(UPLOAD_ROOT)
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid attachment path")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Attachment file is missing")
        return FileResponse(
            path,
            media_type=item.mime_type or "application/octet-stream",
            filename=item.file_name or path.name,
        )


@router.patch("/admin/api/tasks/{task_id}", dependencies=[Depends(require_admin)])
def patch_task(task_id: int, patch: TaskPatch) -> dict:
    allowed_status = {"draft", "todo", "doing", "done", "archived"}
    allowed_priority = {"low", "medium", "high", "urgent"}
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if patch.status is not None:
            if patch.status not in allowed_status:
                raise HTTPException(status_code=400, detail="Invalid status")
            task.status = patch.status
        if patch.priority is not None:
            if patch.priority not in allowed_priority:
                raise HTTPException(status_code=400, detail="Invalid priority")
            task.priority = patch.priority
        task.updated_at = utc_now_iso()
        db.commit()
        return {"ok": True, "id": task.id, "status": task.status, "priority": task.priority}
