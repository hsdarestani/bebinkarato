import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from app.config import get_settings

settings = get_settings()


class AIError(RuntimeError):
    pass


class CloudflareAI:
    def __init__(self) -> None:
        self.token = settings.cloudflare_api_token
        self.account_id = settings.cloudflare_account_id
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(75.0, connect=15.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def _resolve_account_id(self) -> str:
        if self.account_id:
            return self.account_id
        if not self.token:
            raise AIError("CLOUDFLARE_API_TOKEN is not configured.")

        response = await self._client.get(
            "https://api.cloudflare.com/client/v4/accounts",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        data = response.json()
        accounts = data.get("result") or []
        if response.is_success and accounts:
            self.account_id = accounts[0]["id"]
            return self.account_id
        raise AIError(
            "Cloudflare account ID could not be detected. Add CLOUDFLARE_ACCOUNT_ID as a GitHub secret."
        )

    async def _run(self, model: str, payload: dict, raw_audio: bytes | None = None) -> dict:
        account_id = await self._resolve_account_id()
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        if raw_audio is not None:
            headers["Content-Type"] = "application/octet-stream"
            response = await self._client.post(url, headers=headers, content=raw_audio)
        else:
            headers["Content-Type"] = "application/json"
            response = await self._client.post(url, headers=headers, json=payload)
        try:
            data = response.json()
        except Exception as exc:
            raise AIError(f"Cloudflare returned HTTP {response.status_code}.") from exc
        if not response.is_success or data.get("success") is False:
            errors = data.get("errors") or []
            msg = errors[0].get("message") if errors and isinstance(errors[0], dict) else str(errors)
            raise AIError(msg or f"Cloudflare returned HTTP {response.status_code}.")
        return data.get("result") or {}

    async def transcribe(self, audio: bytes) -> str:
        result = await self._run(settings.cloudflare_whisper_model, {}, raw_audio=audio)
        text = result.get("text") or result.get("transcription") or result.get("response") or ""
        if not text.strip():
            raise AIError("No transcription was returned.")
        return text.strip()

    async def parse_tasks(self, text: str, timezone_name: str, language_code: str) -> list[dict]:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo(settings.default_timezone)
            timezone_name = settings.default_timezone

        now = datetime.now(tz)
        system = """You are the planning engine for an AI task manager.
Turn a brain dump into actionable tasks. The user can write Persian, English, German, or mixed text.

Rules:
1. Never invent a hard deadline. due_at is only allowed when the user explicitly gave a date, time, or clear relative deadline.
2. AI may suggest scheduled_at to make the plan feasible. A suggested schedule is not a deadline.
3. Preserve dependencies by scheduling prerequisites before dependent work.
4. Keep task titles short and in the user's language.
5. Use ISO 8601 date-times with timezone offsets.
6. If a date is given without a time, choose a reasonable scheduled time but keep due_at at 23:59 local time.
7. reminder_at should normally be before scheduled_at or due_at, never after it.
8. priority must be low, medium, high, or urgent.
9. Return only valid JSON, no markdown.

Schema:
{"tasks":[{"title":"...","notes":"...","project":"...","priority":"medium","estimated_minutes":30,"due_at":null,"due_source":"explicit|none","scheduled_at":null,"reminder_at":null}]}"""

        user = (
            f"Current local datetime: {now.isoformat()}\n"
            f"Timezone: {timezone_name}\n"
            f"Telegram language hint: {language_code}\n"
            f"Brain dump:\n{text}"
        )
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": 1800,
            },
        )
        raw = result.get("response") or result.get("text") or ""
        obj = self._extract_json(raw)
        tasks = obj.get("tasks") if isinstance(obj, dict) else None
        if not isinstance(tasks, list):
            raise AIError("The planning model returned an invalid task list.")
        clean = []
        for item in tasks[:30]:
            if not isinstance(item, dict) or not str(item.get("title", "")).strip():
                continue
            due_source = "explicit" if item.get("due_source") == "explicit" and item.get("due_at") else "none"
            clean.append(
                {
                    "title": str(item.get("title", "")).strip()[:500],
                    "notes": str(item.get("notes") or "").strip(),
                    "project": str(item.get("project") or "").strip()[:200],
                    "priority": item.get("priority") if item.get("priority") in {"low", "medium", "high", "urgent"} else "medium",
                    "estimated_minutes": self._int(item.get("estimated_minutes"), 30, 5, 1440),
                    "due_at": self._normalize_iso(item.get("due_at"), timezone_name) if due_source == "explicit" else None,
                    "due_source": due_source,
                    "scheduled_at": self._normalize_iso(item.get("scheduled_at"), timezone_name),
                    "reminder_at": self._normalize_iso(item.get("reminder_at"), timezone_name),
                }
            )
        return clean

    async def parse_report(self, text: str, timezone_name: str, language_code: str) -> dict:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo(settings.default_timezone)
            timezone_name = settings.default_timezone

        now = datetime.now(tz)
        system = """You structure a work report for a simple work-log app.
The user may write Persian, English, German, or mixed text.

Rules:
1. Summarize only what the user says they did. Never invent accomplishments.
2. title should be short and concrete.
3. project is optional. Leave it empty if unclear.
4. category should be a short general label such as development, meeting, support, design, research, sales, admin, operations, or work.
5. duration_minutes must be null unless the user explicitly states a duration or a start/end time that makes duration directly calculable.
6. work_date should represent the day the work happened. Resolve today/yesterday using the supplied local time. If no day is mentioned, use the current local date.
7. Return valid JSON only.

Schema:
{"report":{"title":"...","summary":"...","project":"","category":"work","duration_minutes":null,"work_date":"YYYY-MM-DD"}}"""

        user = (
            f"Current local datetime: {now.isoformat()}\n"
            f"Timezone: {timezone_name}\n"
            f"Telegram language hint: {language_code}\n"
            f"Work report:\n{text}"
        )
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": 900,
            },
        )
        raw = result.get("response") or result.get("text") or result
        obj = self._extract_json(raw)
        item = obj.get("report") if isinstance(obj, dict) else None
        if not isinstance(item, dict):
            raise AIError("The planning model returned an invalid report.")

        title = str(item.get("title") or text).strip()[:500]
        summary = str(item.get("summary") or text).strip()
        project = str(item.get("project") or "").strip()[:200]
        category = str(item.get("category") or "work").strip()[:100] or "work"
        duration = item.get("duration_minutes")
        try:
            duration = int(duration) if duration is not None else None
            if duration is not None and (duration < 1 or duration > 1440):
                duration = None
        except Exception:
            duration = None

        work_date = str(item.get("work_date") or now.date().isoformat())[:10]
        try:
            datetime.strptime(work_date, "%Y-%m-%d")
        except Exception:
            work_date = now.date().isoformat()

        return {
            "title": title,
            "summary": summary,
            "project": project,
            "category": category,
            "duration_minutes": duration,
            "work_date": work_date,
        }

    @staticmethod
    def _extract_json(raw) -> dict:
        if isinstance(raw, dict):
            return raw
        raw = str(raw or "").strip()
        raw = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*\x60\x60\x60$", "", raw)
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if not match:
                raise AIError("Could not read the model response.")
            try:
                return json.loads(match.group(0))
            except Exception as exc:
                raise AIError("Could not read the model JSON.") from exc

    @staticmethod
    def _int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except Exception:
            return default

    @staticmethod
    def _normalize_iso(value, timezone_name: str) -> str | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return None
