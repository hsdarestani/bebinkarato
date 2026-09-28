import ast
import base64
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from app.config import get_settings

settings = get_settings()


class AIError(RuntimeError):
    pass


class CloudflareAI:
    @staticmethod
    def _agent_response_format() -> dict:
        task_fields = {
            "title": {"type": "string"},
            "notes": {"type": "string"},
            "project": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "estimated_minutes": {"type": "integer"},
            "due_at": {"type": ["string", "null"]},
            "due_source": {"type": "string", "enum": ["explicit", "none"]},
            "scheduled_at": {"type": ["string", "null"]},
            "reminder_at": {"type": ["string", "null"]},
        }
        change_fields = dict(task_fields)
        return {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["mutate", "today", "upcoming", "today_reports", "reports", "report", "unknown"],
                    },
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["create", "update", "delete", "complete"],
                                },
                                "task_id": {"type": "integer"},
                                "task": {
                                    "type": "object",
                                    "properties": task_fields,
                                    "required": list(task_fields.keys()),
                                },
                                "changes": {
                                    "type": "object",
                                    "properties": change_fields,
                                },
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["mode", "operations"],
            },
        }


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
        raise AIError("Cloudflare account could not be detected.")

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
        result = await self._run(
            settings.cloudflare_whisper_model,
            {
                "audio": base64.b64encode(audio).decode("ascii"),
                "task": "transcribe",
                "language": "fa",
                "vad_filter": True,
                "initial_prompt": "گفتار فارسی محاوره‌ای درباره کارها، برنامه‌ریزی، پروژه‌ها، سایت، ربات، خرید، باشگاه، جلسه و گزارش کار است.",
                "beam_size": 5,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.55,
                "compression_ratio_threshold": 2.2,
                "log_prob_threshold": -0.8
            },
        )
        text = result.get("text") or result.get("transcription") or result.get("response") or ""
        if not str(text).strip():
            raise AIError("No transcription was returned.")
        return await self.clean_transcript(str(text).strip())

    async def clean_transcript(self, text: str) -> str:
        system = """تو فقط متن تبدیل‌شده از ویس فارسی را تمیز می‌کنی.
قوانین خیلی سخت:
1. فقط غلط‌های واضح املایی، فاصله، نیم‌فاصله و اشتباه‌های خیلی روشن گفتاربه‌متن را اصلاح کن.
2. هیچ کار، اسم، پروژه، زمان، مکان یا جزئیاتی از خودت اضافه نکن.
3. اگر درباره یک کلمه مطمئن نیستی، همان متن خام را نگه دار.
4. معنی و لحن محاوره‌ای کاربر را حفظ کن.
5. فقط JSON معتبر برگردان.
Schema: {"text":"متن تمیزشده"}"""
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "max_tokens": 600,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        cleaned = str(obj.get("text") or text).strip()
        return cleaned or text

    async def classify_intent(self, text: str, timezone_name: str = "Asia/Tehran") -> str:
        now = datetime.now(ZoneInfo(timezone_name))
        system = """تو مسیریاب یک دستیار برنامه‌ریزی فارسی هستی.
فقط یکی از این intentها را برگردان:
plan = کاربر درباره کارهایی که باید انجام بدهد، برنامه آینده، ددلاین، یادآوری یا برنامه‌ریزی حرف می‌زند
report = کاربر درباره کاری که انجام داده یا تمام کرده گزارش می‌دهد
today = می‌پرسد امروز چه کارهایی برای انجام دادن دارد
today_reports = می‌پرسد امروز چه کارهایی انجام داده، امروز چه کار کردیم، امروز چی انجام شد، یا خلاصه عملکرد امروز را می‌خواهد
upcoming = می‌پرسد کارهای بعدی یا آینده‌اش چیست
reports = گزارش‌های قبلی یا کارهای انجام‌شده‌اش را به طور کلی می‌خواهد ببیند
unknown = هیچ‌کدام روشن نیست

نکته‌های مهم:
- «امروز چی دارم؟»، «کارای امروزم چیه؟» => today
- «امروز چیکار کردیم؟»، «امروز چی انجام دادم؟»، «کارای انجام‌شده امروز رو بگو» => today_reports
- جمله‌هایی مثل «رباته رو ساختم کامل»، «امروز فلان باگ رو حل کردم»، «جلسه رو انجام دادم» حتما report هستند.
- جمله‌هایی مثل «یه ربات باید بسازم»، «فردا باید...» plan هستند.
- فقط JSON معتبر برگردان.
Schema: {"intent":"plan|report|today|today_reports|upcoming|reports|unknown"}"""
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"الان: {now.isoformat()}\nپیام کاربر:\n{text}"},
                ],
                "temperature": 0,
                "max_tokens": 80,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        intent = str(obj.get("intent") or "unknown")
        return intent if intent in {"plan", "report", "today", "today_reports", "upcoming", "reports", "unknown"} else "unknown"

    async def planning_agent(
        self,
        text: str,
        tasks: list[dict],
        timezone_name: str = "Asia/Tehran",
        current_draft: dict | None = None,
        focus_task_ids: list[int] | None = None,
    ) -> dict:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)

        system = """تو Agent اصلی برنامه‌ریزی روزانه یک کاربر فارسی‌زبان هستی.
وظیفه‌ات این نیست که فقط تسک استخراج کنی؛ باید دستور طبیعی کاربر را روی برنامه واقعی او بفهمی.

کارهایی که باید پشتیبانی کنی:
- ساخت یک یا چند کار جدید
- تغییر عنوان، توضیح، پروژه، اولویت، تخمین زمان
- گذاشتن یا تغییر زمان پیشنهادی انجام
- گذاشتن یا تغییر ددلاین فقط وقتی خود کاربر صریح گفته
- جابه‌جا کردن کارها به امروز، فردا، روز دیگر یا ساعت دیگر
- عقب انداختن یا جلو آوردن کارها
- حذف کار
- انجام‌شده زدن کار
- چیدن برنامه امروز یا فردا با توجه به زمان و اولویت
- سبک‌تر کردن یک روز و منتقل کردن بخشی از کارها
- اولویت‌بندی مجدد
- پاسخ به «امروز چی دارم؟»، «کارای بعدیم چیه؟»، «امروز چیکار کردیم؟»، «گزارش‌هامو بده»
- تشخیص گزارش انجام کار و تطبیق آن با Taskهای باز

اصل محصول:
این Agent کمک برنامه‌ریزی روزانه است. اگر دستور کاربر درباره برنامه‌ریزی قابل اجراست، تا حد ممکن آن را به عملیات مشخص روی Taskها تبدیل کن.

mode فقط یکی از این‌ها:
mutate = لازم است برنامه تغییر کند و operations اجرا شوند
today = فقط برنامه امروز را می‌خواهد
upcoming = فقط کارهای باز/آینده را می‌خواهد
today_reports = کارهای انجام‌شده امروز را می‌خواهد
reports = گزارش‌های انجام‌شده قبلی را می‌خواهد
report = یک گزارش کار جدید است که به Taskهای موجود ربط روشنی ندارد
unknown = واقعاً قابل فهم نیست

قوانین خیلی مهم:
1. برای update/delete/complete فقط از task_idهای موجود در current_tasks استفاده کن.
2. وقتی کاربر به یک Task موجود اشاره می‌کند، Task جدید مشابه نساز.
3. اگر گفت «فلان کارو انجام دادم»، operation نوع complete بساز.
4. اگر گفت «فلان رو حذف کن»، delete بساز.
5. اگر گفت «بندازش فردا ساعت ۱۰»، update با scheduled_at بساز.
6. اگر گفت «ددلاینش جمعه‌ست»، due_at را update کن و due_source را explicit بگذار.
7. اگر فقط گفت «فردا انجامش بده» و از ددلاین حرف نزد، due_at نساز؛ فقط scheduled_at.
8. اگر گفت «برنامه فردامو بچین» یا «امروزمو مرتب کن»، می‌توانی چند update برای scheduled_at بدهی. ترتیب باید منطقی باشد و زمان‌ها روی هم نیفتند.
9. اگر زمان دقیق نداری و کاربر فقط یک کار جدید گفته، ساعت دقیق ساختگی لازم نیست؛ scheduled_at می‌تواند null باشد.
10. اگر کاربر گفت روزم سبک‌تر شود، کارهای کم‌اولویت‌تر را جابه‌جا کن و ددلاین صریح را نقض نکن.
11. تغییراتی که کاربر نخواسته را انجام نده.
12. عنوان، notes و project فارسی باشند مگر اسم خاصی که خود کاربر انگلیسی گفته.
13. priority فقط low, medium, high, urgent.
14. تاریخ‌زمان‌ها ISO 8601 با offset باشند.
15. خروجی فقط JSON معتبر باشد.
16. اگر current_draft وجود دارد، پیام جدید کاربر اصلاح همان پیش‌نمایش قبلی است. عملیات فعلی را حفظ کن و فقط چیزی را که کاربر خواسته تغییر بده؛ مگر اینکه صریحاً بخواهد از نو بچینی.
17. در حالت اصلاح Preview، عملیات نهایی کامل را برگردان، نه فقط delta جدید.
18. اگر focus_task_ids داده شده و کاربر با عباراتی مثل «اینا»، «همینا»، «اون کارا»، «اون دوتا/سه‌تا» اشاره می‌کند، منظور دقیقاً همان Taskهاست.
19. در حالت 18 حق نداری از «فردا»، «امروز» یا عبارت اشاره‌ای یک Task جدید بسازی. باید روی همان Taskهای موجود update/delete/complete انجام بدهی، مگر کاربر صریحاً بگوید یک کار جدید اضافه کن.
20. اگر کاربر گفت «اینا باشه برای فردا»، scheduled_at همان Taskها را به فردا منتقل کن و ساعت قبلی هر Task را تا حد ممکن حفظ کن. due_at را تغییر نده مگر کاربر صریحاً از ددلاین حرف زده باشد.
21. اگر کاربر چند کار را پشت سر هم یا خط‌به‌خط فرستاد، حتی اگر نگفت «برنامه‌ریزی کن»، آن را brain dump برنامه روزانه بدان و mode=mutate بده.
22. در brain dump چندکاری، فقط لیست Task نساز؛ برای هر کار estimated_minutes واقع‌بینانه تخمین بزن و scheduled_at منطقی بچین.
23. همه کارها را ۳۰ دقیقه فرض نکن. کار خیلی کوتاه می‌تواند ۱۰ تا ۲۰ دقیقه، خرید/رفت‌وآمد معمولاً ۳۰ تا ۶۰ دقیقه، جلسه معمولاً ۴۵ تا ۶۰ دقیقه و کار عمیق فنی معمولاً ۶۰ تا ۱۲۰ دقیقه یا بیشتر باشد. بر اساس معنای خود کار تخمین بزن.
24. اگر کاربر ساعت صریح گفته، مثل «جلسه ریشه ساعت ۱۵:۳۰»، همان ساعت fixed است. عبارت ساعت را داخل title نگه ندار؛ title فقط «جلسه ریشه» باشد و scheduled_at روی ۱۵:۳۰ تنظیم شود.
25. برای brain dump امروز، برنامه را از بعدِ now بچین، نه از ساعت گذشته. شروع پیشنهادی را به نزدیک‌ترین بازه ۱۵ دقیقه‌ای بعد از now گرد کن.
26. کارهای بدون ساعت را دور قرارهای fixed بچین و overlap نساز. اگر همه کارها واقع‌بینانه تا آخر روز جا نمی‌شوند، کارهای کم‌اولویت‌تر را به فردا منتقل کن؛ روز را غیرواقعی فشرده نکن.
27. اگر کاربر روز دیگری مثل «فردا» را صریح نگفته، brain dump چندکاری را برنامه امروز فرض کن.
28. ترتیب خام پیام کاربر الزاماً اولویت نیست؛ زمان ثابت، فوریت، وابستگی و منطق اجرا را در چیدمان لحاظ کن.

Schema:
{
  "mode":"mutate|today|upcoming|today_reports|reports|report|unknown",
  "operations":[
    {
      "type":"create",
      "task":{
        "title":"...",
        "notes":"",
        "project":"",
        "priority":"medium",
        "estimated_minutes":30,
        "due_at":null,
        "due_source":"explicit|none",
        "scheduled_at":null,
        "reminder_at":null
      }
    },
    {
      "type":"update",
      "task_id":12,
      "changes":{
        "title":"...",
        "notes":"...",
        "project":"...",
        "priority":"high",
        "estimated_minutes":45,
        "due_at":null,
        "due_source":"none",
        "scheduled_at":"...",
        "reminder_at":null
      }
    },
    {"type":"delete","task_id":13},
    {"type":"complete","task_id":14}
  ]
}"""

        task_payload = []
        valid_ids = set()
        for task in tasks[:100]:
            try:
                task_id = int(task.get("id"))
            except Exception:
                continue
            valid_ids.add(task_id)
            task_payload.append({
                "id": task_id,
                "title": str(task.get("title") or ""),
                "notes": str(task.get("notes") or ""),
                "project": str(task.get("project") or ""),
                "priority": str(task.get("priority") or "medium"),
                "estimated_minutes": task.get("estimated_minutes"),
                "status": str(task.get("status") or ""),
                "scheduled_at": task.get("scheduled_at"),
                "due_at": task.get("due_at"),
                "due_source": str(task.get("due_source") or "none"),
                "original_text": str(task.get("original_text") or ""),
            })

        user_payload = {
            "now": now.isoformat(),
            "timezone": timezone_name,
            "user_message": text,
            "current_tasks": task_payload,
            "focus_task_ids": [int(x) for x in (focus_task_ids or []) if int(x) in valid_ids],
        }
        if current_draft:
            user_payload["current_draft"] = current_draft

        result = await self._run(
            settings.cloudflare_agent_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "response_format": self._agent_response_format(),
                "temperature": 0,
                "max_tokens": 2200,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        mode = str(obj.get("mode") or "unknown")
        allowed_modes = {"mutate", "today", "upcoming", "today_reports", "reports", "report", "unknown"}
        if mode not in allowed_modes:
            mode = "unknown"

        operations = []
        for op in (obj.get("operations") or [])[:40]:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "")
            if op_type == "create":
                task = op.get("task")
                if not isinstance(task, dict):
                    continue
                cleaned = self._clean_tasks([task], timezone_name)
                if cleaned:
                    operations.append({"type": "create", "task": cleaned[0]})
                continue

            if op_type in {"update", "delete", "complete"}:
                try:
                    task_id = int(op.get("task_id"))
                except Exception:
                    continue
                if task_id not in valid_ids:
                    continue

                if op_type == "update":
                    raw_changes = op.get("changes")
                    if not isinstance(raw_changes, dict):
                        continue
                    changes = {}
                    if "title" in raw_changes and str(raw_changes.get("title") or "").strip():
                        changes["title"] = str(raw_changes["title"]).strip()[:500]
                    if "notes" in raw_changes:
                        changes["notes"] = str(raw_changes.get("notes") or "").strip()
                    if "project" in raw_changes:
                        changes["project"] = str(raw_changes.get("project") or "").strip()[:200]
                    if raw_changes.get("priority") in {"low", "medium", "high", "urgent"}:
                        changes["priority"] = raw_changes["priority"]
                    if "estimated_minutes" in raw_changes:
                        changes["estimated_minutes"] = self._int(raw_changes.get("estimated_minutes"), 30, 5, 1440)
                    if "scheduled_at" in raw_changes:
                        changes["scheduled_at"] = self._normalize_iso(raw_changes.get("scheduled_at"), timezone_name)
                    if "reminder_at" in raw_changes:
                        changes["reminder_at"] = self._normalize_iso(raw_changes.get("reminder_at"), timezone_name)
                    if "due_at" in raw_changes:
                        due_source = "explicit" if raw_changes.get("due_source") == "explicit" and raw_changes.get("due_at") else "none"
                        changes["due_at"] = self._normalize_iso(raw_changes.get("due_at"), timezone_name) if due_source == "explicit" else None
                        changes["due_source"] = due_source
                    if changes:
                        operations.append({"type": "update", "task_id": task_id, "changes": changes})
                else:
                    operations.append({"type": op_type, "task_id": task_id})

        if mode == "mutate" and not operations:
            mode = "unknown"

        if mode == "mutate":
            operations = self._deconflict_agent_schedule(operations)
            operations = self._prepare_daily_brain_dump(
                operations,
                text,
                timezone_name,
            )

        return {"mode": mode, "operations": operations}

    @staticmethod
    def _fa_norm(value: str) -> str:
        return (
            str(value or "")
            .replace("ي", "ی")
            .replace("ك", "ک")
            .replace("‌", " ")
            .translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
            .lower()
        )

    @classmethod
    def _parse_clock_from_text(cls, value: str) -> tuple[int, int] | None:
        text = cls._fa_norm(value)
        match = re.search(r"(?:ساعت\s*)?(\d{1,2})(?::(\d{1,2}))?\s*(صبح|ظهر|عصر|شب)?", text)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        if hour > 23 or minute > 59:
            return None
        part = match.group(3) or ""
        if part in {"عصر", "شب"} and 1 <= hour < 12:
            hour += 12
        elif part == "ظهر" and 1 <= hour < 12:
            hour += 12
        elif part == "صبح" and hour == 12:
            hour = 0
        return hour, minute

    @classmethod
    def _fallback_revise_draft_times(
        cls,
        instruction: str,
        current_draft: dict,
        timezone_name: str,
    ) -> dict | None:
        operations = json.loads(json.dumps(current_draft.get("operations") or []))
        if not operations:
            return None

        normalized = cls._fa_norm(instruction)
        if "ساعت" not in normalized and not any(x in normalized for x in ["صبح", "ظهر", "عصر", "شب"]):
            return None

        clauses = [
            part.strip()
            for part in re.split(r"[\n،,;؛]+", instruction)
            if part.strip()
        ]
        timed_clauses = [(clause, cls._parse_clock_from_text(clause)) for clause in clauses]
        timed_clauses = [(clause, clock) for clause, clock in timed_clauses if clock is not None]
        if not timed_clauses:
            return None

        stop = {
            "ساعت","رو","را","بذار","بزار","قرار","بده","کن","هم","در","به","و",
            "صبح","ظهر","عصر","شب","امروز","فردا","پس","اولی","دومی","سومی","چهارمی",
        }
        ordinal_map = {
            "اولی": 0, "اول": 0,
            "دومی": 1, "دوم": 1,
            "سومی": 2, "سوم": 2,
            "چهارمی": 3, "چهارم": 3,
            "پنجمی": 4, "پنجم": 4,
            "ششمی": 5, "ششم": 5,
        }

        def tokens(value: str) -> set[str]:
            cleaned = re.sub(r"[^\w\sآ-ی]", " ", cls._fa_norm(value))
            return {
                t for t in cleaned.split()
                if len(t) >= 2 and t not in stop and not t.isdigit()
            }

        op_titles = []
        for op in operations:
            if op.get("type") == "create":
                title = str((op.get("task") or {}).get("title") or "")
            else:
                title = ""
            op_titles.append((title, tokens(title)))

        used = set()
        changed = 0
        for seq, (clause, clock) in enumerate(timed_clauses):
            clause_n = cls._fa_norm(clause)
            target = None

            for word, idx in ordinal_map.items():
                if word in clause_n and idx < len(operations):
                    target = idx
                    break

            if target is None:
                ct = tokens(clause)
                best_score = 0
                for idx, (title, tt) in enumerate(op_titles):
                    if idx in used or not title:
                        continue
                    score = len(ct & tt)
                    if score > best_score:
                        best_score = score
                        target = idx
                if best_score == 0:
                    target = None

            if target is None and len(timed_clauses) == len(operations) and seq < len(operations):
                target = seq

            if target is None or target >= len(operations):
                continue

            op = operations[target]
            if op.get("type") != "create":
                continue
            task = op.get("task") or {}
            current_raw = task.get("scheduled_at")
            try:
                if current_raw:
                    base = datetime.fromisoformat(str(current_raw).replace("Z", "+00:00")).astimezone(ZoneInfo(timezone_name))
                else:
                    base = datetime.now(ZoneInfo(timezone_name))
            except Exception:
                base = datetime.now(ZoneInfo(timezone_name))

            clause_norm = cls._fa_norm(clause)
            whole_norm = cls._fa_norm(instruction)
            relative = clause_norm if any(x in clause_norm for x in ["امروز", "فردا", "پس فردا"]) else whole_norm
            if "پس فردا" in relative:
                base = base + timedelta(days=2)
            elif "فردا" in relative:
                base = base + timedelta(days=1)
            elif "امروز" in relative:
                base = datetime.now(ZoneInfo(timezone_name))

            hour, minute = clock
            scheduled = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
            task["scheduled_at"] = scheduled.astimezone(timezone.utc).isoformat()
            task["reminder_at"] = None
            op["task"] = task
            used.add(target)
            changed += 1

        if not changed:
            return None
        return {"mode": "mutate", "operations": operations}

    @staticmethod
    def _ceil_quarter(dt: datetime) -> datetime:
        dt = dt.replace(second=0, microsecond=0)
        remainder = dt.minute % 15
        if remainder:
            dt += timedelta(minutes=15 - remainder)
        return dt

    @staticmethod
    def _strip_clock_from_title(title: str) -> str:
        value = str(title or "").strip()
        fa_to_en = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        normalized = value.translate(fa_to_en)
        patterns = [
            r"\s*(?:،|,|-)?\s*ساعت\s*\d{1,2}(?::\d{1,2})?\s*(?:صبح|ظهر|عصر|شب)?\s*$",
            r"\s*(?:،|,|-)?\s*\d{1,2}:\d{2}\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.I)
            if match:
                cut = match.start()
                return value[:cut].strip(" ،,-")
        return value

    @classmethod
    def _clock_in_text(cls, value: str) -> tuple[int, int] | None:
        text = str(value or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        match = re.search(
            r"(?:ساعت\s*)?(\d{1,2})(?::(\d{1,2}))\s*(صبح|ظهر|عصر|شب)?",
            text,
            flags=re.I,
        )
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        part = match.group(3) or ""
        if hour > 23 or minute > 59:
            return None
        if part in {"عصر", "شب"} and 1 <= hour < 12:
            hour += 12
        elif part == "ظهر" and 1 <= hour < 12:
            hour += 12
        elif part == "صبح" and hour == 12:
            hour = 0
        return hour, minute

    @classmethod
    def _prepare_daily_brain_dump(
        cls,
        operations: list[dict],
        source_text: str,
        timezone_name: str,
    ) -> list[dict]:
        creates = [op for op in operations if op.get("type") == "create" and isinstance(op.get("task"), dict)]
        if len(creates) < 2:
            return operations

        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
        now = datetime.now(tz)
        source_norm = str(source_text or "").replace("ي", "ی").replace("ك", "ک")
        if "پس فردا" in source_norm:
            base_date = (now + timedelta(days=2)).date()
        elif "فردا" in source_norm:
            base_date = (now + timedelta(days=1)).date()
        else:
            base_date = now.date()

        # First clean titles and recover explicit clocks if model left them embedded in titles.
        fixed = []
        unscheduled = []
        for op in creates:
            task = op["task"]
            raw_title = str(task.get("title") or "")
            clock = cls._clock_in_text(raw_title)
            if clock and not task.get("scheduled_at"):
                hour, minute = clock
                target = datetime.combine(base_date, datetime.min.time(), tzinfo=tz).replace(hour=hour, minute=minute)
                if base_date == now.date() and target < now:
                    # Explicit past time stays semantically fixed; don't silently move it.
                    pass
                task["scheduled_at"] = target.astimezone(timezone.utc).isoformat()
            task["title"] = cls._strip_clock_from_title(raw_title)

            if task.get("scheduled_at"):
                try:
                    start = datetime.fromisoformat(str(task["scheduled_at"]).replace("Z", "+00:00")).astimezone(tz)
                    fixed.append((start, op))
                    continue
                except Exception:
                    task["scheduled_at"] = None
            unscheduled.append(op)

        fixed.sort(key=lambda x: x[0])

        # If model already scheduled every item with distinct sensible starts, preserve it.
        if not unscheduled:
            starts = []
            for _, op in fixed:
                try:
                    starts.append(datetime.fromisoformat(op["task"]["scheduled_at"].replace("Z", "+00:00")).astimezone(tz))
                except Exception:
                    pass
            if len({x.replace(second=0, microsecond=0) for x in starts}) == len(starts):
                return operations

        cursor = datetime.combine(base_date, datetime.min.time(), tzinfo=tz).replace(hour=9)
        if base_date == now.date():
            cursor = cls._ceil_quarter(now + timedelta(minutes=5))

        # Treat fixed items as blocked intervals.
        blocked = []
        for start, op in fixed:
            duration = max(10, int((op.get("task") or {}).get("estimated_minutes") or 30))
            blocked.append((start, start + timedelta(minutes=duration)))
        blocked.sort(key=lambda x: x[0])

        for op in unscheduled:
            task = op["task"]
            duration = max(10, int(task.get("estimated_minutes") or 30))
            while True:
                collision = None
                end = cursor + timedelta(minutes=duration)
                for b_start, b_end in blocked:
                    if cursor < b_end and end > b_start:
                        collision = (b_start, b_end)
                        break
                if collision:
                    cursor = cls._ceil_quarter(collision[1])
                    continue
                break

            # Don't create an absurdly packed day. Carry overflow to tomorrow morning.
            if cursor.hour >= 23 or (cursor + timedelta(minutes=duration)).date() > cursor.date():
                base_date = cursor.date() + timedelta(days=1)
                cursor = datetime.combine(base_date, datetime.min.time(), tzinfo=tz).replace(hour=9)

            task["scheduled_at"] = cursor.astimezone(timezone.utc).isoformat()
            cursor = cls._ceil_quarter(cursor + timedelta(minutes=duration) + timedelta(minutes=10))

        return operations

    async def revise_planning_draft(
        self,
        instruction: str,
        current_draft: dict,
        tasks: list[dict],
        timezone_name: str = "Asia/Tehran",
    ) -> dict:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)

        valid_ids = set()
        task_payload = []
        for task in tasks[:100]:
            try:
                task_id = int(task.get("id"))
            except Exception:
                continue
            valid_ids.add(task_id)
            task_payload.append({
                "id": task_id,
                "title": str(task.get("title") or ""),
                "notes": str(task.get("notes") or ""),
                "project": str(task.get("project") or ""),
                "priority": str(task.get("priority") or "medium"),
                "estimated_minutes": task.get("estimated_minutes"),
                "status": str(task.get("status") or ""),
                "scheduled_at": task.get("scheduled_at"),
                "due_at": task.get("due_at"),
                "due_source": str(task.get("due_source") or "none"),
            })

        system = """تو فقط ویرایشگر Preview یک Agent برنامه‌ریزی روزانه فارسی هستی.
یک Draft فعلی داری که هنوز روی دیتابیس اعمال نشده و کاربر حالا می‌خواهد همان Preview را اصلاح کند.

قواعد حیاتی:
1. خروجی باید کل Draft نهایی را برگرداند، نه فقط تغییر جدید.
2. ترتیب operations را حفظ کن مگر کاربر صریحاً ترتیب را عوض کرده باشد.
3. operationهای create هنوز Task واقعی نیستند. هرگز create را برای ویرایش به update با task_id خیالی تبدیل نکن.
4. اگر کاربر می‌گوید «اولی»، «دومی»، «سومی»، «چهارمی» منظور ترتیب operationهای قابل مشاهده در Preview است.
5. اگر کاربر اسم کار را می‌گوید، معنایی همان آیتم را پیدا کن.
6. برای create فقط فیلدهای داخل task را عوض کن.
7. برای update/delete/complete مربوط به Task واقعی، task_id موجود را حفظ کن.
8. تغییراتی که کاربر نگفته دست‌نخورده بمانند.
9. «ساعت ۸ شب» یعنی 20:00، «۵ عصر» یعنی 17:00، «۱۰ شب» یعنی 22:00، «۱۱ شب» یعنی 23:00.
10. تاریخ نسبی را با now و timezone ایران حل کن.
11. due_at فقط اگر کاربر صریحاً درباره ددلاین/مهلت حرف زده تغییر کند. تغییر ساعت انجام فقط scheduled_at است.
12. اگر چند زمان جدید داده، همه را در یک پاسخ اعمال کن.
13. هیچ operation جدیدی از خودت اختراع نکن مگر کاربر صریحاً کار جدید اضافه کرده باشد.
14. فقط JSON معتبر برگردان.

Schema:
{
  "mode":"mutate",
  "operations":[
    {"type":"create","task":{"title":"...","notes":"","project":"","priority":"medium","estimated_minutes":30,"due_at":null,"due_source":"none","scheduled_at":null,"reminder_at":null}},
    {"type":"update","task_id":12,"changes":{"scheduled_at":"..."}},
    {"type":"delete","task_id":13},
    {"type":"complete","task_id":14}
  ]
}"""

        result = await self._run(
            settings.cloudflare_agent_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "now": now.isoformat(),
                        "timezone": timezone_name,
                        "current_draft": current_draft,
                        "current_tasks": task_payload,
                        "edit_request": instruction,
                    }, ensure_ascii=False)},
                ],
                "response_format": self._agent_response_format(),
                "temperature": 0,
                "max_tokens": 2200,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        operations = []

        for op in (obj.get("operations") or [])[:40]:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "")

            if op_type == "create":
                item = op.get("task")
                if not isinstance(item, dict):
                    continue
                cleaned = self._clean_tasks([item], timezone_name)
                if cleaned:
                    operations.append({"type": "create", "task": cleaned[0]})
                continue

            if op_type in {"update", "delete", "complete"}:
                try:
                    task_id = int(op.get("task_id"))
                except Exception:
                    continue
                if task_id not in valid_ids:
                    continue

                if op_type == "update":
                    raw = op.get("changes")
                    if not isinstance(raw, dict):
                        continue
                    changes = {}
                    if "title" in raw and str(raw.get("title") or "").strip():
                        changes["title"] = str(raw["title"]).strip()[:500]
                    if "notes" in raw:
                        changes["notes"] = str(raw.get("notes") or "").strip()
                    if "project" in raw:
                        changes["project"] = str(raw.get("project") or "").strip()[:200]
                    if raw.get("priority") in {"low", "medium", "high", "urgent"}:
                        changes["priority"] = raw["priority"]
                    if "estimated_minutes" in raw:
                        changes["estimated_minutes"] = self._int(raw.get("estimated_minutes"), 30, 5, 1440)
                    if "scheduled_at" in raw:
                        changes["scheduled_at"] = self._normalize_iso(raw.get("scheduled_at"), timezone_name)
                    if "reminder_at" in raw:
                        changes["reminder_at"] = self._normalize_iso(raw.get("reminder_at"), timezone_name)
                    if "due_at" in raw:
                        due_source = "explicit" if raw.get("due_source") == "explicit" and raw.get("due_at") else "none"
                        changes["due_at"] = self._normalize_iso(raw.get("due_at"), timezone_name) if due_source == "explicit" else None
                        changes["due_source"] = due_source
                    if changes:
                        operations.append({"type": "update", "task_id": task_id, "changes": changes})
                else:
                    operations.append({"type": op_type, "task_id": task_id})

        if not operations:
            fallback = self._fallback_revise_draft_times(
                instruction,
                current_draft,
                timezone_name,
            )
            if fallback:
                return fallback
            raise AIError("Draft revision returned no valid operations.")

        return {"mode": "mutate", "operations": operations}

    @staticmethod
    def _deconflict_agent_schedule(operations: list[dict]) -> list[dict]:
        # اگر مدل چند کار را دقیقاً روی یک لحظه چیده، آن‌ها را پشت‌سرهم قرار بده.
        # این فقط برای collisionهای واضح است و زمان‌های متفاوت کاربر را دست نمی‌زند.
        last_end_by_start: dict[str, datetime] = {}
        seen_starts: dict[str, datetime] = {}

        for op in operations:
            if op.get("type") != "create":
                continue
            task = op.get("task") or {}
            raw = task.get("scheduled_at")
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except Exception:
                continue

            key = dt.replace(second=0, microsecond=0).isoformat()
            if key not in seen_starts:
                seen_starts[key] = dt
                duration = max(5, int(task.get("estimated_minutes") or 30))
                last_end_by_start[key] = dt + timedelta(minutes=duration)
                continue

            new_start = last_end_by_start[key]
            task["scheduled_at"] = new_start.isoformat()
            duration = max(5, int(task.get("estimated_minutes") or 30))
            last_end_by_start[key] = new_start + timedelta(minutes=duration)

        return operations

    async def parse_tasks(self, text: str, timezone_name: str, language_code: str = "fa") -> list[dict]:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)
        system = """تو مغز برنامه‌ریز یک دستیار کاملاً فارسی هستی.
متن کاربر را به کارهای واقعی و اجرایی تبدیل کن.

قوانین:
1. تمام title، notes و project باید فارسی باشند. هیچ واژه انگلیسی تولید نکن مگر اسم خاصی که خود کاربر دقیقاً انگلیسی گفته باشد.
2. از یک جمله مبهم چند کار ساختگی نساز. فقط چیزهایی را بساز که واقعاً از حرف کاربر درمی‌آید.
3. اگر کاربر فقط گفته «یه ربات باید بسازم فیچراشو درارم و اینا»، نهایتاً 1 یا 2 کار معنادار بساز، نه چهار کار خیالی.
4. due_at فقط وقتی مجاز است که خود کاربر تاریخ یا زمان یا مهلت مشخص گفته باشد.
5. scheduled_at پیشنهاد توست و باید منطقی باشد؛ اگر زمان مشخصی از کاربر نداری، لازم نیست ساعت دقیق الکی بسازی و می‌تواند null باشد.
6. priority فقط low, medium, high, urgent.
7. خروجی فقط JSON معتبر.
Schema:
{"tasks":[{"title":"فارسی","notes":"فارسی","project":"فارسی یا خالی","priority":"medium","estimated_minutes":30,"due_at":null,"due_source":"explicit|none","scheduled_at":null,"reminder_at":null}]}"""
        user = f"الان به وقت ایران: {now.isoformat()}\nمتن کاربر:\n{text}"
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.05,
                "max_tokens": 1600,
            },
        )
        raw = result.get("response") or result.get("text") or result
        obj = self._extract_json(raw)
        tasks = obj.get("tasks") if isinstance(obj, dict) else None
        if not isinstance(tasks, list):
            raise AIError("Invalid task list.")
        return self._clean_tasks(tasks, timezone_name)

    async def revise_tasks(
        self,
        tasks: list[dict],
        instruction: str,
        timezone_name: str,
        original_text: str = "",
    ) -> list[dict]:
        now = datetime.now(ZoneInfo(timezone_name))
        system = """تو مغز ویرایش یک دستیار برنامه‌ریزی فارسی هستی.
سه چیز داری: متن اصلی کاربر، Draft فعلی که قبلاً از آن ساخته شده، و جمله جدیدی که کاربر برای اصلاح گفته.
باید منظور اصلاحی کاربر را معنایی بفهمی، نه اینکه فقط دنبال تطابق کلمه‌به‌کلمه بگردی.

مثال مهم:
Draft فعلی: «باشگاه را بردارید»
کاربر: «پاساژ رو بردارید نه، برم باشگاه»
منظور واقعی: عنوان همان مورد باید بشود «برم باشگاه».
پس ممکن است کاربر عبارتی را اصلاح کند که عیناً در عنوان فعلی وجود ندارد ولی از متن اصلی و context مشخص است کدام مورد را می‌گوید.

مثال:
کاربر: «دومی رو حذف کن» => فقط مورد دوم حذف شود.
کاربر: «اون جلسه با بهنود رو بذار فردا ساعت ۵» => فقط همان مورد و زمانش عوض شود.
کاربر: «نه خرید موز، خرید میوه» => مورد مربوط به خرید موز به خرید میوه اصلاح شود.
کاربر: «همه‌ش خوبه فقط زمان دومی دو ساعت دیرتر» => متن بقیه کارها دست نخورد.

قوانین سخت:
1. همه title، notes و project فارسی باشند؛ مگر اسم خاصی که خود کاربر انگلیسی گفته باشد.
2. فقط تغییر خواسته‌شده را انجام بده. بقیه اطلاعات را تا حد ممکن دقیقاً حفظ کن.
3. کار جدید اختراع نکن مگر خود اصلاح کاربر صریحاً کار تازه‌ای اضافه کرده باشد.
4. اگر کاربر گفت یک مورد حذف شود، همان مورد را حذف کن.
5. تاریخ، ساعت، مدت، اولویت و پروژه‌ای که کاربر درباره‌شان چیزی نگفته را بی‌دلیل تغییر نده.
6. due_at فقط وقتی مجاز است که قبلاً ددلاین صریح وجود داشته یا کاربر در اصلاح جدید ددلاین داده باشد.
7. scheduled_at پیشنهاد برنامه است و می‌تواند null باشد. زمان موجود را بدون دلیل عوض نکن.
8. priority فقط low, medium, high, urgent.
9. تمام کارهای نهایی، حتی موارد تغییرنکرده، باید در خروجی باشند.
10. اگر عبارت اصلاحی عامیانه، ناقص یا دارای اشتباه گفتاری است، با کمک متن اصلی و Draft فعلی منظور را استنباط کن.
11. فقط JSON معتبر برگردان. هیچ توضیح بیرون JSON نده.

Schema دقیق:
{"tasks":[{"title":"فارسی","notes":"فارسی","project":"فارسی یا خالی","priority":"medium","estimated_minutes":30,"due_at":null,"due_source":"explicit|none","scheduled_at":null,"reminder_at":null}]}"""
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "now": now.isoformat(),
                        "timezone": timezone_name,
                        "original_user_text": original_text,
                        "current_tasks": tasks,
                        "edit_request": instruction,
                    }, ensure_ascii=False)},
                ],
                "temperature": 0.05,
                "max_tokens": 1600,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        tasks_out = obj.get("tasks") if isinstance(obj, dict) else None
        if not isinstance(tasks_out, list):
            raise AIError("Invalid revised task list.")
        return self._clean_tasks(tasks_out, timezone_name)

    async def match_completed_tasks(
        self,
        text: str,
        open_tasks: list[dict],
        timezone_name: str,
    ) -> dict:
        if not open_tasks:
            return {"matched_task_ids": [], "unmatched_reports": []}

        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)

        system = """تو مسئول تطبیق گزارش کار فارسی با Taskهای باز کاربر هستی.
کاربر ممکن است خیلی محاوره‌ای بگوید چه کارهایی را انجام داده. باید بررسی کنی آیا هر بخش از حرفش همان یکی از Taskهای باز است یا یک کار جدید و مستقل.

قوانین سخت:
1. تطبیق باید معنایی باشد، نه صرفاً کلمه‌به‌کلمه. مثال: Task «رفتن به باشگاه» با «باشگاهو رفتم» یکی است.
2. فقط وقتی task_id را match کن که از متن کاربر واقعاً معلوم باشد آن کار انجام شده.
3. Taskی که کاربر درباره انجام شدنش حرف نزده را match نکن.
4. اگر بخشی از گزارش با هیچ Task بازی تطبیق ندارد، آن را در unmatched_reports به صورت یک گزارش جدا برگردان.
5. یک گزارش چندکار را به یک عنوان ترکیبی تبدیل نکن. هر کار مستقل باید جدا باشد.
6. تمام title، summary، project و category فارسی باشند.
7. duration_minutes فقط وقتی عدد داشته باشد که کاربر زمان را صریحاً گفته باشد.
8. work_date اگر کاربر تاریخ نگفته، امروز به وقت ایران است.
9. حداکثر 10 Task را match کن و حداکثر 5 گزارش جدید بساز.
10. فقط JSON معتبر برگردان.

Schema:
{
  "matched_task_ids":[1,2],
  "unmatched_reports":[
    {
      "title":"فارسی",
      "summary":"فارسی",
      "project":"",
      "category":"کار",
      "duration_minutes":null,
      "work_date":"YYYY-MM-DD"
    }
  ]
}"""

        payload_tasks = []
        valid_ids = set()
        for task in open_tasks[:30]:
            try:
                task_id = int(task.get("id"))
            except Exception:
                continue
            valid_ids.add(task_id)
            payload_tasks.append({
                "id": task_id,
                "title": str(task.get("title") or ""),
                "notes": str(task.get("notes") or ""),
                "project": str(task.get("project") or ""),
                "original_text": str(task.get("original_text") or ""),
            })

        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "now": now.isoformat(),
                        "timezone": timezone_name,
                        "user_report": text,
                        "open_tasks": payload_tasks,
                    }, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": 1200,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)

        matched = []
        seen = set()
        for raw_id in obj.get("matched_task_ids") or []:
            try:
                task_id = int(raw_id)
            except Exception:
                continue
            if task_id in valid_ids and task_id not in seen:
                matched.append(task_id)
                seen.add(task_id)
            if len(matched) >= 10:
                break

        unmatched = []
        for item in (obj.get("unmatched_reports") or [])[:5]:
            if isinstance(item, dict):
                unmatched.append(self._clean_report(item, text, now))

        return {
            "matched_task_ids": matched,
            "unmatched_reports": unmatched,
        }

    async def refine_completed_task_selection(
        self,
        instruction: str,
        current_task_ids: list[int],
        open_tasks: list[dict],
        timezone_name: str,
    ) -> list[int]:
        if not open_tasks:
            return []

        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)

        valid_ids = set()
        payload_tasks = []
        for task in open_tasks[:30]:
            try:
                task_id = int(task.get("id"))
            except Exception:
                continue
            valid_ids.add(task_id)
            payload_tasks.append({
                "id": task_id,
                "title": str(task.get("title") or ""),
                "notes": str(task.get("notes") or ""),
                "project": str(task.get("project") or ""),
                "original_text": str(task.get("original_text") or ""),
                "currently_selected": task_id in current_task_ids,
            })

        system = """تو ویرایشگر مجموعه کارهای انجام‌شده هستی.
کاربر قبلاً یک پیش‌نمایش از Taskهایی که احتمالاً انجام داده دیده و حالا با زبان محاوره‌ای دارد همان مجموعه را اصلاح می‌کند.

باید در خروجی، کل مجموعه نهایی Taskهای انجام‌شده را برگردانی، نه فقط تغییر جدید را.

مثال‌ها:
- انتخاب فعلی: [خرید موز، جلسه با بهنود]
  کاربر: «درسته، باشگاه هم رفتم»
  خروجی نهایی باید هر سه مورد را شامل شود.
- انتخاب فعلی: [باشگاه، خرید موز، جلسه]
  کاربر: «موز رو انجام ندادم»
  خروجی نهایی فقط باشگاه و جلسه است.
- کاربر: «فقط باشگاه و جلسه»
  خروجی نهایی دقیقاً همان دو مورد است.
- کاربر: «همه‌ش درسته»
  انتخاب فعلی بدون تغییر بماند.
- کاربر: «جلسه هم انجام شد»
  جلسه به انتخاب فعلی اضافه شود.

قوانین:
1. معنایی بفهم، نه صرفاً تطابق کلمه.
2. فقط IDهایی را برگردان که در open_tasks هستند.
3. اگر کاربر گفت «هم»، معمولاً یعنی انتخاب فعلی را نگه دار و مورد جدید را اضافه کن.
4. اگر گفت «نه»، «انجام ندادم»، «بردار»، مورد مربوط را حذف کن و بقیه را نگه دار.
5. اگر گفت «فقط»، انتخاب قبلی را با مواردی که گفته جایگزین کن.
6. اگر حرفش صرفاً تأیید بود، انتخاب فعلی را همان‌طور نگه دار.
7. چیزی را از خودت اضافه نکن.
8. فقط JSON معتبر برگردان.

Schema:
{"selected_task_ids":[1,2,3]}"""

        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "now": now.isoformat(),
                        "timezone": timezone_name,
                        "current_selected_task_ids": current_task_ids,
                        "open_tasks": payload_tasks,
                        "user_correction": instruction,
                    }, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        selected = []
        seen = set()
        for raw_id in obj.get("selected_task_ids") or []:
            try:
                task_id = int(raw_id)
            except Exception:
                continue
            if task_id in valid_ids and task_id not in seen:
                selected.append(task_id)
                seen.add(task_id)
        return selected

    async def parse_report(self, text: str, timezone_name: str, language_code: str = "fa") -> dict:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("Asia/Tehran")
            timezone_name = "Asia/Tehran"
        now = datetime.now(tz)
        system = """تو گزارش کار فارسی را ساختاریافته می‌کنی.
فقط چیزی را ثبت کن که کاربر واقعاً گفته انجام داده است.

قوانین:
1. title، summary، project و category باید فارسی باشند. هیچ ترجمه یا عنوان انگلیسی نساز.
2. اگر کاربر گفت «رباته رو ساختم کامل»، عنوان باید چیزی مثل «ساخت کامل ربات» باشد، نه ترجمه یا حدس نامربوط.
3. duration_minutes فقط اگر کاربر زمان را گفته باشد.
4. work_date اگر تاریخ نگفته، امروز به وقت ایران است.
5. فقط JSON معتبر برگردان.
Schema:
{"report":{"title":"فارسی","summary":"فارسی","project":"","category":"کار","duration_minutes":null,"work_date":"YYYY-MM-DD"}}"""
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"الان به وقت ایران: {now.isoformat()}\nگزارش کاربر:\n{text}"},
                ],
                "temperature": 0.05,
                "max_tokens": 700,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        item = obj.get("report") if isinstance(obj, dict) else None
        if not isinstance(item, dict):
            raise AIError("Invalid report.")
        return self._clean_report(item, text, now)

    async def revise_report(self, report: dict, instruction: str, timezone_name: str) -> dict:
        now = datetime.now(ZoneInfo(timezone_name))
        system = """تو ویرایشگر گزارش کار فارسی هستی.
گزارش فعلی و درخواست اصلاح کاربر را می‌گیری و فقط همان اصلاح را انجام می‌دهی.
هیچ چیز ساختگی اضافه نکن. همه متن‌ها فارسی باشند.
فقط JSON معتبر برگردان.
Schema:
{"report":{"title":"فارسی","summary":"فارسی","project":"","category":"کار","duration_minutes":null,"work_date":"YYYY-MM-DD"}}"""
        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "now": now.isoformat(),
                        "timezone": timezone_name,
                        "current_report": report,
                        "edit_request": instruction,
                    }, ensure_ascii=False)},
                ],
                "temperature": 0.05,
                "max_tokens": 700,
            },
        )
        obj = self._extract_json(result.get("response") or result.get("text") or result)
        item = obj.get("report") if isinstance(obj, dict) else None
        if not isinstance(item, dict):
            raise AIError("Invalid revised report.")
        return self._clean_report(item, report.get("summary") or report.get("title") or "", now)

    def _clean_tasks(self, tasks: list[dict], timezone_name: str) -> list[dict]:
        clean = []
        for item in tasks[:20]:
            if not isinstance(item, dict) or not str(item.get("title", "")).strip():
                continue
            due_source = "explicit" if item.get("due_source") == "explicit" and item.get("due_at") else "none"
            clean.append({
                "title": str(item.get("title", "")).strip()[:500],
                "notes": str(item.get("notes") or "").strip(),
                "project": str(item.get("project") or "").strip()[:200],
                "priority": item.get("priority") if item.get("priority") in {"low", "medium", "high", "urgent"} else "medium",
                "estimated_minutes": self._int(item.get("estimated_minutes"), 30, 5, 1440),
                "due_at": self._normalize_iso(item.get("due_at"), timezone_name) if due_source == "explicit" else None,
                "due_source": due_source,
                "scheduled_at": self._normalize_iso(item.get("scheduled_at"), timezone_name),
                "reminder_at": self._normalize_iso(item.get("reminder_at"), timezone_name),
            })
        return clean

    def _clean_report(self, item: dict, fallback_text: str, now: datetime) -> dict:
        title = str(item.get("title") or fallback_text or "گزارش کار").strip()[:500]
        summary = str(item.get("summary") or fallback_text or "").strip()
        project = str(item.get("project") or "").strip()[:200]
        category = str(item.get("category") or "کار").strip()[:100] or "کار"
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
        raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        raw = re.sub(r"^\x60\x60\x60(?:json|python)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*\x60\x60\x60$", "", raw)

        candidates = [raw]
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match and match.group(0) != raw:
            candidates.append(match.group(0))

        last_error = None
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception as exc:
                last_error = exc

            # Some Workers AI models occasionally return Python-style dicts
            # with single quotes despite an explicit JSON instruction.
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception as exc:
                last_error = exc

            # Repair the most common mixed JSON/Python form:
            # single-quoted keys/strings plus JSON null/true/false.
            try:
                pythonish = re.sub(r"\bnull\b", "None", candidate, flags=re.I)
                pythonish = re.sub(r"\btrue\b", "True", pythonish, flags=re.I)
                pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.I)
                obj = ast.literal_eval(pythonish)
                if isinstance(obj, dict):
                    return obj
            except Exception as exc:
                last_error = exc

        if not match:
            raise AIError("Could not read the model response.")
        raise AIError("Could not read the model JSON.") from last_error

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
