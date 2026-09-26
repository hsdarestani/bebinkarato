import base64
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
today = می‌پرسد امروز چه کارهایی دارد
upcoming = می‌پرسد کارهای بعدی یا آینده‌اش چیست
reports = گزارش‌های قبلی یا کارهای انجام‌شده‌اش را می‌خواهد ببیند
unknown = هیچ‌کدام روشن نیست

نکته‌های مهم:
- جمله‌هایی مثل «رباته رو ساختم کامل»، «امروز فلان باگ رو حل کردم»، «جلسه رو انجام دادم» حتما report هستند.
- جمله‌هایی مثل «یه ربات باید بسازم»، «فردا باید...» plan هستند.
- فقط JSON معتبر برگردان.
Schema: {"intent":"plan|report|today|upcoming|reports|unknown"}"""
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
        return intent if intent in {"plan", "report", "today", "upcoming", "reports", "unknown"} else "unknown"

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
