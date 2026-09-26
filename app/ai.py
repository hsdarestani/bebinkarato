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
        }
        if current_draft:
            user_payload["current_draft"] = current_draft

        result = await self._run(
            settings.cloudflare_llm_model,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
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

        return {"mode": mode, "operations": operations}

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
