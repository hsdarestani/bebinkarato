from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str = ""
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""
    cloudflare_llm_model: str = "@cf/meta/llama-3.1-8b-instruct-fast"
    cloudflare_whisper_model: str = "@cf/openai/whisper"

    database_url: str = "sqlite:////data/bebinkarato.db"
    default_timezone: str = "Europe/Berlin"

    admin_username: str = "admin"
    admin_password: str = ""

    free_ai_requests: int = 60
    free_voice_minutes: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
