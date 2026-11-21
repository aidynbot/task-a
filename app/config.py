from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Document Extractor LLM"
    app_version: str = "1.0.0"

    # Настройки Gemini API
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout: int = Field(default=30, ge=1)

    # Настройки надёжности и режима работы
    use_mock: bool = Field(default=False, alias="USE_MOCK")
    max_validation_retries: int = Field(default=2, ge=0, le=5)

    @property
    def is_mock_mode(self) -> bool:
        """Автоматически включает мок-режим, если флаг USE_MOCK=true или не задан API-ключ."""
        return self.use_mock or not bool(self.gemini_api_key and self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()