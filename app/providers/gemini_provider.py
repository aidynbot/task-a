import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import LLMExtractionResult

logger = logging.getLogger(__name__)


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Повторяем запрос только при временных сбоях сервера (5xx) или rate limit (429)."""
    if isinstance(exc, APIError):
        return exc.code in (429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, ConnectionError))


class GeminiProvider:
    """Провайдер извлечения данных через Gemini API со Structured Outputs."""

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY не задан для инициализации GeminiProvider")
        self._client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(api_version="v1beta"))
        self._model = settings.gemini_model
        self._temperature = settings.temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _call_api(self, user_prompt: str):
        return self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=LLMExtractionResult,
                temperature=self._temperature,
            ),
        )

    def extract(self, text: str, error_feedback: str | None = None) -> LLMExtractionResult:
        user_prompt = build_user_prompt(text, error_feedback=error_feedback)
        response = self._call_api(user_prompt)

        # SDK автоматически парсит ответ в Pydantic-модель через response.parsed
        if isinstance(response.parsed, LLMExtractionResult):
            return response.parsed

        # Запасной парсинг сырого текста ответа, если parsed не заполнился
        if response.text:
            return LLMExtractionResult.model_validate_json(response.text)

        raise ValidationError.from_exception_data(
            title="LLMExtractionResult",
            line_errors=[],
        )
