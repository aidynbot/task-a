import logging
import re
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.guardrails import apply_guardrails
from app.providers.base import ExtractionProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.mock_provider import MockProvider
from app.schemas import DocumentOutput

logger = logging.getLogger(__name__)


def preprocess_text(raw_text: str) -> str:
    """Базовая санитария входного текста без искажения смысловых реквизитов."""
    text = raw_text.replace("\x00", "")
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    # Убираем лишние пробелы в конце строк и схлопываем пустые блоки > 2 строк
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _resolve_provider(settings: Settings, force_mock: bool = False) -> ExtractionProvider:
    if force_mock or settings.is_mock_mode:
        return MockProvider()
    return GeminiProvider(settings=settings)


def _build_fallback_output(reason: str) -> DocumentOutput:
    """Безопасный ответ при пустом тексте или исчерпании попыток валидации."""
    return DocumentOutput(
        doc_type="unknown",
        supplier_name=None,
        supplier_bin=None,
        doc_number=None,
        doc_date=None,
        currency=None,
        total_amount=None,
        vat_amount=None,
        line_items=[],
        field_confidence={
            "doc_type": 0.0,
            "supplier_name": 0.0,
            "supplier_bin": 0.0,
            "doc_number": 0.0,
            "doc_date": 0.0,
            "currency": 0.0,
            "total_amount": 0.0,
            "vat_amount": 0.0,
            "line_items": 0.0,
        },
        warnings=[reason],
    )


def process_text(
    raw_text: str,
    use_mock: bool = False,
    settings: Settings | None = None,
) -> DocumentOutput:
    """
    Полный конвейер обработки документа:
    1. Препроцессинг текста.
    2. Вызов провайдера (Gemini или Mock) с self-correction ретраями при ValidationError.
    3. Прогон через детерминированные guardrails и финальная сборка DocumentOutput.
    """
    active_settings = settings or get_settings()
    cleaned_text = preprocess_text(raw_text)

    if not cleaned_text:
        return _build_fallback_output("Передан пустой текст документа")

    provider = _resolve_provider(active_settings, force_mock=use_mock)
    max_attempts = active_settings.max_validation_retries + 1
    error_feedback: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw_extraction = provider.extract(cleaned_text, error_feedback=error_feedback)
            return apply_guardrails(raw_extraction, source_text=cleaned_text)
        except ValidationError as exc:
            last_error = exc
            error_feedback = str(exc)
            logger.warning(
                "Попытка %d/%d не прошла валидацию Pydantic: %s",
                attempt,
                max_attempts,
                error_feedback,
            )
        except Exception as exc:
            last_error = exc
            logger.exception("Ошибка провайдера при извлечении документа: %s", exc)
            break

    return _build_fallback_output(
        f"Не удалось извлечь валидную структуру документа после {max_attempts} попыток: {last_error}"
    )