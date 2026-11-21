from typing import Protocol, runtime_checkable
from app.schemas import LLMExtractionResult


@runtime_checkable
class ExtractionProvider(Protocol):
    """Единый контракт для провайдеров извлечения данных из текста документа."""

    def extract(self, text: str, error_feedback: str | None = None) -> LLMExtractionResult:
        """
        Извлекает структурированные поля и цитаты (evidence) из сырого текста документа.

        Args:
            text: Предобработанный текст документа.
            error_feedback: Текст ошибки валидации Pydantic с предыдущей попытки (для self-correction).
        """
        ...