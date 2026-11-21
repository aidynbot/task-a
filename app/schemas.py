from datetime import date
import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


DocType = Literal["invoice", "commercial_offer", "act", "unknown"]
CurrencyType = Literal["KZT", "USD", "EUR"]


class LineItem(BaseModel):
    name: str = Field(..., min_length=1, description="Наименование товара или услуги")
    qty: float | None = Field(default=None, description="Количество, null если не указано")
    price: float | None = Field(default=None, description="Цена за единицу, null если не указана")


class FieldEvidence(BaseModel):
    """Точные цитаты из текста документа, подтверждающие извлечённые значения."""
    supplier_name: str | None = Field(default=None, description="Дословная подстрока с названием поставщика")
    supplier_bin: str | None = Field(default=None, description="Дословная подстрока с БИН/ИИН из текста")
    doc_number: str | None = Field(default=None, description="Дословная подстрока с номером документа")
    doc_date: str | None = Field(default=None, description="Дословная подстрока с датой документа")
    total_amount: str | None = Field(default=None, description="Дословная подстрока с итоговой суммой")
    vat_amount: str | None = Field(default=None, description="Дословная подстрока с суммой НДС")


class ConfidenceScores(BaseModel):
    """Оценки уверенности модели от 0.0 до 1.0 по полям (без произвольного dict для совместимости с Gemini)."""
    doc_type: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_name: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_bin: float = Field(default=0.0, ge=0.0, le=1.0)
    doc_number: float = Field(default=0.0, ge=0.0, le=1.0)
    doc_date: float = Field(default=0.0, ge=0.0, le=1.0)
    currency: float = Field(default=0.0, ge=0.0, le=1.0)
    total_amount: float = Field(default=0.0, ge=0.0, le=1.0)
    vat_amount: float = Field(default=0.0, ge=0.0, le=1.0)
    line_items: float = Field(default=0.0, ge=0.0, le=1.0)


# --- Внутренняя модель (передаётся в Gemini response_schema) ---

class LLMExtractionResult(BaseModel):
    """
    Схема для Gemini API.
    Поле evidence идёт ПЕРВЫМ, чтобы заставить модель сначала найти цитату
    в исходном тексте до генерации значений (anti-hallucination CoT).
    """
    evidence: FieldEvidence = Field(
        ...,
        description="Сначала выпиши дословные цитаты из текста. Если фрагмента нет — ставь null."
    )
    doc_type: DocType = Field(
        ...,
        description="Тип документа строго по заголовку/смыслу: invoice, commercial_offer, act или unknown"
    )
    supplier_name: str | None = None
    supplier_bin: str | None = Field(
        default=None,
        description="Ровно 12 цифр БИН/ИИН, если он есть в тексте. Не выдумывать."
    )
    doc_number: str | None = None
    doc_date: str | None = Field(
        default=None,
        description="Дата составления в формате YYYY-MM-DD, иначе null"
    )
    currency: CurrencyType | None = None
    total_amount: float | None = Field(
        default=None,
        description="Итоговая сумма документа, если явно указана. Не суммировать позиции самостоятельно."
    )
    vat_amount: float | None = Field(
        default=None,
        description="Сумма НДС, если явно указана в цифрах. Если 'без НДС' или не указан — null."
    )
    line_items: list[LineItem] = Field(default_factory=list)
    field_confidence: ConfidenceScores = Field(
        default_factory=ConfidenceScores,
        description="Оценка уверенности от 0.0 до 1.0 для каждого извлечённого поля"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Противоречия, дубли дат/сумм, OCR-артефакты и нестыковки"
    )


# --- Публичная модель (строгий контракт ответа сервиса) ---

def validate_kz_bin_checksum(bin_str: str) -> bool:
    """Проверка 12-го контрольного разряда БИН/ИИН РК."""
    if not re.fullmatch(r"\d{12}", bin_str):
        return False
    digits = [int(ch) for ch in bin_str]
    w1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    checksum = sum(d * w for d, w in zip(digits[:11], w1)) % 11
    if checksum == 10:
        w2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        checksum = sum(d * w for d, w in zip(digits[:11], w2)) % 11
    return checksum == digits[11]


class DocumentOutput(BaseModel):
    """Финальный валидированный JSON строго по контракту."""
    doc_type: DocType
    supplier_name: str | None = None
    supplier_bin: str | None = None
    doc_number: str | None = None
    doc_date: str | None = None
    currency: CurrencyType | None = None
    total_amount: float | None = None
    vat_amount: float | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("doc_date")
    @classmethod
    def check_iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"Дата '{v}' не соответствует формату YYYY-MM-DD")
        date.fromisoformat(v)
        return v

    @field_validator("supplier_bin")
    @classmethod
    def check_bin_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not re.fullmatch(r"\d{12}", v):
            raise ValueError(f"БИН '{v}' должен состоять ровно из 12 цифр")
        return v

    @field_validator("field_confidence")
    @classmethod
    def check_confidence_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Confidence для '{key}' ({score}) должен быть в диапазоне [0.0, 1.0]")
        return {k: round(val, 2) for k, val in v.items()}

    @model_validator(mode="after")
    def check_bin_checksum_warning(self) -> "DocumentOutput":
        if self.supplier_bin and not validate_kz_bin_checksum(self.supplier_bin):
            msg = f"БИН {self.supplier_bin} не прошёл проверку контрольного разряда РК (возможна ошибка OCR)"
            if msg not in self.warnings:
                self.warnings.append(msg)
            if "supplier_bin" in self.field_confidence:
                self.field_confidence["supplier_bin"] = min(self.field_confidence["supplier_bin"], 0.5)
        return self


# --- Входные/выходные схемы для FastAPI эндпоинтов ---

class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Сырой текст входящего документа")
    use_mock: bool = Field(default=False, description="Принудительно использовать мок-провайдер без вызова Gemini")