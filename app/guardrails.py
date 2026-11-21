import re
from app.schemas import DocumentOutput, LLMExtractionResult

OCR_DIGIT_MAP = str.maketrans({
    "O": "0",
    "О": "0",  # кириллическая О
    "o": "0",
    "о": "0",
    "l": "1",
    "I": "1",
    "І": "1",  # казахская/украинская І
    "З": "3",  # кириллическая З
    "Б": "6",  # кириллическая Б
})

NO_VAT_MARKERS = (
    "без ндс",
    "ндс не облагается",
    "ққс-сыз",
    "ққс салынбайды",
    "without vat",
    "vat exempt",
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_bin_ocr(raw_value: str) -> tuple[str, bool]:
    """Заменяет визуальные OCR-двойники цифр в кандидате на БИН/ИИН."""
    cleaned = re.sub(r"[\s\-]", "", raw_value)
    normalized = cleaned.translate(OCR_DIGIT_MAP)
    was_modified = normalized != cleaned
    return normalized, was_modified


def apply_ocr_bin_normalization(
    result: LLMExtractionResult,
    warnings: list[str],
) -> None:
    """Фильтр 1: OCR-нормализация 12-значного БИН/ИИН из supplier_bin или evidence."""
    candidate = result.supplier_bin

    # Если supplier_bin пуст, но в evidence.supplier_bin есть 12-символьный токен с OCR-шумом
    if not candidate and result.evidence.supplier_bin:
        m = re.search(r"[0-9A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{12}", result.evidence.supplier_bin)
        if m:
            candidate = m.group(0)

    if not candidate:
        result.supplier_bin = None
        return

    normalized, was_modified = _normalize_bin_ocr(candidate)

    if not re.fullmatch(r"\d{12}", normalized):
        warnings.append(
            f"Извлечённый БИН/ИИН '{candidate}' не соответствует формату 12 цифр и сброшен в null"
        )
        result.supplier_bin = None
        result.field_confidence.supplier_bin = min(result.field_confidence.supplier_bin, 0.5)
        return

    if was_modified:
        warnings.append(
            f"В БИН обнаружены и нормализованы OCR-артефакты ('{candidate}' -> '{normalized}')"
        )
        result.field_confidence.supplier_bin = min(result.field_confidence.supplier_bin, 0.7)

    result.supplier_bin = normalized


def apply_grounding_checks(
    result: LLMExtractionResult,
    source_text: str,
    warnings: list[str],
) -> None:
    """Фильтр 2: Grounding-проверка (защита от выдуманных реквизитов, которых нет в тексте)."""
    norm_source = _normalize_whitespace(source_text)
    norm_source_lower = norm_source.lower()

    # 2.1 Проверка номера документа
    if result.doc_number:
        clean_num = _normalize_whitespace(result.doc_number)
        # Допускаем нормализацию OCR-символов при поиске в исходном тексте
        ocr_norm_source = norm_source.translate(OCR_DIGIT_MAP)
        if clean_num.lower() not in norm_source_lower and clean_num.lower() not in ocr_norm_source.lower():
            warnings.append(
                f"Grounding-фильтр: номер документа '{result.doc_number}' не найден в тексте и сброшен в null"
            )
            result.doc_number = None
            result.field_confidence.doc_number = min(result.field_confidence.doc_number, 0.4)

    # 2.2 Проверка БИН/ИИН в исходном тексте (после перевода OCR-двойников)
    if result.supplier_bin:
        source_digits_only = re.sub(r"[\s\-]", "", norm_source.translate(OCR_DIGIT_MAP))
        if result.supplier_bin not in source_digits_only:
            warnings.append(
                f"Grounding-фильтр: БИН '{result.supplier_bin}' отсутствует в исходном тексте и сброшен в null"
            )
            result.supplier_bin = None
            result.field_confidence.supplier_bin = min(result.field_confidence.supplier_bin, 0.4)

    # 2.3 Проверка наличия признаков даты/года в тексте
    if result.doc_date:
        ocr_norm_source = norm_source.translate(OCR_DIGIT_MAP)
        has_date_tokens = bool(
            re.search(r"\b(19|20)\d{2}\b", ocr_norm_source)
            or re.search(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b", ocr_norm_source)
        )
        if not has_date_tokens:
            warnings.append(
                f"Grounding-фильтр: в тексте не обнаружены признаки даты/года; doc_date '{result.doc_date}' сброшен в null"
            )
            result.doc_date = None
            result.field_confidence.doc_date = min(result.field_confidence.doc_date, 0.4)


def apply_arithmetic_audit(
    result: LLMExtractionResult,
    warnings: list[str],
) -> None:
    """Фильтр 3: Арифметическая сверка позиций и итога без перезаписи исходных данных."""
    if result.total_amount is None or not result.line_items:
        return

    # Считаем сумму только если у всех позиций указаны и количество, и цена
    items_sum = 0.0
    for item in result.line_items:
        if item.qty is None or item.price is None:
            return
        items_sum += item.qty * item.price
    items_sum = round(items_sum, 2)

    total = round(result.total_amount, 2)
    vat = round(result.vat_amount, 2) if result.vat_amount is not None else 0.0

    # Допускаем совпадение с итогом напрямую или с учётом выделенного НДС (погрешность до 1 единицы на округления)
    matches_direct = abs(items_sum - total) <= 1.0
    matches_with_vat = vat > 0 and abs((items_sum + vat) - total) <= 1.0

    if not matches_direct and not matches_with_vat:
        warnings.append(
            f"Арифметическое расхождение: сумма позиций ({items_sum}) не совпадает с total_amount ({total})"
        )
        result.field_confidence.total_amount = min(result.field_confidence.total_amount, 0.7)


def apply_vat_exempt_guard(
    result: LLMExtractionResult,
    source_text: str,
    warnings: list[str],
) -> None:
    """Фильтр 4: Принудительное обнуление НДС при явных маркерах освобождения от налога."""
    lower_source = _normalize_whitespace(source_text).lower()
    has_no_vat_marker = any(marker in lower_source for marker in NO_VAT_MARKERS)

    if has_no_vat_marker and result.vat_amount is not None:
        warnings.append(
            f"В документе указано освобождение от НДС («без НДС» / «ҚҚС-сыз»); значение vat_amount ({result.vat_amount}) сброшено в null"
        )
        result.vat_amount = None
        result.field_confidence.vat_amount = 1.0


def apply_guardrails(raw_result: LLMExtractionResult, source_text: str) -> DocumentOutput:
    """
    Прогоняет сырой результат LLM через все детерминированные защитные фильтры
    и преобразует его в финальный публичный контракт DocumentOutput.
    """
    result = raw_result.model_copy(deep=True)
    warnings: list[str] = list(result.warnings)

    apply_ocr_bin_normalization(result, warnings)
    apply_grounding_checks(result, source_text, warnings)
    apply_arithmetic_audit(result, warnings)
    apply_vat_exempt_guard(result, source_text, warnings)

    # Убираем дубликаты предупреждений с сохранением порядка
    deduped_warnings = list(dict.fromkeys(warnings))

    return DocumentOutput(
        doc_type=result.doc_type,
        supplier_name=result.supplier_name,
        supplier_bin=result.supplier_bin,
        doc_number=result.doc_number,
        doc_date=result.doc_date,
        currency=result.currency,
        total_amount=result.total_amount,
        vat_amount=result.vat_amount,
        line_items=result.line_items,
        field_confidence=result.field_confidence.model_dump(),
        warnings=deduped_warnings,
    )