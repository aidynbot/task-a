import re
from app.schemas import ConfidenceScores, FieldEvidence, LineItem, LLMExtractionResult

KZ_MONTHS = {
    "қаңтар": "01",
    "ақпан": "02",
    "наурыз": "03",
    "сәуір": "04",
    "мамыр": "05",
    "маусым": "06",
    "шілде": "07",
    "тамыз": "08",
    "қыркүйек": "09",
    "қазан": "10",
    "қараша": "11",
    "желтоқсан": "12",
}


class MockProvider:
    """
    Эвристический локальный провайдер для запуска пайплайна и eval.py без ключа Gemini API.
    Имитирует первичный парсинг LLM, оставляя постобработку и валидацию слою guardrails.
    """

    def extract(self, text: str, error_feedback: str | None = None) -> LLMExtractionResult:
        lower_text = text.lower()
        warnings: list[str] = []

        # 1. Определение типа документа
        doc_type = "unknown"
        if any(k in lower_text for k in ("счёт на оплату", "счет на оплату", "шот-фактура", "счет-фактура", "invoice")):
            doc_type = "invoice"
        elif any(k in lower_text for k in ("коммерческое предложение", "commercial offer")):
            doc_type = "commercial_offer"
        elif any(k in lower_text for k in ("акт выполненных работ", "акт оказанных услуг", "актісі")):
            doc_type = "act"

        # 2. Поставщик
        supplier_name = None
        supplier_evidence = None
        m_sup = re.search(
            r'((?:ТОО|ИП|АО|ЖШС|АҚ)\s*[«"][^»"\n]+[»"]|[«"][^»"\n]+[»"]\s*(?:ЖШС|АҚ)|(?:Supplier|от компании)\s*[:\-]?\s*([^\n]+))',
            text,
            re.IGNORECASE,
        )
        if m_sup:
            supplier_evidence = m_sup.group(0).strip()
            supplier_name = (m_sup.group(2) or m_sup.group(1)).strip().strip("«»\"")
            if m_sup.group(1) and any(org in m_sup.group(1) for org in ("ТОО", "ИП", "АО", "ЖШС", "АҚ")):
                supplier_name = m_sup.group(1).strip()

        # 3. БИН / ИИН (намеренно берём сырую 12-символьную строку — её дочистят guardrails)
        supplier_bin = None
        bin_evidence = None
        m_bin = re.search(r"(?:БСН/БИН|БИН|ИИН|БСН)\s*[:№]?\s*([0-9lIOОЗБ]{12})", text)
        if m_bin:
            bin_evidence = m_bin.group(0).strip()
            supplier_bin = m_bin.group(1).strip()

        # 4. Номер документа
        doc_number = None
        num_evidence = None
        m_num = re.search(r"№\s*([A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі0-9\-/]+)", text)
        if m_num:
            num_evidence = m_num.group(0).strip()
            doc_number = m_num.group(1).strip()

        # 5. Дата документа (DD.MM.YYYY, YYYY-MM-DD или казахский формат)
        doc_date = None
        date_evidence = None
        m_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        m_dmy = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", text)
        m_kz_date = re.search(
            r"\b(\d{1,2})\s+(" + "|".join(KZ_MONTHS.keys()) + r")\s+(\d{4})",
            lower_text,
        )
        if m_iso:
            doc_date = m_iso.group(1)
            date_evidence = m_iso.group(0)
        elif m_dmy:
            dd, mm, yyyy = m_dmy.groups()
            doc_date = f"{yyyy}-{mm}-{dd}"
            date_evidence = m_dmy.group(0)
        elif m_kz_date:
            dd, month_kz, yyyy = m_kz_date.groups()
            doc_date = f"{yyyy}-{KZ_MONTHS[month_kz]}-{int(dd):02d}"
            date_evidence = m_kz_date.group(0)

        # 6. Валюта
        currency = None
        if doc_type != "unknown":
            if any(k in lower_text for k in ("usd", "$")):
                currency = "USD"
            elif any(k in lower_text for k in ("eur", "€")):
                currency = "EUR"
            elif any(k in lower_text for k in ("тенге", "теңге", "тг", "kzt")):
                currency = "KZT"

        # 7. Итоговая сумма (ищем только явные маркеры итога)
        total_amount = None
        total_evidence = None
        m_total = re.search(
            r"(?:Всего к оплате|Всего с учётом НДС|Итого к оплате|Total Due|Жалпы сомасы|Итого)\s*[:\-]?\s*\$?\s*([\d\s.,]+)",
            text,
            re.IGNORECASE,
        )
        if m_total:
            raw_num = m_total.group(1).strip().rstrip(".")
            cleaned_num = raw_num.replace(" ", "").replace(",", "")
            if cleaned_num.replace(".", "", 1).isdigit():
                total_amount = float(cleaned_num)
                total_evidence = m_total.group(0).strip()

        # 8. НДС
        vat_amount = None
        vat_evidence = None
        if not any(k in lower_text for k in ("без ндс", "ндс не облагается", "ққс-сыз")):
            m_vat = re.search(r"(?:НДС|VAT)[^:\n]*:\s*([\d\s.,]+)", text, re.IGNORECASE)
            if m_vat:
                raw_vat = m_vat.group(1).strip().rstrip(".")
                cleaned_vat = raw_vat.replace(" ", "").replace(",", "")
                if cleaned_vat.replace(".", "", 1).isdigit():
                    vat_amount = float(cleaned_vat)
                    vat_evidence = m_vat.group(0).strip()

        # 9. Позиции (базовый парсинг строк с ценой)
        line_items: list[LineItem] = []
        for line in text.splitlines():
            m_kp = re.match(r"^\s*[-•]\s*(.+?)\s*[—\-]\s*([\d\s]+)\s*(?:тг|тенге|KZT)", line, re.IGNORECASE)
            if m_kp:
                price_val = float(m_kp.group(2).replace(" ", ""))
                line_items.append(LineItem(name=m_kp.group(1).strip(), qty=None, price=price_val))

        conf_val = 0.85 if doc_type != "unknown" else 1.0
        return LLMExtractionResult(
            evidence=FieldEvidence(
                supplier_name=supplier_evidence,
                supplier_bin=bin_evidence,
                doc_number=num_evidence,
                doc_date=date_evidence,
                total_amount=total_evidence,
                vat_amount=vat_evidence,
            ),
            doc_type=doc_type,
            supplier_name=supplier_name,
            supplier_bin=supplier_bin,
            doc_number=doc_number,
            doc_date=doc_date,
            currency=currency,
            total_amount=total_amount,
            vat_amount=vat_amount,
            line_items=line_items,
            field_confidence=ConfidenceScores(
                doc_type=conf_val,
                supplier_name=conf_val,
                supplier_bin=conf_val,
                doc_number=conf_val,
                doc_date=conf_val,
                currency=conf_val,
                total_amount=conf_val,
                vat_amount=conf_val,
                line_items=0.8 if line_items else 1.0,
            ),
            warnings=warnings,
        )