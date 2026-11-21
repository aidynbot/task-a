from app.guardrails import (
    apply_arithmetic_audit,
    apply_grounding_checks,
    apply_guardrails,
    apply_ocr_bin_normalization,
    apply_vat_exempt_guard,
)
from app.schemas import (
    ConfidenceScores,
    FieldEvidence,
    LLMExtractionResult,
    LineItem,
    validate_kz_bin_checksum,
)


def _base_extraction(**overrides) -> LLMExtractionResult:
    data = {
        "doc_type": "invoice",
        "supplier_name": 'ТОО "Тест"',
        "supplier_bin": "210640900180",
        "doc_number": "100",
        "doc_date": "2026-10-15",
        "currency": "KZT",
        "total_amount": 11200.0,
        "vat_amount": 1200.0,
        "line_items": [LineItem(name="Услуга", qty=1, price=11200.0)],
        "evidence": FieldEvidence(
            supplier_name='ТОО "Тест"',
            supplier_bin="210640900180",
            doc_number="100",
            doc_date="15.10.2026",
            total_amount="11 200",
            vat_amount="1 200",
        ),
        "field_confidence": ConfidenceScores(
            doc_type=0.99,
            supplier_name=0.99,
            supplier_bin=0.99,
            doc_number=0.99,
            doc_date=0.99,
            currency=0.99,
            total_amount=0.99,
            vat_amount=0.99,
            line_items=0.99,
        ),
        "warnings": [],
    }
    data.update(overrides)
    return LLMExtractionResult(**data)


def test_kz_bin_checksum_validation():
    assert validate_kz_bin_checksum("210640900180") is True
    assert validate_kz_bin_checksum("180540019281") is True
    assert validate_kz_bin_checksum("170340021559") is False


def test_ocr_bin_normalization():
    res = _base_extraction(
        supplier_bin="2l064090Ol80",
        evidence=FieldEvidence(supplier_bin="2l064090Ol80"),
    )
    warnings: list[str] = []

    apply_ocr_bin_normalization(res, warnings)
    assert res.supplier_bin == "210640900180"
    assert len(warnings) == 1


def test_grounding_checks_drop_hallucinated_number():
    source = 'Счет на оплату от 15.10.2026 Поставщик ТОО "Тест" БИН 210640900180 Итого 11 200'
    res = _base_extraction(doc_number="999999")
    warnings: list[str] = []

    apply_grounding_checks(res, source, warnings)
    assert res.doc_number is None
    assert res.field_confidence.doc_number <= 0.4
    assert any("номер документа" in w.lower() for w in warnings)


def test_vat_exempt_guard_clears_vat_on_explicit_exemption():
    source = "Акт выполненных работ № 100. Итого: 11 200 тенге. Без НДС."
    res = _base_extraction(vat_amount=1200.0)
    warnings: list[str] = []

    apply_vat_exempt_guard(res, source, warnings)
    assert res.vat_amount is None
    assert res.field_confidence.vat_amount == 1.0
    assert len(warnings) == 1


def test_vat_exempt_guard_preserves_vat_after_subtotal_without_vat():
    source = "Акт № 314. Итого без НДС: 620 000.00 тенге. НДС 16%: 99 200.00 тенге. Всего: 719 200.00"
    res = _base_extraction(total_amount=719200.0, vat_amount=99200.0)
    warnings: list[str] = []

    apply_vat_exempt_guard(res, source, warnings)
    assert res.vat_amount == 99200.0
    assert warnings == []


def test_arithmetic_audit_detects_line_items_mismatch():
    res = _base_extraction(
        total_amount=305000.0,
        line_items=[
            LineItem(name="Позиция 1", qty=2, price=100000.0),
            LineItem(name="Позиция 2", qty=1, price=100000.0),
        ],
    )
    warnings: list[str] = []

    apply_arithmetic_audit(res, warnings)
    assert len(warnings) == 1
    assert "300000.0" in warnings[0]
    assert res.field_confidence.total_amount <= 0.7


def test_apply_guardrails_end_to_end_warns_on_invalid_bin_checksum():
    source = 'Счет № 100 от 15.10.2026 ТОО "Тест" БИН 170340021559 Итого 11 200 KZT'
    res = _base_extraction(
        supplier_bin="170340021559",
        evidence=FieldEvidence(supplier_bin="170340021559"),
    )
    output = apply_guardrails(res, source)
    assert output.field_confidence["supplier_bin"] == 0.5
    assert any("контрольного разряда" in w.lower() for w in output.warnings)


def test_guardrails_do_not_mutate_input_result():
    res = _base_extraction(supplier_bin="170340021559")
    original_bin = res.supplier_bin

    output = apply_guardrails(res, "Счет № 100 от 15.10.2026 БИН 170340021559")

    assert res.supplier_bin == original_bin
    assert output.supplier_bin == original_bin


def test_vat_exemption_without_amount_clears_vat():
    res = _base_extraction(vat_amount=1200.0)
    warnings: list[str] = []

    apply_vat_exempt_guard(res, "Счет: НДС не облагается", warnings)

    assert res.vat_amount is None
    assert len(warnings) == 1
