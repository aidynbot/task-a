import pytest
from pydantic import ValidationError

from app.schemas import ConfidenceScores, DocumentOutput, LineItem


def test_confidence_scores_reject_values_outside_unit_interval():
    with pytest.raises(ValidationError):
        ConfidenceScores(doc_type=1.01)


def test_document_output_rejects_invalid_date_and_bin_format():
    with pytest.raises(ValidationError):
        DocumentOutput(doc_type="invoice", doc_date="15.10.2026")

    with pytest.raises(ValidationError):
        DocumentOutput(doc_type="invoice", supplier_bin="12345")


def test_document_output_warns_for_invalid_bin_checksum():
    output = DocumentOutput(
        doc_type="invoice",
        supplier_bin="170340021559",
        field_confidence={"supplier_bin": 0.9},
    )

    assert output.field_confidence["supplier_bin"] == 0.5
    assert any("контрольного разряда" in warning.lower() for warning in output.warnings)


def test_line_item_requires_a_non_empty_name():
    with pytest.raises(ValidationError):
        LineItem(name="", qty=1, price=10)
