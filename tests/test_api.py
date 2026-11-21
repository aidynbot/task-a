from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] in {"mock", "gemini"}


def test_parse_endpoint_accepts_mock_request_and_returns_contract():
    response = client.post(
        "/parse",
        json={"text": "Коммерческое предложение. Итого: 12 500 тенге.", "use_mock": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["doc_type"] == "commercial_offer"
    assert body["currency"] == "KZT"
    assert body["total_amount"] == 12500.0
    assert set(body) == {
        "doc_type", "supplier_name", "supplier_bin", "doc_number", "doc_date",
        "currency", "total_amount", "vat_amount", "line_items", "field_confidence", "warnings",
    }


def test_parse_endpoint_rejects_missing_text():
    response = client.post("/parse", json={"use_mock": True})

    assert response.status_code == 422
