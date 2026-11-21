import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from app.pipeline import process_text

SCALAR_FIELDS = (
    "doc_type",
    "supplier_name",
    "supplier_bin",
    "doc_number",
    "doc_date",
    "currency",
    "total_amount",
    "vat_amount",
)


def _norm_str(val: str) -> str:
    """Мягкая нормализация кавычек и пробелов при сравнении названий компаний."""
    s = val.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"').lower()
    s = s.replace("индивидуальный предприниматель", "ип")
    s = s.replace("товарищество с ограниченной ответственностью", "тоо")
    s = s.replace("жауапкершілігі шектеулі серіктестігі", "жшс")
    return re.sub(r"\s+", " ", s).strip().rstrip(".,;:")


def _values_match(field: str, pred: Any, exp: Any) -> bool:
    if pred is None and exp is None:
        return True
    if pred is None or exp is None:
        return False

    if isinstance(exp, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(pred) - float(exp)) <= 0.01

    if field == "supplier_name" and isinstance(pred, str) and isinstance(exp, str):
        return _norm_str(pred) == _norm_str(exp)

    if isinstance(pred, str) and isinstance(exp, str):
        return pred.strip() == exp.strip()

    return pred == exp


def _line_items_match(pred_items: list[dict[str, Any]], exp_items: list[dict[str, Any]]) -> bool:
    if len(pred_items) != len(exp_items):
        return False

    for p, e in zip(pred_items, exp_items):
        if _norm_str(str(p.get("name", ""))) != _norm_str(str(e.get("name", ""))):
            return False
        if not _values_match("qty", p.get("qty"), e.get("qty")):
            return False
        if not _values_match("price", p.get("price"), e.get("price")):
            return False
    return True


def run_evaluation(
    inputs_dir: Path,
    expected_dir: Path,
    use_mock: bool = False,
    delay_sec: float = 0.0,
) -> dict[str, Any]:
    input_files = sorted(inputs_dir.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"В директории {inputs_dir} не найдены входные .txt файлы")

    field_hits: dict[str, int] = {f: 0 for f in (*SCALAR_FIELDS, "line_items")}
    total_docs = len(input_files)

    expected_nulls = 0
    hallucinations = 0
    expected_present = 0
    omissions = 0
    warning_expected_docs = 0
    warning_detected_docs = 0
    latencies_ms: list[float] = []
    doc_summaries: list[dict[str, Any]] = []

    for idx, input_path in enumerate(input_files):
        if idx > 0 and delay_sec > 0 and not use_mock:
            time.sleep(delay_sec)

        expected_path = expected_dir / f"{input_path.stem}.json"
        if not expected_path.exists():
            raise FileNotFoundError(f"Не найден эталонный файл: {expected_path}")

        raw_text = input_path.read_text(encoding="utf-8")
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        t0 = time.perf_counter()
        output = process_text(raw_text=raw_text, use_mock=use_mock)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        predicted = output.model_dump()
        mismatches: list[str] = []

        for field in SCALAR_FIELDS:
            exp_val = expected.get(field)
            pred_val = predicted.get(field)

            if exp_val is None:
                expected_nulls += 1
                if pred_val is not None:
                    hallucinations += 1
            else:
                expected_present += 1
                if pred_val is None:
                    omissions += 1

            if _values_match(field, pred_val, exp_val):
                field_hits[field] += 1
            else:
                mismatches.append(f"{field}: got={pred_val!r}, exp={exp_val!r}")

        exp_items = expected.get("line_items", [])
        pred_items = predicted.get("line_items", [])
        if not exp_items:
            expected_nulls += 1
            if pred_items:
                hallucinations += 1
        else:
            expected_present += 1
            if not pred_items:
                omissions += 1

        if _line_items_match(pred_items, exp_items):
            field_hits["line_items"] += 1
        else:
            mismatches.append(f"line_items: got={len(pred_items)} items, exp={len(exp_items)} items")

        if expected.get("warnings"):
            warning_expected_docs += 1
            if predicted.get("warnings"):
                warning_detected_docs += 1

        doc_summaries.append(
            {
                "doc": input_path.name,
                "latency_ms": round(elapsed_ms, 1),
                "mismatches": mismatches,
                "warnings_count": len(predicted.get("warnings", [])),
            }
        )

    # Печать отчёта
    mode_label = "MOCK" if use_mock else "AUTO / GEMINI"
    print(f"\n=== ОЦЕНКА ПАЙПЛАЙНА ИЗВЛЕЧЕНИЯ ДОКУМЕНТОВ (Режим: {mode_label}, Документов: {total_docs}) ===\n")

    print(f"{'Поле':<18} | {'Совпадений':<12} | {'Точность (%)':<12}")
    print("-" * 48)
    for field, hits in field_hits.items():
        acc = (hits / total_docs) * 100
        print(f"{field:<18} | {f'{hits}/{total_docs}':<12} | {acc:>10.1f}%")

    overall_accuracy = (
        sum(field_hits.values()) / (total_docs * len(field_hits))
    ) * 100
    hallucination_rate = (hallucinations / expected_nulls * 100) if expected_nulls else 0.0
    omission_rate = (omissions / expected_present * 100) if expected_present else 0.0
    warning_recall = (
        (warning_detected_docs / warning_expected_docs * 100) if warning_expected_docs else 100.0
    )
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    print("-" * 48)
    print(f"Общая точность по всем полям : {overall_accuracy:.1f}%")
    print(f"Галлюцинации (False Positives): {hallucinations}/{expected_nulls} ({hallucination_rate:.1f}%)")
    print(f"Пропуски (False Negatives)     : {omissions}/{expected_present} ({omission_rate:.1f}%)")
    print(f"Детекция конфликтов (warnings): {warning_detected_docs}/{warning_expected_docs} ({warning_recall:.1f}%)")
    print(f"Средняя задержка на документ  : {avg_latency:.1f} мс\n")

    print("=== ДЕТАЛИЗАЦИЯ ПО ДОКУМЕНТАМ ===")
    for item in doc_summaries:
        status = "OK" if not item["mismatches"] else f"DIFF ({len(item['mismatches'])})"
        print(f"• {item['doc']:<28} | {status:<10} | {item['latency_ms']:>6.1f} мс | warnings: {item['warnings_count']}")
        for err in item["mismatches"]:
            print(f"    ↳ {err}")

    return {
        "mode": mode_label,
        "total_docs": total_docs,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "hallucinations": hallucinations,
        "expected_nulls": expected_nulls,
        "hallucination_rate_pct": round(hallucination_rate, 2),
        "omissions": omissions,
        "expected_present": expected_present,
        "omission_rate_pct": round(omission_rate, 2),
        "warning_recall_pct": round(warning_recall, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "field_accuracy_pct": {
            k: round((v / total_docs) * 100, 2) for k, v in field_hits.items()
        },
        "documents": doc_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества извлечения финансовых документов")
    parser.add_argument("--inputs", type=Path, default=Path("dataset/inputs"), help="Папка со входными .txt")
    parser.add_argument("--expected", type=Path, default=Path("dataset/expected"), help="Папка с эталонными .json")
    parser.add_argument("--mock", action="store_true", help="Запустить оценку в принудительном Mock-режиме")
    parser.add_argument("--delay", type=float, default=0.0, help="Пауза в секундах между запросами (для Free Tier RPM)")
    parser.add_argument("--save-report", type=Path, default=None, help="Путь для сохранения JSON-отчёта")
    args = parser.parse_args()

    report = run_evaluation(
        inputs_dir=args.inputs,
        expected_dir=args.expected,
        use_mock=args.mock,
        delay_sec=args.delay,
    )

    if args.save_report:
        args.save_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт сохранён в {args.save_report}")

if __name__ == "__main__":
    main()
