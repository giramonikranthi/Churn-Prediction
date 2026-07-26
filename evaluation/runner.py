from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evaluation.judge import JudgeResult, judge_case_with_llm
from evaluation.metrics import MetricResult, evaluate_all_metrics
from evaluation.test_suite import EvaluationCase, load_default_test_suite
from src.retention_agent.orchestrator import run_retention_orchestrator


IST_TZ = timezone(timedelta(hours=5, minutes=30), name="IST")


def _now_ist_text() -> str:
    return datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class EvaluationConfig:
    use_judge: bool
    max_cases: int | None
    output_dir: Path


def _auto_case_score(row: dict[str, Any]) -> float:
    metrics = row["metrics"]
    weighted = (
        0.25 * float(metrics["status_match"]["score"])
        + 0.25 * float(metrics["tool_selection_accuracy"]["score"])
        + 0.20 * float(metrics["parameter_extraction_accuracy"]["score"])
        + 0.20 * float(metrics["response_completeness"]["score"])
        + 0.10 * float(metrics["hallucination_detection"]["score"])
    )
    return round(weighted * 100.0, 2)


@contextmanager
def _disable_response_polish() -> Any:
    # Prevent expensive optional polishing calls during benchmark execution.
    import src.retention_agent.orchestrator as orchestrator_module

    original = orchestrator_module.optional_llm_polish
    orchestrator_module.optional_llm_polish = lambda text: text
    try:
        yield
    finally:
        orchestrator_module.optional_llm_polish = original


def _metric_dict(metric_map: dict[str, MetricResult]) -> dict[str, Any]:
    return {
        name: {
            "score": metric.score,
            "details": metric.details,
        }
        for name, metric in metric_map.items()
    }


def _status_match_score(expected_status: str, actual_status: str) -> float:
    return 1.0 if expected_status == actual_status else 0.0


def _category_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        categories.setdefault(str(row["category"]), []).append(row)

    category_summary: dict[str, Any] = {}
    for category, items in categories.items():
        count = len(items)
        metric_names = [
            "status_match",
            "tool_selection_accuracy",
            "parameter_extraction_accuracy",
            "response_completeness",
            "hallucination_detection",
        ]

        averaged_metrics: dict[str, float] = {}
        for metric_name in metric_names:
            averaged_metrics[metric_name] = round(
                sum(float(item["metrics"][metric_name]["score"]) for item in items) / count,
                4,
            )

        judge_items = [item for item in items if item.get("judge", {}).get("status") == "ok"]
        judge_average = None
        if judge_items:
            judge_average = round(
                sum(float(item["judge"]["weighted_total"]) for item in judge_items) / len(judge_items),
                2,
            )

        category_summary[category] = {
            "count": count,
            "metrics": averaged_metrics,
            "judge_weighted_total_avg": judge_average,
        }

    return category_summary


def _overall_summary(rows: list[dict[str, Any]], started_at: float, ended_at: float) -> dict[str, Any]:
    count = max(len(rows), 1)

    avg_status = sum(float(row["metrics"]["status_match"]["score"]) for row in rows) / count
    avg_tool = sum(float(row["metrics"]["tool_selection_accuracy"]["score"]) for row in rows) / count
    avg_param = sum(float(row["metrics"]["parameter_extraction_accuracy"]["score"]) for row in rows) / count
    avg_comp = sum(float(row["metrics"]["response_completeness"]["score"]) for row in rows) / count
    avg_hallu = sum(float(row["metrics"]["hallucination_detection"]["score"]) for row in rows) / count
    avg_latency = sum(float(row["latency_seconds"]) for row in rows) / count

    judge_rows = [row for row in rows if row.get("judge", {}).get("status") == "ok"]
    judge_avg = None
    if judge_rows:
        judge_avg = round(
            sum(float(row["judge"]["weighted_total"]) for row in judge_rows) / len(judge_rows),
            2,
        )

    return {
        "run_duration_seconds": round(ended_at - started_at, 3),
        "total_cases": len(rows),
        "automated_metrics": {
            "status_match": round(avg_status, 4),
            "tool_selection_accuracy": round(avg_tool, 4),
            "parameter_extraction_accuracy": round(avg_param, 4),
            "response_completeness": round(avg_comp, 4),
            "hallucination_detection": round(avg_hallu, 4),
            "latency_seconds": round(avg_latency, 4),
        },
        "judge_weighted_total_avg": judge_avg,
    }


def _evaluate_quality_gates(
    rows: list[dict[str, Any]],
    overall: dict[str, Any],
    by_category: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    hard_cases = [
        row
        for row in rows
        if str(row.get("category")) in {"escalation_trigger", "adversarial"}
    ]
    hard_hallucination_pass = all(
        float(row["metrics"]["hallucination_detection"]["score"]) >= 1.0 for row in hard_cases
    )
    checks.append(
        {
            "name": "hard_hallucination_guard",
            "type": "hard",
            "passed": hard_hallucination_pass,
            "scope": "escalation_trigger + adversarial",
            "threshold": ">= 1.0",
            "actual": [
                {
                    "case_id": row["case_id"],
                    "score": row["metrics"]["hallucination_detection"]["score"],
                }
                for row in hard_cases
            ],
        }
    )

    hard_status_pass = all(float(row["metrics"]["status_match"]["score"]) >= 1.0 for row in rows)
    checks.append(
        {
            "name": "hard_status_match",
            "type": "hard",
            "passed": hard_status_pass,
            "scope": "all_cases",
            "threshold": "== 1.0",
            "actual": round(float(overall["automated_metrics"]["status_match"]), 4),
        }
    )

    soft_checks = [
        ("soft_tool_accuracy_overall", "tool_selection_accuracy", 0.95),
        ("soft_response_completeness_overall", "response_completeness", 0.90),
        ("soft_hallucination_overall", "hallucination_detection", 0.98),
    ]
    for name, metric_key, threshold in soft_checks:
        actual = float(overall["automated_metrics"][metric_key])
        checks.append(
            {
                "name": name,
                "type": "soft",
                "passed": actual >= threshold,
                "scope": "overall",
                "threshold": f">= {threshold}",
                "actual": round(actual, 4),
            }
        )

    for category in ("single_tool_happy_path", "multi_step_chaining"):
        category_metrics = by_category.get(category, {}).get("metrics", {})
        actual = float(category_metrics.get("tool_selection_accuracy", 0.0))
        checks.append(
            {
                "name": f"soft_{category}_tool_accuracy",
                "type": "soft",
                "passed": actual >= 0.95,
                "scope": category,
                "threshold": ">= 0.95",
                "actual": round(actual, 4),
            }
        )

    model_disagreement_metrics = by_category.get("model_disagreement", {}).get("metrics", {})
    if model_disagreement_metrics:
        actual = float(model_disagreement_metrics.get("response_completeness", 0.0))
        checks.append(
            {
                "name": "soft_model_disagreement_completeness",
                "type": "soft",
                "passed": actual >= 0.80,
                "scope": "model_disagreement",
                "threshold": ">= 0.80",
                "actual": round(actual, 4),
            }
        )

    if overall.get("judge_weighted_total_avg") is not None:
        judge_avg = float(overall["judge_weighted_total_avg"])
        checks.append(
            {
                "name": "soft_judge_weighted_total",
                "type": "soft",
                "passed": judge_avg >= 80.0,
                "scope": "overall",
                "threshold": ">= 80.0",
                "actual": round(judge_avg, 2),
            }
        )

    hard_failed = [check for check in checks if check["type"] == "hard" and not check["passed"]]
    soft_failed = [check for check in checks if check["type"] == "soft" and not check["passed"]]

    return {
        "pass": len(hard_failed) == 0 and len(soft_failed) == 0,
        "hard_failures": hard_failed,
        "soft_failures": soft_failed,
        "checks": checks,
    }


def _build_leaderboard_markdown(rows: list[dict[str, Any]], report: dict[str, Any]) -> str:
    sorted_rows = sorted(rows, key=_auto_case_score, reverse=True)
    lines: list[str] = []
    lines.append("# Evaluation Leaderboard")
    lines.append("")
    lines.append(f"Run At (IST): {report['run_at_ist']}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    overall = report["overall"]
    lines.append(f"- Total Cases: {overall['total_cases']}")
    lines.append(f"- Avg Tool Selection Accuracy: {overall['automated_metrics']['tool_selection_accuracy']}")
    lines.append(f"- Avg Parameter Extraction Accuracy: {overall['automated_metrics']['parameter_extraction_accuracy']}")
    lines.append(f"- Avg Response Completeness: {overall['automated_metrics']['response_completeness']}")
    lines.append(f"- Avg Hallucination Detection: {overall['automated_metrics']['hallucination_detection']}")
    lines.append(f"- Avg Latency (s): {overall['automated_metrics']['latency_seconds']}")
    lines.append(f"- Judge Weighted Avg: {overall['judge_weighted_total_avg']}")
    lines.append("")

    gate_state = report.get("quality_gates", {})
    lines.append("## Quality Gates")
    lines.append("")
    lines.append(f"- Pass: {gate_state.get('pass')}")
    lines.append(f"- Hard Failures: {len(gate_state.get('hard_failures', []))}")
    lines.append(f"- Soft Failures: {len(gate_state.get('soft_failures', []))}")
    lines.append("")

    lines.append("## Case Ranking")
    lines.append("")
    lines.append("| Rank | Case ID | Category | Auto Score | Judge Score | Status |")
    lines.append("|---|---|---|---:|---:|---|")
    for index, row in enumerate(sorted_rows, start=1):
        auto_score = _auto_case_score(row)
        judge_score = row.get("judge", {}).get("weighted_total")
        if row.get("judge", {}).get("status") != "ok":
            judge_score = "-"
        lines.append(
            "| "
            f"{index} | {row['case_id']} | {row['category']} | {auto_score} | {judge_score} | {row['actual'].get('status')} |"
        )

    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Cases | Tool Acc | Param Acc | Completeness | Hallucination |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, summary in report["by_category"].items():
        metrics = summary["metrics"]
        lines.append(
            "| "
            f"{category} | {summary['count']} | {metrics['tool_selection_accuracy']} | "
            f"{metrics['parameter_extraction_accuracy']} | {metrics['response_completeness']} | "
            f"{metrics['hallucination_detection']} |"
        )

    lines.append("")
    lines.append("## Lowest Scoring Cases")
    lines.append("")
    lowest = sorted(rows, key=_auto_case_score)[:3]
    for row in lowest:
        lines.append(
            f"- {row['case_id']} ({row['category']}): auto_score={_auto_case_score(row)}"
        )

    lines.append("")
    return "\n".join(lines)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    cases = load_default_test_suite()
    if config.max_cases is not None:
        cases = cases[: max(0, config.max_cases)]

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []

    with _disable_response_polish():
        for case in cases:
            case_start = time.perf_counter()
            actual = run_retention_orchestrator(
                user_message=case.user_input,
                selected_customer_id=case.selected_customer_id,
                prediction_model=False,
                session_id=f"EVAL-{case.case_id}",
            )
            case_end = time.perf_counter()

            metric_map = evaluate_all_metrics(case=case, actual_output=actual)
            status_score = _status_match_score(
                expected_status=case.expected_status,
                actual_status=str(actual.get("status") or ""),
            )

            metrics_payload = _metric_dict(metric_map)
            metrics_payload["status_match"] = {
                "score": status_score,
                "details": {
                    "expected_status": case.expected_status,
                    "actual_status": str(actual.get("status") or ""),
                },
            }

            judge_result: JudgeResult
            if config.use_judge:
                judge_result = judge_case_with_llm(case=case, actual_output=actual, metric_snapshot=metric_map)
            else:
                judge_result = JudgeResult(
                    status="skipped",
                    scores={},
                    weighted_total=0.0,
                    confidence="low",
                    failure_modes=["judge_disabled"],
                    improvement_actions=["Run with --use-judge to enable LLM-based scoring."],
                )

            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "notes": case.notes,
                    "expected": case.to_dict(),
                    "actual": actual,
                    "metrics": metrics_payload,
                    "judge": judge_result.to_dict(),
                    "latency_seconds": round(case_end - case_start, 4),
                }
            )

    ended_at = time.perf_counter()

    overall = _overall_summary(rows=rows, started_at=started_at, ended_at=ended_at)
    by_category = _category_aggregates(rows)
    quality_gates = _evaluate_quality_gates(rows=rows, overall=overall, by_category=by_category)

    report = {
        "run_at_ist": _now_ist_text(),
        "overall": overall,
        "by_category": by_category,
        "quality_gates": quality_gates,
        "cases": rows,
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = config.output_dir / "aggregate_report.json"
    rows_jsonl = config.output_dir / "case_results.jsonl"
    leaderboard_md = config.output_dir / "leaderboard.md"

    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)

    with rows_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    leaderboard_text = _build_leaderboard_markdown(rows=rows, report=report)
    with leaderboard_md.open("w", encoding="utf-8") as handle:
        handle.write(leaderboard_text)

    return {
        "report_path": str(report_json),
        "rows_path": str(rows_jsonl),
        "leaderboard_path": str(leaderboard_md),
        "report": report,
    }


def _default_output_dir() -> Path:
    stamp = datetime.now(IST_TZ).strftime("%Y%m%d_%H%M%S")
    return Path("data") / "evaluation_metrics" / f"eval_run_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation for retention agent.")
    parser.add_argument("--use-judge", action="store_true", help="Enable separate LLM-as-judge scoring.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for number of cases to run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output artifacts. Default creates timestamped folder in data/evaluation_metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    config = EvaluationConfig(use_judge=bool(args.use_judge), max_cases=args.max_cases, output_dir=output_dir)

    result = run_evaluation(config=config)
    report = result["report"]
    overall = report["overall"]

    print("Evaluation complete")
    print(f"- report: {result['report_path']}")
    print(f"- rows:   {result['rows_path']}")
    print(f"- leaderboard: {result['leaderboard_path']}")
    print(f"- total_cases: {overall['total_cases']}")
    print(f"- avg_tool_selection_accuracy: {overall['automated_metrics']['tool_selection_accuracy']}")
    print(f"- avg_parameter_extraction_accuracy: {overall['automated_metrics']['parameter_extraction_accuracy']}")
    print(f"- avg_response_completeness: {overall['automated_metrics']['response_completeness']}")
    print(f"- avg_hallucination_detection: {overall['automated_metrics']['hallucination_detection']}")
    print(f"- avg_latency_seconds: {overall['automated_metrics']['latency_seconds']}")
    print(f"- judge_weighted_total_avg: {overall['judge_weighted_total_avg']}")
    print(f"- quality_gates_pass: {report['quality_gates']['pass']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
