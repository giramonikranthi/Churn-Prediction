from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evaluation.test_suite import EvaluationCase, ExpectedToolCall


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    details: dict[str, Any]


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, left_item in enumerate(left, start=1):
        for j, right_item in enumerate(right, start=1):
            if left_item == right_item:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _extract_tool_calls(actual_output: dict[str, Any]) -> list[dict[str, Any]]:
    trace = actual_output.get("tool_trace", [])
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


def tool_selection_accuracy(case: EvaluationCase, actual_output: dict[str, Any]) -> MetricResult:
    expected_names = [call.name for call in case.expected_tool_calls]
    actual_calls = _extract_tool_calls(actual_output)
    actual_names = [str(item.get("tool") or "") for item in actual_calls]

    lcs = _lcs_length(expected_names, actual_names)
    precision = lcs / len(actual_names) if actual_names else (1.0 if not expected_names else 0.0)
    recall = lcs / len(expected_names) if expected_names else 1.0
    exact_order_match = expected_names == actual_names

    score = 0.6 * recall + 0.4 * precision
    if exact_order_match:
        score = 1.0

    return MetricResult(
        name="tool_selection_accuracy",
        score=round(max(0.0, min(1.0, score)), 4),
        details={
            "expected_sequence": expected_names,
            "actual_sequence": actual_names,
            "lcs": lcs,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "exact_order_match": exact_order_match,
        },
    )


def _match_expected_to_actual(
    expected_calls: list[ExpectedToolCall], actual_calls: list[dict[str, Any]]
) -> list[tuple[ExpectedToolCall, dict[str, Any]]]:
    matches: list[tuple[ExpectedToolCall, dict[str, Any]]] = []
    used_indices: set[int] = set()

    for expected in expected_calls:
        for idx, actual in enumerate(actual_calls):
            if idx in used_indices:
                continue
            if str(actual.get("tool") or "") != expected.name:
                continue
            used_indices.add(idx)
            matches.append((expected, actual))
            break

    return matches


def parameter_extraction_accuracy(case: EvaluationCase, actual_output: dict[str, Any]) -> MetricResult:
    actual_calls = _extract_tool_calls(actual_output)
    matches = _match_expected_to_actual(case.expected_tool_calls, actual_calls)

    total_required = 0
    total_correct = 0
    mismatch_items: list[dict[str, Any]] = []

    for expected, actual in matches:
        actual_args = actual.get("arguments", {})
        if not isinstance(actual_args, dict):
            actual_args = {}

        for key, expected_value in expected.required_params.items():
            total_required += 1
            actual_value = actual_args.get(key)
            if _normalize(actual_value) == _normalize(expected_value):
                total_correct += 1
            else:
                mismatch_items.append(
                    {
                        "tool": expected.name,
                        "param": key,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

    if total_required == 0:
        score = 1.0
    else:
        score = total_correct / total_required

    return MetricResult(
        name="parameter_extraction_accuracy",
        score=round(max(0.0, min(1.0, score)), 4),
        details={
            "matched_expected_calls": len(matches),
            "total_expected_calls": len(case.expected_tool_calls),
            "required_params_checked": total_required,
            "required_params_correct": total_correct,
            "mismatches": mismatch_items,
        },
    )


def response_completeness(case: EvaluationCase, actual_output: dict[str, Any]) -> MetricResult:
    text = str(actual_output.get("assistant_message") or "")
    lowered = text.lower()

    required_hits = []
    for phrase in case.quality_criteria.required_phrases:
        required_hits.append({"phrase": phrase, "present": phrase.lower() in lowered})

    forbidden_hits = []
    for phrase in case.quality_criteria.forbidden_phrases:
        forbidden_hits.append({"phrase": phrase, "present": phrase.lower() in lowered})

    checks: list[bool] = []
    checks.extend(item["present"] for item in required_hits)
    checks.extend(not item["present"] for item in forbidden_hits)

    if case.quality_criteria.must_reference_next_action:
        checks.append("next action" in lowered)

    if case.quality_criteria.must_be_non_json_text:
        looks_like_json = (
            (text.strip().startswith("{") and text.strip().endswith("}"))
            or (text.strip().startswith("[") and text.strip().endswith("]"))
            or '"tool"' in text
        )
        checks.append(not looks_like_json)

    score = sum(1 for item in checks if item) / len(checks) if checks else 1.0

    return MetricResult(
        name="response_completeness",
        score=round(max(0.0, min(1.0, score)), 4),
        details={
            "required_hits": required_hits,
            "forbidden_hits": forbidden_hits,
            "must_reference_next_action": case.quality_criteria.must_reference_next_action,
            "next_action_present": "next action" in lowered,
            "text_length": len(text),
        },
    )


def hallucination_detection(case: EvaluationCase, actual_output: dict[str, Any]) -> MetricResult:
    text = str(actual_output.get("assistant_message") or "")
    lowered = text.lower()
    tools = [str(call.get("tool") or "") for call in _extract_tool_calls(actual_output)]

    penalties: list[dict[str, Any]] = []

    if ("escalation" in lowered or "supervisor" in lowered) and "escalate_to_supervisor" not in tools:
        penalties.append({"rule": "escalation_claim_without_tool", "weight": 0.35})

    has_probability_phrase = bool(re.search(r"\b\d{1,3}(?:\.\d+)?%\b", text)) or "churn probability" in lowered
    if has_probability_phrase and "predict_churn" not in tools:
        penalties.append({"rule": "probability_claim_without_prediction", "weight": 0.35})

    has_offer_id = bool(re.search(r"\bOFR-[A-Z0-9-]+\b", text.upper()))
    if has_offer_id and "get_retention_offers" not in tools:
        penalties.append({"rule": "offer_claim_without_offer_tool", "weight": 0.2})

    if case.expected_status == "not_found" and "not found" not in lowered:
        penalties.append({"rule": "missing_not_found_grounding", "weight": 0.1})

    penalty_total = sum(item["weight"] for item in penalties)
    score = max(0.0, 1.0 - penalty_total)

    return MetricResult(
        name="hallucination_detection",
        score=round(score, 4),
        details={
            "penalties": penalties,
            "detected_tools": tools,
        },
    )


def evaluate_all_metrics(case: EvaluationCase, actual_output: dict[str, Any]) -> dict[str, MetricResult]:
    results = [
        tool_selection_accuracy(case, actual_output),
        parameter_extraction_accuracy(case, actual_output),
        response_completeness(case, actual_output),
        hallucination_detection(case, actual_output),
    ]
    return {result.name: result for result in results}
