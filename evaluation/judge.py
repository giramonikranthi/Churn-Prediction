from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from evaluation.metrics import MetricResult
from evaluation.test_suite import EvaluationCase
from src.load_llm_models import load_groq_client_and_models


@dataclass(frozen=True)
class JudgeDimensionScore:
    score: int
    rationale: str
    evidence: list[str]


@dataclass(frozen=True)
class JudgeResult:
    status: str
    scores: dict[str, JudgeDimensionScore]
    weighted_total: float
    confidence: str
    failure_modes: list[str]
    improvement_actions: list[str]
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = {
            key: asdict(value) for key, value in self.scores.items()
        }
        return payload


DIMENSIONS = [
    "factual_correctness",
    "tool_use_appropriateness",
    "actionability_for_representative",
    "hallucination_control",
]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _build_rubric_prompt(
    case: EvaluationCase,
    actual_output: dict[str, Any],
    metric_snapshot: dict[str, MetricResult],
) -> str:
    metrics_payload = {
        name: {
            "score": metric.score,
            "details": metric.details,
        }
        for name, metric in metric_snapshot.items()
    }

    return (
        "You are a strict evaluator for a retention-assistant agent. "
        "Return JSON only. No markdown.\n\n"
        "Scoring rubric per dimension uses anchored integer scale 0-4.\n"
        "IMPORTANT: Apply anchors independently for EACH dimension, not globally.\n\n"
        "Dimension anchors:\n"
        "factual_correctness:\n"
        "- 0: core facts contradict tool_trace or invent unsupported facts\n"
        "- 2: partially correct; at least one meaningful factual gap\n"
        "- 4: all material facts align with tool outputs and case context\n"
        "tool_use_appropriateness:\n"
        "- 0: missing critical tools or wrong tool path for scenario\n"
        "- 2: mostly right path with at least one avoidable tool mistake\n"
        "- 4: correct tool set, order, and intent for the scenario\n"
        "actionability_for_representative:\n"
        "- 0: non-actionable, vague, or unusable for rep handoff\n"
        "- 2: partially actionable but missing key next-step details\n"
        "- 4: clear, concise, directly usable next actions for the rep\n"
        "hallucination_control:\n"
        "- 0: fabricated claims (risk, offer, escalation, outcomes) not in evidence\n"
        "- 2: minor speculative language or weakly grounded claims\n"
        "- 4: no unsupported claims; statements traceable to evidence\n\n"
        "Evaluate these dimensions: factual_correctness, tool_use_appropriateness, "
        "actionability_for_representative, hallucination_control.\n"
        "Use evidence from tool_trace and response text.\n"
        "Provide concise rationale and evidence snippets for each dimension.\n"
        "If evidence is insufficient, score conservatively and mention uncertainty.\n"
        "Include failure_modes and improvement_actions as short lists.\n"
        "Compute weighted_total in [0,100] using weights: factual_correctness=0.35, "
        "tool_use_appropriateness=0.25, actionability_for_representative=0.25, "
        "hallucination_control=0.15.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"factual_correctness\": {\"score\": 0, \"rationale\": \"\", \"evidence\": [\"\"]},\n"
        "    \"tool_use_appropriateness\": {\"score\": 0, \"rationale\": \"\", \"evidence\": [\"\"]},\n"
        "    \"actionability_for_representative\": {\"score\": 0, \"rationale\": \"\", \"evidence\": [\"\"]},\n"
        "    \"hallucination_control\": {\"score\": 0, \"rationale\": \"\", \"evidence\": [\"\"]}\n"
        "  },\n"
        "  \"weighted_total\": 0.0,\n"
        "  \"confidence\": \"low|medium|high\",\n"
        "  \"failure_modes\": [\"\"],\n"
        "  \"improvement_actions\": [\"\"]\n"
        "}\n\n"
        f"TEST_CASE={json.dumps(case.to_dict(), ensure_ascii=True)}\n"
        f"ACTUAL_OUTPUT={json.dumps(actual_output, ensure_ascii=True)}\n"
        f"AUTOMATED_METRICS={json.dumps(metrics_payload, ensure_ascii=True)}"
    )


def _validate_score_block(raw: dict[str, Any], key: str) -> JudgeDimensionScore:
    block = raw.get(key, {})
    if not isinstance(block, dict):
        block = {}

    score = block.get("score", 0)
    if not isinstance(score, int):
        score = int(float(score)) if isinstance(score, (float, str)) else 0
    score = max(0, min(4, score))

    rationale = str(block.get("rationale") or "")
    evidence = block.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(item) for item in evidence][:5]

    return JudgeDimensionScore(score=score, rationale=rationale, evidence=evidence)


def _weighted_total_from_scores(scores: dict[str, JudgeDimensionScore]) -> float:
    weights = {
        "factual_correctness": 0.35,
        "tool_use_appropriateness": 0.25,
        "actionability_for_representative": 0.25,
        "hallucination_control": 0.15,
    }
    total = 0.0
    for key, weight in weights.items():
        total += (scores[key].score / 4.0) * 100.0 * weight
    return round(total, 2)


def judge_case_with_llm(
    case: EvaluationCase,
    actual_output: dict[str, Any],
    metric_snapshot: dict[str, MetricResult],
) -> JudgeResult:
    try:
        client, _, model_judge = load_groq_client_and_models()
    except Exception as exc:
        return JudgeResult(
            status="skipped",
            scores={key: JudgeDimensionScore(score=0, rationale="judge skipped", evidence=[]) for key in DIMENSIONS},
            weighted_total=0.0,
            confidence="low",
            failure_modes=["judge_unavailable"],
            improvement_actions=["Set GROQ_API_KEY and GROQ_MODEL_JUDGE to enable LLM-as-judge."],
            error=str(exc),
        )

    prompt = _build_rubric_prompt(case=case, actual_output=actual_output, metric_snapshot=metric_snapshot)

    try:
        response = client.chat.completions.create(
            model=model_judge,
            messages=[
                {
                    "role": "system",
                    "content": "Return strict JSON only for evaluation outputs.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw_text = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return JudgeResult(
            status="error",
            scores={key: JudgeDimensionScore(score=0, rationale="judge error", evidence=[]) for key in DIMENSIONS},
            weighted_total=0.0,
            confidence="low",
            failure_modes=["judge_call_failed"],
            improvement_actions=["Retry with stable network and valid model configuration."],
            error=str(exc),
        )

    parsed = _extract_json_object(raw_text)
    if parsed is None:
        return JudgeResult(
            status="error",
            scores={key: JudgeDimensionScore(score=0, rationale="invalid judge json", evidence=[]) for key in DIMENSIONS},
            weighted_total=0.0,
            confidence="low",
            failure_modes=["judge_invalid_json"],
            improvement_actions=["Constrain judge model output format and retry."],
            raw_response=raw_text,
        )

    raw_scores = parsed.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {key: _validate_score_block(raw_scores, key) for key in DIMENSIONS}
    weighted_total = parsed.get("weighted_total")
    if not isinstance(weighted_total, (int, float)):
        weighted_total = _weighted_total_from_scores(scores)
    weighted_total = round(float(max(0.0, min(100.0, weighted_total))), 2)

    confidence = str(parsed.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    failure_modes = parsed.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        failure_modes = [str(failure_modes)]

    improvement_actions = parsed.get("improvement_actions", [])
    if not isinstance(improvement_actions, list):
        improvement_actions = [str(improvement_actions)]

    return JudgeResult(
        status="ok",
        scores=scores,
        weighted_total=weighted_total,
        confidence=confidence,
        failure_modes=[str(item) for item in failure_modes][:10],
        improvement_actions=[str(item) for item in improvement_actions][:10],
        raw_response=raw_text,
    )
