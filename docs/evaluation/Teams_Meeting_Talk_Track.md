# Teams Meeting Talk Track (10-15 Minutes)

## Slide/Section 1: Problem statement (1 minute)
- We need confidence that the retention agent is correct, safe, and actionable.
- Manual spot checks are not enough.
- We added a repeatable evaluation pipeline.

## Slide/Section 2: Framework overview (2 minutes)
- Structured test suite with 14 representative scenarios.
- Automated metrics for deterministic quality checks.
- LLM-as-judge for nuanced response evaluation.
- Quality gates for release readiness.

## Slide/Section 3: Scenario coverage (2 minutes)
- Walk through the 8 scenario categories.
- Highlight high-risk buckets: escalation, adversarial, model disagreement.
- Explain why each category matters operationally.

## Slide/Section 4: Scoring model (3 minutes)
### Automated metrics
- status_match
- tool_selection_accuracy
- parameter_extraction_accuracy
- response_completeness
- hallucination_detection

### Judge rubric (0-4 per dimension)
- factual_correctness
- tool_use_appropriateness
- actionability_for_representative
- hallucination_control

Weighted judge total is normalized to 0-100.

## Slide/Section 5: Governance with quality gates (2 minutes)
- Hard failures block confidence (safety-critical checks).
- Soft failures flag improvement targets.
- Category-level thresholds prevent average-score masking.

## Slide/Section 6: Demo flow (2 minutes)
1. Run evaluation command.
2. Open aggregate_report.json for metrics and gates.
3. Open leaderboard.md for quick narrative summary.
4. Show one strong case and one weak case (for learning loop).

## Slide/Section 7: What we learned already (2 minutes)
- Most categories score strongly on deterministic checks.
- Model disagreement category is useful for exposing policy/model conflicts.
- Framework is now suitable for regression tracking before changes.

## Quick commands for live demo
```powershell
python scripts/run_evaluation.py --max-cases 5
python scripts/run_evaluation.py --use-judge --max-cases 3
```

## Q&A prompts you can use
- Which category should we treat as release-blocking?
- Do we want stricter thresholds for escalation cases?
- Should judge score be mandatory or advisory for CI?
- Which failures should auto-create improvement tickets?

## Suggested next team action
- Agree on threshold policy by category.
- Add 5 domain-specific scenarios from real calls.
- Start weekly trend review from leaderboard and gate results.
