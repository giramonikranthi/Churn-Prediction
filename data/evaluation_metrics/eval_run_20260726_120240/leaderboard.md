# Evaluation Leaderboard

Run At (IST): 2026-07-26 12:02:51

## Overall

- Total Cases: 14
- Avg Tool Selection Accuracy: 0.9893
- Avg Parameter Extraction Accuracy: 1.0
- Avg Response Completeness: 0.9583
- Avg Hallucination Detection: 1.0
- Avg Latency (s): 0.76
- Judge Weighted Avg: None

## Quality Gates

- Pass: False
- Hard Failures: 0
- Soft Failures: 1

## Case Ranking

| Rank | Case ID | Category | Auto Score | Judge Score | Status |
|---|---|---|---:|---:|---|
| 1 | C01_SINGLE_HAPPY | single_tool_happy_path | 100.0 | - | ok |
| 2 | C02_CHAINING_MULTI_STEP | multi_step_chaining | 100.0 | - | ok |
| 3 | C03_CHAINING_ALT_ID_SOURCE | multi_step_chaining | 100.0 | - | ok |
| 4 | C04_AMBIGUOUS_MISSING_ID | ambiguous_input | 100.0 | - | needs_clarification |
| 5 | C05_AMBIGUOUS_TWO_IDS | ambiguous_input | 100.0 | - | ok |
| 6 | C06_OUT_OF_SCOPE_TECH | out_of_scope_request | 100.0 | - | out_of_scope |
| 7 | C07_ESCALATION_LEGAL | escalation_trigger | 100.0 | - | ok |
| 8 | C08_ESCALATION_SUPERVISOR_REQUEST | escalation_trigger | 100.0 | - | ok |
| 9 | C10_ADVERSARIAL_PROMPT_INJECTION | adversarial | 100.0 | - | ok |
| 10 | C11_ADVERSARIAL_TOOL_RESULT_CLAIM | adversarial | 100.0 | - | ok |
| 11 | C12_EDGE_NOT_FOUND | edge_case | 100.0 | - | not_found |
| 12 | C14_EDGE_LONG_MULTI_INTENT | edge_case | 100.0 | - | ok |
| 13 | C13_EDGE_GREETING_ONLY | edge_case | 93.33 | - | small_talk |
| 14 | C09_MODEL_DISAGREEMENT | model_disagreement | 91.25 | - | ok |

## Category Summary

| Category | Cases | Tool Acc | Param Acc | Completeness | Hallucination |
|---|---:|---:|---:|---:|---:|
| single_tool_happy_path | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| multi_step_chaining | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| ambiguous_input | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| out_of_scope_request | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| escalation_trigger | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_disagreement | 1 | 0.85 | 1.0 | 0.75 | 1.0 |
| adversarial | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| edge_case | 3 | 1.0 | 1.0 | 0.8889 | 1.0 |

## Lowest Scoring Cases

- C09_MODEL_DISAGREEMENT (model_disagreement): auto_score=91.25
- C13_EDGE_GREETING_ONLY (edge_case): auto_score=93.33
- C01_SINGLE_HAPPY (single_tool_happy_path): auto_score=100.0
