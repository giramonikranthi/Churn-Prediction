from __future__ import annotations

from typing import Any

from src.llm_tools import dispatch_tool_call
from src.retention_agent.runtime import log_step
from src.retention_agent.schema_guardrails import TOOL_ARG_SCHEMAS, TOOL_RESULT_SCHEMAS, validate_schema


def tool_call(
    session_id: str,
    tool_trace: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    arg_schema = TOOL_ARG_SCHEMAS.get(tool_name)
    normalized_arguments = arguments
    if arg_schema is not None:
        args_ok, normalized_arguments, args_error = validate_schema(arguments, arg_schema)
        if not args_ok:
            log_step(
                session_id,
                "tool_call_rejected",
                {
                    "tool": tool_name,
                    "reason": "argument_schema_validation_failed",
                    "error": args_error,
                },
            )
            rejected_result = {
                "status": "error",
                "error": f"Argument schema validation failed for {tool_name}: {args_error}",
            }
            tool_trace.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": rejected_result,
                }
            )
            return rejected_result

    log_step(
        session_id,
        "tool_call_requested",
        {
            "tool": tool_name,
            "arguments": normalized_arguments,
        },
    )
    result = dispatch_tool_call(tool_name, normalized_arguments)

    result_schema = TOOL_RESULT_SCHEMAS.get(tool_name)
    if result_schema is not None and isinstance(result, dict):
        result_ok, normalized_result, result_error = validate_schema(result, result_schema)
        if result_ok:
            result = normalized_result
        else:
            result = {
                "status": "error",
                "error": f"Result schema validation failed for {tool_name}: {result_error}",
                "raw_result": result,
            }

    log_step(
        session_id,
        "tool_call_completed",
        {
            "tool": tool_name,
            "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
        },
    )
    tool_trace.append(
        {
            "tool": tool_name,
            "arguments": normalized_arguments,
            "result": result,
        }
    )
    return result
