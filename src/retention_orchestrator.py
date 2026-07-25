"""Compatibility wrapper for retention orchestrator.

Main orchestration logic was modularized under src/retention_agent/ to keep files
short, reusable, and easier to monitor.
"""

from src.retention_agent.orchestrator import run_retention_orchestrator

__all__ = ["run_retention_orchestrator"]
