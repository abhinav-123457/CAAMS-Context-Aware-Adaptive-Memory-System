# multi_agent_orchestrator.py — CAAMS Shared Skill Registry
#
# PURPOSE: Provides a fail-safe skill execution boundary used by:
#   - selftest_skill_failsafe.py  (validates skill error handling)
#   - selftest_graph_failsafe.py  (validates LangGraph continuity under failure)
#   - memory_manager.py           (evaluation harness)
#
# This is NOT the production agent runtime.
# Production agents run as separate OS processes via pipeline_runner.py.
# Each agent communicates exclusively via MCP HTTP — see agents/ directory.
#
# License: Apache 2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Callable


SkillFn = Callable[[dict], dict]


class Agent(Protocol):
    name: str

    def step(self, state: dict) -> dict: ...


@dataclass
class SkillRegistry:
    """
    Shared skill registry for CAAMS prototype agents.
    A single runtime catalog enables real-agent style reusable skill calls.
    """
    skills: dict[str, SkillFn]
    primary_owner: dict[str, str]
    cross_agent_allowed: bool = True

    def has(self, skill_name: str) -> bool:
        return skill_name in self.skills

    def get(self, skill_name: str) -> SkillFn:
        if skill_name not in self.skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return self.skills[skill_name]

    def can_call(self, skill_name: str, caller_agent: str) -> bool:
        if not self.has(skill_name):
            return False
        owner = self.primary_owner.get(skill_name)
        if caller_agent == owner:
            return True
        return self.cross_agent_allowed


@dataclass
class SkillExecutor:
    """
    Fail-safe executor:
    - Never throws into the graph runtime.
    - Returns deterministic fallback state on any skill failure.
    """
    registry: SkillRegistry

    def execute(self, skill_name: str, caller_agent: str, input_state: dict) -> dict:
        state = dict(input_state)
        state.setdefault("skill_trace", [])
        trace_entry = {"skill": skill_name, "caller": caller_agent, "status": "ok"}

        if not self.registry.has(skill_name):
            trace_entry["status"] = "fallback"
            trace_entry["reason"] = "unknown_skill"
            state["skill_trace"].append(trace_entry)
            state["skill_fallback"] = {
                "skill": skill_name,
                "reason": "unknown_skill",
                "caller": caller_agent,
            }
            return state

        if not self.registry.can_call(skill_name, caller_agent):
            trace_entry["status"] = "fallback"
            trace_entry["reason"] = "permission_denied"
            state["skill_trace"].append(trace_entry)
            state["skill_fallback"] = {
                "skill": skill_name,
                "reason": "permission_denied",
                "caller": caller_agent,
            }
            return state

        fn = self.registry.get(skill_name)
        try:
            result = fn(state)
            if not isinstance(result, dict):
                trace_entry["status"] = "fallback"
                trace_entry["reason"] = "invalid_skill_output"
                state["skill_trace"].append(trace_entry)
                state["skill_fallback"] = {
                    "skill": skill_name,
                    "reason": "invalid_skill_output",
                    "caller": caller_agent,
                }
                return state
            result.setdefault("skill_trace", state.get("skill_trace", []))
            result["skill_trace"].append(trace_entry)
            return result
        except Exception as exc:  # defensive runtime boundary
            trace_entry["status"] = "fallback"
            trace_entry["reason"] = "exception"
            trace_entry["error"] = str(exc)
            state["skill_trace"].append(trace_entry)
            state["skill_fallback"] = {
                "skill": skill_name,
                "reason": "exception",
                "caller": caller_agent,
                "error": str(exc),
            }
            return state


@dataclass
class ContextPredictorAgent:
    """
    Owns context inference: predictions + intensity signals + path decision.
    """
    name: str = "context_predictor"
    cp_assess_context_fn: Any = None
    skill_executor: SkillExecutor | None = None

    def step(self, state: dict) -> dict:
        if self.skill_executor is not None:
            return self.skill_executor.execute(
                skill_name="context_window_maintenance",
                caller_agent=self.name,
                input_state=state,
            )
        return self.cp_assess_context_fn(state)


@dataclass
class MemoryAllocationAgent:
    """
    Owns planning + execution: rule engine, optional cold reasoning, execute, validate.
    """
    name: str = "memory_allocator"
    rule_engine_fn: Any = None
    qwen_eviction_fn: Any = None
    execute_fn: Any = None
    validate_fn: Any = None
    skill_executor: SkillExecutor | None = None

    def rule_engine(self, state: dict) -> dict:
        if self.skill_executor is not None:
            return self.skill_executor.execute(
                skill_name="preload_candidate_ranking",
                caller_agent=self.name,
                input_state=state,
            )
        return self.rule_engine_fn(state)

    def qwen_eviction(self, state: dict) -> dict:
        if self.skill_executor is not None:
            return self.skill_executor.execute(
                skill_name="adaptive_eviction_policy",
                caller_agent=self.name,
                input_state=state,
            )
        return self.qwen_eviction_fn(state)

    def execute(self, state: dict) -> dict:
        return self.execute_fn(state)

    def validate(self, state: dict) -> dict:
        return self.validate_fn(state)


@dataclass
class Supervisor:
    """
    Chooses which agent owns the next step.
    In the current CAAMS design, we always assess context first, then allocate.
    """
    name: str = "supervisor"
    skill_executor: SkillExecutor | None = None

    def run_skill(self, skill_name: str, state: dict) -> dict:
        if self.skill_executor is None:
            return state
        return self.skill_executor.execute(
            skill_name=skill_name,
            caller_agent=self.name,
            input_state=state,
        )

    def dispatch(self, state: dict) -> str:
        # Mirrors existing behavior: if we haven't predicted yet, go to context.
        if not state.get("predicted_apps"):
            return "context_predictor"
        return "memory_allocator"

