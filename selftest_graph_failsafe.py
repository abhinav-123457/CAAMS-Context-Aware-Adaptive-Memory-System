"""
Minimal LangGraph integration self-test for CAAMS fail-safe skills.

This test intentionally fails a shared skill during graph execution and verifies
the graph still reaches the terminal node.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from multi_agent_orchestrator import (
    SkillRegistry,
    SkillExecutor,
    Supervisor,
    ContextPredictorAgent,
    MemoryAllocationAgent,
)


class TestState(TypedDict, total=False):
    predicted_apps: list
    active_agent: str
    executed: bool
    skill_trace: list
    skill_fallback: dict


def _triage_ok(state: dict) -> dict:
    out = dict(state)
    out.setdefault("predicted_apps", [])
    return out


def _context_skill_fails(state: dict) -> dict:
    raise RuntimeError("forced context skill failure")


def _execute_node(state: dict) -> dict:
    out = dict(state)
    out["executed"] = True
    return out


def main() -> int:
    registry = SkillRegistry(
        skills={
            "memory_pressure_triage": _triage_ok,
            "context_window_maintenance": _context_skill_fails,
            "preload_candidate_ranking": _triage_ok,
        },
        primary_owner={
            "memory_pressure_triage": "supervisor",
            "context_window_maintenance": "context_predictor",
            "preload_candidate_ranking": "memory_allocator",
        },
        cross_agent_allowed=True,
    )
    executor = SkillExecutor(registry)

    supervisor = Supervisor(skill_executor=executor)
    cp_agent = ContextPredictorAgent(
        cp_assess_context_fn=lambda s: s,
        skill_executor=executor,
    )
    ma_agent = MemoryAllocationAgent(
        rule_engine_fn=lambda s: s,
        qwen_eviction_fn=lambda s: s,
        execute_fn=_execute_node,
        validate_fn=lambda s: s,
        skill_executor=executor,
    )

    graph = StateGraph(TestState)

    def _supervisor_node(state: TestState) -> TestState:
        state = supervisor.run_skill("memory_pressure_triage", state)
        state["active_agent"] = supervisor.dispatch(state)
        return state

    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("cp_assess_context", cp_agent.step)
    graph.add_node("ma_execute", ma_agent.execute)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "cp_assess_context")
    graph.add_edge("cp_assess_context", "ma_execute")
    graph.add_edge("ma_execute", END)

    app = graph.compile()
    result = app.invoke(TestState(predicted_apps=[]))

    # Graph should complete despite forced skill failure in cp_assess_context.
    assert result.get("executed") is True, "Graph did not reach execute node"
    assert "skill_fallback" in result, "Expected fallback metadata missing"
    assert result["skill_fallback"]["skill"] == "context_window_maintenance"
    assert result["skill_fallback"]["reason"] == "exception"

    print("PASS: LangGraph continuity under forced skill failure")
    print(" - forced skill failure captured with fallback metadata")
    print(" - graph reached ma_execute and completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

