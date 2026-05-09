"""
Fail-safe skill runtime self-test for CAAMS.

Goal:
- Prove shared skills can fail without crashing prototype control flow.
- Demonstrate fallback behavior and trace emission.
"""

from multi_agent_orchestrator import SkillRegistry, SkillExecutor, Supervisor


def _ok_skill(state: dict) -> dict:
    out = dict(state)
    out["ok_ran"] = True
    return out


def _bad_skill_exception(state: dict) -> dict:
    raise RuntimeError("simulated skill failure")


def _bad_skill_output(state: dict):
    return "not_a_dict"


def main() -> int:
    registry = SkillRegistry(
        skills={
            "ok_skill": _ok_skill,
            "bad_exception": _bad_skill_exception,
            "bad_output": _bad_skill_output,
        },
        primary_owner={
            "ok_skill": "supervisor",
            "bad_exception": "supervisor",
            "bad_output": "supervisor",
        },
        cross_agent_allowed=True,
    )
    executor = SkillExecutor(registry=registry)
    supervisor = Supervisor(skill_executor=executor)

    base_state = {"predicted_apps": [], "query_pressure": 0.2}

    # 1) Happy path
    s1 = supervisor.run_skill("ok_skill", base_state)
    assert s1.get("ok_ran") is True, "ok_skill should set ok_ran=True"
    assert s1.get("skill_trace"), "ok_skill should append skill_trace"
    assert s1["skill_trace"][-1]["status"] == "ok"

    # 2) Unknown skill should not crash
    s2 = supervisor.run_skill("does_not_exist", base_state)
    assert "skill_fallback" in s2, "unknown skill should trigger fallback metadata"
    assert s2["skill_fallback"]["reason"] == "unknown_skill"

    # 3) Skill exception should not crash
    s3 = supervisor.run_skill("bad_exception", base_state)
    assert "skill_fallback" in s3, "exception should trigger fallback metadata"
    assert s3["skill_fallback"]["reason"] == "exception"

    # 4) Invalid output should not crash
    s4 = supervisor.run_skill("bad_output", base_state)
    assert "skill_fallback" in s4, "invalid output should trigger fallback metadata"
    assert s4["skill_fallback"]["reason"] == "invalid_skill_output"

    # 5) Dispatch still works after failures
    dispatch_state = dict(s4)
    next_agent = supervisor.dispatch(dispatch_state)
    assert next_agent in {"context_predictor", "memory_allocator"}

    print("PASS: fail-safe skill runtime validated")
    print(" - success path: ok")
    print(" - unknown skill fallback: ok")
    print(" - exception fallback: ok")
    print(" - invalid output fallback: ok")
    print(f" - dispatch continuity: ok ({next_agent})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

