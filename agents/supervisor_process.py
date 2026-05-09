# agents/supervisor_process.py — CAAMS SupervisorAgent (separate OS process)
#
# Called by pipeline_runner.py as a subprocess per step.
# Reads live MCP telemetry → calls Qwen for directive → writes to MCP pipeline state.
# Qwen is ONLY called here, when device is NOT under critical memory pressure.
# License: Apache 2.0

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.client.sse import sse_client
from mcp import ClientSession

MCP_URL = os.getenv("CAAMS_MCP_URL", "http://127.0.0.1:8765/sse")


async def _call(tool: str, args: dict) -> dict:
    async with sse_client(MCP_URL) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.content:
                try:
                    return json.loads(result.content[0].text)
                except Exception:
                    return {}
    return {}


def mcp(tool: str, args: dict) -> dict:
    return asyncio.run(_call(tool, args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",       type=int,   required=True)
    parser.add_argument("--app",        type=str,   required=True)
    parser.add_argument("--prev_app",   type=str,   required=True)
    parser.add_argument("--hour",       type=int,   required=True)
    parser.add_argument("--pressure",   type=float, default=0.3)
    parser.add_argument("--intensity",  type=float, default=0.5)
    parser.add_argument("--loop_count", type=int,   default=0)
    args = parser.parse_args()

    print(f"[SupervisorAgent PID={os.getpid()}] step={args.step} app={args.app}")

    telemetry = mcp("get_telemetry_report", {})
    snap      = mcp("get_memory_snapshot",  {})

    hit_rate = float(telemetry.get("cache_hit_rate_pct", 100.0))
    drift    = telemetry.get("drift_flags", [])
    free_pct = float(snap.get("free_pct", 100.0))

    print(f"  MCP: hit_rate={hit_rate}% free={free_pct}% drift={drift}")

    # --- Architectural constraint: LLM is only safe when NOT under pressure ---
    # If free_pct < 25, device is memory-starved. Running Qwen here risks OOM.
    # In that regime, MemoryAgent uses RL Q-table exclusively (< 1ms).
    # Qwen is reserved for idle-time advisory only.
    llm_safe = free_pct >= 25.0 and args.pressure <= 0.85

    qwen_result = {}
    if llm_safe:
        try:
            from local_llm import get_local_llm, parse_json_response
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = get_local_llm()
            if llm:
                resp = llm.invoke([
                    SystemMessage(content=(
                        "You are the Supervisor of a Samsung on-device memory manager. "
                        "Reply ONLY with valid JSON, no markdown."
                    )),
                    HumanMessage(content=(
                        f"free_pct={free_pct:.1f}, pressure={args.pressure:.3f}, "
                        f"intensity={args.intensity:.3f}, hit_rate={hit_rate}, "
                        f"drift={drift}, loop_count={args.loop_count}, "
                        f"app={args.app!r}. "
                        'Return: {"context_task": "predict_standard|predict_aggressive", '
                        '"top_k": 3, "memory_task": "preload_predicted|evict_and_preload", '
                        '"max_preloads": 2, "eviction_urgency": "none|low|high", '
                        '"path": "hot|cold", "reason": "one sentence"}'
                    )),
                ])
                qwen_result = parse_json_response(resp)
                print(f"  [Qwen] {qwen_result}")
        except Exception as e:
            print(f"  [Qwen] failed: {e} → deterministic fallback")

    # --- Deterministic fallback (always used if Qwen unavailable or unsafe) ---
    is_cold   = free_pct < 25 or args.pressure > 0.85
    top_k     = 5 if hit_rate < 75 else 3
    max_pre   = 0 if free_pct < 15 else (1 if is_cold else (3 if hit_rate < 75 else 2))
    ev_urg    = "high" if free_pct < 25 else ("low" if free_pct < 40 else "none")

    directive = {
        "context_task":       qwen_result.get("context_task", "predict_standard"),
        "top_k":              int(qwen_result.get("top_k", top_k)),
        "memory_task":        qwen_result.get("memory_task",
                                "evict_and_preload" if is_cold else "preload_predicted"),
        "max_preloads":       int(qwen_result.get("max_preloads", max_pre)),
        "eviction_urgency":   qwen_result.get("eviction_urgency", ev_urg),
        "protect_apps":       [args.app],
        "path":               qwen_result.get("path", "cold" if is_cold else "hot"),
        "reason":             qwen_result.get("reason",
                                f"[fallback] free={free_pct:.1f}% pressure={args.pressure:.2f}"),
        "llm_used":           llm_safe and bool(qwen_result),
    }

    print(f"  Directive: {directive}")

    mcp("set_pipeline_state", {
        "key":   "supervisor_directive",
        "value": json.dumps(directive),
    })
    print(f"[SupervisorAgent] Done — directive written to MCP pipeline state")


if __name__ == "__main__":
    main()