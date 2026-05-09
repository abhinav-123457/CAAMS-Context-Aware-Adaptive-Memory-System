# agents/supervisor_process.py
# Runs as: python -m agents.supervisor_process --step 1 --app chrome --prev_app gmail --hour 14
#
# Reads:  MCP telemetry (get_telemetry_report, get_memory_snapshot)
#         MCP pipeline state: "telemetry_output" (from previous TelemetryAgent run)
# Writes: MCP pipeline state: "supervisor_directive" (consumed by ContextAgent)
# License: Apache 2.0

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_llm import get_local_llm, parse_json_response
from langchain_core.messages import SystemMessage, HumanMessage

import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

MCP_URL = os.getenv("CAAMS_MCP_URL", "http://127.0.0.1:8765/sse")


async def _call(tool: str, args: dict):
    async with sse_client(MCP_URL) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.content:
                try:
                    return json.loads(result.content[0].text)
                except Exception:
                    return {"_raw": result.content[0].text}
    return {}


def mcp(tool: str, args: dict) -> dict:
    return asyncio.run(_call(tool, args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",        type=int,   required=True)
    parser.add_argument("--app",         type=str,   required=True)
    parser.add_argument("--prev_app",    type=str,   required=True)
    parser.add_argument("--hour",        type=int,   required=True)
    parser.add_argument("--pressure",    type=float, default=0.3)
    parser.add_argument("--intensity",   type=float, default=0.5)
    parser.add_argument("--loop_count",  type=int,   default=0)
    args = parser.parse_args()

    print(f"[SupervisorAgent PID={os.getpid()}] step={args.step} app={args.app}")

    # Read live telemetry from MCP
    telemetry = mcp("get_telemetry_report", {})
    snap      = mcp("get_memory_snapshot",  {})

    hit_rate = float(telemetry.get("cache_hit_rate_pct", 100.0))
    drift    = telemetry.get("drift_flags", [])
    free_pct = float(snap.get("free_pct", 100.0))

    # Read previous TelemetryAgent recommendation (inter-process message)
    prev_raw = mcp("get_pipeline_state", {"key": "telemetry_output"})
    prev_rec = "none"
    if prev_raw.get("value"):
        try:
            prev_rec = json.loads(prev_raw["value"]).get("recommendation", "none")
        except Exception:
            pass

    print(f"  [MCP] hit_rate={hit_rate}% free={free_pct}% drift={drift}")
    print(f"  [MCP] prev_recommendation={prev_rec}")

    # Call Qwen to produce directive
    llm = get_local_llm()
    directive = {}

    if llm is not None:
        try:
            resp = llm.invoke([
                SystemMessage(content=(
                    "You are the Supervisor of a Samsung on-device memory manager. "
                    "Reply ONLY with valid JSON, no markdown."
                )),
                HumanMessage(content=(
                    f"free_pct={free_pct:.1f}, query_pressure={args.pressure:.3f}, "
                    f"chronos_intensity={args.intensity:.3f}, "
                    f"hit_rate={hit_rate}, drift={drift}, "
                    f"prev_recommendation=\"{prev_rec}\", "
                    f"loop_count={args.loop_count}, current_app=\"{args.app}\"\n\n"
                    "Rules:\n"
                    "- path=\"cold\" if free_pct<25 or query_pressure>0.85\n"
                    "- top_k=5 and max_preloads=3 if hit_rate<75\n"
                    "- eviction_urgency=\"critical\" if free_pct<15\n"
                    "- route_after_memory=\"supervisor\" if hit_rate<85 and loop_count<2\n"
                    "- route_after_memory=\"telemetry\" otherwise\n\n"
                    "Return JSON:\n"
                    "{\n"
                    "  \"context_task\": \"predict_standard|predict_aggressive|predict_conservative\",\n"
                    "  \"top_k\": <1-5>,\n"
                    "  \"memory_task\": \"preload_predicted|evict_and_preload|evict_only|hold\",\n"
                    "  \"max_preloads\": <0-3>,\n"
                    "  \"eviction_urgency\": \"none|low|high|critical\",\n"
                    f"  \"protect_apps\": [\"{args.app}\"],\n"
                    "  \"route_after_memory\": \"telemetry|supervisor\",\n"
                    "  \"reason\": \"<one sentence>\",\n"
                    f"  \"detected_drift\": {drift},\n"
                    "  \"path\": \"hot|cold\"\n"
                    "}"
                )),
            ])
            directive = parse_json_response(resp)
            print(f"  [Qwen] directive={json.dumps(directive, indent=2)}")
        except Exception as e:
            print(f"  [Qwen] failed: {e} — using deterministic fallback")

    # Deterministic fallback
    if not directive or "_error" in directive:
        is_cold = free_pct < 25 or args.pressure > 0.85
        directive = {
            "context_task":      "predict_aggressive" if hit_rate < 75 else "predict_standard",
            "top_k":             5 if hit_rate < 75 else 3,
            "memory_task":       "evict_only" if free_pct < 15 else
                                 ("evict_and_preload" if is_cold else "preload_predicted"),
            "max_preloads":      0 if free_pct < 15 else (1 if is_cold else (3 if hit_rate < 75 else 2)),
            "eviction_urgency":  "critical" if free_pct < 15 else ("high" if free_pct < 25 else "none"),
            "protect_apps":      [args.app],
            "route_after_memory": "telemetry" if (args.loop_count >= 2 or hit_rate >= 85) else "supervisor",
            "reason":            f"[deterministic] free={free_pct:.1f}% hit={hit_rate}%",
            "detected_drift":    drift,
            "path":              "cold" if is_cold else "hot",
        }
        print(f"  [fallback] directive={directive['reason']}")

    # Write directive to MCP pipeline state — ContextAgent will read this
    mcp("set_pipeline_state", {
        "key":   "supervisor_directive",
        "value": json.dumps(directive),
    })
    print(f"  [MCP] wrote supervisor_directive -> ContextAgent will read this")
    print(f"  [SupervisorAgent] done | path={directive['path']} task={directive['context_task']}")


if __name__ == "__main__":
    main()
