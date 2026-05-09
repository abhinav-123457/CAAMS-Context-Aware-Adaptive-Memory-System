# agents/telemetry_agent_process.py
# Reads:  MCP pipeline state: "memory_output", "supervisor_directive"
# Writes: MCP pipeline state: "telemetry_output"
#         MCP record_telemetry (persistent log)
# License: Apache 2.0

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    parser.add_argument("--step",     type=int, required=True)
    parser.add_argument("--next_app", type=str, required=True)
    args = parser.parse_args()

    print(f"[TelemetryAgent PID={os.getpid()}] step={args.step} next_app={args.next_app}")

    # Read both inputs from MCP
    mem_raw = mcp("get_pipeline_state", {"key": "memory_output"})
    dir_raw = mcp("get_pipeline_state", {"key": "supervisor_directive"})

    if not mem_raw.get("value"):
        print("  [ERROR] No memory_output in MCP state.")
        sys.exit(1)

    mem_out   = json.loads(mem_raw["value"])
    directive = json.loads(dir_raw["value"]) if dir_raw.get("value") else {}

    print(f"  [MCP] read memory_output: evictions={[e['app'] for e in mem_out.get('evictions',[])]}")
    print(f"  [MCP] read memory_output: preloads={[p['app'] for p in mem_out.get('preloads',[])]}")

    # Check if next app is resident
    snap     = mcp("get_memory_snapshot", {})
    resident = (set(snap.get("allocated_apps", {}).keys()) |
                set(snap.get("preloaded_apps", {}).keys()))
    is_hit   = args.next_app in resident

    # Write step to persistent MCP telemetry log
    mcp("record_telemetry", {
        "step":       args.step,
        "hit":        is_hit,
        "path":       directive.get("path", "hot"),
        "latency_ms": 0.0,
        "util_pct":   mem_out.get("util_pct", 0.0),
        "thrash":     False,
        "notes":      f"preloads={[p['app'] for p in mem_out.get('preloads', [])]}",
    })

    # Read aggregate from MCP
    aggregate = mcp("get_telemetry_report", {})
    hit_rate  = float(aggregate.get("cache_hit_rate_pct", 100.0))
    drift     = aggregate.get("drift_flags", [])

    # Produce recommendation for Supervisor next round
    if hit_rate < 75:
        recommendation = "increase_top_k_and_preloads"
    elif hit_rate < 85:
        recommendation = "increase_preloads"
    elif mem_out.get("free_pct_after", 100) < 20:
        recommendation = "trigger_eviction"
    else:
        recommendation = "maintain_current_policy"

    icon = "HIT" if is_hit else "MISS"
    print(f"  [{icon}] next_app={args.next_app!r}")
    print(f"  [MCP] aggregate hit_rate={hit_rate}% drift={drift}")
    print(f"  [MCP] recommendation -> {recommendation}")

    telemetry_output = {
        "next_app_resident": is_hit,
        "aggregate_hit_rate": hit_rate,
        "drift_flags":        drift,
        "recommendation":     recommendation,
    }

    # Write recommendation back — Supervisor reads this next step
    mcp("set_pipeline_state", {
        "key":   "telemetry_output",
        "value": json.dumps(telemetry_output),
    })
    print(f"  [MCP] wrote telemetry_output -> Supervisor reads this next step")
    print(f"  [TelemetryAgent] done")


if __name__ == "__main__":
    main()
