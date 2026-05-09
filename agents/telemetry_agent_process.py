# agents/telemetry_agent_process.py — CAAMS TelemetryAgent (separate OS process)
#
# Called by pipeline_runner.py as a subprocess per step.
# Reads memory_output from MCP → records telemetry → produces recommendation
# for SupervisorAgent to consume on the next step.
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
    parser.add_argument("--step",     type=int, required=True)
    parser.add_argument("--next_app", type=str, required=True)
    args = parser.parse_args()

    print(f"[TelemetryAgent PID={os.getpid()}] step={args.step} next_app={args.next_app}")

    # Read upstream memory output
    raw    = mcp("get_pipeline_state", {"key": "memory_output"})
    mem    = json.loads(raw.get("value", "{}")) if raw.get("value") else {}

    raw_dir = mcp("get_pipeline_state", {"key": "supervisor_directive"})
    directive = json.loads(raw_dir.get("value", "{}")) if raw_dir.get("value") else {}

    print(f"  memory_output read: evictions={[e['app'] for e in mem.get('evictions',[])]}")
    print(f"  preloads={[p['app'] for p in mem.get('preloads',[])]}")

    # Check if next_app is resident after this step
    snap     = mcp("get_memory_snapshot", {})
    resident = (set(snap.get("allocated_apps", {}).keys()) |
                set(snap.get("preloaded_apps", {}).keys()))
    is_hit   = args.next_app in resident
    icon     = "HIT" if is_hit else "MISS"
    print(f"  next_app='{args.next_app}' → [{icon}]")

    # Real hot-path latency from MemoryAgent (not a hardcoded constant)
    # This is the actual time MemoryAgent spent on RL + MCP calls
    hot_lat = float(mem.get("hot_latency_ms", 0.0))

    # Record to MCP telemetry log
    mcp("record_telemetry", {
        "step":       args.step,
        "hit":        is_hit,
        "path":       directive.get("path", "hot"),
        "latency_ms": hot_lat,
        "util_pct":   float(mem.get("util_pct", 0.0)),
        "thrash":     False,
        "notes":      (
            f"eviction_method={mem.get('eviction_method','rl_qtable')} "
            f"preloads={[p['app'] for p in mem.get('preloads',[])]} "
            f"hot_lat={hot_lat}ms"
        ),
    })

    # Read aggregate KPIs
    aggregate = mcp("get_telemetry_report", {})
    hit_rate  = float(aggregate.get("cache_hit_rate_pct", 100.0))
    drift     = aggregate.get("drift_flags", [])
    free_pct  = float(snap.get("free_pct", 100.0))

    # Produce recommendation for SupervisorAgent (reads this at next step start)
    if hit_rate < 75:
        rec = "increase_top_k_and_preloads: hit rate critically low"
    elif hit_rate < 85:
        rec = "increase_preloads: hit rate below 85 percent target"
    elif free_pct < 20:
        rec = "trigger_eviction: memory pressure high — RL eviction recommended"
    else:
        rec = "maintain_current_policy: system healthy"

    telemetry_output = {
        "next_app_resident":   is_hit,
        "aggregate_hit_rate":  hit_rate,
        "drift_flags":         drift,
        "recommendation":      rec,
        "from_agent":          "telemetry_agent",
    }

    mcp("set_pipeline_state", {
        "key":   "telemetry_output",
        "value": json.dumps(telemetry_output),
    })

    print(f"  aggregate: hit_rate={hit_rate}% drift={drift}")
    print(f"  recommendation → {rec}")
    print(f"[TelemetryAgent] Done — telemetry_output written to MCP pipeline state")


if __name__ == "__main__":
    main()