# agents/memory_agent_process.py — CAAMS MemoryAgent (separate OS process)
#
# Called by pipeline_runner.py as a subprocess per step.
# Reads directive + context_output from MCP → executes memory operations.
#
# ARCHITECTURAL CONSTRAINT (enforced here):
#   RL Q-agent is the UNCONDITIONAL primary eviction mechanism.
#   It runs in < 1ms and requires zero additional RAM.
#   Qwen is NEVER called from this process.
#   Rationale: cold path triggers at free_pct < 25%. Loading a 1.5B
#   parameter model at that point causes OOM or latency spikes that
#   defeat the entire purpose of the memory manager.
#   Qwen's role in the system is Supervisor-only, on the hot path,
#   when free_pct >= 25%.
#
# License: Apache 2.0

import argparse
import asyncio
import json
import os
import sys
import time

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
    parser.add_argument("--app",      type=str,   required=True)
    parser.add_argument("--pressure", type=float, default=0.3)
    args = parser.parse_args()

    print(f"[MemoryAgent PID={os.getpid()}] app={args.app}")

    # Read both upstream outputs
    raw_dir = mcp("get_pipeline_state", {"key": "supervisor_directive"})
    raw_ctx = mcp("get_pipeline_state", {"key": "context_output"})

    directive = json.loads(raw_dir.get("value", "{}")) if raw_dir.get("value") else {}
    ctx       = json.loads(raw_ctx.get("value", "{}")) if raw_ctx.get("value") else {}

    preds          = ctx.get("predictions", [])
    max_preloads   = int(directive.get("max_preloads", 2))
    ev_urgency     = directive.get("eviction_urgency", "none")
    protect        = set(directive.get("protect_apps", [args.app]))
    intensity      = float(ctx.get("chronos_intensity", 0.5))

    print(f"  Directive: ev_urgency={ev_urgency} max_preloads={max_preloads}")
    print(f"  Predictions: {[p['app'] for p in preds]}")

    snap     = mcp("get_memory_snapshot", {})
    free_pct = float(snap.get("free_pct", 100.0))

    t_hot_start = time.perf_counter()

    evictions = []
    preloads  = []

    # --- EVICTION: RL Q-agent ALWAYS, no LLM, no exceptions ---
    # The RL Q-table lookup is sub-millisecond and zero-RAM.
    # This is the ONLY eviction mechanism in this agent regardless of path.
    if ev_urgency in ("low", "high"):
        all_apps  = (list(snap.get("allocated_apps", {}).keys()) +
                     list(snap.get("preloaded_apps", {}).keys()))
        evictable = [a for a in all_apps if a not in protect]

        if evictable:
            n_evict = 1 if ev_urgency == "low" else 2
            ranked  = mcp("rank_eviction", {
                "candidates":      evictable[:5],
                "memory_free_pct": free_pct,
            })
            to_evict = ranked.get("ranked", evictable)[:n_evict]
            print(f"  RL eviction (urgency={ev_urgency}): {to_evict}")

            for ev_app in to_evict:
                res = mcp("evict_app", {"app_name": ev_app})
                if res.get("freed_mb", 0) > 0:
                    evictions.append({"app": ev_app, "freed_mb": res["freed_mb"]})

    # --- CACHE LOOKUP ---
    pred_map  = {p["app"]: p.get("prob", 0.0) for p in preds}
    cache_hit = mcp("cache_lookup", {
        "app_name":  args.app,
        "pred_prob": float(pred_map.get(args.app, 0.0)),
    })
    is_hit = bool(cache_hit.get("hit", False))
    print(f"  Cache lookup: hit={is_hit} rate={cache_hit.get('hit_rate', '?')}%")

    # --- ALLOCATE FOREGROUND ---
    mcp("allocate_app", {"app_name": args.app})

    # --- PRELOAD PREDICTED APPS ---
    # Scale preload count with Chronos intensity signal
    if intensity > 0.7:
        effective_max = min(max_preloads, 3)
    elif intensity < 0.3:
        effective_max = min(max_preloads, 1)
    else:
        effective_max = max_preloads

    loaded = 0
    for p in preds:
        if loaded >= effective_max:
            break
        if p["app"] != args.app and p.get("prob", 0) > 0.01:
            res = mcp("preload_app", {
                "app_name":  p["app"],
                "pred_prob": float(p.get("prob", 0.0)),
            })
            if res.get("success"):
                preloads.append({"app": p["app"], "prob": p.get("prob", 0)})
                loaded += 1

    # --- ADAPT CACHE CAPACITY ---
    snap_new = mcp("get_memory_snapshot", {})
    mcp("adapt_cache_capacity", {
        "free_device_pct": float(snap_new.get("free_pct", 50.0)),
        "query_pressure":  args.pressure,
    })

    hot_latency_ms = round((time.perf_counter() - t_hot_start) * 1000, 2)

    memory_output = {
        "evictions":       evictions,
        "preloads":        preloads,
        "cache_hit":       is_hit,
        "util_pct":        float(snap_new.get("utilization_pct", 0.0)),
        "free_pct_after":  float(snap_new.get("free_pct", 100.0)),
        "hot_latency_ms":  hot_latency_ms,
        "eviction_method": "rl_qtable",   # always RL, never LLM in this agent
        "from_agent":      "memory_agent",
    }

    print(f"  Evicted: {[e['app'] for e in evictions]}")
    print(f"  Preloaded: {[p['app'] for p in preloads]}")
    print(f"  Hot-path latency (RL+MCP only): {hot_latency_ms}ms")

    mcp("set_pipeline_state", {
        "key":   "memory_output",
        "value": json.dumps(memory_output),
    })
    print(f"[MemoryAgent] Done — memory_output written to MCP pipeline state")


if __name__ == "__main__":
    main()