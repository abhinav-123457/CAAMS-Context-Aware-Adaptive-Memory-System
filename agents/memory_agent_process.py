# agents/memory_agent_process.py
# Reads:  MCP pipeline state: "supervisor_directive", "context_output"
# Writes: MCP pipeline state: "memory_output"
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
    parser.add_argument("--app",      type=str,   required=True)
    parser.add_argument("--pressure", type=float, default=0.3)
    args = parser.parse_args()

    print(f"[MemoryAgent PID={os.getpid()}] app={args.app}")

    # Read both inputs from MCP — inter-process messages
    dir_raw = mcp("get_pipeline_state", {"key": "supervisor_directive"})
    ctx_raw = mcp("get_pipeline_state", {"key": "context_output"})

    if not dir_raw.get("value") or not ctx_raw.get("value"):
        print("  [ERROR] Missing directive or context_output in MCP state.")
        sys.exit(1)

    directive = json.loads(dir_raw["value"])
    ctx       = json.loads(ctx_raw["value"])
    preds     = ctx.get("predictions", [])
    protect   = set(directive.get("protect_apps", [])) | {args.app}

    print(f"  [MCP] read supervisor_directive: task={directive.get('memory_task')}")
    print(f"  [MCP] read context_output: predictions={[p['app'] for p in preds]}")

    snap     = mcp("get_memory_snapshot", {})
    free_pct = float(snap.get("free_pct", 100.0))
    evictions, preloads = [], []

    # Eviction logic
    if directive.get("eviction_urgency") in ("low", "high", "critical"):
        all_apps  = (list(snap.get("allocated_apps", {}).keys()) +
                     list(snap.get("preloaded_apps", {}).keys()))
        evictable = [a for a in all_apps if a not in protect]

        if evictable:
            if directive.get("path") == "cold" and directive.get("eviction_urgency") in ("high", "critical"):
                # Cold path: Qwen decides
                llm = get_local_llm()
                to_evict = []
                if llm:
                    try:
                        resp = llm.invoke([
                            SystemMessage(content="Samsung memory manager. Reply ONLY JSON."),
                            HumanMessage(content=(
                                f"free_pct={free_pct:.1f}, urgency={directive['eviction_urgency']}, "
                                f"protect={list(protect)}, evictable={evictable}, "
                                f"predicted={[p['app'] for p in preds[:3]]}. "
                                "Return: {\"evict\": [\"app1\"], \"reasoning\": \"one sentence\"}"
                            )),
                        ])
                        ev = parse_json_response(resp)
                        if ev and "evict" in ev:
                            to_evict = [a for a in ev["evict"] if a not in protect and a in evictable]
                            print(f"  [Qwen cold] evict={to_evict} | {ev.get('reasoning','')}")
                    except Exception as e:
                        print(f"  [Qwen] failed: {e}")

                if not to_evict:
                    ranked   = mcp("rank_eviction", {"candidates": evictable[:4], "memory_free_pct": free_pct})
                    to_evict = ranked.get("ranked", evictable)[:2]
                    print(f"  [RL fallback] evict={to_evict}")
            else:
                # Hot path: RL agent via MCP
                ranked   = mcp("rank_eviction", {"candidates": evictable[:4], "memory_free_pct": free_pct})
                to_evict = ranked.get("ranked", evictable)[:2]
                print(f"  [RL hot] evict={to_evict}")

            for app in to_evict:
                res = mcp("evict_app", {"app_name": app})
                if res.get("freed_mb", 0) > 0:
                    evictions.append({"app": app, "freed_mb": res["freed_mb"]})

    # Cache lookup + foreground allocation
    pred_map     = {p["app"]: p.get("prob", 0.0) for p in preds}
    cache_result = mcp("cache_lookup", {
        "app_name":  args.app,
        "pred_prob": float(pred_map.get(args.app, 0.0)),
    })
    cache_hit = bool(cache_result.get("hit", False))
    print(f"  [MCP] cache_lookup -> hit={cache_hit} rate={cache_result.get('hit_rate','?')}%")

    mcp("allocate_app", {"app_name": args.app})
    print(f"  [MCP] allocate_app -> {args.app} (foreground)")

    # Preload predicted apps
    if directive.get("memory_task") in ("preload_predicted", "evict_and_preload"):
        max_pre   = int(directive.get("max_preloads", 2))
        candidates = [p for p in preds
                      if p["app"] != args.app
                      and p["app"] not in protect
                      and p.get("prob", 0) > 0.01][:max_pre]
        for p in candidates:
            res = mcp("preload_app", {"app_name": p["app"], "pred_prob": float(p.get("prob", 0.0))})
            if res.get("success"):
                preloads.append({"app": p["app"], "prob": p.get("prob", 0)})
                print(f"  [MCP] preload_app -> {p['app']} (prob={p.get('prob',0):.3f})")

    snap_new = mcp("get_memory_snapshot", {})
    mcp("adapt_cache_capacity", {
        "free_device_pct": float(snap_new.get("free_pct", 50.0)),
        "query_pressure":  args.pressure,
    })

    memory_output = {
        "evictions":      evictions,
        "preloads":       preloads,
        "cache_hit":      cache_hit,
        "util_pct":       float(snap_new.get("utilization_pct", 0.0)),
        "free_pct_after": float(snap_new.get("free_pct", 100.0)),
    }

    # Write to MCP — TelemetryAgent reads this
    mcp("set_pipeline_state", {
        "key":   "memory_output",
        "value": json.dumps(memory_output),
    })
    print(f"  [MCP] wrote memory_output -> TelemetryAgent will read this")
    print(f"  [MemoryAgent] done | evicted={[e['app'] for e in evictions]} preloaded={[p['app'] for p in preloads]}")


if __name__ == "__main__":
    main()
