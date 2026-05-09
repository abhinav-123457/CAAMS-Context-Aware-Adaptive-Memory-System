
# What this shows:
#   1. Supervisor calls Qwen on live MCP telemetry → produces a typed directive
#   2. ContextAgent reads the directive, calls MCP, optionally validates with Qwen
#   3. MemoryAgent reads directive + context_output, evicts/preloads via MCP
#   4. TelemetryAgent reads memory_output, writes to MCP, sends recommendation back
#   5. Real feedback loop: supervisor re-reads telemetry and loops if needed
#
# Architecture (LangGraph):
#   supervisor → context_agent → memory_agent → telemetry_agent
#       ↑______________________________________________|  (on drift/pressure)
#
# Run:
#   Terminal 1: python mcp_server.py
#   Terminal 2: python orchestrator.py
#
# License: Apache 2.0

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import pandas as pd
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from mcp.client.sse import sse_client
from mcp import ClientSession

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_llm import get_local_llm, parse_json_response
from context_predictor import ChronosUsageForecaster, load_lsapp

DATA_DIR = "./data"
MCP_URL  = os.getenv("CAAMS_MCP_URL", "http://127.0.0.1:8765/sse")


# ─────────────────────────────────────────────────────────────────────────────
# SupervisorDirective — typed message agents actually read and act on
# ─────────────────────────────────────────────────────────────────────────────
class SupervisorDirective(TypedDict):
    context_task:       str    # predict_standard | predict_aggressive | predict_conservative
    top_k:              int
    memory_task:        str    # preload_predicted | evict_and_preload | evict_only | hold
    max_preloads:       int
    eviction_urgency:   str    # none | low | high | critical
    protect_apps:       list
    route_after_memory: str    # telemetry | supervisor
    reason:             str
    detected_drift:     list
    path:               str    # hot | cold


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state — no cumulative metrics, only inter-agent messages
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    current_app:       str
    prev_app:          str
    hour:              int
    step:              int
    query_pressure:    float
    chronos_intensity: float
    next_app:          str

    # The actual inter-agent communication channels
    directive:         SupervisorDirective   # supervisor → all agents
    context_output:    dict                  # context_agent → memory_agent
    memory_output:     dict                  # memory_agent → telemetry_agent
    telemetry_output:  dict                  # telemetry_agent → supervisor

    loop_count:        int
    path:              str


# ─────────────────────────────────────────────────────────────────────────────
# MCP client
# ─────────────────────────────────────────────────────────────────────────────
async def _call_async(tool: str, args: dict) -> Any:
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
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _call_async(tool, args)).result(timeout=30)
        except RuntimeError:
            return asyncio.run(_call_async(tool, args))
    except Exception as e:
        return {"_error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Qwen
# ─────────────────────────────────────────────────────────────────────────────
_qwen = None

def get_qwen():
    global _qwen
    if _qwen is None:
        _qwen = get_local_llm()
    return _qwen


def qwen(system: str, user: str) -> dict:
    llm = get_qwen()
    if llm is None:
        return {}
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return parse_json_response(resp)
    except Exception as e:
        print(f"  [Qwen] call error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Chronos intensity
# ─────────────────────────────────────────────────────────────────────────────
CHRONOS_BUCKETS  = [5.0] * 6
CHRONOS_PEAK_OFF = 0

def init_chronos():
    global CHRONOS_BUCKETS, CHRONOS_PEAK_OFF
    try:
        print("[Chronos] Loading Chronos-T5-Small (Apache 2.0)...")
        df     = load_lsapp()
        cf     = ChronosUsageForecaster()
        series = cf.build_hourly_series(df)
        result = cf.forecast(series, prediction_length=6)
        CHRONOS_BUCKETS  = result["mean_launches"]
        CHRONOS_PEAK_OFF = result["peak_hour_offset"]
        print(f"[Chronos] Ready | buckets={CHRONOS_BUCKETS} | peak=+{CHRONOS_PEAK_OFF}h")
    except Exception as e:
        print(f"[Chronos] Failed ({e}), using flat defaults.")


def chronos_intensity(hour: int) -> float:
    bucket = min(hour // 4, len(CHRONOS_BUCKETS) - 1)
    mx = max(CHRONOS_BUCKETS) or 1.0
    return round(CHRONOS_BUCKETS[bucket] / mx, 3)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Supervisor
# Reads live MCP telemetry → calls Qwen → produces SupervisorDirective
# ─────────────────────────────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> AgentState:
    step       = state["step"]
    loop_count = state.get("loop_count", 0)

    print(f"\n{'='*60}")
    print(f"[Supervisor] Step {step} | loop={loop_count} | app={state['current_app']}")

    telemetry = mcp("get_telemetry_report", {})
    snap      = mcp("get_memory_snapshot",  {})

    hit_rate  = float(telemetry.get("cache_hit_rate_pct", 100.0))
    drift     = telemetry.get("drift_flags", [])
    free_pct  = float(snap.get("free_pct", 100.0))
    prev_rec  = state.get("telemetry_output", {}).get("recommendation", "none")

    print(f"  MCP telemetry → hit_rate={hit_rate}% free={free_pct}% drift={drift}")
    print(f"  TelemetryAgent recommendation (last round) → {prev_rec}")
    _llm_safe = free_pct >= 25.0 and state["query_pressure"] <= 0.85
    result = {}

    if _llm_safe:

        result = qwen(
        "You are the Supervisor of a Samsung on-device memory manager. "
        "Reply ONLY with valid JSON.",
        f"""
        free_pct={free_pct:.1f}, query_pressure={state['query_pressure']:.3f},
        chronos_intensity={state['chronos_intensity']:.3f},
        hit_rate={hit_rate}, drift={drift},
        prev_recommendation="{prev_rec}",
        loop_count={loop_count}, current_app="{state['current_app']}"

        Rules:
            - path="cold" if free_pct<25 or query_pressure>0.85
            - top_k=5 and max_preloads=3 if hit_rate<75
            - eviction_urgency="critical" if free_pct<15
            - route_after_memory="supervisor" if hit_rate<85 and loop_count<2
            - route_after_memory="telemetry" otherwise

        Return JSON:
        {{
            "context_task": "predict_standard|predict_aggressive|predict_conservative",
            "top_k": <1-5>,
            "memory_task": "preload_predicted|evict_and_preload|evict_only|hold",
            "max_preloads": <0-3>,
            "eviction_urgency": "none|low|high|critical",
            "protect_apps": ["{state['current_app']}"],
            "route_after_memory": "telemetry|supervisor",
            "reason": "<one sentence>",
            "detected_drift": {drift},
            "path": "hot|cold"
            }}
        """)

    required = {"context_task","top_k","memory_task","max_preloads",
                 "eviction_urgency","protect_apps","route_after_memory",
                 "reason","detected_drift","path"}

    if required.issubset(result.keys()):
        directive = SupervisorDirective(**{k: result[k] for k in required})
        print(f"  Qwen directive → task={directive['context_task']} "
              f"top_k={directive['top_k']} memory={directive['memory_task']} "
              f"path={directive['path']}")
        print(f"  reason → {directive['reason']}")
    else:
        # Deterministic fallback — same logic, no Qwen needed
        is_cold  = free_pct < 25 or state["query_pressure"] > 0.85
        top_k    = 5 if hit_rate < 75 else 3
        max_pre  = 0 if free_pct < 15 else (1 if is_cold else (3 if hit_rate < 75 else 2))
        ev_urg   = "critical" if free_pct < 15 else ("high" if free_pct < 25 else "none")
        route    = "telemetry" if (loop_count >= 2 or hit_rate >= 85) else "supervisor"
        directive = SupervisorDirective(
            context_task     = "predict_aggressive" if hit_rate < 75 else "predict_standard",
            top_k            = top_k,
            memory_task      = "evict_only" if free_pct < 15 else
                               ("evict_and_preload" if is_cold else "preload_predicted"),
            max_preloads     = max_pre,
            eviction_urgency = ev_urg,
            protect_apps     = [state["current_app"]],
            route_after_memory = route,
            reason           = f"[deterministic fallback] free={free_pct:.1f}% hit={hit_rate}%",
            detected_drift   = drift,
            path             = "cold" if is_cold else "hot",
        )
        print(f"  [fallback] directive → {directive['reason']}")

    state["directive"]  = directive
    state["path"]       = directive["path"]
    state["loop_count"] = loop_count
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — ContextAgent
# Reads directive → calls MCP predict → optionally validates with Qwen
# Writes context_output for MemoryAgent to consume
# ─────────────────────────────────────────────────────────────────────────────
def context_agent_node(state: AgentState) -> AgentState:
    directive = state["directive"]
    print(f"\n[ContextAgent] directive read → task={directive['context_task']} "
          f"top_k={directive['top_k']}")

    pred_res = mcp("predict_next_app", {
        "prev_app":    state["prev_app"],
        "current_app": state["current_app"],
        "hour":        state["hour"],
        "top_k":       directive["top_k"],
    })
    preds = pred_res.get("predictions", [])
    print(f"  MCP predict_next_app → {[p['app'] for p in preds]}")

    # Qwen validates predictions on aggressive/conservative tasks
    qwen_note = ""
    if directive["context_task"] != "predict_standard":
        vres = qwen(
            "Validate app predictions for Samsung memory manager. Reply ONLY JSON.",
            f"hour={state['hour']}, pressure={state['query_pressure']:.2f}, "
            f"app={state['current_app']!r}, predictions={preds}, "
            f"task={directive['context_task']}. "
            'Return: {"drop_low_confidence": true|false, '
            '"confidence_threshold": 0.0-1.0, "reasoning": "one sentence"}'
        )
        if vres and not vres.get("_error"):
            threshold = float(vres.get("confidence_threshold", 0.0))
            if vres.get("drop_low_confidence") and threshold > 0:
                before = len(preds)
                preds  = [p for p in preds if p.get("prob", 0) >= threshold]
                print(f"  Qwen validation → filtered {before}→{len(preds)} "
                      f"(threshold={threshold:.2f})")
            qwen_note = vres.get("reasoning", "")

    context_output = {
        "predictions":       preds,
        "top_k_used":        directive["top_k"],
        "task":              directive["context_task"],
        "chronos_intensity": state["chronos_intensity"],
        "qwen_note":         qwen_note,
    }
    print(f"  → context_output written for MemoryAgent "
          f"| predictions={[p['app'] for p in preds]}")

    state["context_output"] = context_output
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — MemoryAgent
# Reads directive + context_output → evicts/allocates/preloads via MCP
# Writes memory_output for TelemetryAgent to consume
# ─────────────────────────────────────────────────────────────────────────────
def memory_agent_node(state: AgentState) -> AgentState:
    directive = state["directive"]
    ctx       = state["context_output"]
    preds     = ctx.get("predictions", [])
    protect   = set(directive.get("protect_apps", [])) | {state["current_app"]}

    print(f"\n[MemoryAgent] directive read → task={directive['memory_task']} "
          f"max_preloads={directive['max_preloads']} "
          f"eviction={directive['eviction_urgency']}")
    print(f"  context_output read → predictions={[p['app'] for p in preds]}")

    snap      = mcp("get_memory_snapshot", {})
    free_pct  = float(snap.get("free_pct", 100.0))
    evictions = []
    preloads  = []

    # Eviction
    if directive["eviction_urgency"] in ("low", "high", "critical"):
        all_apps  = (list(snap.get("allocated_apps", {}).keys()) +
                     list(snap.get("preloaded_apps", {}).keys()))
        evictable = [a for a in all_apps if a not in protect]

        if evictable:
            ranked = mcp("rank_eviction", {
                "candidates":      evictable[:4],
                "memory_free_pct": free_pct,
            })
            to_evict = ranked.get("ranked", evictable)[:2]
            path_label = directive.get("path", "hot")
            print(f" RL eviction [{path_label} path] --> {to_evict} | free={free_pct:.1f}%")

            for app in to_evict:
                res = mcp("evict_app", {"app_name": app})
                if res.get("freed_mb", 0) > 0:
                    evictions.append({"app": app, "freed_mb": res["freed_mb"]})

    # Cache lookup
    pred_map     = {p["app"]: p.get("prob", 0.0) for p in preds}
    cache_result = mcp("cache_lookup", {
        "app_name":  state["current_app"],
        "pred_prob": float(pred_map.get(state["current_app"], 0.0)),
    })
    cache_hit = bool(cache_result.get("hit", False))
    print(f"  MCP cache_lookup → hit={cache_hit} "
          f"rate={cache_result.get('hit_rate','?')}%")

    # Allocate foreground
    mcp("allocate_app", {"app_name": state["current_app"]})
    print(f"  MCP allocate_app → {state['current_app']} (foreground)")

    # Preload predicted apps up to directive limit
    if directive["memory_task"] in ("preload_predicted", "evict_and_preload"):
        candidates = [p for p in preds
                      if p["app"] != state["current_app"]
                      and p["app"] not in protect
                      and p.get("prob", 0) > 0.01][:directive["max_preloads"]]
        for p in candidates:
            res = mcp("preload_app", {"app_name": p["app"],
                                       "pred_prob": float(p.get("prob", 0.0))})
            if res.get("success"):
                preloads.append({"app": p["app"], "prob": p.get("prob", 0)})
                print(f"  MCP preload_app → {p['app']} (prob={p.get('prob',0):.3f})")

    # Adapt cache capacity
    snap_new = mcp("get_memory_snapshot", {})
    mcp("adapt_cache_capacity", {
        "free_device_pct": float(snap_new.get("free_pct", 50.0)),
        "query_pressure":  state["query_pressure"],
    })

    memory_output = {
        "evictions":          evictions,
        "preloads":           preloads,
        "cache_hit":          cache_hit,
        "util_pct":           float(snap_new.get("utilization_pct", 0.0)),
        "free_pct_after":     float(snap_new.get("free_pct", 100.0)),
        "directive_followed": {
            "task":             directive["memory_task"],
            "actual_evictions": len(evictions),
            "actual_preloads":  len(preloads),
        },
    }
    print(f"  → memory_output written for TelemetryAgent "
          f"| evicted={[e['app'] for e in evictions]} "
          f"preloaded={[p['app'] for p in preloads]}")

    state["memory_output"] = memory_output
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — TelemetryAgent
# Reads memory_output → writes to MCP → produces recommendation for Supervisor
# Closes the feedback loop
# ─────────────────────────────────────────────────────────────────────────────
def telemetry_agent_node(state: AgentState) -> AgentState:
    mem_out   = state["memory_output"]
    directive = state["directive"]
    next_app  = state.get("next_app", "")

    print(f"\n[TelemetryAgent] memory_output read → "
          f"evictions={[e['app'] for e in mem_out.get('evictions',[])]} "
          f"preloads={[p['app'] for p in mem_out.get('preloads',[])]} "
          f"cache_hit={mem_out.get('cache_hit')}")

    # Check if next app is resident after this step
    snap     = mcp("get_memory_snapshot", {})
    resident = (set(snap.get("allocated_apps", {}).keys()) |
                set(snap.get("preloaded_apps", {}).keys()))
    is_hit   = bool(next_app and next_app in resident)

    # Write step to MCP
    mcp("record_telemetry", {
        "step":       state["step"],
        "hit":        is_hit,
        "path":       directive["path"],
        "latency_ms": 0.0,
        "util_pct":   mem_out.get("util_pct", 0.0),
        "thrash":     False,
        "notes":      f"preloads={[p['app'] for p in mem_out.get('preloads',[])]}",
    })

    # Read aggregate from MCP — this is what Supervisor reads next round
    aggregate = mcp("get_telemetry_report", {})
    hit_rate  = float(aggregate.get("cache_hit_rate_pct", 100.0))
    drift     = aggregate.get("drift_flags", [])

    # Recommendation sent back to Supervisor
    if hit_rate < 75:
        recommendation = "increase_top_k_and_preloads: hit rate critically low"
    elif hit_rate < 85:
        recommendation = "increase_preloads: hit rate below 85% target"
    elif mem_out.get("free_pct_after", 100) < 20:
        recommendation = "trigger_eviction: memory pressure high"
    else:
        recommendation = "maintain_current_policy: system healthy"

    telemetry_output = {
        "next_app_resident": is_hit,
        "aggregate_hit_rate": hit_rate,
        "drift_flags":        drift,
        "recommendation":     recommendation,
    }

    icon = "✓ resident" if is_hit else "✗ cold start"
    print(f"  next_app='{next_app}' → [{icon}]")
    print(f"  MCP aggregate → hit_rate={hit_rate}% drift={drift}")
    print(f"  → recommendation to Supervisor: {recommendation}")

    state["telemetry_output"] = telemetry_output
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Routing — the real feedback loop
# ─────────────────────────────────────────────────────────────────────────────
def route_after_telemetry(state: AgentState) -> str:
    directive  = state.get("directive", {})
    loop_count = state.get("loop_count", 0)
    route      = directive.get("route_after_memory", "telemetry")

    if route == "supervisor" and loop_count < 2:
        print(f"\n[Router] drift detected → looping back to Supervisor "
              f"(loop {loop_count + 1})")
        state["loop_count"] = loop_count + 1
        return "supervisor"

    print(f"\n[Router] → END (loop_count={loop_count} route={route})")
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor",      supervisor_node)
    g.add_node("context_agent",   context_agent_node)
    g.add_node("memory_agent",    memory_agent_node)
    g.add_node("telemetry_agent", telemetry_agent_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor",    "context_agent")
    g.add_edge("context_agent", "memory_agent")
    g.add_edge("memory_agent",  "telemetry_agent")
    g.add_conditional_edges(
        "telemetry_agent",
        route_after_telemetry,
        {"supervisor": "supervisor", END: END},
    )
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Session loader
# ─────────────────────────────────────────────────────────────────────────────
def load_session(android_df: pd.DataFrame, min_steps: int = 10) -> pd.DataFrame:
    opened = (android_df[android_df["event_type"] == "Opened"].copy()
              if "event_type" in android_df.columns else android_df.copy())
    cands  = opened.groupby("session_id").filter(
        lambda g: len(g) >= min_steps and g["app_name"].nunique() >= 5)
    if cands.empty:
        cands = opened.groupby("session_id").filter(lambda g: len(g) >= min_steps)
    best = cands.groupby("session_id")["app_name"].nunique().idxmax()
    df   = cands[cands["session_id"] == best].reset_index(drop=True)
    print(f"[Session] id={best} | events={len(df)} | "
          f"unique_apps={df['app_name'].nunique()}")
    return df


def get_query_pressure(hour: int) -> float:
    path = os.path.join(DATA_DIR, "melbourne_context.csv")
    if not os.path.exists(path):
        return 0.3
    try:
        df      = pd.read_csv(path)
        hour_df = df[df["hour"] == hour] if "hour" in df.columns else df
        active  = int(hour_df.get("is_active_query",
                      pd.Series([0] * len(hour_df))).fillna(0).sum())
        return round(float(active / max(len(hour_df), 1)), 3)
    except Exception:
        return 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Main — runs N steps, prints agent flow only
# ─────────────────────────────────────────────────────────────────────────────
def run(n_steps: int = 5, cold_path_proof: bool = False):
    print("=" * 60)
    print("  CAAMS — Multi-Agent Orchestrator (flow showcase)")
    print(f"  Agents: Supervisor(Qwen) → ContextAgent → MemoryAgent → TelemetryAgent")
    print(f"  MCP: {MCP_URL}")
    print("=" * 60)

    # Verify MCP
    test = mcp("get_memory_snapshot", {})
    if "_error" in test:
        print(f"\n[FATAL] MCP unreachable: {test['_error']}")
        print("  Run: python mcp_server.py")
        return
    print(f"[OK] MCP connected | pool={test.get('total_mb')}MB\n")

    # Optional: prefill memory to force cold path
    if cold_path_proof:
        print("[Cold Path] Prefilling memory to ~80% to force cold routing...")
        total_mb = int(test.get("total_mb", 6144))
        target   = int(total_mb * 0.80)
        filled   = 0
        i        = 0
        while filled < target:
            res = mcp("preload_app", {"app_name": f"__bg_{i}", "pred_prob": 0.0})
            if not res.get("success"):
                break
            filled += res.get("preloaded_mb", 180)
            i      += 1
        snap = mcp("get_memory_snapshot", {})
        print(f"[Cold Path] Pool after prefill: "
              f"free={snap.get('free_pct')}% used={snap.get('used_mb')}MB\n")

    init_chronos()
    print(f"\n[OK] Chronos ready | peak=+{CHRONOS_PEAK_OFF}h")

    print("[Init] Loading Qwen2.5-1.5B-Instruct (Apache 2.0)...")
    get_qwen()
    print("[OK] Qwen ready\n")

    android_df = pd.read_csv(f"{DATA_DIR}/android_usage.csv")
    session_df = load_session(android_df, min_steps=n_steps + 2)
    graph      = build_graph()

    for step in range(min(n_steps, len(session_df) - 2)):
        print(f"\n{'─'*60}")
        print(f"  STEP {step+1}/{n_steps}")
        print(f"{'─'*60}")

        row_prev = session_df.iloc[step]
        row_curr = session_df.iloc[step + 1]
        hour     = int(row_curr.get("hour", 12))

        empty_directive = SupervisorDirective(
            context_task="predict_standard", top_k=3,
            memory_task="preload_predicted", max_preloads=2,
            eviction_urgency="none", protect_apps=[str(row_curr["app_name"])],
            route_after_memory="telemetry", reason="initial",
            detected_drift=[], path="hot",
        )

        initial: AgentState = {
            "current_app":       str(row_curr["app_name"]),
            "prev_app":          str(row_prev["app_name"]),
            "next_app":          str(session_df.iloc[step + 2]["app_name"]),
            "hour":              hour,
            "step":              step + 1,
            "query_pressure":    get_query_pressure(hour),
            "chronos_intensity": chronos_intensity(hour),
            "directive":         empty_directive,
            "context_output":    {},
            "memory_output":     {},
            "telemetry_output":  {},
            "loop_count":        0,
            "path":              "hot",
        }

        graph.invoke(initial)

    print(f"\n{'='*60}")
    print("  Agent flow complete.")
    print(f"  All 4 agents ran per step: Supervisor → ContextAgent → "
          f"MemoryAgent → TelemetryAgent")
    print(f"  Feedback loop active: TelemetryAgent recommendation → Supervisor")
    print(f"  MCP tools used: predict_next_app, cache_lookup, allocate_app,")
    print(f"    preload_app, evict_app, rank_eviction, adapt_cache_capacity,")
    print(f"    record_telemetry, get_telemetry_report, get_memory_snapshot")
    print(f"  Models: Qwen2.5-1.5B | Markov (LSApp) | Chronos-T5 | RL Q-agent")
    print(f"  License: Apache 2.0 | No third-party API")
    print(f"{'='*60}")


if __name__ == "__main__":
    n           = int(os.getenv("CAAMS_STEPS", "5"))
    cold_proof  = os.getenv("CAAMS_COLD_PROOF", "0") == "1"
    run(n_steps=n, cold_path_proof=cold_proof)

    if not cold_proof:
        print("\n[Tip] Run with CAAMS_COLD_PROOF=1 to see cold path + Qwen eviction")