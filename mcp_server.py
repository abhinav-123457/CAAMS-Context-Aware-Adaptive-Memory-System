# mcp_server.py — CAAMS MCP Tool Server (SSE Transport)
#
# Runs as an INDEPENDENT PROCESS on http://127.0.0.1:8765
# Agents connect via HTTP/SSE — one persistent server, no per-call subprocess.
#
# HOW TO USE:
#   Terminal 1:  python mcp_server.py        (keep running)
#   Terminal 2:  python orchestrator.py      (agents connect via SSE)
#
# License: Apache 2.0

import os
import sys
import time
import pandas as pd

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_pool import DeviceMemoryPool
from cache_manager import AdaptiveLRUFCache, KVCachePressureEstimator
from context_predictor import HourAwareMarkovPredictor, load_lsapp
from rl_eviction_policy import (
    EvictionQAgent, rl_rank_eviction_candidates, QTABLE_PATH
)

DATA_DIR = "./data"
MCP_HOST = "127.0.0.1"
MCP_PORT = 8765

# ── Shared state (lives in this server process only) ─────────────────────────
print("[MCP] Booting CAAMS tool server (SSE)...")

df_lsapp = load_lsapp()
markov   = HourAwareMarkovPredictor(hour_buckets=6)
markov.fit(df_lsapp)

pool  = DeviceMemoryPool()
cache = AdaptiveLRUFCache(max_mb=2048, min_mb=512)

rl_agent = EvictionQAgent()
if os.path.isfile(QTABLE_PATH):
    rl_agent.load(QTABLE_PATH)
else:
    print("[MCP] WARNING: Q-table missing. Run bootstrap.py first.")

try:
    kv_df        = pd.read_csv(f"{DATA_DIR}/kv_cache_workloads.csv")
    kv_estimator = KVCachePressureEstimator(kv_df)
except FileNotFoundError:
    print("[MCP] WARNING: kv_cache_workloads.csv missing.")
    class _StubKV:
        def sample_pressure(self, n_concurrent=1): return 50.0
    kv_estimator = _StubKV()

_telemetry_log: list[dict] = []
print(f"[MCP] All components ready. Serving on http://{MCP_HOST}:{MCP_PORT}\n")

# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP(name="CAAMS-MemoryServer", host=MCP_HOST, port=MCP_PORT)


@mcp.tool()
def predict_next_app(prev_app: str, current_app: str,
                     hour: int, top_k: int = 3) -> dict:
    """Predicts top-k next apps using second-order hour-aware Markov chain trained on LSApp."""
    preds = markov.predict(prev_app, current_app, hour, top_k=top_k)
    return {
        "predictions": preds,
        "model":       "HourAwareMarkov-2nd-order",
        "context":     {"prev": prev_app, "current": current_app, "hour": hour},
    }


@mcp.tool()
def get_memory_snapshot() -> dict:
    """Returns current Samsung device memory state (allocated, preloaded, free MB, utilization)."""
    return pool.snapshot()


@mcp.tool()
def allocate_app(app_name: str) -> dict:
    """Allocates app as foreground (1.5x weight). Reclaims stale foreground apps."""
    freed = []
    for app in list(pool.allocated.keys()):
        if app != app_name:
            res = pool.evict(app)
            if res["freed_mb"] > 0:
                freed.append(res)
    result = pool.allocate(app_name, priority="foreground")
    result["reclaimed"] = freed
    return result


@mcp.tool()
def preload_app(app_name: str, pred_prob: float = 0.0) -> dict:
    """Preloads predicted app into memory (background). Updates LRU-F cache."""
    result = pool.preload(app_name)
    if result.get("success"):
        cache.insert(app_name, mb=pool.app_footprint(app_name), pred_prob=pred_prob)
    return result


@mcp.tool()
def evict_app(app_name: str) -> dict:
    """Evicts an app from memory pool, freeing its MB."""
    return pool.evict(app_name)


@mcp.tool()
def rank_eviction(candidates: list, memory_free_pct: float) -> dict:
    """Ranks eviction candidates using trained RL Q-agent (offline on LSApp)."""
    ranked = rl_rank_eviction_candidates(
        agent           = rl_agent,
        candidates      = candidates,
        memory_free_pct = memory_free_pct,
        last_used       = pool.last_access,
        use_counts      = pool.access_count,
        current_step    = int(time.time()),
    )
    return {"ranked": ranked, "agent": "EvictionQAgent-tabular",
            "memory_free_pct": memory_free_pct}


@mcp.tool()
def get_cache_snapshot() -> dict:
    """Returns LRU-F adaptive cache state with per-app retention scores."""
    return cache.snapshot()


@mcp.tool()
def adapt_cache_capacity(free_device_pct: float, query_pressure: float) -> dict:
    """Adapts cache capacity using KV pressure + free memory + query load signals."""
    old_cap = cache.capacity_mb
    kv_mb   = kv_estimator.sample_pressure(n_concurrent=1)
    cache.adapt_capacity(free_device_pct=free_device_pct,
                         kv_pressure_mb=kv_mb, query_pressure=query_pressure)
    return {"old_capacity_mb": old_cap, "new_capacity_mb": cache.capacity_mb,
            "changed": old_cap != cache.capacity_mb, "kv_pressure_mb": round(kv_mb, 1)}


@mcp.tool()
def cache_lookup(app_name: str, pred_prob: float = 0.0) -> dict:
    """Checks if app is in LRU-F cache. Returns hit/miss and current hit rate."""
    hit = cache.lookup(app_name, pred_prob=pred_prob)
    return {"app": app_name, "hit": hit, "hit_rate": cache.hit_rate}


@mcp.tool()
def record_telemetry(step: int, hit: bool, path: str,
                     latency_ms: float, util_pct: float,
                     thrash: bool, notes: str = "") -> dict:
    """TelemetryAgent writes one step KPI snapshot here. Supervisor reads aggregate."""
    _telemetry_log.append({
        "step": step, "hit": hit, "path": path,
        "latency_ms": latency_ms, "util_pct": util_pct,
        "thrash": thrash, "notes": notes, "ts": time.time(),
    })
    return {"recorded": True, "total_steps": len(_telemetry_log)}


@mcp.tool()
def get_telemetry_report() -> dict:
    """Aggregate KPI report. Supervisor reads this to detect drift and decide routing."""
    if not _telemetry_log:
        return {"status": "no_data", "steps": 0}
    total   = len(_telemetry_log)
    hits    = sum(1 for e in _telemetry_log if e["hit"])
    thrash  = sum(1 for e in _telemetry_log if e["thrash"])
    cold    = sum(1 for e in _telemetry_log if e["path"] == "cold")
    avg_lat = sum(e["latency_ms"] for e in _telemetry_log) / total
    avg_ut  = sum(e["util_pct"]   for e in _telemetry_log) / total
    hit_rt  = round(hits / total * 100, 1)
    drift   = []
    if hit_rt < 85.0:   drift.append("hit_rate_below_85pct")
    if avg_lat > 10.0:  drift.append("latency_above_10ms")
    if thrash > 0:      drift.append(f"thrash_events:{thrash}")
    return {
        "cache_hit_rate_pct": hit_rt,  "hit_target_met": hit_rt >= 85.0,
        "avg_latency_ms": round(avg_lat, 2), "latency_target_met": avg_lat < 10.0,
        "avg_util_pct": round(avg_ut, 1),    "util_target_met": avg_ut <= 40.0,
        "thrash_events": thrash, "cold_path_steps": cold,
        "total_steps": total, "drift_flags": drift,
        "overall_pass": len(drift) == 0,
    }

@mcp.tool()
def reset_state() -> dict:
    """Resets device pool and LRU-F cache to clean state between agent simulations."""
    global pool, cache
    pool  = DeviceMemoryPool()
    cache = AdaptiveLRUFCache(max_mb=2048, min_mb=512)
    _telemetry_log.clear()
    return {"status": "reset", "free_mb": pool.free_mb}

_pipeline_state: dict = {}

@mcp.tool()
def set_pipeline_state(key: str, value: str) -> dict:
    """Agents write their output here. Key = agent name. Value = JSON string."""
    _pipeline_state[key] = value
    return {"ok": True, "key": key}

@mcp.tool()
def get_pipeline_state(key: str) -> dict:
    """Agents read input from previous agent. Returns JSON string or empty."""
    return {"key": key, "value": _pipeline_state.get(key, "")}

@mcp.tool()
def clear_pipeline_state() -> dict:
    """Resets inter-agent state between steps."""
    _pipeline_state.clear()
    return {"ok": True}


if __name__ == "__main__":
    print(f"[MCP] Starting SSE server → http://{MCP_HOST}:{MCP_PORT}")
    print("[MCP] Keep this terminal open. Run orchestrator.py in a second terminal.")
    mcp.run(transport="sse")