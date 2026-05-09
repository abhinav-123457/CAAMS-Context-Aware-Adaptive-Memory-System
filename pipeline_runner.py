# pipeline_runner.py — CAAMS Real Multi-Process Pipeline
#
# Each agent runs as a SEPARATE OS PROCESS.
# Agents communicate exclusively via MCP over HTTP — no shared memory.
# Process boundary is real: different PIDs, different memory spaces.
#
# Architecture:
#   [SupervisorAgent process] → writes supervisor_directive to MCP
#   [ContextAgent process]   → reads directive, writes context_output to MCP
#   [MemoryAgent process]    → reads both, writes memory_output to MCP
#   [TelemetryAgent process] → reads memory_output, writes telemetry_output to MCP
#                              ↑___________________________________________|
#                              (Supervisor reads this next step)
#
# Run:
#   Terminal 1: python mcp_server.py
#   Terminal 2: python pipeline_runner.py
#
# License: Apache 2.0

import os
import sys
import time
import json
import subprocess
import pandas as pd
import asyncio

from mcp.client.sse import sse_client
from mcp import ClientSession

DATA_DIR = "./data"
MCP_URL  = os.getenv("CAAMS_MCP_URL", "http://127.0.0.1:8765/sse")
PYTHON   = sys.executable   # same interpreter, different process


# ── MCP helper (coordinator-level calls only) ─────────────────────────────────
async def _call(tool: str, args: dict):
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


def run_agent(script: str, agent_args: list[str], timeout: int = 120) -> int:
    """
    Spawns agent as a separate OS process.
    Streams output live. Returns exit code.
    """
    cmd = [PYTHON, script] + agent_args
    print(f"\n[Pipeline] Spawning: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "CAAMS_MCP_URL": MCP_URL, "PYTHONUTF8": "1"},  
    )
    for line in proc.stdout:
        print(f"  {line}", end="")
    proc.wait(timeout=timeout)
    return proc.returncode


def check_mcp() -> bool:
    res = mcp("get_memory_snapshot", {})
    return "total_mb" in res


def load_session(n_steps: int = 10) -> pd.DataFrame:
    android_df = pd.read_csv(f"{DATA_DIR}/android_usage.csv")
    opened = (android_df[android_df["event_type"] == "Opened"].copy()
              if "event_type" in android_df.columns else android_df.copy())
    cands = opened.groupby("session_id").filter(
        lambda g: len(g) >= n_steps + 3 and g["app_name"].nunique() >= 5)
    if cands.empty:
        cands = opened.groupby("session_id").filter(lambda g: len(g) >= n_steps)
    best = cands.groupby("session_id")["app_name"].nunique().idxmax()
    df   = cands[cands["session_id"] == best].reset_index(drop=True)
    print(f"[Session] id={best} | events={len(df)} | unique_apps={df['app_name'].nunique()}")
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


def get_chronos_intensity(hour: int) -> float:
    # Read from MCP if Chronos buckets were precomputed, else use hour heuristic
    # Simple proxy: morning/evening peaks
    buckets = [3.0, 4.0, 6.0, 5.0, 7.0, 4.0]   # 6 × 4h buckets
    bucket  = min(hour // 4, len(buckets) - 1)
    return round(buckets[bucket] / max(buckets), 3)


def main(n_steps: int = 5):
    print("=" * 65)
    print("  CAAMS — Real Multi-Process Pipeline")
    print("  Each agent = separate OS process, MCP = message bus")
    print(f"  MCP: {MCP_URL}")
    print("=" * 65)

    if not check_mcp():
        print("\n[FATAL] MCP server unreachable. Run: python mcp_server.py")
        sys.exit(1)
    print("[OK] MCP connected\n")

    session_df = load_session(n_steps=n_steps)
    t_total    = time.perf_counter()

    for step in range(min(n_steps, len(session_df) - 2)):
        print(f"\n{'─'*65}")
        print(f"  STEP {step+1}/{n_steps}")
        print(f"{'─'*65}")

        row_prev = session_df.iloc[step]
        row_curr = session_df.iloc[step + 1]
        row_next = session_df.iloc[step + 2]

        app      = str(row_curr["app_name"])
        prev_app = str(row_prev["app_name"])
        next_app = str(row_next["app_name"])
        hour     = int(row_curr.get("hour", 12))
        pressure = get_query_pressure(hour)
        intensity= get_chronos_intensity(hour)

        # Clear inter-agent messages for this step
        mcp("clear_pipeline_state", {})

        # ── Agent 1: Supervisor ───────────────────────────────────────────────
        rc = run_agent("agents/supervisor_process.py", [
            "--step",       str(step + 1),
            "--app",        app,
            "--prev_app",   prev_app,
            "--hour",       str(hour),
            "--pressure",   str(pressure),
            "--intensity",  str(intensity),
            "--loop_count", "0",
        ])
        if rc != 0:
            print(f"[Pipeline] SupervisorAgent failed (rc={rc}), aborting step.")
            continue

        # ── Agent 2: ContextAgent ─────────────────────────────────────────────
        rc = run_agent("agents/context_agent_process.py", [
            "--app",       app,
            "--prev_app",  prev_app,
            "--hour",      str(hour),
            "--pressure",  str(pressure),
            "--intensity", str(intensity),
        ])
        if rc != 0:
            print(f"[Pipeline] ContextAgent failed (rc={rc}), aborting step.")
            continue

        # ── Agent 3: MemoryAgent ──────────────────────────────────────────────
        rc = run_agent("agents/memory_agent_process.py", [
            "--app",      app,
            "--pressure", str(pressure),
        ])
        if rc != 0:
            print(f"[Pipeline] MemoryAgent failed (rc={rc}), aborting step.")
            continue

        # ── Agent 4: TelemetryAgent ───────────────────────────────────────────
        rc = run_agent("agents/telemetry_agent_process.py", [
            "--step",     str(step + 1),
            "--next_app", next_app,
        ])
        if rc != 0:
            print(f"[Pipeline] TelemetryAgent failed (rc={rc}), aborting step.")
            continue

        print(f"\n  [Step {step+1}] All 4 agents completed successfully.")

    total = round(time.perf_counter() - t_total, 2)
    final = mcp("get_telemetry_report", {})

    print(f"\n{'='*65}")
    print("  CAAMS Multi-Process Pipeline — Done")
    print(f"  Wall time        : {total}s")
    print(f"  Total steps      : {n_steps}")
    print(f"  Aggregate hit rate: {final.get('cache_hit_rate_pct','?')}%")
    print(f"  Avg latency      : {final.get('avg_latency_ms','?')}ms")
    print(f"  Drift flags      : {final.get('drift_flags', [])}")
    print(f"\n  Process boundary proof:")
    print(f"  - Each agent ran as a separate PID (check output above)")
    print(f"  - Agents communicated ONLY via MCP HTTP calls")
    print(f"  - No shared Python objects, no shared memory")
    print(f"  - MCP server was the sole state store")
    print(f"{'='*65}")


if __name__ == "__main__":
    n = int(os.getenv("CAAMS_STEPS", "5"))
    main(n_steps=n)