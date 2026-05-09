# agents/context_agent_process.py — CAAMS ContextAgent (separate OS process)
#
# Called by pipeline_runner.py as a subprocess per step.
# Reads supervisor_directive from MCP → calls predict_next_app tool →
# optionally filters predictions → writes context_output to MCP.
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
    parser.add_argument("--app",       type=str,   required=True)
    parser.add_argument("--prev_app",  type=str,   required=True)
    parser.add_argument("--hour",      type=int,   required=True)
    parser.add_argument("--pressure",  type=float, default=0.3)
    parser.add_argument("--intensity", type=float, default=0.5)
    args = parser.parse_args()

    print(f"[ContextAgent PID={os.getpid()}] app={args.app} hour={args.hour}")

    # Read supervisor directive
    raw       = mcp("get_pipeline_state", {"key": "supervisor_directive"})
    directive = json.loads(raw.get("value", "{}")) if raw.get("value") else {}
    top_k     = int(directive.get("top_k", 3))
    task      = directive.get("context_task", "predict_standard")

    print(f"  Directive read: top_k={top_k} task={task}")

    # Call MCP predict tool — this is the ONLY prediction mechanism
    pred_res = mcp("predict_next_app", {
        "prev_app":    args.prev_app,
        "current_app": args.app,
        "hour":        args.hour,
        "top_k":       top_k,
    })
    preds = pred_res.get("predictions", [])
    print(f"  MCP predict_next_app → {[p['app'] for p in preds]}")

    # On aggressive task: filter very low-confidence predictions
    # This does NOT require Qwen — pure probability threshold
    if task == "predict_aggressive":
        before = len(preds)
        preds  = [p for p in preds if p.get("prob", 0) >= 0.05]
        print(f"  Aggressive filter: {before} → {len(preds)} predictions (threshold=0.05)")

    context_output = {
        "predictions":       preds,
        "top_k_used":        top_k,
        "task":              task,
        "chronos_intensity": args.intensity,
        "from_agent":        "context_agent",
    }

    mcp("set_pipeline_state", {
        "key":   "context_output",
        "value": json.dumps(context_output),
    })
    print(f"[ContextAgent] Done — context_output written to MCP pipeline state")


if __name__ == "__main__":
    main()