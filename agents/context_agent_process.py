# agents/context_agent_process.py
# Reads:  MCP pipeline state: "supervisor_directive"
# Writes: MCP pipeline state: "context_output"
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
    parser.add_argument("--app",       type=str, required=True)
    parser.add_argument("--prev_app",  type=str, required=True)
    parser.add_argument("--hour",      type=int, required=True)
    parser.add_argument("--pressure",  type=float, default=0.3)
    parser.add_argument("--intensity", type=float, default=0.5)
    args = parser.parse_args()

    print(f"[ContextAgent PID={os.getpid()}] app={args.app} hour={args.hour}")

    # Read supervisor directive from MCP — this is the inter-process message
    raw = mcp("get_pipeline_state", {"key": "supervisor_directive"})
    if not raw.get("value"):
        print("  [ERROR] No supervisor_directive found in MCP state. Did SupervisorAgent run?")
        sys.exit(1)

    directive = json.loads(raw["value"])
    top_k     = int(directive.get("top_k", 3))
    task      = directive.get("context_task", "predict_standard")
    print(f"  [MCP] read supervisor_directive: task={task} top_k={top_k}")

    # Call MCP predict tool
    pred_res = mcp("predict_next_app", {
        "prev_app":    args.prev_app,
        "current_app": args.app,
        "hour":        args.hour,
        "top_k":       top_k,
    })
    preds = pred_res.get("predictions", [])
    print(f"  [MCP] predict_next_app -> {[p['app'] for p in preds]}")

    # Qwen validates predictions on non-standard tasks
    qwen_note = ""
    if task != "predict_standard":
        llm = get_local_llm()
        if llm:
            try:
                resp = llm.invoke([
                    SystemMessage(content="Validate app predictions. Reply ONLY JSON."),
                    HumanMessage(content=(
                        f"hour={args.hour}, pressure={args.pressure:.2f}, "
                        f"app={args.app!r}, predictions={preds}, task={task}. "
                        "Return: {\"drop_low_confidence\": true|false, "
                        "\"confidence_threshold\": 0.0-1.0, \"reasoning\": \"one sentence\"}"
                    )),
                ])
                vres = parse_json_response(resp)
                if vres and not vres.get("_error"):
                    threshold = float(vres.get("confidence_threshold", 0.0))
                    if vres.get("drop_low_confidence") and threshold > 0:
                        before = len(preds)
                        preds  = [p for p in preds if p.get("prob", 0) >= threshold]
                        print(f"  [Qwen] filtered {before}->{len(preds)} (threshold={threshold:.2f})")
                    qwen_note = vres.get("reasoning", "")
            except Exception as e:
                print(f"  [Qwen] validation skipped: {e}")

    context_output = {
        "predictions":       preds,
        "top_k_used":        top_k,
        "task":              task,
        "chronos_intensity": args.intensity,
        "qwen_note":         qwen_note,
    }

    # Write to MCP — MemoryAgent will read this
    mcp("set_pipeline_state", {
        "key":   "context_output",
        "value": json.dumps(context_output),
    })
    print(f"  [MCP] wrote context_output -> MemoryAgent will read this")
    print(f"  [ContextAgent] done | predictions={[p['app'] for p in preds]}")


if __name__ == "__main__":
    main()
