# CAAMS Agent Specification
# Context-Aware Adaptive Memory System for Samsung Edge Devices
# License: Apache 2.0

## Agent Identity
name: CAAMS-MemoryAgent
version: 3.0
target_device: Samsung Galaxy S/A series (6-8 GB RAM), Samsung edge devices
runtime: Separate OS processes per agent, MCP over HTTP as message bus

## Architecture: Real Process-Level Multi-Agent System

Each agent runs as a **separate OS process** with its own PID and memory space.
Agents communicate **exclusively via MCP HTTP tool calls** — no shared Python
objects, no shared memory, no direct function calls between agents.

The MCP server is the **sole state store and message bus**.
[SupervisorAgent process]
|
| writes supervisor_directive → MCP set_pipeline_state
v
[ContextAgent process]
|
| reads supervisor_directive ← MCP get_pipeline_state
| calls predict_next_app    ← MCP tool
| writes context_output     → MCP set_pipeline_state
v
[MemoryAgent process]
|
| reads supervisor_directive ← MCP get_pipeline_state
| reads context_output       ← MCP get_pipeline_state
| calls evict_app, preload_app, allocate_app ← MCP tools
| writes memory_output       → MCP set_pipeline_state
v
[TelemetryAgent process]
|
| reads memory_output        ← MCP get_pipeline_state
| calls record_telemetry     ← MCP tool
| reads get_telemetry_report ← MCP tool
| writes telemetry_output    → MCP set_pipeline_state
|
v
[SupervisorAgent next step reads telemetry_output]

**Pipeline coordinator**: `pipeline_runner.py` spawns each agent as a
subprocess using Python `subprocess.Popen`. It does not share state with agents.

## Agent Processes

### SupervisorAgent (`agents/supervisor_process.py`)
- **Input**: MCP `get_telemetry_report`, `get_memory_snapshot`,
  `get_pipeline_state("telemetry_output")` from previous step
- **LLM**: Calls Qwen2.5-1.5B-Instruct locally to produce directive
- **Output**: Writes `supervisor_directive` JSON to MCP `set_pipeline_state`
- **Fallback**: If Qwen fails, deterministic policy based on free_pct and hit_rate

### ContextAgent (`agents/context_agent_process.py`)
- **Input**: Reads `supervisor_directive` from MCP `get_pipeline_state`
- **Tools**: Calls MCP `predict_next_app` (Markov chain)
- **LLM**: Calls Qwen2.5-1.5B-Instruct for prediction validation on
  non-standard tasks (aggressive/conservative modes)
- **Output**: Writes `context_output` JSON to MCP `set_pipeline_state`

### MemoryAgent (`agents/memory_agent_process.py`)
- **Input**: Reads `supervisor_directive` + `context_output` from MCP
- **Tools**: Calls MCP `rank_eviction` (RL Q-agent), `evict_app`,
  `allocate_app`, `preload_app`, `cache_lookup`, `adapt_cache_capacity`
- **LLM**: Calls Qwen2.5-1.5B-Instruct for cold path eviction reasoning
- **Output**: Writes `memory_output` JSON to MCP `set_pipeline_state`

### TelemetryAgent (`agents/telemetry_agent_process.py`)
- **Input**: Reads `memory_output` + `supervisor_directive` from MCP
- **Tools**: Calls MCP `record_telemetry`, `get_telemetry_report`,
  `get_memory_snapshot`
- **Output**: Writes `telemetry_output` (recommendation) to MCP
  `set_pipeline_state` — Supervisor reads this next step

## Agent Goal
Minimize app cold-start latency on Samsung smartphones and edge devices by
predicting the next app and pre-loading it before the switch happens.
Simultaneously manage eviction to prevent memory thrashing.

## Tools Available (via MCP HTTP)

### Prediction Tools
- `predict_next_app(prev_app, current_app, hour, top_k)`
  - Model: HourAwareMarkovPredictor (second-order, hour-conditioned)
  - Trained on: LSApp dataset (3.6M real Android app transitions)
  - Latency: <1ms

### Memory Tools
- `get_memory_snapshot()` — Samsung device pool state
- `allocate_app(app_name)` — foreground allocation (1.5x weight)
- `preload_app(app_name, pred_prob)` — background preload
- `evict_app(app_name)` — free memory
- `rank_eviction(candidates, memory_free_pct)` — RL Q-agent ranking

### Cache Tools
- `cache_lookup(app_name, pred_prob)` — LRU-F hit/miss check
- `get_cache_snapshot()` — cache state with retention scores
- `adapt_cache_capacity(free_device_pct, query_pressure)` — adaptive sizing

### Telemetry Tools
- `record_telemetry(step, hit, path, latency_ms, util_pct, thrash, notes)`
- `get_telemetry_report()` — aggregate KPIs with drift flags

### Pipeline State Tools (inter-agent message bus)
- `set_pipeline_state(key, value)` — agent writes output
- `get_pipeline_state(key)` — agent reads previous agent output
- `clear_pipeline_state()` — coordinator resets between steps

## Routing Logic

### Hot Path (default, <10ms target)
- Triggered when: `free_pct >= 25%` AND `query_pressure <= 0.85`
- Flow: Supervisor → ContextAgent → MemoryAgent (RL eviction) → TelemetryAgent
- LLM: Only Supervisor uses Qwen (directive generation)

### Cold Path (memory pressure)
- Triggered when: `free_pct < 25%` OR `query_pressure > 0.85`
- Flow: Same as hot path + MemoryAgent uses Qwen for eviction reasoning
- LLM: Supervisor + MemoryAgent both use Qwen

## Evaluation Harnesses (Single Process, Not Production Runtime)

These files use LangGraph for controlled KPI measurement and flow tracing.
They are evaluation tools, not the agent runtime.

- `kpi_scenarios.py` — measures all 7 KPIs against baseline
- `orchestrator.py` — traces agent communication flow step by step
- `memory_manager.py` — core simulation engine used by KPI harness

## Shared Skill Runtime (Evaluation Harness Only)

`multi_agent_orchestrator.py` provides a fail-safe skill execution boundary
used by `selftest_skill_failsafe.py` and `selftest_graph_failsafe.py`.
These self-tests prove the system handles skill failures without crashing.
This is not used by the production process-level agents.

## Open Weight Models Used
- **Qwen2.5-1.5B-Instruct** (Alibaba, Apache 2.0) — local CPU/GPU inference
- **HourAwareMarkovPredictor** (custom, Apache 2.0) — trained on LSApp
- **Chronos-T5-Small** (Amazon, Apache 2.0) — usage intensity forecasting
- **EvictionQAgent** (custom Q-table, Apache 2.0) — RL eviction ranking

## Real Datasets Used
- **LSApp** (Apache 2.0) — 3.6M Android app transitions, 87 apps
- **Melbourne Parking** (CC BY 4.0) — context query pressure signal
- **ShareGPT/LMSYS** (Apache 2.0) — KV cache workload sizing

## Target KPIs

| KPI | Target | Benchmark |
|-----|--------|-----------|
| Application Load Time Improvement | 20% | No optimization baseline |
| App Launch Time Improvement | 10%+ | No optimization baseline |
| Memory Thrashing Reduction | 50%+ | No optimization baseline |
| System Stability | 0 issues | No optimization baseline |
| Next Context Prediction Top-1 | >=75% | Random prediction baseline |
| Caching Hit Rate | >=85% | Static caching baseline |
| Memory Utilization Efficiency | 30%+ improvement | No optimization baseline |