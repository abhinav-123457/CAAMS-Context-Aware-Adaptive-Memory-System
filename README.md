# CAAMS — Context-Aware Adaptive Memory System

**Phase 1 Prototype** for: *Context-Aware, Adaptive Memory Solution for
Mobile Agentic Systems* (Samsung ennovateX AX Hackathon)

**Important**: This is a PC simulation prototype. KPI numbers are simulation
harness results, not Android/Samsung smartphone measurements. Phase 2 targets
on-device deployment on Samsung Galaxy S/A series.

---

## What This Prototype Demonstrates

### 1. Real Process-Level Multi-Agent System

Each agent runs as a **separate OS process**. Agents communicate **only via
MCP HTTP tool calls** - no shared Python memory and no shared objects.

```text
SupervisorAgent
  reads MCP telemetry
  calls Qwen
  writes supervisor_directive
        |
        v
ContextAgent
  reads supervisor_directive
  calls predict_next_app MCP tool
  writes context_output
        |
        v
MemoryAgent
  reads supervisor_directive + context_output
  evicts/preloads via MCP tools
  writes memory_output
        |
        v
TelemetryAgent
  reads memory_output
  writes telemetry log
  sends recommendation back to SupervisorAgent
        |
        v
SupervisorAgent next step
```

**Run this with**: `python pipeline_runner.py`

### 2. LLM Placement & Safety Gates

Qwen2.5-1.5B-Instruct is **only** called by the SupervisorAgent and **only** when:
- `free_pct >= 25%` (device has breathing room)
- **AND** `query_pressure <= 0.85` (system not saturated)

Below these thresholds, the Supervisor uses deterministic fallback rules. 
**MemoryAgent never calls any LLM** — it uses the EvictionQAgent (tabular Q-learning,
<1ms, zero RAM). Rationale: running a 1.5B model on a memory-constrained device
causes OOM, thrashing, or latency spikes that defeat the purpose of the memory manager.

### 3. Eviction Strategy: RL Q-Agent on All Paths

The EvictionQAgent unconditionally ranks eviction candidates on **hot and cold paths**.

- **Why RL?** Sub-millisecond latency, zero additional memory, trained offline on 3.6M real Android app transitions
- **Why RL?** Sub-millisecond latency, zero additional memory, trained offline on 1.67M real Android app transitions
- **Why not Qwen?** Cold path triggers at free_pct < 25%. A 1.5B model call at that point would thrash memory further
- **Skill name**: `rl_cold_eviction` in skill registry (see [skills.md](skills.md))

### 4. Context-Aware Memory Allocation

Memory manager dynamically allocates based on:
- Current and predicted app context
- Hour-of-day usage patterns (Markov + Chronos)
- Real query pressure from Melbourne parking dataset
- KV cache pressure from ShareGPT workloads

### 5. Predictive Pre-Loading

- HourAwareMarkovPredictor (second-order) trained on 3.6M real Android
  app transitions from LSApp dataset
- HourAwareMarkovPredictor (second-order) trained on 1.67M real Android
  app transitions from LSApp dataset
- Predicts top-k next apps before user switches
- Chronos-T5-Small forecasts usage intensity per hour bucket

### 6. Adaptive Caching & Eviction

- LRU-F cache: recency × frequency × prediction bonus scoring
- Adaptive capacity: shrinks under KV cache pressure, expands under query load
- RL Q-agent (tabular Q-learning, trained offline) ranks eviction candidates on **all paths** (hot and cold)
  - Hot path: deterministic RL ranking, <1ms latency
  - Cold path: same RL ranking, no LLM involvement in eviction decisions

---

## Repository Structure

```text
caams/
├── agents/                          # Production: separate OS processes
│   ├── supervisor_process.py        # Reads MCP telemetry, Qwen directive
│   ├── context_agent_process.py     # Reads directive, calls predict MCP tool
│   ├── memory_agent_process.py      # Reads both, executes via MCP tools
│   └── telemetry_agent_process.py   # Reads output, writes recommendation
├── pipeline_runner.py               # Spawns agents as subprocesses
├── mcp_server.py                    # MCP tool server (SSE transport)
│                                    # All agent tools + pipeline state bus
├── device_pool.py                   # Samsung device memory pool simulation
├── cache_manager.py                 # Adaptive LRU-F cache
├── context_predictor.py             # Markov predictor + Chronos forecaster
├── rl_eviction_policy.py            # Q-learning eviction agent
├── local_llm.py                     # Qwen2.5-1.5B local inference
├── kpi_scenarios.py                 # All 7 KPIs measured vs baseline
├── LangGraph flow showcase.py       # Agent flow showcase (LangGraph)
├── memory_manager.py                # Core simulation engine
├── selftest_skill_failsafe.py       # Proves skill failure handling
├── selftest_graph_failsafe.py       # Proves LangGraph continuity under failure
├── agents.md                        # Agent architecture specification
├── skills.md                        # Skill contracts specification
├── multi_agent_orchestrator.py      # Skill registry for evaluation harness
├── data_loader.py                   # Downloads and prepares real datasets
├── data/
│   ├── android_usage.csv           # LSApp (generated by data_loader.py)
│   ├── melbourne_context.csv       # Melbourne parking (generated)
│   └── kv_cache_workloads.csv      # ShareGPT workloads (generated)
└── models/
    ├── eviction_qtable.pkl         # Trained RL Q-table
    ├── chronos-t5-small/           # Chronos weights (if local)
    └── qwen2.5-1.5b-instruct/      # Qwen weights (if local)
```

---

## Open Weight Models (All Apache 2.0)

| Model | Use | Constraint |
|-------|-----|------------|
| Qwen2.5-1.5B-Instruct (Alibaba) | Supervisor directive generation only | Only when free_pct ≥ 25% AND query_pressure ≤ 0.85 |
| Chronos-T5-Small (Amazon) | Usage intensity forecasting | Always safe (<100ms) |
| HourAwareMarkovPredictor (custom) | Next app prediction | Always safe (<1ms) |
| EvictionQAgent (custom Q-table) | Eviction ranking, all paths | Always safe (<1ms, zero RAM) |

---

## Real Datasets Used (No Synthetic Data)

| Dataset | License | Use |
|---------|---------|-----|
| LSApp (3.6M Android app events) | Apache 2.0 | Markov training + session replay |
| LSApp (1.67M Android app events) | Apache 2.0 | Markov training + session replay |
| Melbourne Parking Sensors | CC BY 4.0 | Query pressure signal |
| ShareGPT 52K conversations | Apache 2.0 | KV cache workload sizing |

---

## First-Time Setup (Run Once)

```bash
# Install dependencies
pip install -r requirements.txt

# Download data + verify models + train RL agent
python setup.py
```

That's it. `setup.py` handles everything in the right order:
- Downloads LSApp, Melbourne Parking, ShareGPT from their public sources
- Verifies Qwen2.5-1.5B loads (~3GB, auto-cached by HuggingFace)
- Verifies Chronos-T5-Small loads (~250MB, auto-cached)
- Trains and saves the RL eviction Q-table on real LSApp data

If something fails, setup.py tells you exactly which step and why.

### What a Fresh User Actually Does

1. `git clone <repo>`
2. `pip install -r requirements.txt`
3. `python setup.py`          ← one command, handles everything
4. `python mcp_server.py`     ← Terminal 1, keep open
5. `python pipeline_runner.py` ← Terminal 2

---

## Running the System

### Option 1: Real Multi-Process Pipeline (Primary Demo)

Each agent is a separate OS process. PIDs are printed for verification.

```bash
# Terminal 1 — keep running
python mcp_server.py

# Terminal 2
# Windows PowerShell:
$env:CAAMS_STEPS="5"; $env:PYTHONUTF8="1"; python pipeline_runner.py

# Linux/Mac:
CAAMS_STEPS=5 python pipeline_runner.py
```

What you will see:
- 4 different PIDs printed (one per agent per step)
- Each agent reads from MCP, writes to MCP
- No agent directly calls another agent's code

### Option 2: KPI Measurement (Evaluation Harness)

```bash
# Terminal 1 — keep running
python mcp_server.py

# Terminal 2
python kpi_scenarios.py
```

Measures all 7 KPIs against honest baselines. LangGraph single process
for timing accuracy.

### Option 3: Agent Flow Trace (LangGraph Showcase)

```bash
python "LangGraph flow showcase.py"
```

Traces agent communication step by step. Single process. Useful for
debugging agent logic without process overhead.

### Self-Tests

```bash
python selftest_skill_failsafe.py
python selftest_graph_failsafe.py
```

---

## AI Development Guidelines Compliance

| Guideline | How Met |
|-----------|---------|
| Agentic workflows | 4 agents, assess → plan → execute → validate per step |
| Reasoning & planning | Supervisor calls Qwen, produces typed directive each step |
| Tool use / tool chaining | All agents call MCP tools only, no direct calls |
| MCP servers | `mcp_server.py` exposes 15 tools via SSE transport |
| agents.md + skills.md | Full specification in repo root |
| Memory / context handling | Stateful pool + cache across steps in MCP server |
| Multi-agent orchestration | 4 separate OS processes, MCP message bus |
| Open weight models | Qwen2.5-1.5B + Chronos-T5-Small + custom RL agent |
| No third-party APIs | All inference is local, all data is open license |

---

## KPI Targets

| KPI | Target | Benchmark |
|-----|--------|-----------|
| Application Load Time Improvement | 20% | No optimization baseline |
| App Launch Time Improvement | 10%+ | No optimization baseline |
| Memory Thrashing Reduction | 50%+ | No optimization baseline |
| System Stability | 0 issues | No optimization baseline |
| Next Context Prediction Top-1 | >=75% | Random prediction baseline |
| Caching Hit Rate | >=85% | Static caching baseline |
| Memory Utilization Efficiency | 30%+ improvement | No optimization baseline |

---

## Phase 2 Migration Path (Samsung Edge Devices)

- Replace simulation memory pool with Android memory APIs
- Move Qwen cold path to on-device runtime (llama.cpp on Exynos/Snapdragon NPU)
- Replace Melbourne proxy signal with on-device context sensors
- Measure KPIs directly on Samsung Galaxy S/A series hardware

---

## License

Apache License 2.0