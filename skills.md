# CAAMS Skills Specification
# Context-Aware Adaptive Memory System
# License: Apache 2.0

## Purpose
Defines the reusable skill modules used by CAAMS production agents.
Skills map to MCP tool calls made by process-level agents in the `agents/`
directory. Each skill has a primary owner agent and a defined input/output
contract.

## Execution Model

In production (process-level agents):
- Skills are MCP tool calls over HTTP
- Each agent process calls MCP tools directly
- No shared state between agents — MCP server is the only state store
- Failure handling: if MCP call fails, agent uses deterministic fallback
  and continues — no crash, no pipeline abort

In evaluation harness (single-process, LangGraph):
- Skills are registered in `SkillRegistry` in `multi_agent_orchestrator.py`
- `SkillExecutor` provides fail-safe boundary (unknown skill, exception,
  invalid output all produce deterministic fallback, not crash)
- Used by self-tests to prove resilience

## Production Agent Skills

### skill: memory_pressure_triage
- **purpose**: classify pressure level, decide hot vs cold path
- **owner**: SupervisorAgent
- **callable by**: all agents
- **MCP tools used**: `get_telemetry_report`, `get_memory_snapshot`
- **inputs**:
  - `free_pct` — device memory free percentage
  - `query_pressure` — Melbourne parking signal (0-1)
  - `hit_rate` — current cache hit rate from telemetry
- **outputs**:
  - `path` in `{hot, cold}`
  - `directive` JSON written to MCP pipeline state
- **trigger rule**: cold when `free_pct < 25` OR `query_pressure > 0.85`
- **latency target**: <1ms (deterministic fallback path)

### skill: preload_candidate_ranking
- **purpose**: rank predicted apps for preloading within memory budget
- **owner**: ContextAgent
- **callable by**: all agents
- **MCP tools used**: `predict_next_app`
- **inputs**:
  - `prev_app`, `current_app`, `hour` — Markov state
  - `top_k` — from supervisor directive
  - `task` — predict_standard / predict_aggressive / predict_conservative
- **outputs**:
  - `predictions` list with app names and probabilities
  - written to MCP `context_output` pipeline state
- **model**: HourAwareMarkovPredictor (second-order, Apache 2.0)
- **latency target**: <1ms on hot path

### skill: adaptive_eviction_policy
- **purpose**: evict lowest-value residents, prevent thrashing
- **owner**: MemoryAgent
- **callable by**: all agents
- **MCP tools used**: `rank_eviction`, `evict_app`, `allocate_app`,
  `preload_app`, `cache_lookup`, `adapt_cache_capacity`
- **inputs**:
  - `supervisor_directive` — from MCP pipeline state
  - `context_output` — predictions from ContextAgent via MCP
  - `memory_snapshot` — from MCP `get_memory_snapshot`
- **outputs**:
  - evictions executed via MCP `evict_app`
  - preloads executed via MCP `preload_app`
  - `memory_output` written to MCP pipeline state
- **hot path**: RL Q-agent via MCP `rank_eviction`
- **cold path**: Qwen2.5-1.5B reasoning + RL fallback
- **safety rule**: never evict `protect_apps` from supervisor directive

### skill: context_window_maintenance
- **purpose**: keep Markov transition context fresh per app switch
- **owner**: ContextAgent
- **callable by**: all agents
- **inputs**:
  - `prev_app`, `current_app`, `hour` — updated each step
- **outputs**:
  - updated second-order state used by next predict call
  - stateful within MCP server (Markov model lives in server process)
- **persistence**: Markov model state is maintained in MCP server process
  across all app switch events

### skill: telemetry_validation
- **purpose**: validate runtime KPIs, detect drift, produce recommendations
- **owner**: TelemetryAgent
- **callable by**: all agents
- **MCP tools used**: `record_telemetry`, `get_telemetry_report`,
  `get_memory_snapshot`
- **inputs**:
  - `memory_output` — from MCP pipeline state
  - `next_app` — ground truth for hit/miss evaluation
- **outputs**:
  - step written to MCP telemetry log via `record_telemetry`
  - `telemetry_output` recommendation written to MCP pipeline state
  - Supervisor reads this at start of next step
- **thresholds**:
  - hit rate `>= 85%` → maintain policy
  - hit rate `< 85%` → increase preloads
  - hit rate `< 75%` → increase top_k and preloads
  - free_pct `< 20%` → trigger eviction

## Fail-Safe Behavior

All production agents implement this fallback chain:
1. Call MCP tool
2. If MCP unavailable or returns error → use deterministic local policy
3. Log the fallback in output JSON
4. Continue pipeline — never abort on single skill failure

Deterministic fallbacks:
- Supervisor: rule-based directive from free_pct + hit_rate thresholds
- ContextAgent: top-3 predictions from first-order Markov fallback
- MemoryAgent: LRU-based eviction without Qwen
- TelemetryAgent: local hit/miss check from MCP snapshot

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