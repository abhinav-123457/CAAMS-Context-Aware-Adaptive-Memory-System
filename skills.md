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

## LLM Safety Constraint (applies to all skills)

No skill invokes Qwen2.5-1.5B when `free_pct < 25%` or
`query_pressure > 0.85`. Below these thresholds every skill falls through
to its deterministic path unconditionally. This is enforced in code,
not by convention.

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
- **LLM use**: Qwen called only on hot path (`free_pct >= 25` AND
  `pressure <= 0.85`). Deterministic rule is the fallback and the
  unconditional cold-path mechanism.
- **latency target**: <1ms on deterministic path

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
- **LLM use**: None. Prediction filtering on aggressive task uses
  probability threshold only (>= 0.05), no Qwen call.
- **latency target**: <1ms

### skill: rl_cold_eviction
- **purpose**: rank and execute evictions to free memory, prevent thrashing
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
- **eviction mechanism**: RL Q-agent via MCP `rank_eviction` — unconditional
  on both hot and cold paths. Sub-millisecond, zero additional RAM.
- **LLM use**: None. Qwen is never called in this skill or the agent that
  owns it. The previous design calling Qwen on cold path was a system design
  error: running a 1.5B model when free_pct < 25% risks OOM on edge devices.
- **safety rule**: never evict `protect_apps` from supervisor directive
- **previously named**: `adaptive_eviction_policy` (renamed to reflect
  actual mechanism — RL Q-table, not LLM)

### skill: context_window_maintenance
- **purpose**: keep Markov transition context fresh per app switch
- **owner**: ContextAgent
- **callable by**: all agents
- **inputs**:
  - `prev_app`, `current_app`, `hour` — updated each step
- **outputs**:
  - updated second-order state used by next predict call
  - stateful within MCP server (Markov model lives in server process)
- **LLM use**: None.
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
- **LLM use**: None.
- **thresholds**:
  - hit rate `>= 85%` → maintain policy
  - hit rate `< 85%` → increase preloads
  - hit rate `< 75%` → increase top_k and preloads
  - free_pct `< 20%` → trigger RL eviction

## Fail-Safe Behavior

All production agents implement this fallback chain:
1. Call MCP tool
2. If MCP unavailable or returns error → use deterministic local policy
3. Log the fallback in output JSON
4. Continue pipeline — never abort on single skill failure

Deterministic fallbacks:
- Supervisor: rule-based directive from `free_pct` + `hit_rate` thresholds
- ContextAgent: top-3 predictions from first-order Markov fallback
- MemoryAgent: LRU candidate list from `pool.lru_candidates()`, no Qwen
- TelemetryAgent: local hit/miss check from MCP snapshot

## Skill Registry (multi_agent_orchestrator.py)

```python
SkillRegistry(
    skills={
        "memory_pressure_triage":    skill_memory_pressure_triage,
        "preload_candidate_ranking":  cp_assess_context,
        "rl_cold_eviction":          ma_rl_eviction,     # was: ma_qwen_eviction
        "context_window_maintenance": cp_assess_context,
        "telemetry_validation":       ma_validate,
    },
    primary_owner={
        "memory_pressure_triage":    "supervisor",
        "preload_candidate_ranking":  "context_predictor",
        "rl_cold_eviction":          "memory_allocator",
        "context_window_maintenance": "context_predictor",
        "telemetry_validation":       "supervisor",
    },
    cross_agent_allowed=True,
)
```

## KPI Targets

| KPI | Target | Measurement |
|-----|--------|-------------|
| Application Load Time Improvement | 20% | Simulated bound — see latency_probe.py |
| App Launch Time Improvement | 10%+ | Simulated bound — see latency_probe.py |
| Memory Thrashing Reduction | 50%+ | Measured on real LSApp sessions |
| System Stability | 0 issues | Measured — self-tests pass |
| Next Context Prediction Top-1 | >=75% | Measured on held-out LSApp test set |
| Caching Hit Rate | >=85% | Measured on real LSApp sessions |
| Memory Utilization Efficiency | 30%+ improvement | Measured on real LSApp sessions |

Load time and launch time figures are derived from measured hit rates combined
with published Android cold-start benchmarks (AOSP/Samsung Knox, 180-280ms).
They are labeled as simulated bounds, not hardware measurements.
Phase 2 replaces these with on-device profiling on Samsung Galaxy hardware.