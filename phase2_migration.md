# CAAMS Phase 2 Migration Guide
# PC Prototype → Samsung Edge Device / Smartphone
# License: Apache 2.0

## What Changes in Phase 2

### 1. Real Memory API (Biggest Change)

Replace DeviceMemoryPool simulation with real Android APIs:

PC simulation:
  pool.allocated = {}  # Python dict

Android replacement:
  ActivityManager.getMemoryInfo()           # real free RAM
  ActivityManager.getRunningAppProcesses()  # real resident apps
  ComponentCallbacks2.onTrimMemory()        # system pressure signal
  ActivityManager.killBackgroundProcesses() # real eviction

Bridge layer needed:
  A thin Python-to-Android JNI bridge or
  Rewrite memory_manager.py in Kotlin/Java for the Android layer
  and keep prediction/cache logic in Python via Chaquopy or similar.

### 2. Qwen Quantization

Current: Qwen2.5-1.5B in float32 = ~6GB RAM on CPU
Target:  Qwen2.5-1.5B in 4-bit  = ~800MB RAM

Tools:
  llama.cpp GGUF format (Q4_K_M quantization)
  bitsandbytes 4-bit NF4 quantization
  Samsung NPU via ONE-SDK (if Exynos target)

local_llm.py change needed:
  Replace AutoModelForCausalLM with llama-cpp-python
  or use Samsung's on-device AI APIs

### 3. Agent Process → Android Service

Current: subprocess.Popen per agent
Android: each agent becomes an Android Service

  SupervisorAgent   → Bound Service
  ContextAgent      → Intent Service
  MemoryAgent       → Foreground Service (needs persistent access)
  TelemetryAgent    → Scheduled Job (WorkManager)

pipeline_runner.py → Android ServiceConnection + Intent dispatch

### 4. MCP Server as Android Background Service

Current: FastMCP SSE on 127.0.0.1:8765
Android: FastMCP inside a Foreground Service with:
  startForegroundService()
  WAKE_LOCK permission
  REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
  Doze mode whitelist

### 5. File Paths

Current:
  DATA_DIR   = "./data"
  MODELS_DIR = "./models"

Android:
  DATA_DIR   = context.getFilesDir() + "/data"
  MODELS_DIR = context.getFilesDir() + "/models"

Add a config.py:
  import os
  DATA_DIR   = os.getenv("CAAMS_DATA_DIR",   "./data")
  MODELS_DIR = os.getenv("CAAMS_MODELS_DIR", "./models")

This makes paths injectable without code changes.

### 6. What Does NOT Change

  HourAwareMarkovPredictor  — pure Python, runs as-is on ARM
  RL Q-table (pickle)       — runs as-is on ARM, <1ms
  LRU-F cache logic         — runs as-is on ARM
  MCP tool definitions      — run as-is, just different transport
  Agent logic               — runs as-is, just different process model
  KPI measurement logic     — runs as-is for on-device benchmarking

## Recommended Phase 2 Stack

  Language:   Python (Chaquopy) + Kotlin for Android layer
  LLM:        llama.cpp GGUF Q4_K_M or Samsung On-Device AI API
  Transport:  Unix domain socket instead of TCP (lower latency on-device)
  Memory API: Android ActivityManager + ComponentCallbacks2
  Deployment: APK with foreground service for MCP server
