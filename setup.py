# setup.py — CAAMS First-Time Setup
#
# Run this once before anything else.
# Downloads all data, verifies models, trains RL agent.
#
# Usage: python setup.py
# License: Apache 2.0

import os
import sys
import time

print("=" * 60)
print("  CAAMS — First-Time Setup")
print("=" * 60)
print()
print("This will:")
print("  1. Download real datasets (LSApp, Melbourne, ShareGPT)")
print("  2. Verify Qwen2.5-1.5B can load (downloads ~3GB if not cached)")
print("  3. Verify Chronos-T5-Small can load (~250MB if not cached)")
print("  4. Train and save the RL eviction Q-table")
print()
print("Estimated time on first run: 10-30 min (mostly model downloads)")
print("Subsequent runs: ~2 min")
print()

# ── Step 1: Data ──────────────────────────────────────────────────────────────
print("─" * 60)
print("STEP 1/4 — Downloading real datasets")
print("─" * 60)

DATA_DIR = "./data"
MODELS_DIR = "./models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

data_files = [
    f"{DATA_DIR}/android_usage.csv",
    f"{DATA_DIR}/melbourne_context.csv",
    f"{DATA_DIR}/kv_cache_workloads.csv",
]

all_data_present = all(os.path.exists(f) for f in data_files)

if all_data_present:
    print("[SKIP] All data files already present.")
    for f in data_files:
        size_mb = round(os.path.getsize(f) / 1024 / 1024, 1)
        print(f"  {f}  ({size_mb} MB)")
else:
    print("[RUN] Running data_loader.py ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "data_loader.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[FAIL] data_loader.py failed.")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    missing = [f for f in data_files if not os.path.exists(f)]
    if missing:
        print(f"\n[FAIL] These files were not created: {missing}")
        sys.exit(1)

    print("\n[OK] All datasets downloaded.")

# ── Step 2: Qwen ──────────────────────────────────────────────────────────────
print()
print("─" * 60)
print("STEP 2/4 — Verifying Qwen2.5-1.5B-Instruct (Apache 2.0)")
print("  Source: Qwen/Qwen2.5-1.5B-Instruct on HuggingFace")
print("  Size:   ~3 GB (downloads once, cached in HuggingFace cache)")
print("─" * 60)

try:
    from local_llm import get_local_llm
    llm = get_local_llm()
    if llm is None:
        print("[WARN] Qwen failed to load.")
        print("  The system will use deterministic fallback policies.")
        print("  Cold path will work but without LLM reasoning.")
        print("  To fix: ensure transformers and torch are installed correctly.")
    else:
        print("[OK] Qwen2.5-1.5B ready.")
except Exception as e:
    print(f"[WARN] Qwen load error: {e}")
    print("  Deterministic fallback will be used.")

# ── Step 3: Chronos ───────────────────────────────────────────────────────────
print()
print("─" * 60)
print("STEP 3/4 — Verifying Chronos-T5-Small (Amazon, Apache 2.0)")
print("  Source: amazon/chronos-t5-small on HuggingFace")
print("  Size:   ~250 MB (downloads once, cached)")
print("─" * 60)

try:
    import pandas as pd
    from context_predictor import ChronosUsageForecaster, load_lsapp
    df     = load_lsapp()
    cf     = ChronosUsageForecaster()
    series = cf.build_hourly_series(df)
    result = cf.forecast(series, prediction_length=6)
    print(f"[OK] Chronos-T5-Small ready.")
    print(f"     Forecast: {result['mean_launches']}")
    print(f"     Peak at +{result['peak_hour_offset']}h")
except Exception as e:
    print(f"[WARN] Chronos load error: {e}")
    print("  Uniform defaults will be used for intensity signal.")

# ── Step 4: RL Q-table ────────────────────────────────────────────────────────
print()
print("─" * 60)
print("STEP 4/4 — Training RL Eviction Q-table")
print("  Dataset: LSApp (real Android app transitions)")
print("  Time:    ~1-2 min on CPU")
print("─" * 60)

QTABLE_PATH = f"{MODELS_DIR}/eviction_qtable.pkl"

if os.path.exists(QTABLE_PATH):
    size_kb = round(os.path.getsize(QTABLE_PATH) / 1024, 1)
    print(f"[SKIP] Q-table already exists ({size_kb} KB).")
    print(f"  Path: {QTABLE_PATH}")
    print("  Delete this file and re-run setup.py to retrain.")
else:
    print("[RUN] Training Q-table on LSApp data ...")
    try:
        import pandas as pd
        from context_predictor import load_lsapp
        from rl_eviction_policy import train_eviction_agent

        df    = load_lsapp()
        agent = train_eviction_agent(df, episodes=20, max_steps=1000)
        agent.save(QTABLE_PATH)
        print(f"[OK] Q-table trained and saved to {QTABLE_PATH}")
    except Exception as e:
        print(f"[FAIL] RL training failed: {e}")
        print("  Run manually: python rl_eviction_policy.py")
        sys.exit(1)

# ── Final checklist ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Setup Complete — Checklist")
print("=" * 60)

checks = {
    "android_usage.csv":        f"{DATA_DIR}/android_usage.csv",
    "melbourne_context.csv":    f"{DATA_DIR}/melbourne_context.csv",
    "kv_cache_workloads.csv":   f"{DATA_DIR}/kv_cache_workloads.csv",
    "eviction_qtable.pkl":      QTABLE_PATH,
}

all_ok = True
for label, path in checks.items():
    exists = os.path.exists(path)
    icon   = "OK  " if exists else "FAIL"
    print(f"  [{icon}] {label}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("  All checks passed. Ready to run.")
    print()
    print("  How to run:")
    print()
    print("  Terminal 1 (keep open):")
    print("    python mcp_server.py")
    print()
    print("  Terminal 2 — Real multi-process pipeline:")
    print("    Windows: $env:PYTHONUTF8='1'; python pipeline_runner.py")
    print("    Linux:   python pipeline_runner.py")
    print()
    print("  Terminal 2 — KPI measurement:")
    print("    python kpi_scenarios.py")
    print()
    print("  Terminal 2 — Agent flow trace:")
    print("    python orchestrator.py")
else:
    print("  Some checks failed. Fix errors above and re-run setup.py")
    sys.exit(1)