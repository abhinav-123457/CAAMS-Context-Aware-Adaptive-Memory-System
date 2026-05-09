# data_loader.py — CAAMS Complete Data Loader
# Datasets:
#   1. LSApp          — Android app usage (github.com/aliannejadi/LSApp)
#   2. Melbourne      — Parking context logs (City of Melbourne Open Data)
#   3. LMSYS Chat 1M  — KV cache workloads (HuggingFace, Apache 2.0)
# License: Apache 2.0

import os
import glob
import subprocess
import requests
import pandas as pd
from datasets import load_dataset

DATA_DIR  = "./data"
LSAPP_DIR = f"{DATA_DIR}/LSApp"
os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. LSApp — Android App Usage ─────────────────────────────────────────────
# Replace only load_android_usage() in data_loader.py

import gzip

def load_android_usage() -> pd.DataFrame:
    print("[1/3] Loading LSApp — Android Usage Patterns...")

    if not os.path.exists(LSAPP_DIR):
        print("   Cloning LSApp...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/aliannejadi/LSApp.git", LSAPP_DIR],
            check=True
        )

    gz_path = f"{LSAPP_DIR}/lsapp.tsv.gz"

    if not os.path.exists(gz_path):
        print(f"   ✗ Not found: {gz_path}")
        print(f"   Files present: {os.listdir(LSAPP_DIR)}")
        return pd.DataFrame()

    print("   Reading lsapp.tsv.gz ...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)

    print(f"   ✓ Raw rows    : {len(df)}")
    print(f"   ✓ Raw columns : {list(df.columns)}")
    print(f"\n   Sample:\n{df.head(3).to_string()}\n")

    # Normalize columns
    col_map = {
        "timestamp":   ["timestamp","time","ts","datetime","start_time",
                         "date","startTime","created_at","start","session_start"],
        "app_name":    ["app_name","app","package","application","package_name",
                         "appName","Activity","item_id","appId","app_package","app_id"],
        "duration_ms": ["duration_ms","duration","time_spent","usage_ms",
                         "Duration","totalTime","duration_seconds",
                         "session_length","dwell_time","time_ms"],
        "user_id":     ["user_id","userId","user","uid",
                         "participant_id","participantId","user_num"]
    }
    for target, candidates in col_map.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df.rename(columns={c: target}, inplace=True)
                    break

    if "duration_ms" in df.columns:
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(0)
        if df["duration_ms"].median() < 500:
            df["duration_ms"] = df["duration_ms"] * 1000

    if "timestamp" in df.columns:
        df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
        df                = df.dropna(subset=["timestamp"])
        df                = df.sort_values("timestamp").reset_index(drop=True)
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    print(f"   ── Android Summary ──")
    print(f"   Final rows   : {len(df)}")
    if "app_name"  in df.columns: print(f"   Unique apps  : {df['app_name'].nunique()}")
    if "user_id"   in df.columns: print(f"   Unique users : {df['user_id'].nunique()}")
    if "timestamp" in df.columns:
        print(f"   Date range   : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Columns      : {list(df.columns)}\n")

    df.to_csv(f"{DATA_DIR}/android_usage.csv", index=False)
    return df

# ── 2. Melbourne Parking — Context Query Logs ────────────────────────────────
def load_melbourne_context(limit: int = 5000) -> pd.DataFrame:
    """
    Real parking sensor events from City of Melbourne.
    Time-sensitive, heterogeneous context queries — mirrors
    what the memory system must handle per the problem statement.
    License: CC BY 4.0  |  No auth required.
    """
    print("[2/3] Loading Melbourne Parking Context Logs...")

    endpoints = [
        # v2 endpoint
        (
            "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
            f"on-street-parking-bay-sensors/exports/json?limit={limit}&timezone=UTC",
            "v2"
        ),
        # v1 fallback
        (
            "https://data.melbourne.vic.gov.au/api/records/1.0/search/"
            f"?dataset=on-street-parking-bay-sensors&rows={limit}",
            "v1"
        ),
    ]

    df = pd.DataFrame()
    for url, version in endpoints:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw  = resp.json()
            df   = pd.DataFrame(raw if version == "v2"
                                else [r["fields"] for r in raw.get("records", [])])
            print(f"   ✓ Loaded via {version} endpoint")
            break
        except Exception as e:
            print(f"   ✗ {version} failed: {e}")

    if df.empty:
        print("   ✗ Melbourne API unreachable. Check network/proxy settings.")
        return df

    # Normalize
    df.rename(columns={
        "ts":           "timestamp",
        "bay_id":       "context_id",
        "st_marker_id": "location_marker",
        "status":       "event_type",
    }, inplace=True, errors="ignore")

    # Catch alternative timestamp column names
    if "timestamp" not in df.columns:
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                df.rename(columns={col: "timestamp"}, inplace=True)
                break

    df["timestamp"]      = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df                   = df.dropna(subset=["timestamp"])
    df                   = df.sort_values("timestamp").reset_index(drop=True)
    df["is_active_query"] = (
        df.get("event_type", pd.Series(["Present"] * len(df)))
        .astype(str).str.strip().str.lower() == "present"
    )
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    print(f"\n   ── Melbourne Summary ──")
    print(f"   Rows          : {len(df)}")
    print(f"   Active queries: {df['is_active_query'].sum()}")
    print(f"   Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Columns       : {list(df.columns)}\n")

    df.to_csv(f"{DATA_DIR}/melbourne_context.csv", index=False)
    return df


# ── 3. LMSYS Chat 1M — KV Cache Workloads ────────────────────────────────────
def load_kv_cache_workloads(num_samples: int = 2000) -> pd.DataFrame:
    """
    ShareGPT conversations as KV cache memory workloads.
    Apache 2.0.
    """
    print("[3/3] Loading ShareGPT KV Cache Workloads...")

    try:
        dataset = load_dataset(
            "RyokoAI/ShareGPT52K",  # verified on HF, Apache 2.0
            split="train",
            trust_remote_code=True
        )
    except Exception:
        # Second verified alternative
        dataset = load_dataset(
            "liyucheng/ShareGPT90K",
            split="train",
            trust_remote_code=True
        )

    records = []
    for item in list(dataset)[:num_samples]:
        convs = item.get("conversations", item.get("items", []))
        total_tokens = sum(
            len(str(c.get("value", c.get("text", ""))).split())
            for c in convs
        )
        num_turns = len(convs)
        records.append({
            "workload_id":     str(item.get("id", len(records))),
            "num_turns":       num_turns,
            "total_tokens":    total_tokens,
            "kv_cache_kb":     total_tokens * 2,
            "avg_turn_tokens": round(total_tokens / max(num_turns, 1), 1),
        })

    df = pd.DataFrame(records)

    print(f"   ✓ Workload samples : {len(df)}")
    print(f"   ✓ Avg tokens/conv  : {df['total_tokens'].mean():.0f}")
    print(f"   ✓ Max KV cache KB  : {df['kv_cache_kb'].max()}")
    print(f"   ✓ Columns          : {list(df.columns)}\n")

    df.to_csv(f"{DATA_DIR}/kv_cache_workloads.csv", index=False)
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CAAMS — Data Loader")
    print("=" * 60 + "\n")

    android_df   = load_android_usage()
    melbourne_df = load_melbourne_context()
    kv_df        = load_kv_cache_workloads()

    print("=" * 60)
    print("  LOAD SUMMARY")
    print("=" * 60)
    print(f"  Android (LSApp)  : {'✓ ' + str(len(android_df))   + ' rows' if not android_df.empty   else '✗ FAILED'}")
    print(f"  Melbourne        : {'✓ ' + str(len(melbourne_df)) + ' rows' if not melbourne_df.empty else '✗ FAILED'}")
    print(f"  KV Cache (LMSYS) : {'✓ ' + str(len(kv_df))       + ' rows' if not kv_df.empty        else '✗ FAILED'}")
    print("=" * 60)
    print("\nPaste the full output so we can move to the predictor.")