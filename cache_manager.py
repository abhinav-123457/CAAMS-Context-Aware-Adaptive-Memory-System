# cache_manager.py  — CAAMS Adaptive Cache Manager
#
# Implements LRU-F (LRU + Frequency weighted eviction) with:
#   - Context-aware retention scoring from Markov predictions
#   - KV cache pressure awareness (LMSYS/ShareGPT workload data)
#   - Melbourne query pressure as load signal
#   - Adaptive policy: tightens/relaxes based on real memory pressure
#
# No LLM involved here — cache decisions must be sub-millisecond.
# Agentic integration: cache_manager is called by memory_manager on each
# app switch to update cache state and return hit/miss outcome.
#
# Open resources used:
#   - LSApp (Apache 2.0) — real Android usage for frequency weights
#   - Melbourne Parking (CC BY 4.0) — context query pressure
#   - ShareGPT / LMSYS (Apache 2.0) — KV cache workload sizing
#
# License: Apache 2.0

import os
import time
import math
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Optional

DATA_DIR = "./data"


# ─────────────────────────────────────────────────────────────────────────────
# LRU-F Cache Entry
# ─────────────────────────────────────────────────────────────────────────────
class CacheEntry:
    """
    Single entry in the LRU-F cache.
    Score = frequency_weight * recency_weight * prediction_bonus
    Higher score = more valuable to keep.
    """
    __slots__ = ["app", "mb", "freq", "last_ts", "load_ts", "pred_prob"]

    def __init__(self, app: str, mb: int, pred_prob: float = 0.0):
        self.app       = app
        self.mb        = mb
        self.freq      = 1
        self.last_ts   = time.time()
        self.load_ts   = time.time()
        self.pred_prob = pred_prob  # current Markov probability, 0–1

    def touch(self, pred_prob: float = 0.0):
        self.freq     += 1
        self.last_ts   = time.time()
        self.pred_prob = pred_prob

    def score(self, now: float, decay_half_life_s: float = 300.0) -> float:
        """
        Composite retention score.
        - Recency: exponential decay from last access
        - Frequency: log-scaled hit count
        - Prediction bonus: Markov probability × 2 (strong signal)
        """
        age_s     = max(now - self.last_ts, 1e-6)
        recency   = math.exp(-age_s / decay_half_life_s)
        frequency = math.log1p(self.freq)
        return recency * frequency + self.pred_prob * 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive LRU-F Cache
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveLRUFCache:
    """
    App cache with adaptive capacity.

    Capacity adapts based on:
    - KV cache pressure (large LLM prompts consume memory → shrink app cache)
    - Melbourne query pressure (high → expand app cache for faster switching)
    - Current free memory percentage

    Eviction policy: lowest composite score (recency × frequency × prediction).
    """

    def __init__(self, max_mb: int = 2048, min_mb: int = 512):
        self.max_mb       = max_mb    # upper bound for cache budget
        self.min_mb       = min_mb    # lower bound (always keep some)
        self.capacity_mb  = max_mb    # current adaptive capacity
        self.entries:     dict[str, CacheEntry] = {}
        self.used_mb      = 0

        # Metrics
        self.hits         = 0
        self.misses       = 0
        self.evictions    = 0
        self.policy_log:  list = []   # records capacity adjustments

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / max(total, 1) * 100, 1)

    @property
    def free_mb(self) -> int:
        return max(self.capacity_mb - self.used_mb, 0)

    # ── Core operations ──────────────────────────────────────────────────────

    def lookup(self, app: str, pred_prob: float = 0.0) -> bool:
        """
        Returns True (hit) if app is in cache, touches entry.
        Returns False (miss) if not.
        """
        if app in self.entries:
            self.entries[app].touch(pred_prob)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def insert(self, app: str, mb: int, pred_prob: float = 0.0) -> dict:
        """
        Insert app into cache.
        Evicts lowest-scored entries if over capacity.
        """
        if app in self.entries:
            self.entries[app].touch(pred_prob)
            return {"status": "refreshed", "app": app}

        # Evict until we have room
        evicted = []
        while self.used_mb + mb > self.capacity_mb and self.entries:
            victim = self._pick_eviction_victim(protect=[app])
            if victim is None:
                break
            removed = self.entries.pop(victim)
            self.used_mb  -= removed.mb
            self.evictions += 1
            evicted.append({"app": victim, "freed_mb": removed.mb})

        if self.used_mb + mb > self.capacity_mb:
            return {"status": "rejected_oom", "app": app,
                    "needed_mb": mb, "free_mb": self.free_mb}

        self.entries[app] = CacheEntry(app, mb, pred_prob)
        self.used_mb     += mb
        return {"status": "inserted", "app": app, "mb": mb, "evicted": evicted}

    def _pick_eviction_victim(self, protect: list[str]) -> Optional[str]:
        """Lowest composite score wins eviction."""
        now = time.time()
        candidates = {
            name: entry.score(now)
            for name, entry in self.entries.items()
            if name not in protect
        }
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    # ── Adaptive capacity ────────────────────────────────────────────────────

    def adapt_capacity(self, free_device_pct: float,
                       kv_pressure_mb: float,
                       query_pressure: float):
        """
        Adjust cache capacity based on system-wide signals.

        free_device_pct: % of device RAM free (0–100)
        kv_pressure_mb : MB consumed by active KV caches (LLM inference)
        query_pressure : 0–1 from Melbourne data
        """
        # Base: start at max_mb
        new_cap = self.max_mb

        # Shrink if device is low on memory
        if free_device_pct < 15:
            new_cap = self.min_mb
        elif free_device_pct < 30:
            new_cap = int(self.min_mb + (self.max_mb - self.min_mb) * 0.4)

        # Further shrink proportional to KV cache pressure
        # KV cache and app cache compete for the same pool
        kv_fraction = min(kv_pressure_mb / 1024, 0.5)   # cap at 50% reduction
        new_cap = int(new_cap * (1.0 - kv_fraction * 0.5))

        # Expand slightly under high query pressure (more app switches expected)
        if query_pressure > 0.7 and free_device_pct > 40:
            new_cap = min(int(new_cap * 1.15), self.max_mb)

        new_cap = max(new_cap, self.min_mb)

        if new_cap != self.capacity_mb:
            self.policy_log.append({
                "ts":          time.time(),
                "old_cap_mb":  self.capacity_mb,
                "new_cap_mb":  new_cap,
                "free_pct":    free_device_pct,
                "kv_mb":       kv_pressure_mb,
                "q_pressure":  query_pressure,
            })
            self.capacity_mb = new_cap

            # If capacity shrank below current usage, evict to fit
            while self.used_mb > self.capacity_mb and self.entries:
                victim = self._pick_eviction_victim(protect=[])
                if victim is None:
                    break
                removed = self.entries.pop(victim)
                self.used_mb   -= removed.mb
                self.evictions += 1

    def snapshot(self) -> dict:
        now = time.time()
        return {
            "capacity_mb":  self.capacity_mb,
            "used_mb":      self.used_mb,
            "free_mb":      self.free_mb,
            "entries":      len(self.entries),
            "hit_rate_pct": self.hit_rate,
            "hits":         self.hits,
            "misses":       self.misses,
            "evictions":    self.evictions,
            "cached_apps":  {
                name: round(e.score(now), 3)
                for name, e in sorted(
                    self.entries.items(),
                    key=lambda x: x[1].score(now),
                    reverse=True
                )
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# KV Cache Pressure Estimator (from ShareGPT workload data)
# ─────────────────────────────────────────────────────────────────────────────
class KVCachePressureEstimator:
    """
    Estimates KV cache memory pressure from real LLM workload patterns.
    Data source: ShareGPT/LMSYS (Apache 2.0) sampled in data_loader.py.

    On-device LLM inference (e.g., Samsung Gauss running on NPU) competes
    with app cache for device memory. This estimator gives a pressure signal.
    """

    def __init__(self, kv_df: pd.DataFrame):
        self.kv_df     = kv_df
        self.mean_kb   = float(kv_df["kv_cache_kb"].mean())
        self.p90_kb    = float(kv_df["kv_cache_kb"].quantile(0.90))
        self.max_kb    = float(kv_df["kv_cache_kb"].max())
        self.active_mb = 0.0

        print(f"[KVCache] Mean workload: {self.mean_kb:.0f} KB")
        print(f"[KVCache] P90 workload : {self.p90_kb:.0f} KB")

    def sample_pressure(self, n_concurrent: int = 1) -> float:
        """
        Returns estimated KV cache pressure in MB for n concurrent LLM queries.
        Uses real workload distribution, not synthetic.
        """
        sample = self.kv_df["kv_cache_kb"].sample(n=n_concurrent, replace=True).sum()
        self.active_mb = sample / 1024.0
        return self.active_mb


# ─────────────────────────────────────────────────────────────────────────────
# Cache Simulation — runs on real LSApp sequences
# ─────────────────────────────────────────────────────────────────────────────
def run_cache_simulation(android_df: pd.DataFrame,
                         kv_df:      pd.DataFrame,
                         melbourne_df: pd.DataFrame,
                         markov,          # HourAwareMarkovPredictor
                         n_steps: int = 50) -> dict:
    """
    Replays a real LSApp session through the cache manager.
    Each step:
      1. Receive current app launch
      2. Lookup cache (hit/miss)
      3. Update cache with predicted apps
      4. Adapt capacity based on KV pressure + query pressure
    """
    print("\n" + "=" * 60)
    print("  CAAMS — Cache Manager Simulation")
    print(f"  Steps: {n_steps} | Policy: LRU-F Adaptive")
    print("=" * 60)

    # Pick a diverse session (same logic as memory_manager)
    opened = android_df[android_df["event_type"] == "Opened"].copy() \
             if "event_type" in android_df.columns else android_df.copy()

    best_session = (
        opened.groupby("session_id")["app_name"]
        .nunique()
        .nlargest(5)
        .sample(1)   # randomize among top-5 diverse sessions
        .index[0]
    )
    session_df = opened[opened["session_id"] == best_session].reset_index(drop=True)
    print(f"\n[Session] id={best_session} | events={len(session_df)} | "
          f"unique_apps={session_df['app_name'].nunique()}")

    # Build app-level frequency weights from full LSApp dataset
    app_freq = opened["app_name"].value_counts(normalize=True).to_dict()

    # KV pressure estimator
    kv_estimator = KVCachePressureEstimator(kv_df)

    # Melbourne pressure (time-of-day matched)
    current_hour = int(session_df.iloc[0].get("hour", 12))
    hour_df = melbourne_df[melbourne_df["hour"] == current_hour] \
              if "hour" in melbourne_df.columns else melbourne_df
    active_q  = hour_df["is_active_query"].sum() if "is_active_query" in hour_df.columns else 0
    query_pressure = float(active_q / max(len(hour_df), 1))

    # Adaptive cache (2GB budget, Samsung 8GB device)
    APP_MB = {
        "chrome": 400, "instagram": 250, "facebook": 300,
        "whatsapp": 200, "youtube": 350, "gmail": 150,
        "reddit": 220, "telegram": 180, "spotify": 180,
        "maps": 250, "camera": 280, "settings": 80,
        "default": 180,
    }

    def app_mb(name: str) -> int:
        n = name.lower()
        for k, v in APP_MB.items():
            if k in n:
                return v
        return APP_MB["default"]

    cache = AdaptiveLRUFCache(max_mb=2048, min_mb=512)

    # Tracking
    step_results = []
    t_total = time.perf_counter()

    for step in range(min(n_steps, len(session_df) - 1)):
        row  = session_df.iloc[step]
        app  = str(row["app_name"])
        hour = int(row.get("hour", 12))

        # Get Markov prediction prob for this app (how valuable to cache it)
        prev_app = str(session_df.iloc[max(step-1, 0)]["app_name"])
        preds    = markov.predict(prev_app, app, hour, top_k=3)
        pred_map = {p["app"]: p["prob"] for p in preds}
        curr_prob = pred_map.get(app, 0.0)

        # Step 1: Lookup
        t0    = time.perf_counter()
        is_hit = cache.lookup(app, pred_prob=curr_prob)
        hit_latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        # Step 2: Insert current app if missed
        if not is_hit:
            cache.insert(app, mb=app_mb(app), pred_prob=curr_prob)

        # Step 3: Pre-insert predicted apps (preloading into cache)
        for p in preds:
            predicted_app = p["app"]
            if predicted_app != app:
                cache.insert(predicted_app, mb=app_mb(predicted_app),
                             pred_prob=p["prob"])

        # Step 4: Adapt cache capacity
        kv_mb = kv_estimator.sample_pressure(n_concurrent=1)
        # Simulate realistic free device memory (starts at 97%, decreases)
        sim_free_pct = max(100 - step * 1.5, 25.0)
        cache.adapt_capacity(
            free_device_pct = sim_free_pct,
            kv_pressure_mb  = kv_mb,
            query_pressure  = query_pressure,
        )

        step_results.append({
            "step":          step + 1,
            "app":           app,
            "hit":           is_hit,
            "hit_latency_ms": hit_latency_ms,
            "cache_used_mb": cache.used_mb,
            "cache_cap_mb":  cache.capacity_mb,
            "kv_mb":         round(kv_mb, 1),
        })

        status = "HIT " if is_hit else "MISS"
        print(f"  [{status}] {app:<25} | cache {cache.used_mb}/{cache.capacity_mb} MB "
              f"| KV {kv_mb:.0f} MB | free {sim_free_pct:.0f}%")

    # ── Final Report ──────────────────────────────────────────────────────────
    total_time = round(time.perf_counter() - t_total, 2)
    snap       = cache.snapshot()

    print("\n" + "=" * 60)
    print("  ── Cache Manager Results ──")
    print(f"  Cache Hit Rate      : {snap['hit_rate_pct']}%    (target ≥85%)")
    print(f"  Total Hits          : {snap['hits']}")
    print(f"  Total Misses        : {snap['misses']}")
    print(f"  Total Evictions     : {snap['evictions']}")
    print(f"  Cache Capacity Mb   : {snap['capacity_mb']} (adaptive)")
    print(f"  Cache Used MB       : {snap['used_mb']}")
    print(f"  Policy Adjustments  : {len(cache.policy_log)}")
    print(f"  Total Sim Time      : {total_time}s")
    print(f"  Cached apps (score) : {snap['cached_apps']}")
    print("=" * 60)

    # ── Policy adjustments log ────────────────────────────────────────────────
    if cache.policy_log:
        print("\n  ── Capacity Adjustments ──")
        for adj in cache.policy_log[-5:]:  # last 5
            print(f"    {adj['old_cap_mb']} MB → {adj['new_cap_mb']} MB "
                  f"| free={adj['free_pct']:.0f}% kv={adj['kv_mb']:.0f}MB "
                  f"pressure={adj['q_pressure']:.2f}")

    return {
        "hit_rate":     snap["hit_rate_pct"],
        "hits":         snap["hits"],
        "misses":       snap["misses"],
        "evictions":    snap["evictions"],
        "policy_changes": len(cache.policy_log),
        "step_results": step_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: LRU-F vs Static LRU (to show improvement)
# ─────────────────────────────────────────────────────────────────────────────
def run_static_lru_baseline(android_df: pd.DataFrame, n_steps: int = 50) -> float:
    """
    Static LRU baseline — no frequency weighting, no prediction bonus,
    fixed capacity. Used to prove LRU-F improvement.
    """
    print("\n[Baseline] Running static LRU...")
    opened = android_df[android_df["event_type"] == "Opened"].copy() \
             if "event_type" in android_df.columns else android_df.copy()

    best_session = (
        opened.groupby("session_id")["app_name"]
        .nunique()
        .nlargest(5)
        .sample(1)
        .index[0]
    )
    session_df = opened[opened["session_id"] == best_session].reset_index(drop=True)

    # Fixed-capacity LRU (OrderedDict)
    lru: OrderedDict = OrderedDict()
    MAX_ENTRIES = 8
    hits, misses = 0, 0

    for step in range(min(n_steps, len(session_df))):
        app = str(session_df.iloc[step]["app_name"])
        if app in lru:
            lru.move_to_end(app)
            hits += 1
        else:
            misses += 1
            lru[app] = True
            if len(lru) > MAX_ENTRIES:
                lru.popitem(last=False)  # evict LRU

    hit_rate = round(hits / max(hits + misses, 1) * 100, 1)
    print(f"[Baseline] Static LRU hit rate: {hit_rate}%  (hits={hits}, misses={misses})")
    return hit_rate


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CAAMS — Cache Manager")
    print("=" * 60)

    # Load real data (written by data_loader.py)
    print("\nLoading data...")
    android_df   = pd.read_csv(f"{DATA_DIR}/android_usage.csv")
    melbourne_df = pd.read_csv(f"{DATA_DIR}/melbourne_context.csv")
    kv_df        = pd.read_csv(f"{DATA_DIR}/kv_cache_workloads.csv")

    print(f"  Android  : {len(android_df)} rows")
    print(f"  Melbourne: {len(melbourne_df)} rows")
    print(f"  KV Cache : {len(kv_df)} rows")

    # Train Markov predictor (same weights as memory_manager)
    from context_predictor import HourAwareMarkovPredictor, load_lsapp
    df_lsapp = load_lsapp()
    markov   = HourAwareMarkovPredictor(hour_buckets=6)
    markov.fit(df_lsapp)

    # Run LRU-F adaptive simulation
    results = run_cache_simulation(
        android_df   = android_df,
        kv_df        = kv_df,
        melbourne_df = melbourne_df,
        markov       = markov,
        n_steps      = 50,
    )

    # Run static LRU baseline for comparison
    baseline_hit_rate = run_static_lru_baseline(android_df, n_steps=50)

    # Summary comparison
    print("\n" + "=" * 60)
    print("  ── LRU-F vs Static LRU ──")
    print(f"  LRU-F Adaptive : {results['hit_rate']}%")
    print(f"  Static LRU     : {baseline_hit_rate}%")
    improvement = round(results["hit_rate"] - baseline_hit_rate, 1)
    print(f"  Improvement    : +{improvement}pp")
    print("=" * 60)
    print("\nPaste full output — prototype complete, moving to PPT blueprint.")