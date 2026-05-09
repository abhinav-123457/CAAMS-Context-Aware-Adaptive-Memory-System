# device_pool.py — CAAMS Samsung Device Memory Pool
# Standalone — no LangGraph, no LLM, no Chronos imports.
# Imported by both mcp_server.py and agent nodes.
# Simulates Samsung Galaxy S/A series (6–8 GB RAM).
# License: Apache 2.0

import time
from collections import defaultdict


class DeviceMemoryPool:
    """
    Simulates a Samsung Galaxy-class device memory pool.
    Total RAM: 8GB | OS reserved: ~2GB | App-available: 6GB (6144 MB)
    """

    TOTAL_MB = 6144

    APP_PROFILES = {
        "chrome": 400,    "browser": 400,
        "instagram": 250, "facebook": 300,
        "whatsapp": 200,  "messenger": 200,
        "youtube": 350,   "netflix": 300,
        "gmail": 150,     "email": 150,
        "maps": 250,      "navigation": 250,
        "camera": 280,    "gallery": 200,
        "spotify": 180,   "music": 150,
        "game": 400,      "minesweeper": 120,
        "clock": 60,      "alarm": 60,
        "settings": 80,   "calculator": 50,
        "calendar": 100,  "contacts": 80,
        "reddit": 220,    "twitter": 200,
        "telegram": 180,  "signal": 150,
        "default": 180,
    }

    def __init__(self):
        self.allocated:    dict[str, int]   = {}
        self.preloaded:    dict[str, int]   = {}
        self.eviction_log: list             = []
        self.access_count: dict[str, int]   = defaultdict(int)
        self.last_access:  dict[str, float] = {}

    @property
    def used_mb(self)         -> int:   return sum(self.allocated.values()) + sum(self.preloaded.values())
    @property
    def free_mb(self)         -> int:   return self.TOTAL_MB - self.used_mb
    @property
    def free_pct(self)        -> float: return round(self.free_mb / self.TOTAL_MB * 100, 1)
    @property
    def utilization_pct(self) -> float: return round(self.used_mb / self.TOTAL_MB * 100, 1)

    def app_footprint(self, app_name: str) -> int:
        name = str(app_name).lower()
        for kw, mb in self.APP_PROFILES.items():
            if kw in name:
                return mb
        return self.APP_PROFILES["default"]

    def allocate(self, app_name: str, priority: str = "foreground") -> dict:
        mb = self.app_footprint(app_name)
        if priority == "foreground":
            mb = int(mb * 1.5)
        if mb > self.free_mb:
            return {"success": False, "reason": "oom",
                    "needed_mb": mb, "free_mb": self.free_mb}
        self.allocated[app_name] = mb
        self.access_count[app_name] += 1
        self.last_access[app_name]   = time.time()
        return {"success": True, "app": app_name,
                "allocated_mb": mb, "priority": priority}

    def preload(self, app_name: str) -> dict:
        if app_name in self.allocated or app_name in self.preloaded:
            return {"success": True, "app": app_name, "status": "already_loaded"}
        mb = self.app_footprint(app_name)
        if mb > self.free_mb:
            return {"success": False, "reason": "oom",
                    "needed_mb": mb, "free_mb": self.free_mb}
        self.preloaded[app_name] = mb
        self.access_count[app_name] += 1
        self.last_access[app_name]   = time.time()
        return {"success": True, "app": app_name, "preloaded_mb": mb}

    def evict(self, app_name: str) -> dict:
        freed = 0
        if app_name in self.preloaded:
            freed = self.preloaded.pop(app_name)
        elif app_name in self.allocated:
            freed = self.allocated.pop(app_name)
        if freed > 0:
            self.eviction_log.append({"app": app_name, "freed_mb": freed,
                                       "ts": time.time()})
        return {"evicted": app_name, "freed_mb": freed}

    def lru_candidates(self, exclude: list[str], top_n: int = 3) -> list[str]:
        candidates = {
            app: self.last_access.get(app, 0.0)
            for app in list(self.allocated) + list(self.preloaded)
            if app not in exclude
        }
        return sorted(candidates, key=lambda a: candidates[a])[:top_n]

    def snapshot(self) -> dict:
        return {
            "total_mb":        self.TOTAL_MB,
            "used_mb":         self.used_mb,
            "free_mb":         self.free_mb,
            "free_pct":        self.free_pct,
            "utilization_pct": self.utilization_pct,
            "allocated_apps":  dict(self.allocated),
            "preloaded_apps":  dict(self.preloaded),
            "eviction_count":  len(self.eviction_log),
        }

    def reset(self):
        self.allocated.clear()
        self.preloaded.clear()
        self.eviction_log.clear()
        self.access_count.clear()
        self.last_access.clear()

    def prefill_to_utilization(self, target_util_pct: float,
                                protect: list[str] | None = None):
        """Pre-fills pool to target% — used by harness to trigger cold path."""
        protect = protect or []
        target_used = int(self.TOTAL_MB * (target_util_pct / 100.0))
        i = 0
        while self.used_mb < target_used and self.free_mb > 0:
            name = f"__prefill_bg_{i}"
            if name in protect:
                i += 1
                continue
            mb = min(self.APP_PROFILES["default"], self.free_mb)
            if mb <= 0:
                break
            self.preloaded[name] = mb
            i += 1