# rl_eviction_policy.py — CAAMS RL Eviction Policy
#
# Implements a Q-learning based eviction decision agent.
# Trained offline on LSApp session data, used at runtime by
# MemoryAllocationAgent to decide WHICH app to evict under pressure.
#
# Why Q-learning over TD3/DDPG:
#   - Discrete action space (evict app A, B, or C) suits Q-learning
#   - TD3/DDPG are for continuous action spaces (e.g. exact MB to allocate)
#   - Q-table runs in <1ms on ARM CPU — no GPU/NPU needed
#   - Satisfies RL requirement from problem statement
#
# State space  : (memory_pressure_bucket, app_recency_bucket, app_freq_bucket)
# Action space : evict_rank (0=evict this app first, 1=second, 2=keep)
# Reward       : +1 if evicted app not needed in next 3 steps, -2 if it was
#
# License: Apache 2.0

import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)
QTABLE_PATH = f"{MODELS_DIR}/eviction_qtable.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# State Encoder
# ─────────────────────────────────────────────────────────────────────────────
def encode_state(memory_free_pct: float,
                 steps_since_last_use: int,
                 use_frequency: int) -> tuple:
    """
    Discretizes continuous values into state buckets.
    Keeps state space small enough for a Q-table on-device.
    """
    # Memory pressure: 0=critical(<20%), 1=tight(20-40%), 2=ok(>40%)
    if memory_free_pct < 20:
        mem_bucket = 0
    elif memory_free_pct < 40:
        mem_bucket = 1
    else:
        mem_bucket = 2

    # Recency: 0=used_very_recently(<3), 1=moderate(3-10), 2=stale(>10)
    if steps_since_last_use < 3:
        rec_bucket = 0
    elif steps_since_last_use < 10:
        rec_bucket = 1
    else:
        rec_bucket = 2

    # Frequency: 0=rarely_used(<3), 1=sometimes(3-10), 2=frequently(>10)
    if use_frequency < 3:
        freq_bucket = 0
    elif use_frequency < 10:
        freq_bucket = 1
    else:
        freq_bucket = 2

    return (mem_bucket, rec_bucket, freq_bucket)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Table Agent
# ─────────────────────────────────────────────────────────────────────────────
class EvictionQAgent:
    """
    Tabular Q-learning agent for app eviction ranking.
    Actions: 0=evict_first, 1=evict_second, 2=keep
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.1):
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # exploration rate (low — mostly exploit)
        # Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(3))
        self.trained = False

    def select_action(self, state: tuple, training: bool = False) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self.Q[state]))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def save(self, path: str = QTABLE_PATH):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)
        print(f"[RL] Q-table saved: {len(self.Q)} states → {path}")

    def load(self, path: str = QTABLE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(3), data)
        self.trained = True
        print(f"[RL] Q-table loaded: {len(self.Q)} states from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Offline Training on LSApp
# ─────────────────────────────────────────────────────────────────────────────
def train_eviction_agent(df: pd.DataFrame,
                         episodes: int = 5,
                         max_steps: int = 300) -> EvictionQAgent:
    print(f"\n[RL] Training eviction Q-agent on LSApp data...")
    print(f"[RL] Episodes: {episodes} | Steps/episode: {max_steps}")

    agent  = EvictionQAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
    # Only use sessions with enough diversity to trigger eviction
    valid_sessions = (
        df.groupby("session_id")
          .filter(lambda g: len(g) >= 20 and g["app_name"].nunique() >= 6)
          ["session_id"].unique()
    )
    print(f"[RL] Valid sessions for training: {len(valid_sessions)}")

    POOL_LIMIT    = 3    # tight pool → forces eviction decisions
    total_updates = 0

    for ep in range(episodes):
        ep_reward = 0
        sid       = np.random.choice(valid_sessions)
        sess      = df[df["session_id"] == sid].reset_index(drop=True)

        resident_apps  = []
        use_count      = defaultdict(int)
        last_used_step = defaultdict(lambda: -999)

        for step in range(min(max_steps, len(sess) - 3)):
            app = str(sess.iloc[step]["app_name"])
            use_count[app]      += 1
            last_used_step[app]  = step

            if app not in resident_apps:
                resident_apps.append(app)

            if len(resident_apps) > POOL_LIMIT:
                n_resident   = len(resident_apps)
                free_pct     = max(0.0, (POOL_LIMIT - n_resident) / POOL_LIMIT * 100 + 20)
                candidates   = [a for a in resident_apps if a != app]
                if not candidates:
                    continue

                scored = []
                for cand in candidates[:4]:
                    staleness = step - last_used_step.get(cand, 0)
                    freq      = use_count.get(cand, 0)
                    state     = encode_state(free_pct, staleness, freq)
                    action    = agent.select_action(state, training=True)
                    scored.append((cand, state, action))

                to_evict                    = sorted(scored, key=lambda x: x[2])[0]
                evicted_app, evict_state, evict_action = to_evict

                future_apps = [
                    str(sess.iloc[step + k]["app_name"])
                    for k in range(1, 4)
                    if step + k < len(sess)
                ]
                reward     = -2.0 if evicted_app in future_apps else 1.0
                ep_reward += reward

                staleness_next = step + 1 - last_used_step.get(evicted_app, 0)
                next_state     = encode_state(
                    min(free_pct + 20, 100),
                    staleness_next,
                    use_count.get(evicted_app, 0)
                )
                agent.update(evict_state, evict_action, reward, next_state)
                total_updates += 1
                resident_apps.remove(evicted_app)

        print(f"[RL] Episode {ep+1}/{episodes} | session={sid} | "
              f"reward={ep_reward:.1f} | Q-states={len(agent.Q)}")

    agent.trained = True
    print(f"[RL] Training complete | total Q-updates: {total_updates}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Interface — called by MemoryAllocationAgent
# ─────────────────────────────────────────────────────────────────────────────
def rl_rank_eviction_candidates(agent: EvictionQAgent,
                                candidates: list[str],
                                memory_free_pct: float,
                                last_used: dict,
                                use_counts: dict,
                                current_step: int) -> list[str]:
    """
    Ranks eviction candidates using the trained Q-agent.
    Returns list sorted from most-evictable to least.
    Called by ma_rule_engine in memory_manager.py under memory pressure.
    """
    if not candidates:
        return []

    scored = []
    for app in candidates:
        staleness = current_step - last_used.get(app, 0)
        freq      = use_counts.get(app, 0)
        state     = encode_state(memory_free_pct, staleness, freq)
        # Lower action value = higher eviction priority
        q_vals    = agent.Q[state]
        evict_score = q_vals[0] - q_vals[2]   # prefer_evict - prefer_keep
        scored.append((app, evict_score))

    # Sort descending by evict_score — most evictable first
    scored.sort(key=lambda x: x[1], reverse=True)
    return [app for app, _ in scored]


# ─────────────────────────────────────────────────────────────────────────────
# Main: Train and save
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CAAMS — RL Eviction Policy Training")
    print("=" * 60)

    from context_predictor import load_lsapp
    df = load_lsapp()

    agent = train_eviction_agent(df, episodes=20, max_steps=1000)
    agent.save()

    # Quick sanity check
    print("\n[RL] Sanity check — sample eviction decisions:")
    test_cases = [
        (15.0, 20, 2,  "critical memory, stale, rare app → should evict"),
        (50.0, 1,  15, "ok memory, just used, frequent → should keep"),
        (25.0, 8,  5,  "tight memory, moderate age, moderate freq"),
    ]
    for free, stale, freq, desc in test_cases:
        state  = encode_state(free, stale, freq)
        action = agent.select_action(state)
        label  = ["EVICT_FIRST", "EVICT_SECOND", "KEEP"][action]
        print(f"  {label:15} | {desc}")

    print("\n[RL] Done. Load this agent in memory_manager.py")
    print("=" * 60)