# context_predictor.py
# Predicts next app user will open — feeds memory pre-loader
#
# Open weight models used:
#   - Chronos-T5-Small (Amazon, Apache 2.0) → usage intensity forecasting
#   - Markov Chain (hour-aware) → next app prediction on LSApp real data
#
# Real data: LSApp (3.6M events, 87 apps, Sep 2017 - May 2018)
# License: Apache 2.0

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from chronos import ChronosPipeline

DATA_DIR   = "./data"
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Load & Clean LSApp ───────────────────────────────────────────────────────
def load_lsapp() -> pd.DataFrame:
    df = pd.read_csv(f"{DATA_DIR}/android_usage.csv")

    # Drop junk column from tsv header bleed
    if "lsapp.tsv" in df.columns:
        df.drop(columns=["lsapp.tsv"], inplace=True)

    # Only Opened events — these are actual app launches
    df = df[df["event_type"] == "Opened"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    print(f"[Data] Opened events  : {len(df)}")
    print(f"[Data] Unique apps    : {df['app_name'].nunique()}")
    print(f"[Data] Unique sessions: {df['session_id'].nunique()}")
    return df


# ── Hour-Aware Markov Chain Predictor ────────────────────────────────────────
class HourAwareMarkovPredictor:
    """
    Second-order Markov chain conditioned on hour-of-day.
    State = (prev_app, current_app, hour_bucket)
    Predicts: next_app

    Why Markov + hour-aware:
    - App usage is strongly time-dependent (e.g., alarm at 7am, Spotify commuting)
    - Second-order captures "I was on Gmail then Calendar → likely next is Meet"
    - No training infrastructure needed, works directly on real LSApp sequences
    """

    def __init__(self, hour_buckets: int = 6):
        # Bucket 24 hours into 6 periods (4hr each):
        # 0=night, 1=early_morning, 2=morning, 3=afternoon, 4=evening, 5=late_night
        self.hour_buckets  = hour_buckets
        self.transitions   = defaultdict(lambda: defaultdict(float))
        self.app_index     = {}
        self.index_app     = {}
        self.trained       = False

    def _hour_bucket(self, hour: int) -> int:
        return hour // (24 // self.hour_buckets)

    def _state_key(self, prev: str, curr: str, hour: int) -> str:
        return f"{prev}||{curr}||{self._hour_bucket(hour)}"

    def fit(self, df: pd.DataFrame):
        print("\n[Markov] Building transition matrix...")

        # Build app vocabulary
        apps = df["app_name"].unique().tolist()
        self.app_index = {a: i for i, a in enumerate(apps)}
        self.index_app = {i: a for a, i in self.app_index.items()}

        # Build transitions per session
        total_transitions = 0
        for sid, grp in df.groupby("session_id"):
            apps_seq  = grp["app_name"].tolist()
            hours_seq = grp["hour"].tolist()

            # Need at least 3 events for second-order
            if len(apps_seq) < 3:
                continue

            for i in range(len(apps_seq) - 2):
                prev    = apps_seq[i]
                curr    = apps_seq[i + 1]
                nxt     = apps_seq[i + 2]
                hour    = hours_seq[i + 1]
                key     = self._state_key(prev, curr, hour)
                self.transitions[key][nxt] += 1
                total_transitions += 1

        # Normalize to probabilities
        for key in self.transitions:
            total = sum(self.transitions[key].values())
            for app in self.transitions[key]:
                self.transitions[key][app] /= total

        self.trained = True
        print(f"[Markov] Transitions built : {total_transitions:,}")
        print(f"[Markov] Unique states     : {len(self.transitions):,}")
        print(f"[Markov] Apps in vocab     : {len(self.app_index)}")

    def predict(self, prev_app: str, curr_app: str,
                hour: int, top_k: int = 3) -> list[dict]:
        """
        Returns top-k predicted next apps with probabilities.
        Falls back to first-order then frequency if state unseen.
        """
        if not self.trained:
            raise RuntimeError("Call fit() first")

        # Try second-order state
        key = self._state_key(prev_app, curr_app, hour)
        if key in self.transitions:
            probs = self.transitions[key]
        else:
            # Fallback: first-order (any prev_app)
            bucket  = self._hour_bucket(hour)
            fo_key  = f"__any__||{curr_app}||{bucket}"
            # Aggregate all transitions from curr_app in this hour bucket
            probs = defaultdict(float)
            for k, v in self.transitions.items():
                parts = k.split("||")
                if parts[1] == curr_app and parts[2] == str(bucket):
                    for app, p in v.items():
                        probs[app] += p
            if not probs:
                # Last resort: most common apps in this hour bucket
                return [{"app": a, "prob": 1/len(self.app_index)}
                        for a in list(self.app_index.keys())[:top_k]]

        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [{"app": a, "prob": round(p, 4)} for a, p in ranked[:top_k]]

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Top-1 and Top-3 accuracy on held-out sessions.
        """
        correct_top1 = 0
        correct_top3 = 0
        total        = 0

        for sid, grp in df.groupby("session_id"):
            apps_seq  = grp["app_name"].tolist()
            hours_seq = grp["hour"].tolist()
            if len(apps_seq) < 3:
                continue

            for i in range(len(apps_seq) - 2):
                prev   = apps_seq[i]
                curr   = apps_seq[i + 1]
                actual = apps_seq[i + 2]
                hour   = hours_seq[i + 1]

                preds = self.predict(prev, curr, hour, top_k=3)
                pred_apps = [p["app"] for p in preds]

                if pred_apps and pred_apps[0] == actual:
                    correct_top1 += 1
                if actual in pred_apps:
                    correct_top3 += 1
                total += 1

        return {
            "top1_accuracy": round(correct_top1 / max(total, 1), 4),
            "top3_accuracy": round(correct_top3 / max(total, 1), 4),
            "total_predictions": total
        }


# ── Chronos — Usage Intensity Forecaster ─────────────────────────────────────
class ChronosUsageForecaster:
    """
    Uses Amazon Chronos-T5-Small to forecast app switch frequency per hour.
    Output: predicted number of app launches in next N hours.
    This tells the memory manager HOW MUCH memory pressure to expect,
    so it can scale pre-loading aggressively or conservatively.
    Model: amazon/chronos-t5-small (Apache 2.0, ~250M params)
    """

    def __init__(self, model_id: str = "amazon/chronos-t5-small"):
        # Prefer local weights if present; otherwise allow auto-download.
        local_path = os.path.join(MODELS_DIR, "chronos-t5-small")
        if os.path.isdir(local_path):
            model_id = local_path

        print(f"\n[Chronos] Loading {model_id} ...")
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=torch.float32,
        )
        print(f"[Chronos] Loaded on {self.device}")

    def build_hourly_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Aggregates LSApp Opened events into hourly app-launch counts.
        This is the time series Chronos will forecast.
        """
        df = df.copy()
        df["hour_slot"] = df["timestamp"].dt.floor("h")
        series = (
            df.groupby("hour_slot")
            .size()
            .asfreq("h", fill_value=0)
        )
        return series

    def forecast(self, series: pd.Series,
                 prediction_length: int = 6) -> dict:
        """
        Forecasts app launch count for next `prediction_length` hours.
        Returns mean forecast + confidence interval.
        """
        context = torch.tensor(
            series.values[-72:],   # last 72 hours as context
            dtype=torch.float32
        ).unsqueeze(0)             # shape: [1, 72]

        with torch.no_grad():
            forecast = self.pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=20,
            )

        # forecast shape: [1, num_samples, prediction_length]
        samples = forecast[0].numpy()    # [20, 6]
        mean    = samples.mean(axis=0)
        low     = np.percentile(samples, 10, axis=0)
        high    = np.percentile(samples, 90, axis=0)

        result = {
            "forecast_hours":     prediction_length,
            "mean_launches":      [round(float(v), 1) for v in mean],
            "low_launches":       [round(float(v), 1) for v in low],
            "high_launches":      [round(float(v), 1) for v in high],
            "peak_hour_offset":   int(np.argmax(mean)),
            "total_expected":     round(float(mean.sum()), 1),
        }
        return result


# ── Main: Train + Evaluate ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CAAMS — Context Predictor")
    print("=" * 60)

    df = load_lsapp()

    # Train/test split by session (80/20)
    sessions      = df["session_id"].unique()
    train_sess, test_sess = train_test_split(
        sessions, test_size=0.2, random_state=42
    )
    train_df = df[df["session_id"].isin(train_sess)]
    test_df  = df[df["session_id"].isin(test_sess)]

    print(f"\n[Split] Train sessions: {len(train_sess):,}")
    print(f"[Split] Test  sessions: {len(test_sess):,}")

    # ── Train Markov Predictor ────────────────────────────────────────────────
    markov = HourAwareMarkovPredictor(hour_buckets=6)
    markov.fit(train_df)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n[Eval] Running on test set...")
    metrics = markov.evaluate(test_df)
    print(f"\n  ── Markov Predictor Results ──")
    print(f"  Top-1 Accuracy : {metrics['top1_accuracy']*100:.1f}%  (target: ≥75%)")
    print(f"  Top-3 Accuracy : {metrics['top3_accuracy']*100:.1f}%")
    print(f"  Total Preds    : {metrics['total_predictions']:,}")

    # ── Sample Predictions ────────────────────────────────────────────────────
    print("\n  ── Sample Predictions ──")
    sample = test_df.groupby("session_id").filter(lambda x: len(x) >= 3).head(9)
    if len(sample) >= 3:
        row1 = sample.iloc[0]
        row2 = sample.iloc[1]
        preds = markov.predict(row1["app_name"], row2["app_name"],
                               int(row2["hour"]), top_k=3)
        actual = sample.iloc[2]["app_name"]
        print(f"  Prev app   : {row1['app_name']}")
        print(f"  Curr app   : {row2['app_name']} (hour {int(row2['hour'])})")
        print(f"  Predicted  : {[p['app'] for p in preds]}")
        print(f"  Actual     : {actual}")

    # ── Chronos Forecast ─────────────────────────────────────────────────────
    print("\n[Chronos] Building hourly usage series...")
    forecaster = ChronosUsageForecaster()
    series     = forecaster.build_hourly_series(df)

    print(f"[Chronos] Series length : {len(series)} hours")
    print(f"[Chronos] Avg launches/hr: {series.mean():.1f}")

    forecast = forecaster.forecast(series, prediction_length=6)
    print(f"\n  ── Chronos Forecast (next 6 hours) ──")
    print(f"  Mean launches   : {forecast['mean_launches']}")
    print(f"  Low  (10th pct) : {forecast['low_launches']}")
    print(f"  High (90th pct) : {forecast['high_launches']}")
    print(f"  Peak at hour +{forecast['peak_hour_offset']}")
    print(f"  Total expected  : {forecast['total_expected']} launches")

    print("\n" + "=" * 60)
    print("  Paste full output — moving to memory_manager.py next")
    print("=" * 60)