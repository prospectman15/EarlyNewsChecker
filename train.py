# train.py
import os, json, math, joblib, pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from feature_utils import MODEL_FEATURES

DATA_DIR = "dataset"
SIGNALS_CSV = os.path.join(DATA_DIR, "signals.csv")   # produced by scanner
OUTCOMES_CSV = "outcomes.csv"                         # produced by scanner follow-ups
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")     # optional human labels
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ---------------- helpers ----------------
def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def _load_df(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def _merge_targets(signals: pd.DataFrame, outcomes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = signals.copy()
    df["y"] = 0

    # Auto target: did |return| at +60m exceed 3%?
    if not outcomes.empty:
        o60 = outcomes[outcomes["label"]=="t+60m"].copy()
        o60["due_iso"] = pd.to_datetime(o60["due_iso"], errors="coerce", utc=True)
        o60["ret_pct"] = pd.to_numeric(o60["ret_pct"], errors="coerce")
        o60["auto_y"] = (o60["ret_pct"].abs() >= 3.0).astype(int)
        # join by nearest ticker+time (same alert id if present)
        if "id" in df.columns and "id" in o60.columns:
            df = df.merge(o60[["id","auto_y"]], on="id", how="left")
        else:
            df["auto_y"] = 0
        df["auto_y"].fillna(0, inplace=True)
        df["y"] = df[["y","auto_y"]].max(axis=1)

    # Human labels: labels.csv with columns [ticker, iso_time, window_min, y]
    # We mark all signals for that ticker within ±window_min of iso_time as y=1 (or y=0 if you add negatives)
    if not labels.empty:
        labels["iso_time"] = pd.to_datetime(labels["iso_time"], errors="coerce", utc=True)
        labels["window_min"] = pd.to_numeric(labels.get("window_min", 90), errors="coerce").fillna(90)
        for _, row in labels.iterrows():
            t = str(row.get("ticker","")).upper()
            yv = int(row.get("y",1))
            ts = row["iso_time"]
            win = int(row["window_min"])
            if pd.isna(ts) or not t:
                continue
            lo = ts - pd.Timedelta(minutes=win)
            hi = ts + pd.Timedelta(minutes=win)
            mask = (df["ticker"]==t) & (pd.to_datetime(df["signal_iso"], utc=True).between(lo, hi))
            if yv == 1:
                df.loc[mask, "y"] = 1
            else:
                df.loc[mask, "y"] = 0

    return df

# ---------------- main ----------------
def main():
    _ensure_dirs()
    signals = _load_df(SIGNALS_CSV)
    outcomes = _load_df(OUTCOMES_CSV)
    labels = _load_df(LABELS_CSV)

    if signals.empty:
        print("No signals to train on yet.")
        return

    df = _merge_targets(signals, outcomes, labels)

    # Build X/y
    X = df[MODEL_FEATURES].astype(float).values
    y = df["y"].astype(int).values

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.sum() and y.sum()<len(y) else None)

    # Simple, fast model
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # Eval
    proba = clf.predict_proba(Xte)[:,1]
    auc = float(roc_auc_score(yte, proba)) if len(set(yte))>1 else None

    # Choose threshold that targets ~70% precision on holdout (fallback 0.6)
    thresholds = [i/100 for i in range(30, 91, 5)]
    best = (0.6, 0, 0, 0)  # (thr, prec, rec, f1)
    import numpy as np
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        if prec >= best[1] and f1 >= best[3]:
            best = (thr, prec, rec, f1)

    metrics = {
        "auc": auc,
        "best_threshold": best[0],
        "precision_at_best": best[1],
        "recall_at_best": best[2],
        "f1_at_best": best[3],
        "n_signals": int(len(df)),
        "pos_rate": float(df["y"].mean()) if len(df) else 0.0,
        "features": MODEL_FEATURES,
    }

    # Save model + metrics
    joblib.dump({"clf": clf, "features": MODEL_FEATURES, "default_threshold": best[0]}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
