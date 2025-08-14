# train.py  — robust to tiny datasets
import os, json, joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from feature_utils import MODEL_FEATURES

DATA_DIR = "dataset"
SIGNALS_CSV = os.path.join(DATA_DIR, "signals.csv")   # from scanner logging
OUTCOMES = "outcomes.csv"                             # from scanner follow-ups
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")     # optional manual labels
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def _load_df(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def _merge_targets(signals: pd.DataFrame, outcomes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = signals.copy()
    if df.empty:
        return df
    df["y"] = 0

    # Auto target from +60m follow-up: |return| >= 3% => positive
    if not outcomes.empty:
        o = outcomes[outcomes.get("label","")=="t+60m"].copy()
        if not o.empty:
            o["auto_y"] = (pd.to_numeric(o["ret_pct"], errors="coerce").abs() >= 3.0).astype(int)
            if "id" in df.columns and "id" in o.columns:
                df = df.merge(o[["id","auto_y"]], on="id", how="left")
                df["auto_y"].fillna(0, inplace=True)
                df["y"] = df[["y","auto_y"]].max(axis=1)

    # Human labels: ticker, iso_time, window_min, y (1 or 0)
    if not labels.empty:
        labels["iso_time"] = pd.to_datetime(labels["iso_time"], errors="coerce", utc=True)
        labels["window_min"] = pd.to_numeric(labels.get("window_min", 90), errors="coerce").fillna(90)
        sig_ts = pd.to_datetime(df["signal_iso"], errors="coerce", utc=True)
        for _, row in labels.iterrows():
            t = str(row.get("ticker","")).upper()
            yv = int(row.get("y",1))
            ts = row["iso_time"]; win = int(row["window_min"])
            if pd.isna(ts) or not t: continue
            lo = ts - pd.Timedelta(minutes=win)
            hi = ts + pd.Timedelta(minutes=win)
            mask = (df["ticker"]==t) & (sig_ts.between(lo, hi))
            df.loc[mask, "y"] = yv
    return df

def _fit_with_fallback(X, y):
    """
    Returns (model_dict, metrics_dict)
    model_dict has keys: clf, features, default_threshold
    """
    n = len(y)
    pos = int(y.sum())
    neg = int(n - pos)

    # Not enough data or single-class -> fallback to DummyClassifier
    if n < 4 or pos == 0 or neg == 0:
        clf = DummyClassifier(strategy="prior")  # predicts class probas by class frequency
        clf.fit(X, y)
        pos_rate = float(pos / n) if n else 0.0
        metrics = {
            "mode": "fallback_dummy",
            "auc": None,
            "best_threshold": 0.60,   # keep scanner gate sane
            "precision_at_best": None,
            "recall_at_best": None,
            "f1_at_best": None,
            "n_rows": int(n),
            "pos_rate": pos_rate,
            "features": MODEL_FEATURES
        }
        model = {"clf": clf, "features": MODEL_FEATURES, "default_threshold": 0.60}
        return model, metrics

    # Regular path: train/test split
    strat = y if (pos and neg) else None
    # ensure the test set has at least 1 sample and train set >=1
    test_size = 0.25 if n >= 8 else max(1 / n, 0.2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # Evaluate + choose threshold (favor precision)
    proba = clf.predict_proba(Xte)[:,1]
    auc = float(roc_auc_score(yte, proba)) if len(set(yte))>1 else None

    best = (0.6, 0, 0, 0)  # thr, prec, rec, f1
    for thr in [i/100 for i in range(30, 91, 5)]:
        pred = (proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        if prec >= best[1] and f1 >= best[3]:
            best = (thr, float(prec), float(rec), float(f1))

    metrics = {
        "mode": "logreg",
        "auc": auc,
        "best_threshold": best[0],
        "precision_at_best": best[1],
        "recall_at_best": best[2],
        "f1_at_best": best[3],
        "n_rows": int(n),
        "pos_rate": float(pos / n),
        "features": MODEL_FEATURES
    }
    model = {"clf": clf, "features": MODEL_FEATURES, "default_threshold": best[0]}
    return model, metrics

def main():
    _ensure_dirs()
    signals  = _load_df(SIGNALS_CSV)
    outcomes = _load_df(OUTCOMES)
    labels   = _load_df(LABELS_CSV)

    if signals.empty:
        print("No signals to train on yet.")
        return

    df = _merge_targets(signals, outcomes, labels)
    if df.empty:
        print("No merged rows to train on yet.")
        return

    X = df[MODEL_FEATURES].astype(float).values
    y = df["y"].astype(int).values

    model, metrics = _fit_with_fallback(X, y)

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
