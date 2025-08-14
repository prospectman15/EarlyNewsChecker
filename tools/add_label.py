#!/usr/bin/env python3
# tools/add_label.py
import os, csv, sys, argparse
from datetime import datetime
from zoneinfo import ZoneInfo

DATA_DIR = "dataset"
LABELS = os.path.join(DATA_DIR, "labels.csv")

def to_utc_iso(ts_str: str, assume_tz="America/New_York") -> str:
    """
    Accepts:
      - Full ISO like '2025-08-12T20:10:00Z' or '2025-08-12T16:10:00-04:00'
      - Naive local like '2025-08-12 16:10' (assumed ET by default)
    Returns ISO-8601 in UTC with 'Z'
    """
    s = ts_str.strip().replace("T", " ")
    # If it ends with 'Z' or has an explicit offset, let fromisoformat parse it
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1])
            return dt.replace(tzinfo=ZoneInfo("UTC")).isoformat().replace("+00:00","Z")
        if "+" in s or "-" in s[10:]:
            dt = datetime.fromisoformat(ts_str)
            return dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00","Z")
    except Exception:
        pass
    # Fallback: assume local ET
    et = ZoneInfo(assume_tz)
    dt = datetime.fromisoformat(s).replace(tzinfo=et)
    return dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00","Z")

def ensure_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LABELS):
        with open(LABELS, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ticker","iso_time","window_min","y","notes"])

def append_label(ticker: str, iso_time: str, window_min: int, y: int, notes: str):
    ensure_header()
    # simple de-dupe (exact row)
    row = [ticker.upper(), iso_time, str(int(window_min)), str(int(y)), notes]
    if os.path.exists(LABELS):
        with open(LABELS, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r == row:
                    print("Label already exists; skipping.")
                    return
    with open(LABELS, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    print("Added label:", row)

def main():
    ap = argparse.ArgumentParser(description="Append a training label row")
    ap.add_argument("--ticker", required=True, help="e.g. CRWV")
    ap.add_argument("--time", required=True, help="UTC ISO like 2025-08-12T20:10:00Z or '2025-08-12 16:10' (assumes ET)")
    ap.add_argument("--window", type=int, default=90, help="minutes around time to mark (default 90)")
    ap.add_argument("--y", type=int, choices=[0,1], default=1, help="1=important/actionable, 0=noise")
    ap.add_argument("--notes", default="", help="optional note")
    args = ap.parse_args()

    iso_utc = to_utc_iso(args.time)
    append_label(args.ticker, iso_utc, args.window, args.y, args.notes)

if __name__ == "__main__":
    main()
