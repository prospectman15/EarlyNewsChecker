#!/usr/bin/env python3
"""
Seed your dataset with MANY synthetic-but-realistic examples.

Creates/append to:
  - dataset/signals.csv  (features your model trains on)
  - dataset/spikes.csv   (spike telemetry; not required for training)
  - dataset/labels.csv   (positives & negatives for supervised learning)

Usage (locally or in Actions):
  python tools/seed_dataset.py --pos 120 --neg 160 --days 21 --tickers AMD,NVDA,BA,CRWD,CRWV,SRFM,UAL,DAL,AAL
"""

import os, csv, random
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

DATASET_DIR = "dataset"
SIGNALS = os.path.join(DATASET_DIR, "signals.csv")
SPIKES  = os.path.join(DATASET_DIR, "spikes.csv")
LABELS  = os.path.join(DATASET_DIR, "labels.csv")

WATCH = ["AMD","NVDA","BA","CRWD","CRWV","SRFM","UAL","DAL","AAL"]

POS_TITLES = {
    "generic": [
        "upgrade; price target raised",
        "large call flow; breakout chatter",
        "major partnership rumored; PR expected",
        "guidance raised per leak",
        "analyst reiterates buy; target hiked",
    ],
    "earn": [
        "earnings beat; revenue above consensus",
        "EPS surprise; guide above",
        "results strong; margin up",
        "pre-announce upside",
    ],
    "crisis": [
        "emergency landing; FAA update incoming",
        "major outage report; cyber incident",
        "production issue; recall chatter",
        "CEO resign rumor",
    ],
}

NEG_TITLES = [
    "to the moon!! no source",
    "is a squeeze coming? vague rumors",
    "anyone buying this dip??",
    "chart looks bullish AF",
    "random DD thread (no links)",
]

PRIM_LINKS = [
    "https://www.businesswire.com",
    "https://www.prnewswire.com",
    "https://www.sec.gov/Archives/edgar/",
]
REDDIT_LINK = "https://www.reddit.com/r/stocks"
TWITTER     = "https://twitter.com/someacct"

def ensure_headers():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(SIGNALS):
        with open(SIGNALS, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id","ticker","signal_iso",
                "hits_in_window","score_max","score_sum","comments_max","comments_sum",
                "has_positive","has_crisis","has_earnings","has_primary_source_link",
                "titles_concat","links_concat",
                "score_max","comments_max","hits"  # trailing dup fields (scanner-compatible)
            ])
    if not os.path.exists(SPIKES):
        with open(SPIKES, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id","ticker","spike_iso","last","from_open_pct","from_prev_close_pct",
                "ten_min_pct","vol_mult","direction"
            ])
    if not os.path.exists(LABELS):
        with open(LABELS, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ticker","iso_time","window_min","y","notes"])

def rand_market_dt(days_back=14) -> datetime:
    # random business day/time in the last N days, during RTH 09:35–15:45 ET
    for _ in range(200):
        d = datetime.now(ET) - timedelta(days=random.randint(1, days_back))
        if d.weekday() >= 5:  # weekend
            continue
        hh = random.randint(9, 15)
        mm = random.randint(0, 59)
        if hh == 9 and mm < 35: mm = 35
        if hh == 15 and mm > 45: mm = 45
        dt_et = d.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return dt_et.astimezone(UTC)
    return datetime.now(UTC) - timedelta(days=1)

def pick(seq): return random.choice(seq)

def make_positive_row(ticker: str) -> tuple[dict, dict, dict]:
    """
    Returns (signals_row_dict, spikes_row_dict, label_row_dict)
    Positive means: clear online activity -> spike in next 15–75 minutes
    """
    # Categorize: earnings vs positive vs crisis (crisis => DOWN spike)
    kind = random.choices(["earn","generic","crisis"], weights=[0.35, 0.45, 0.20])[0]
    base_t = rand_market_dt()

    # chatter features
    hits = random.randint(2, 4)
    score_max = random.randint(80, 220)
    comments_max = random.randint(60, 160)
    score_sum = score_max + random.randint(50, 200)
    comments_sum = comments_max + random.randint(40, 180)
    has_pos  = 1 if kind in ("earn","generic") else 0
    has_cri  = 1 if kind == "crisis" else 0
    has_earn = 1 if kind == "earn" else 0
    prim     = 1 if (has_earn or random.random() < 0.5) else 0

    titles = []
    if kind == "earn":   titles.append(pick(POS_TITLES["earn"]))
    if kind == "generic":titles.append(pick(POS_TITLES["generic"]))
    if kind == "crisis": titles.append(pick(POS_TITLES["crisis"]))
    # add a second corroborating title
    titles.append(pick(POS_TITLES["generic" if kind!="crisis" else "crisis"]))

    links = [REDDIT_LINK]
    if prim: links.append(pick(PRIM_LINKS))
    if random.random() < 0.4: links.append(TWITTER)

    sig_iso = base_t.isoformat()
    sid = f"SEED-{ticker}-{int(base_t.timestamp())}"

    signals = {
        "id": sid, "ticker": ticker, "signal_iso": sig_iso,
        "hits_in_window": hits, "score_max": score_max, "score_sum": score_sum,
        "comments_max": comments_max, "comments_sum": comments_sum,
        "has_positive": has_pos, "has_crisis": has_cri, "has_earnings": has_earn,
        "has_primary_source_link": prim,
        "titles_concat": " || ".join(titles),
        "links_concat":  " || ".join(links),
        "score_max_dup": score_max, "comments_max_dup": comments_max, "hits_dup": hits
    }

    # spike ~15–75 minutes later
    delta_min = random.randint(15, 75)
    spike_t = base_t + timedelta(minutes=delta_min)
    direction_up = (kind != "crisis")
    move_big = random.random() < 0.65
    from_open = (random.uniform(4.0, 9.0) if move_big else random.uniform(2.0, 4.5))
    ten_min   = random.uniform(1.5, 4.0)
    vol_mult  = random.uniform(1.5, 3.2)
    if not direction_up:
        from_open *= -1.0
        ten_min   *= -1.0
    from_prev = from_open * (0.7 + random.uniform(-0.15, 0.15))

    spikes = {
        "id": f"SPIKE-{ticker}-{int(spike_t.timestamp())}",
        "ticker": ticker,
        "spike_iso": spike_t.isoformat(),
        "last": f"{random.uniform(5.0, 350.0):.2f}",
        "from_open_pct": f"{from_open:.3f}",
        "from_prev_close_pct": f"{from_prev:.3f}",
        "ten_min_pct": f"{ten_min:.3f}",
        "vol_mult": f"{vol_mult:.3f}",
        "direction": "UP" if direction_up else "DOWN"
    }

    label = {
        "ticker": ticker, "iso_time": spike_t.isoformat(),
        "window_min": "90", "y": "1",
        "notes": f"auto positive seed via spike {spikes['direction']}"
    }

    return signals, spikes, label

def make_negative_row(ticker: str) -> tuple[dict, dict|None, dict]:
    """
    Negative means: chatter but no meaningful move soon (or opposite tiny move).
    We don't add a spike row (or add a tiny/noisy one very rarely).
    """
    base_t = rand_market_dt()
    hits = random.randint(2, 3)
    score_max = random.randint(20, 60)
    comments_max = random.randint(8, 28)
    score_sum = score_max + random.randint(10, 40)
    comments_sum = comments_max + random.randint(6, 24)

    titles = [pick(NEG_TITLES), pick(NEG_TITLES)]
    links = [REDDIT_LINK]
    if random.random() < 0.2: links.append(TWITTER)

    sig_iso = base_t.isoformat()
    sid = f"SEED-{ticker}-{int(base_t.timestamp())}"

    signals = {
        "id": sid, "ticker": ticker, "signal_iso": sig_iso,
        "hits_in_window": hits, "score_max": score_max, "score_sum": score_sum,
        "comments_max": comments_max, "comments_sum": comments_sum,
        "has_positive": 0, "has_crisis": 0, "has_earnings": 0,
        "has_primary_source_link": 0,
        "titles_concat": " || ".join(titles),
        "links_concat":  " || ".join(links),
        "score_max_dup": score_max, "comments_max_dup": comments_max, "hits_dup": hits
    }

    label = {
        "ticker": ticker, "iso_time": sig_iso,
        "window_min": "90", "y": "0",
        "notes": "noise/hype; no move"
    }

    # very rarely add a tiny counter-move spike for realism
    spikes = None
    if random.random() < 0.08:
        spike_t = base_t + timedelta(minutes=random.randint(20, 90))
        from_open = random.uniform(-0.8, 0.8)
        ten_min   = random.uniform(-0.5, 0.5)
        vol_mult  = random.uniform(0.8, 1.3)
        spikes = {
            "id": f"SPIKE-{ticker}-{int(spike_t.timestamp())}",
            "ticker": ticker,
            "spike_iso": spike_t.isoformat(),
            "last": f"{random.uniform(5.0, 350.0):.2f}",
            "from_open_pct": f"{from_open:.3f}",
            "from_prev_close_pct": f"{from_open*0.7:.3f}",
            "ten_min_pct": f"{ten_min:.3f}",
            "vol_mult": f"{vol_mult:.3f}",
            "direction": "UP" if from_open >= 0 else "DOWN"
        }

    return signals, spikes, label

def append_row(path, row_list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row_list)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=120, help="number of positive examples")
    ap.add_argument("--neg", type=int, default=160, help="number of negative examples")
    ap.add_argument("--days", type=int, default=21, help="lookback days for timestamps")
    ap.add_argument("--tickers", type=str, default=",".join(WATCH), help="comma list")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    ensure_headers()

    # Load existing IDs to avoid exact dupes
    seen_ids = set()
    if os.path.exists(SIGNALS):
        with open(SIGNALS, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                seen_ids.add(row.get("id",""))

    def write_signal(d):
        append_row(SIGNALS, [
            d["id"], d["ticker"], d["signal_iso"],
            d["hits_in_window"], d["score_max"], d["score_sum"], d["comments_max"], d["comments_sum"],
            d["has_positive"], d["has_crisis"], d["has_earnings"], d["has_primary_source_link"],
            d["titles_concat"], d["links_concat"],
            d["score_max_dup"], d["comments_max_dup"], d["hits_dup"]
        ])

    def write_spike(s):
        append_row(SPIKES, [
            s["id"], s["ticker"], s["spike_iso"], s["last"], s["from_open_pct"],
            s["from_prev_close_pct"], s["ten_min_pct"], s["vol_mult"], s["direction"]
        ])

    def write_label(l):
        append_row(LABELS, [l["ticker"], l["iso_time"], l["window_min"], l["y"], l["notes"]])

    # positives
    npos = 0
    while npos < args.pos:
        t = random.choice(tickers)
        sig, spk, lab = make_positive_row(t)
        if sig["id"] in seen_ids: continue
        write_signal(sig); seen_ids.add(sig["id"])
        write_spike(spk); write_label(lab)
        npos += 1

    # negatives
    nneg = 0
    while nneg < args.neg:
        t = random.choice(tickers)
        sig, spk, lab = make_negative_row(t)
        if sig["id"] in seen_ids: continue
        write_signal(sig); seen_ids.add(sig["id"])
        if spk: write_spike(spk)  # rare tiny/noisy spike
        write_label(lab)
        nneg += 1

    print(f"Seeded: {args.pos} positives, {args.neg} negatives into dataset/.")

if __name__ == "__main__":
    main()
