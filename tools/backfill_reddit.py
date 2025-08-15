#!/usr/bin/env python3
"""
Backfill historical chatter from Reddit, create signals & labels, then your normal train step
can learn from "chatter -> later price move" patterns.

- Pulls posts via Pushshift mirror (free): api.pullpush.io
- Clusters posts in a rolling window (default 60m)
- Applies your rules (>=2 corroborating posts OR one consensus Reddit post)
- Writes model-ready rows to dataset/signals.csv
- Computes t+60m outcome using yfinance 1m data (RTH-aligned), labels y=1 if |return| >= min_abs_ret_pct
- Also appends a t+60m line to outcomes.csv (optional but nice for summaries)

Usage (locally or in Actions):
  python tools/backfill_reddit.py --tickers AMD,NVDA,BA,CRWD,CRWV,SRFM,TSLA,PLTR,SMCI,COIN \
    --days 30 --subs stocks,StockMarket,wallstreetbets,investing,options,swingtrading
"""

import os, csv, math, time, json, argparse, re
from datetime import datetime, timedelta, timezone, time as dtime
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import requests
import yfinance as yf

DATASET_DIR = "dataset"
SIGNALS = os.path.join(DATASET_DIR, "signals.csv")
LABELS  = os.path.join(DATASET_DIR, "labels.csv")
SPIKES  = os.path.join(DATASET_DIR, "spikes.csv")  # not required, left here for parity
OUTCOMES = "outcomes.csv"  # for summary/follow-up consistency

# --- your rules (keep consistent with scanner) ---
CLUSTER_WINDOW_MIN = 60
REQUIRED_MATCHES = 2
SINGLE_POST_MIN_SCORE = 80
SINGLE_POST_MIN_COMMENTS = 40
MIN_ABS_RET_PCT = 3.0  # label positive if |t+60m move| >= this
LABEL_WINDOW_MIN = 90

MEDIA_BLACKLIST = {
    "bloomberg.com","reuters.com","wsj.com","nytimes.com","cnn.com","cnbc.com","apnews.com",
    "foxnews.com","marketwatch.com","seekingalpha.com","financialtimes.com","ft.com","benzinga.com",
    "yahoo.com","forbes.com","theguardian.com","investing.com","washingtonpost.com","usatoday.com",
    "nbcnews.com","abcnews.go.com","bbc.com","barrons.com","coindesk.com","cointelegraph.com",
}
MEDIA_ALLOW = {"sec.gov","businesswire.com","prnewswire.com"}

CRISIS = ["crash","emergency landing","mayday","grounded","evacuate","explosion","fire","smoke",
          "engine failure","faa","ntsb","probe","recall","lawsuit","hack","breach","ransomware",
          "outage","strike","walkout","ceo resign"]
POSITIVE = ["upgrade","upgraded","raises guidance","raised guidance","raise price target",
            "price target","initiated coverage","beats","beat estimates","contract win","buyback",
            "approval","fda"]
EARNINGS = [r"\bearnings\b", r"\beps\b","results","miss","beat","guide","guidance","revenue",
            "margin","outlook","forecast","loss","profit"]

ET = datetime.now().astimezone().tzinfo  # OK for Actions; we RTH-align explicitly

def ensure_headers():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(SIGNALS):
        with open(SIGNALS,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id","ticker","signal_iso",
                "hits_in_window","score_max","score_sum","comments_max","comments_sum",
                "has_positive","has_crisis","has_earnings","has_primary_source_link",
                "titles_concat","links_concat",
                "score_max","comments_max","hits"
            ])
    if not os.path.exists(LABELS):
        with open(LABELS,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["ticker","iso_time","window_min","y","notes"])
    if not os.path.exists(OUTCOMES):
        with open(OUTCOMES,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["id","ticker","label","due_iso","alert_price","cur_price","ret_pct"])

def is_allowed(url: str) -> bool:
    try:
        host = urlparse(url or "").netloc.lower()
    except Exception:
        return True
    if not host:
        return True
    if host in MEDIA_ALLOW or any(host.endswith("."+d) for d in MEDIA_ALLOW):
        return True
    if host in MEDIA_BLACKLIST or any(host.endswith("."+d) for d in MEDIA_BLACKLIST):
        return False
    if host.startswith("investor.") or host.startswith("ir."):
        return True
    return True

def contains_any(s: str, terms: List[str]) -> bool:
    t = (s or "").lower()
    return any(term in t for term in terms)

def contains_regex_any(s: str, rx: List[str]) -> bool:
    t = (s or "").lower()
    return any(re.search(p, t) for p in rx)

def rth_align(t_utc: datetime) -> datetime:
    """Return the first REGULAR-HOURS minute >= t_utc. If outside, move to next open 09:30 ET."""
    # Convert to ET clock
    et = t_utc.astimezone(datetime.now().astimezone().tzinfo)
    # weekend
    if et.weekday() >= 5:
        # move to next Monday 09:30
        days = 7 - et.weekday()
        et = (et + timedelta(days=days)).replace(hour=9,minute=30,second=0,microsecond=0)
        return et.astimezone(timezone.utc)
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    if et < market_open:
        return market_open.astimezone(timezone.utc)
    if et > market_close:
        # next day open
        nxt = et + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        nxt = nxt.replace(hour=9,minute=30,second=0,microsecond=0)
        return nxt.astimezone(timezone.utc)
    return et.astimezone(timezone.utc)

def get_price_at(ticker: str, t_utc: datetime) -> float | None:
    """Get the last trade at/just before t_utc using 1m bars around that time."""
    # yfinance 1m covers ~30 days; we download a small pad
    start = (t_utc - timedelta(minutes=10)).strftime("%Y-%m-%d")
    try:
        df = yf.download(tickers=ticker, period="5d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None
        # Make sure index is tz-naive UTC-ish
        idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) else df.index
        df = df.copy()
        df.index = idx
        # Find the bar at or before timestamp minute
        tgt = t_utc.replace(second=0, microsecond=0).replace(tzinfo=None)
        sub = df.loc[:tgt].tail(1)
        if sub.empty:
            return None
        return float(sub["Close"].iloc[-1])
    except Exception:
        return None

def get_t60_return(ticker: str, alert_ts: datetime) -> Tuple[float|None, float|None, float|None]:
    px0 = get_price_at(ticker, alert_ts)
    if px0 is None or px0 == 0.0:
        return None, None, None
    due = rth_align(alert_ts + timedelta(minutes=60))
    px1 = get_price_at(ticker, due)
    if px1 is None or px1 == 0.0:
        return px0, None, None
    ret = (px1/px0 - 1.0) * 100.0
    return px0, px1, ret

def pushshift_fetch(sub: str, q: str, after_ts: int, before_ts: int, size: int = 250) -> List[Dict[str, Any]]:
    url = ("https://api.pullpush.io/reddit/search/submission/"
           f"?subreddit={sub}&q={requests.utils.quote(q)}&after={after_ts}&before={before_ts}"
           f"&size={size}&sort=asc")
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"early-news-backfill/1.0"})
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        out=[]
        for d in data:
            out.append({
                "title": d.get("title",""),
                "url": d.get("url") or "",
                "created_utc": int(d.get("created_utc", 0) or 0),
                "num_comments": int(d.get("num_comments", 0) or 0),
                "score": int(d.get("score", 0) or 0),
                "source": f"r/{sub}"
            })
        return out
    except Exception:
        return []

def collect_posts(tickers: List[str], subs: List[str], start_utc: datetime, end_utc: datetime) -> Dict[str, List[Dict[str, Any]]]:
    by_ticker = {t: [] for t in tickers}
    after_ts = int(start_utc.timestamp())
    before_ts = int(end_utc.timestamp())
    for sub in subs:
        for t in tickers:
            # query tries both cashtag style and name
            q = f'"{t}" OR {t}'
            rows = pushshift_fetch(sub, q, after_ts, before_ts)
            # keep only allowed-link posts mentioning ticker in title
            for p in rows:
                if t.lower() not in (p["title"] or "").lower():  # simple filter
                    continue
                if p["url"] and not is_allowed(p["url"]):
                    continue
                by_ticker[t].append(p)
        time.sleep(0.4)  # be nice to the endpoint
    # sort
    for t in tickers:
        by_ticker[t].sort(key=lambda x: x["created_utc"])
    return by_ticker

def cluster_and_emit(ticker: str, posts: List[Dict[str, Any]], writer_sig, writer_lbl, writer_out):
    if not posts:
        return 0, 0
    n_sig = n_lbl = 0
    i = 0
    W = CLUSTER_WINDOW_MIN * 60
    seen_windows = []
    while i < len(posts):
        anchor = posts[i]["created_utc"]
        # window = [anchor, anchor+W]
        cluster = []
        j = i
        while j < len(posts) and posts[j]["created_utc"] <= anchor + W:
            cluster.append(posts[j]); j += 1
        i += 1  # move anchor

        # rules
        hits = len(cluster)
        score_max = max([p.get("score",0) for p in cluster]) if cluster else 0
        comments_max = max([p.get("num_comments",0) for p in cluster]) if cluster else 0
        score_sum = sum([p.get("score",0) for p in cluster])
        comments_sum = sum([p.get("num_comments",0) for p in cluster])
        titles = [p["title"] for p in cluster[:4]]
        links  = [p.get("url","") for p in cluster[:4]]

        has_pos  = 1 if any(contains_any(" "+p["title"]+" ", POSITIVE) for p in cluster) else 0
        has_cri  = 1 if any(contains_any(" "+p["title"]+" ", CRISIS) for p in cluster) else 0
        has_earn = 1 if any(contains_regex_any(p["title"], EARNINGS) for p in cluster) else 0
        has_primary = 1 if any(is_allowed(u) and any(dom in (u or "") for dom in MEDIA_ALLOW) for u in links) else 0

        cluster_ok = hits >= REQUIRED_MATCHES
        consensus_ok = (score_max >= SINGLE_POST_MIN_SCORE and comments_max >= SINGLE_POST_MIN_COMMENTS)

        if not (cluster_ok or consensus_ok):
            continue

        # de-dup overlapping windows by 30m buckets
        bucket = anchor // 1800
        key = (ticker, bucket)
        if key in seen_windows:
            continue
        seen_windows.append(key)

        ts_iso = datetime.fromtimestamp(anchor, tz=timezone.utc).isoformat()
        sig_id = f"BF-{ticker}-{anchor}"
        writer_sig.writerow([
            sig_id, ticker, ts_iso,
            hits, score_max, score_sum, comments_max, comments_sum,
            has_pos, has_cri, has_earn, has_primary,
            " || ".join(titles), " || ".join(links),
            score_max, comments_max, hits
        ])
        n_sig += 1

        # compute t+60m outcome and label
        alert_ts = datetime.fromtimestamp(anchor, tz=timezone.utc)
        px0, px1, ret = get_t60_return(ticker, alert_ts)
        if px0 is not None and px1 is not None and ret is not None:
            writer_out.writerow([sig_id, ticker, "t+60m",
                                 (alert_ts + timedelta(minutes=60)).isoformat(),
                                 f"{px0:.4f}", f"{px1:.4f}", f"{ret:.3f}"])
            y = 1 if abs(ret) >= MIN_ABS_RET_PCT else 0
            writer_lbl.writerow([ticker, ts_iso, str(LABEL_WINDOW_MIN), str(y),
                                 f"backfill t+60m |Δ|={ret:.2f}%"])
            n_lbl += 1
        else:
            # if price missing, still add a neutral label (y=0) so it doesn't bias positives
            writer_lbl.writerow([ticker, ts_iso, str(LABEL_WINDOW_MIN), "0",
                                 "backfill: price missing"])
            n_lbl += 1
    return n_sig, n_lbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma list (e.g., AMD,NVDA,TSLA,PLTR,SMCI)")
    ap.add_argument("--days", type=int, default=30, help="how many trailing calendar days to fetch")
    ap.add_argument("--subs", default="stocks,StockMarket,wallstreetbets,investing,options,swingtrading",
                    help="comma list of subreddits")
    ap.add_argument("--cluster_window", type=int, default=CLUSTER_WINDOW_MIN)
    ap.add_argument("--required_matches", type=int, default=REQUIRED_MATCHES)
    ap.add_argument("--single_post_min_score", type=int, default=SINGLE_POST_MIN_SCORE)
    ap.add_argument("--single_post_min_comments", type=int, default=SINGLE_POST_MIN_COMMENTS)
    ap.add_argument("--min_abs_ret_pct", type=float, default=MIN_ABS_RET_PCT)
    args = ap.parse_args()

    # update globals from inputs
    global CLUSTER_WINDOW_MIN, REQUIRED_MATCHES, SINGLE_POST_MIN_SCORE, SINGLE_POST_MIN_COMMENTS, MIN_ABS_RET_PCT
    CLUSTER_WINDOW_MIN = args.cluster_window
    REQUIRED_MATCHES = args.required_matches
    SINGLE_POST_MIN_SCORE = args.single_post_min_score
    SINGLE_POST_MIN_COMMENTS = args.single_post_min_comments
    MIN_ABS_RET_PCT = args.min_abs_ret_pct

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    subs = [s.strip() for s in args.subs.split(",") if s.strip()]

    ensure_headers()

    # open writers once (append mode)
    with open(SIGNALS,"a",newline="",encoding="utf-8") as fs, \
         open(LABELS,"a",newline="",encoding="utf-8") as fl, \
         open(OUTCOMES,"a",newline="",encoding="utf-8") as fo:
        ws = csv.writer(fs); wl = csv.writer(fl); wo = csv.writer(fo)

        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=args.days)

        print(f"Backfilling {args.days}d for {len(tickers)} tickers across {len(subs)} subs...")
        counts = []
        for day in range(args.days, 0, -1):
            day_end = end_utc - timedelta(days=day-1)
            day_start = end_utc - timedelta(days=day)
            posts_by_tkr = collect_posts(tickers, subs, day_start, day_end)
            for tkr, posts in posts_by_tkr.items():
                ns, nl = cluster_and_emit(tkr, posts, ws, wl, wo)
                if ns or nl:
                    counts.append((tkr, ns, nl))
            # light pause per day chunk
            time.sleep(0.5)

        print("Done. Wrote signals/labels/outcomes rows.")
        if counts:
            print("Per-ticker counts (signals, labels):")
            for tkr, ns, nl in counts:
                print(f"  {tkr}: {ns} signals, {nl} labels")

if __name__ == "__main__":
    main()
