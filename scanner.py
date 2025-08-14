#!/usr/bin/env python3
"""
Early Event Scanner (Discord)

What it does:
- MARKET HOURS ONLY (US RTH 9:30–16:00 ET, Mon–Fri) for alerts & follow-ups
- NEWS/EARNINGS ALERT when:
    A) >= REQUIRED_MATCHES posts in last CLUSTER_WINDOW_MIN minutes, OR
    B) one "consensus" Reddit post (min score/comments)
  Filters mainstream media links; allows IR/SEC/PR early sources
- PRICE SPIKE ALERT when intraday move/volume crosses thresholds
- FOLLOW-UPS at +60m and +24h (deferred until market open)
- WEEKLY SUMMARY if MODE=weekly
- TEST webhook if MODE=test
- OPTIONAL ML GATE: logs features to dataset/signals.csv and, if model/model.pkl exists,
  requires score >= learned threshold in addition to the rules

ENV (optional overrides):
  DISCORD_WEBHOOK_URL
  MODE                        # "", "weekly", "test"
  SUMMARY_LOOKBACK_DAYS       # default 7
  REQUIRED_MATCHES            # default 2
  CLUSTER_WINDOW_MIN          # default 60
  SINGLE_POST_MIN_SCORE       # default 50
  SINGLE_POST_MIN_COMMENTS    # default 20
  MODEL_THRESHOLD             # default 0.60 (used if model bundle lacks its own)
  PRICE_SPIKES_ENABLED        # "1" (default) or "0" to disable spike alerts
"""

import os
import re
import csv
import json
import time
import html
import hashlib
import statistics
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta, time as dtime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

# --------------------------- Watchlist ---------------------------

WATCH_TICKERS: Dict[str, List[str]] = {
    "AMD":  ["amd", "advanced micro devices"],
    "NVDA": ["nvidia", "nvda"],
    "CRWD": ["crowdstrike", "crwd"],
    "CRWV": ["coreweave", "crwv"],
    "SRFM": ["surfair", "surf air mobility", "srfm"],
    "BA":   ["boeing", "737", "787", "777", "max 9", "door plug"],
    "UAL":  ["united airlines", "ual", "united flight", "united air"],
    "DAL":  ["delta", "dal", "delta air lines"],
    "AAL":  ["american airlines", "aal"],
}

# --------------------------- Keywords ---------------------------

POSITIVE_WORDS: List[str] = [
    "upgrade","upgraded","raises guidance","raised guidance",
    "raise price target","price target","initiated coverage",
    "beats","beat estimates","contract win","buyback","approval","fda",
]
CRISIS_WORDS: List[str] = [
    "crash","emergency landing","mayday","grounded","evacuate",
    "explosion","fire","smoke","engine failure","bird strike",
    "faa","ntsb","probe","recall","lawsuit","hack","breach",
    "ransomware","outage","strike","walkout","ceo resign",
]
EARNINGS_WORDS: List[str] = [
    r"\bearnings\b", r"\beps\b", "results", "miss", "beat", "guide", "guidance",
    "revenue", "margin", "outlook", "forecast", "loss", "profit",
]

# --------------------- Allowed / Blocked Sources ---------------------

MEDIA_BLACKLIST: List[str] = [
    "bloomberg.com","reuters.com","wsj.com","nytimes.com","cnn.com",
    "cnbc.com","apnews.com","foxnews.com","marketwatch.com",
    "seekingalpha.com","financialtimes.com","ft.com","benzinga.com",
    "yahoo.com","forbes.com","theguardian.com","investing.com",
    "washingtonpost.com","usatoday.com","nbcnews.com","abcnews.go.com",
    "bbc.com","barrons.com","coindesk.com","cointelegraph.com",
]
MEDIA_ALLOW: List[str] = ["sec.gov","businesswire.com","prnewswire.com"]  # IR/SEC/PR allowed

RSS_SOURCES: List[str] = [
    "https://www.reddit.com/r/StockMarket/.rss",
    "https://www.reddit.com/r/markets/.rss",
    "https://www.reddit.com/r/investing/.rss",
    "https://www.reddit.com/r/swingtrading/.rss",
    "https://www.reddit.com/r/options/.rss",
    "https://www.reddit.com/r/stocks/.rss",
    "https://www.reddit.com/r/wallstreetbets/.rss",
]

# ------------------------ Tunables (ENV) ------------------------

REQUIRED_MATCHES       = int(os.getenv("REQUIRED_MATCHES", "2"))
CLUSTER_WINDOW_MIN     = int(os.getenv("CLUSTER_WINDOW_MIN", "60"))
SINGLE_POST_MIN_SCORE  = int(os.getenv("SINGLE_POST_MIN_SCORE", "50"))
SINGLE_POST_MIN_COMMS  = int(os.getenv("SINGLE_POST_MIN_COMMENTS", "20"))

COOLDOWN_MIN_NEWS = 90  # minutes per-ticker

PRICE_SPIKE = {
    "from_open_abs_pct": 4.0,      # |last/open - 1| >= 4%
    "ten_min_abs_pct": 1.5,        # |last/last_10m - 1| >= 1.5%
    "volume_mult": 1.4,            # 10m avg vol vs today's per-minute avg
    "big_move_abs_pct": 5.0,       # bypass volume if >= 5%
    "cooldown_min": 45,
}
PRICE_SPIKES_ENABLED = os.getenv("PRICE_SPIKES_ENABLED", "1") == "1"

STATE_FILE   = "state.json"
ALERTS_CSV   = "alerts.csv"
OUTCOMES_CSV = "outcomes.csv"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ---------------------- Learning (optional) ----------------------

MODEL_PATH = "model/model.pkl"
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.60"))
DATASET_DIR = "dataset"
SIGNALS_CSV = os.path.join(DATASET_DIR, "signals.csv")

_loaded_model = None
_have_features = False
try:
    from feature_utils import build_features, MODEL_FEATURES  # optional module
    _have_features = True
except Exception:
    pass

def load_model():
    global _loaded_model
    if _loaded_model is None and os.path.exists(MODEL_PATH):
        try:
            import joblib
            _loaded_model = joblib.load(MODEL_PATH)
        except Exception:
            _loaded_model = None
    return _loaded_model

def ensure_signals_header():
    if not _have_features:
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(SIGNALS_CSV):
        with open(SIGNALS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id","ticker","signal_iso", *MODEL_FEATURES,
                "titles_concat","links_concat","score_max","comments_max","hits"
            ])

def log_signal_row(sig_id, ticker, ts_iso, feats_vec, titles, links, score_max, comm_max, hits):
    if not _have_features:
        return
    ensure_signals_header()
    with open(SIGNALS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            sig_id, ticker, ts_iso, *feats_vec,
            " || ".join(titles or []), " || ".join(links or []),
            score_max, comm_max, hits
        ])

# --------------------- Time / general utils ---------------------

ET = ZoneInfo("America/New_York")

def is_market_open(dt_utc: datetime | None = None) -> bool:
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    et = dt_utc.astimezone(ET)
    if et.weekday() >= 5:  # Sat/Sun
        return False
    return dtime(9, 30) <= et.time() <= dtime(16, 0)

def clean_text(s: str) -> str:
    return re.sub(r"\s+"," ", html.unescape((s or "")).strip()).lower()

def contains_any(text: str, terms: List[str]) -> bool:
    t = clean_text(text)
    return any(re.search(term if term.startswith(r"\b") else re.escape(term), t) for term in terms)

def allowed_host(host: str) -> bool:
    if not host: return False
    host = host.lower()
    if any(host == d or host.endswith("." + d) for d in MEDIA_ALLOW):
        return True
    if host.startswith("investor.") or host.startswith("ir."):
        return True
    return False

def is_blacklisted(url: str) -> bool:
    try:
        host = urlparse((url or "").lower()).netloc
    except Exception:
        return False
    if allowed_host(host):
        return False
    return any(host == d or host.endswith("." + d) for d in MEDIA_BLACKLIST)

def ensure_csv_headers(path: str, headers: List[str]) -> None:
    if not os.path.exists(path):
        with open(path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def append_csv(path: str, row: List[Any]) -> None:
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def send_discord(message: str, title: str = "Early Event Scanner") -> None:
    if not DISCORD_WEBHOOK_URL:
        print("[DRY RUN] ALERT:", title, "\n", message); return
    payload = {"embeds":[{"title": title, "description": message}]}
    for i in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
            if r.status_code == 429 and i < 2:
                time.sleep(int(r.headers.get("Retry-After","1"))); continue
            break
        except Exception as e:
            if i == 2: print("Discord post failed:", e)

def sig_of_post(p: Dict[str, Any]) -> str:
    raw = f"{p.get('title','')}|{p.get('url','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ------------------------ Price utilities ------------------------

def get_intraday_df(ticker: str):
    try:
        df = yf.download(tickers=ticker, period="2d", interval="1m", progress=False, threads=False)
        return df if df is not None and len(df)>0 else None
    except Exception:
        return None

def get_last_price(ticker: str) -> Tuple[float,str]:
    try:
        df = yf.download(tickers=ticker, period="2d", interval="1m", progress=False, threads=False)
        if df is not None and len(df)>0:
            return float(df["Close"].dropna().iloc[-1]), "1m"
    except Exception:
        pass
    try:
        df = yf.download(tickers=ticker, period="5d", interval="1d", progress=False, threads=False)
        if df is not None and len(df)>0:
            return float(df["Close"].dropna().iloc[-1]), "1d"
    except Exception:
        pass
    return float("nan"), "na"

def get_prev_close(ticker: str) -> float:
    try:
        df = yf.download(tickers=ticker, period="5d", interval="1d", progress=False, threads=False)
        if df is not None and len(df)>1:
            return float(df["Close"].dropna().iloc[-2])
        elif df is not None and len(df)>0:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return float("nan")

# ------------------------- Sources (web) -------------------------

def fetch_reddit_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/search.json?q={requests.utils.quote(query)}&sort=new&limit={limit}&t=day"
    headers = {"User-Agent":"early-event-scanner/0.9 by github-actions"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200: return []
        out=[]
        for c in r.json().get("data",{}).get("children",[]):
            d=c.get("data",{})
            out.append({
                "title": d.get("title",""),
                "url": d.get("url_overridden_by_dest") or d.get("url") or "",
                "permalink": "https://www.reddit.com"+d.get("permalink",""),
                "created_utc": d.get("created_utc",0),
                "num_comments": d.get("num_comments", 0),
                "score": d.get("score", 0),
                "source": "reddit_search",
            })
        return out
    except Exception:
        return []

def fetch_rss(url: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return []
        txt = r.text
        items=[]
        for m in re.finditer(r"<item>(.*?)</item>", txt, re.S|re.I):
            chunk=m.group(1)
            tit = re.search(r"<title>(.*?)</title>", chunk, re.S|re.I)
            link= re.search(r"<link>(.*?)</link>", chunk, re.S|re.I)
            pub = re.search(r"<pubDate>(.*?)</pubDate>", chunk, re.S|re.I)
            ts = time.time()
            if pub:
                import email.utils as eut
                try:
                    ts = eut.parsedate_to_datetime(pub.group(1)).astimezone(timezone.utc).timestamp()
                except Exception:
                    ts = time.time()
            items.append({
                "title": html.unescape(tit.group(1)) if tit else "",
                "url": html.unescape(link.group(1)) if link else "",
                "created_utc": ts,
                "source": url,
            })
        return items
    except Exception:
        return []

# ----------------------- News-based scanning -----------------------

def scan_news(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not is_market_open():
        return []

    t_now = datetime.now(timezone.utc)
    window_start = t_now - timedelta(minutes=CLUSTER_WINDOW_MIN)
    candidates: List[Dict[str, Any]] = []

    # Reddit search
    for ticker, terms in WATCH_TICKERS.items():
        q = " OR ".join(list({*terms, ticker.lower()}))
        posts = fetch_reddit_search(q, limit=25)
        for p in posts:
            created = datetime.fromtimestamp(p.get("created_utc",0) or 0, tz=timezone.utc)
            if created < window_start: continue
            title = p.get("title",""); url = p.get("url") or p.get("permalink","")
            if is_blacklisted(url): continue
            if contains_any(title, terms + [ticker.lower()]) and (
               contains_any(title, CRISIS_WORDS) or
               contains_any(title, POSITIVE_WORDS) or
               contains_any(title, EARNINGS_WORDS)
            ):
                p["_ticker"] = ticker
                p["_created"] = created
                candidates.append(p)

    # RSS feeds (no reddit scores/comments)
    for feed in RSS_SOURCES:
        items = fetch_rss(feed)
        for it in items:
            created = datetime.fromtimestamp(it.get("created_utc",time.time()), tz=timezone.utc)
            if created < window_start: continue
            title = it.get("title",""); url = it.get("url","")
            if is_blacklisted(url): continue
            for ticker, terms in WATCH_TICKERS.items():
                if contains_any(title, terms + [ticker.lower()]) and (
                   contains_any(title, CRISIS_WORDS) or
                   contains_any(title, POSITIVE_WORDS) or
                   contains_any(title, EARNINGS_WORDS)
                ):
                    candidates.append({
                        "title": title, "url": url, "_ticker": ticker, "_created": created,
                        "num_comments": 0, "score": 0, "source": feed
                    })

    # De-dup
    seen=set(); uniq=[]
    for c in candidates:
        sig = sig_of_post({"title": c["title"], "url": c["url"]})
        if sig in seen: continue
        seen.add(sig); uniq.append(c)

    # Group by ticker and apply rules (+ optional model gate)
    new_alerts=[]
    state.setdefault("cooldowns_news", {})
    state.setdefault("pending", [])
    state.setdefault("metrics", {})

    by_ticker: Dict[str,List[Dict[str, Any]]] = {}
    for c in uniq:
        by_ticker.setdefault(c["_ticker"], []).append(c)

    mdl = load_model()
    learned_thr = float(mdl.get("default_threshold", MODEL_THRESHOLD)) if mdl else None

    for ticker, posts in by_ticker.items():
        posts = [p for p in posts if p["_created"] >= window_start]
        if not posts: continue
        posts.sort(key=lambda x: x["_created"], reverse=True)

        cluster_ok = len(posts) >= REQUIRED_MATCHES
        consensus_ok = any(
            (p.get("source") == "reddit_search") and
            (int(p.get("score",0)) >= SINGLE_POST_MIN_SCORE) and
            (int(p.get("num_comments",0)) >= SINGLE_POST_MIN_COMMS)
            for p in posts
        )
        if not (cluster_ok or consensus_ok):
            continue

        now_ts = int(datetime.now(timezone.utc).timestamp())
        last_ts = state["cooldowns_news"].get(ticker, 0)
        minutes_since = ((now_ts - last_ts) / 60.0) if last_ts else 1e9
        if minutes_since < COOLDOWN_MIN_NEWS:
            continue

        top = posts[:min(5, len(posts))]
        titles = [p['title'] for p in top]
        links  = [(p.get('url') or "") for p in top]
        scores = [int(p.get("score",0)) for p in top]
        comms  = [int(p.get("num_comments",0)) for p in top]

        spot, src = get_last_price(ticker)
        msg = (f"**{ticker}** — early *news/earnings* "
               f"({len(posts)} hits/{CLUSTER_WINDOW_MIN}m; req={REQUIRED_MATCHES}, consensus={'yes' if consensus_ok else 'no'})\n"
               f"Spot: ${spot:.2f} (src: {src})\n" +
               "\n".join([f"• {p['_created'].strftime('%H:%M UTC')} — {p['title']}" for p in top]) +
               (("\nLinks: " + " | ".join(links)) if links else ""))

        # Log signal for training
        sig_id = f"SIGNAL-{ticker}-{now_ts}"
        if _have_features:
            feats_vec, _ = build_features(titles, links, scores, comms, hits_in_window=len(posts))
            log_signal_row(sig_id, ticker, datetime.now(timezone.utc).isoformat(),
                           feats_vec, titles, links, max(scores or [0]), max(comms or [0]), len(posts))
        else:
            feats_vec = None

        # Model gate if model present
        if mdl and feats_vec is not None:
            try:
                prob = float(mdl["clf"].predict_proba([feats_vec])[0][1])
                use_thr = learned_thr if learned_thr is not None else MODEL_THRESHOLD
                msg += f"\nModel score: {prob:.2f} (thr {use_thr:.2f})"
                if prob < use_thr:
                    continue
            except Exception:
                pass

        # Send + schedule follow-ups
        send_discord(msg, title="Early Event Scanner")
        state["cooldowns_news"][ticker] = now_ts

        aid = f"NEWS-{ticker}-{now_ts}"
        t60  = now_ts + 60*60
        t24h = now_ts + 24*60*60
        state["pending"].extend([
            {"id": aid, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t60, "label": "t+60m", "done": False},
            {"id": aid, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t24h, "label": "t+24h", "done": False},
        ])

        new_alerts.append({"id": aid, "ticker": ticker, "ts": datetime.now(timezone.utc).isoformat(),
                           "price": spot, "hits": len(posts)})
    return new_alerts

# ----------------------- Price spike scanning -----------------------

def price_spike_alert(ticker: str, state: Dict[str, Any]) -> Dict[str, Any] | None:
    # Only during market hours
    if not is_market_open():
        return None

    df = get_intraday_df(ticker)
    if df is None or len(df) < 15:
        return None

    # yfinance sometimes returns tz-aware index
    try:
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_convert(None)
    except Exception:
        pass

    # today's bars only
    today = df.index.date[-1]
    tdf = df[df.index.date == today]
    if tdf is None or tdf.empty or len(tdf) < 12:
        return None

    def _f(x, default=float("nan")):
        try:
            v = float(x)
            return v if v == v else default
        except Exception:
            return default

    last = _f(tdf["Close"].iloc[-1])
    day_open = _f(tdf["Open"].iloc[0])
    if not (last == last and day_open == day_open) or day_open == 0.0:
        return None

    from_open = (last / day_open - 1.0) * 100.0

    ten_idx = max(0, len(tdf) - 11)
    base_close = _f(tdf["Close"].iloc[ten_idx])
    if not (base_close == base_close) or base_close == 0.0:
        return None
    ten_pct = (last / base_close - 1.0) * 100.0

    vol_series = tdf.get("Volume")
    if vol_series is None:
        return None
    vol_tail = vol_series.tail(10).fillna(0)
    n_tail = max(1, len(vol_tail))
    vol10 = _f(vol_tail.sum()) / n_tail

    avg_per_min = _f(vol_series.fillna(0).mean())
    if not (avg_per_min == avg_per_min) or avg_per_min <= 0:
        avg_per_min = 1.0
    vol_mult = vol10 / max(1.0, avg_per_min)

    prev_close = get_prev_close(ticker)
    from_prev_close = ((last / prev_close) - 1.0) * 100.0 if (prev_close == prev_close and prev_close != 0.0) else 0.0

    big_move = abs(from_open) >= PRICE_SPIKE["big_move_abs_pct"] or abs(from_prev_close) >= PRICE_SPIKE["big_move_abs_pct"]
    base_move = (abs(from_open) >= PRICE_SPIKE["from_open_abs_pct"] or abs(ten_pct) >= PRICE_SPIKE["ten_min_abs_pct"])
    vol_ok = (vol_mult >= PRICE_SPIKE["volume_mult"])

    if not (big_move or (base_move and vol_ok)):
        return None

    # Cooldown per ticker
    state.setdefault("cooldowns_price", {})
    last_ts = state["cooldowns_price"].get(ticker, 0)
    if last_ts:
        mins = (datetime.now(timezone.utc).timestamp() - last_ts) / 60.0
        if mins < PRICE_SPIKE["cooldown_min"]:
            return None
    state["cooldowns_price"][ticker] = int(datetime.now(timezone.utc).timestamp())

    return {
        "ticker": ticker,
        "last": last,
        "from_open": from_open,
        "from_prev_close": from_prev_close,
        "ten_pct": ten_pct,
        "vol_mult": vol_mult,
        "direction": "UP" if (from_open >= 0) else "DOWN",
    }

def scan_price_spikes(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not is_market_open() or not PRICE_SPIKES_ENABLED:
        return []
    out = []
    t_now = datetime.now(timezone.utc)
    for ticker in WATCH_TICKERS.keys():
        try:
            a = price_spike_alert(ticker, state)
        except Exception as e:
            print(f"[price_spike_alert] {ticker} error: {e}")
            a = None
        if not a:
            continue
        msg = (
            f"**{ticker}** — *price spike* **{a['direction']}**\n"
            f"From open: {a['from_open']:+.2f}% | From prev close: {a['from_prev_close']:+.2f}% | "
            f"Last 10m: {a['ten_pct']:+.2f}% | Vol ≈ {a['vol_mult']:.1f}× avg\n"
            f"Spot: ${a['last']:.2f}"
        )
        send_discord(msg, title="Price Spike")

        aid = f"PRC-{ticker}-{int(t_now.timestamp())}"
        t60  = int((t_now + timedelta(minutes=60)).timestamp())
        t24h = int((t_now + timedelta(hours=24)).timestamp())
        state.setdefault("pending", []).extend([
            {"id": aid, "ticker": ticker, "alert_ts": int(t_now.timestamp()), "alert_px": a["last"],
             "due_ts": t60, "label": "t+60m", "done": False},
            {"id": aid, "ticker": ticker, "alert_ts": int(t_now.timestamp()), "alert_px": a["last"],
             "due_ts": t24h, "label": "t+24h", "done": False},
        ])
        out.append({"id": aid, "ticker": ticker, "ts": t_now.isoformat(), "price": a["last"], "hits": -1})
    return out

# --------------------------- Follow-ups ---------------------------

def process_followups() -> List[Dict[str, Any]]:
    if not is_market_open():
        return []

    # Load state
    state = {}
    if os.path.exists(STATE_FILE):
        try: state = json.load(open(STATE_FILE,"r",encoding="utf-8"))
        except Exception: state = {}
    state.setdefault("pending", []); state.setdefault("metrics", {})

    t_now = int(datetime.now(timezone.utc).timestamp())
    results=[]; new_pending=[]
    for item in state["pending"]:
        if item.get("done"): continue
        if t_now >= int(item["due_ts"]):
            ticker = item["ticker"]
            alert_px = float(item["alert_px"])
            cur_px, _ = get_last_price(ticker)
            if not (cur_px and cur_px==cur_px):
                new_pending.append(item); continue
            ret_pct = ((cur_px/alert_px)-1.0) * 100.0
            label = item["label"]
            ts_iso = datetime.fromtimestamp(item["due_ts"], tz=timezone.utc).isoformat()

            results.append({
                "id": item["id"], "ticker": ticker, "label": label,
                "due_ts": ts_iso, "alert_px": alert_px, "cur_px": cur_px, "ret_pct": ret_pct
            })

            if label == "t+24h":
                m = state["metrics"].get(ticker, {"count":0,"win_count":0,"avg_abs_move":0.0})
                m["count"] = int(m.get("count",0)) + 1
                win = 1 if abs(ret_pct) >= 5.0 else 0
                m["win_count"] = int(m.get("win_count",0)) + win
                prev_n = m["count"] - 1
                prev_avg = float(m.get("avg_abs_move",0.0))
                m["avg_abs_move"] = ((prev_avg*prev_n) + abs(ret_pct)) / m["count"]
                state["metrics"][ticker] = m
                send_discord(
                    f"**{ticker}** follow-up ({label}): {ret_pct:+.2f}% (alert ${alert_px:.2f} → now ${cur_px:.2f})\n"
                    f"Updated: win_rate={(m['win_count']/m['count']*100):.0f}%  avg |Δ|={m['avg_abs_move']:.2f}%  (n={m['count']})",
                    title="Price Follow-up"
                )
            elif label == "t+60m":
                send_discord(f"**{ticker}** follow-up ({label}): {ret_pct:+.2f}% (alert ${alert_px:.2f} → now ${cur_px:.2f})",
                             title="Price Follow-up")

            item["done"] = True
        else:
            new_pending.append(item)

    state["pending"] = new_pending
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(state,f)
    return results

# ------------------------- Weekly summary -------------------------

def parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z","")).replace(tzinfo=timezone.utc) if "Z" in s else datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)

def post_weekly_summary():
    lookback_days = int(os.getenv("SUMMARY_LOOKBACK_DAYS","7"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    alerts=[]; results=[]
    if os.path.exists(ALERTS_CSV):
        with open(ALERTS_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts = parse_iso(row["alert_iso"])
                except Exception:
                    continue
                if ts >= cutoff: alerts.append({"ticker": row["ticker"], "ts": ts})

    if os.path.exists(OUTCOMES_CSV):
        with open(OUTCOMES_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("label")!="t+24h": continue
                try:
                    ts = parse_iso(row["due_iso"])
                except Exception:
                    continue
                if ts >= cutoff:
                    try:   ret = float(row["ret_pct"])
                    except Exception: continue
                    results.append({"ticker": row["ticker"], "ret": ret})

    total_alerts = len(alerts)
    by_ticker_alerts: Dict[str,int] = {}
    for a in alerts: by_ticker_alerts[a["ticker"]] = by_ticker_alerts.get(a["ticker"], 0) + 1

    by_ticker_returns: Dict[str, List[float]] = {}
    for rrow in results: by_ticker_returns.setdefault(rrow["ticker"], []).append(rrow["ret"])

    lines = [f"**Weekly Summary (last {lookback_days} days)**", f"Total alerts: {total_alerts}"]
    if not by_ticker_alerts and not by_ticker_returns:
        lines.append("_No data yet. Come back next week._"); send_discord("\n".join(lines), title="Weekly Summary"); return

    lines.append("\n**Per-ticker:**")
    tickers = sorted(set(list(by_ticker_alerts.keys()) + list(by_ticker_returns.keys())))
    for t in tickers:
        a = by_ticker_alerts.get(t,0); rets = by_ticker_returns.get(t,[])
        if rets:
            win_rate = 100.0 * sum(1 for x in rets if abs(x) >= 5.0) / len(rets)
            avg_abs = statistics.mean(abs(x) for x in rets)
            avg_ret = statistics.mean(rets)
            lines.append(f"- {t}: alerts={a}, 24h n={len(rets)}, win%={win_rate:.0f}%, avg|Δ|={avg_abs:.2f}%, avg={avg_ret:+.2f}%")
        else:
            lines.append(f"- {t}: alerts={a}, 24h n=0")

    all_moves = [(t, v) for t, arr in by_ticker_returns.items() for v in arr]
    if all_moves:
        best = sorted(all_moves, key=lambda x: x[1], reverse=True)[:3]
        worst = sorted(all_moves, key=lambda x: x[1])[:3]
        lines.append("\n**Top movers (24h):**")
        lines.append("Best: " + "; ".join([f"{t} {v:+.2f}%" for t, v in best]))
        lines.append("Worst: " + "; ".join([f"{t} {v:+.2f}%" for t, v in worst]))
    send_discord("\n".join(lines), title="Weekly Summary")

# ------------------------------ Main ------------------------------

def main():
    mode = os.getenv("MODE","").strip().lower()

    # CSV headers
    ensure_csv_headers(ALERTS_CSV,   ["id","ticker","alert_iso","alert_price","hits"])
    ensure_csv_headers(OUTCOMES_CSV, ["id","ticker","label","due_iso","alert_price","cur_price","ret_pct"])

    if mode == "weekly":
        post_weekly_summary(); print("Weekly summary posted."); return
    if mode == "test":
        send_discord("**TEST** — webhook is working ✅", title="Test Alert"); print("Sent test message."); return

    # Hard gate: skip scan if market is closed
    if not is_market_open():
        print("Market closed — skipping scan and follow-ups.")
        return

    # Load state
    state = {}
    if os.path.exists(STATE_FILE):
        try: state = json.load(open(STATE_FILE,"r",encoding="utf-8"))
        except Exception: state = {}
    state.setdefault("pending", []); state.setdefault("metrics", {})

    # 1) News-based alerts
    news_alerts = scan_news(state)

    # 2) Price spike alerts (optional)
    price_alerts = scan_price_spikes(state) if PRICE_SPIKES_ENABLED else []

    # Log alerts
    for a in news_alerts + price_alerts:
        append_csv(ALERTS_CSV, [a["id"], a["ticker"], a["ts"], f"{a['price']:.4f}", a["hits"]])

    # Persist state after scheduling follow-ups
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(state,f)

    # 3) Process follow-ups
    outcomes = process_followups()
    for o in outcomes:
        append_csv(OUTCOMES_CSV, [
            o["id"], o["ticker"], o["label"], o["due_ts"],
            f"{o['alert_px']:.4f}", f"{o['cur_px']:.4f}", f"{o['ret_pct']:.3f}"
        ])

    print("Scan complete.",
          "News alerts:", len(news_alerts),
          "Price alerts:", len(price_alerts),
          "Follow-ups processed:", len(outcomes))

if __name__ == "__main__":
    main()
