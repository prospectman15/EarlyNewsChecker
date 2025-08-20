#!/usr/bin/env python3
# scanner.py — EarlyNewsChecker
# - Watchlist news scan (Reddit RSS + Reddit search) with model gating & follow-ups
# - SILENT price-spike logger (yfinance minute data)
# - Market-wide hype scanner (dynamic ticker discovery across subs)
#
# Env knobs (defaults shown):
# REQUIRED_MATCHES=2
# CLUSTER_WINDOW_MIN=60
# SINGLE_POST_MIN_SCORE=20
# SINGLE_POST_MIN_COMMENTS=10
# COOLDOWN_MIN_NEWS=9
# PRICE_SPIKES_ENABLED=1
# DISCORD_WEBHOOK_URL=<optional>
#
# Hype scan:
# HYPE_SCAN_ENABLED=1
# HYPE_MIN_HITS=2
# HYPE_MIN_Z=1.5
# HYPE_SEND_DISCORD=1

import os, re, csv, json, time, math, random, textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, urlencode, quote_plus

import requests
import yfinance as yf

# ---- local helpers
from feature_utils import build_features
from hype_utils import (
    extract_entities, cluster_by_entity, build_cluster_features, hype_score
)

# ------------------------ Files ------------------------
ALERTS_CSV   = "alerts.csv"
OUTCOMES_CSV = "outcomes.csv"
SIGNALS_CSV  = "signals.csv"
SPIKES_CSV   = "spikes.csv"
LABELS_CSV   = "labels.csv"
STATE_FILE   = "state.json"
METRICS_JSON = "metrics.json"   # for learned model threshold (optional)
MODEL_FILE   = "model.pkl"      # optional sklearn pipeline

# ------------------------ Tunables (ENV) ------------------------
REQUIRED_MATCHES       = int(os.getenv("REQUIRED_MATCHES", "2"))
CLUSTER_WINDOW_MIN     = int(os.getenv("CLUSTER_WINDOW_MIN", "60"))

# Lowered so you actually get consensus alerts
SINGLE_POST_MIN_SCORE  = int(os.getenv("SINGLE_POST_MIN_SCORE", "20"))
SINGLE_POST_MIN_COMMS  = int(os.getenv("SINGLE_POST_MIN_COMMENTS", "10"))

COOLDOWN_MIN_NEWS      = int(os.getenv("COOLDOWN_MIN_NEWS", "9"))
MODEL_THRESHOLD        = float(os.getenv("MODEL_THRESHOLD", "0.60"))

PRICE_SPIKES_ENABLED   = os.getenv("PRICE_SPIKES_ENABLED", "1") == "1"

DISCORD_WEBHOOK_URL    = os.getenv("DISCORD_WEBHOOK_URL","").strip()

# ---- Hype scan tunables ----
HYPE_SCAN_ENABLED  = os.getenv("HYPE_SCAN_ENABLED","1") == "1"
HYPE_MIN_HITS      = int(os.getenv("HYPE_MIN_HITS","2"))
HYPE_MIN_Z         = float(os.getenv("HYPE_MIN_Z","1.5"))
HYPE_SEND_DISCORD  = os.getenv("HYPE_SEND_DISCORD","1") == "1"

# ------------------------ Watchlist ------------------------
WATCH_TICKERS: Dict[str, List[str]] = {
    "AMD": ["amd","advanced micro devices"],
    "NVDA": ["nvda","nvidia"],
    "CRWD": ["crwd","crowdstrike"],
    "CRWV": ["crwv","coreweave"],
    "SRFM": ["srfm","surf air mobility","surfair"],
    "BA":   ["ba","boeing","737","737 max","max 9","door plug","787","777"],
    "UAL":  ["ual","united airlines","united flight","united air"],
    "DAL":  ["dal","delta","delta air lines"],
    "AAL":  ["aal","american airlines","american air"],
}

# --------------------- Lexicons ---------------------
POSITIVE_WORDS: List[str] = [
    "upgrade","upgrades","raised to buy","initiated with buy","raises guidance",
    "beats","beat estimates","contract win","buyback","approval","approved","fda",
    "acquisition","merger","record revenue","record profit","launch"
]
CRISIS_WORDS: List[str] = [
    "crash","emergency landing","mayday","grounded","evacuate",
    "explosion","fire","smoke","engine failure","bird strike",
    "faa","ntsb","probe","recall","lawsuit","hack","breach",
    "ransomware","outage","strike","walkout","ceo resign","bankruptcy","chapter 11"
]
EARNINGS_WORDS: List[str] = [
    r"\bearnings\b", r"\beps\b", "results", "miss", "beat", "guide", "guidance",
    "revenue", "margin", "outlook", "forecast", "loss", "profit",
]

# --------------------- Allowed / Blocked Sources ---------------------
MEDIA_ALLOW: List[str] = [
    "sec.gov","businesswire.com","prnewswire.com","globenewswire.com",
    "seekingalpha.com","wsj.com","bloomberg.com","reuters.com","ft.com",
    "investors.com","marketwatch.com","techcrunch.com","theverge.com",
    "investorplace.com","fool.com","cnbc.com","forbes.com",
]
MEDIA_BLACKLIST: List[str] = [
    "zerohedge.com","benzinga.com","thestreet.com","yahoo.com","yahoo.co",
    "reddit.com","old.reddit.com","www.reddit.com","imgur.com","medium.com",
]

# ------------------ RSS Sources (expanded) ------------------
RSS_SOURCES: List[str] = [
    # Investing / trading
    "https://www.reddit.com/r/StockMarket/.rss",
    "https://www.reddit.com/r/markets/.rss",
    "https://www.reddit.com/r/investing/.rss",
    "https://www.reddit.com/r/swingtrading/.rss",
    "https://www.reddit.com/r/options/.rss",
    "https://www.reddit.com/r/stocks/.rss",
    "https://www.reddit.com/r/wallstreetbets/.rss",
    # Tech / AI where narratives start
    "https://www.reddit.com/r/technology/.rss",
    "https://www.reddit.com/r/ArtificialIntelligence/.rss",
    "https://www.reddit.com/r/datascience/.rss",
    "https://www.reddit.com/r/singularity/.rss",
    "https://www.reddit.com/r/MachineLearning/.rss",
    "https://www.reddit.com/r/technews/.rss",
]

# ------------------ Utility helpers ------------------

def ensure_csv_headers(path: str, headers: List[str]) -> None:
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_csv(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def send_discord(message: str, title: str = "Early Event Scanner") -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {
        "embeds": [{
            "title": title,
            "description": message[:4000],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception:
        pass

def clean_text(text: str) -> str:
    t = (text or "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def contains_any(text: str, terms: List[str]) -> bool:
    t = clean_text(text).lower()
    for term in terms:
        if term.startswith(r"\b"):
            if re.search(term, t):
                return True
        else:
            if term.lower() in t:
                return True
    return False

def allowed_host(host: str) -> bool:
    if not host: return False
    h = host.lower()
    if any(h == d or h.endswith("." + d) for d in MEDIA_ALLOW):
        return True
    if h.startswith("investor.") or h.startswith("ir."):
        return True
    return False

def is_blacklisted(url: str) -> bool:
    try:
        host = urlparse((url or "").lower()).netloc
    except Exception:
        return False
    if allowed_host(host):
        return False
    if any(host == d or host.endswith("." + d) for d in MEDIA_BLACKLIST):
        return True
    return False

# ------------------ Market time ------------------

def is_market_open(now: datetime | None = None) -> bool:
    # Eastern Time window 9:30–16:00, Mon–Fri; coarse check using UTC offsets
    # User is in America/New_York per project settings.
    now = now or datetime.now(timezone.utc)
    # Approx ET: UTC-4 in Aug; we don't import pytz here, so adjust roughly
    et = now - timedelta(hours=4)
    if et.weekday() >= 5:   # Sat/Sun
        return False
    h, m = et.hour, et.minute
    minutes = h*60 + m
    return (9*60 + 30) <= minutes <= (16*60)

# ------------------ Pricing ------------------

def get_last_price(ticker: str) -> float:
    try:
        df = yf.download(tickers=ticker, period="1d", interval="1m", progress=False)
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])
    except Exception:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="1d", interval="1m")
            if df is None or df.empty:
                return 0.0
            return float(df["Close"].iloc[-1])
        except Exception:
            return 0.0

def get_intraday_df(ticker: str):
    try:
        df = yf.download(tickers=ticker, period="1d", interval="1m", progress=False)
        return df if df is not None and not df.empty else None
    except Exception:
        return None

# ------------------ Fetchers ------------------

def fetch_rss(url: str) -> List[Dict[str, Any]]:
    """Fetch a Reddit .rss feed and return lightweight items."""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"EarlyNewsChecker/1.0"})
        r.raise_for_status()
        text = r.text
        # super simple RSS item extraction
        items = []
        for raw in re.split(r"<item>|<entry>", text)[1:]:
            title_m = re.search(r"<title>(<!\[CDATA\[)?(.*?)(\]\]>)?</title>", raw, re.S|re.I)
            link_m  = re.search(r"<link.*?>(.*?)</link>", raw, re.S|re.I)
            if not title_m:
                continue
            title = re.sub(r"<.*?>","", title_m.group(2) if title_m.group(2) else "").strip()
            link  = ""
            if link_m:
                link = re.sub(r"<.*?>","", link_m.group(1)).strip()
            items.append({
                "title": title,
                "url": link,
                "permalink": link,
                "created_utc": time.time(),
                "num_comments": 0,
                "score": 0,
                "source": url,
                "subreddit": url.split("/r/")[1].split("/")[0] if "/r/" in url else "reddit",
            })
        return items
    except Exception:
        return []

def fetch_reddit_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Use Reddit search JSON (no auth). This returns richer fields (score/comments)."""
    try:
        params = {
            "q": query, "sort": "new", "restrict_sr": False, "limit": limit,
            "include_over_18": False
        }
        u = "https://www.reddit.com/search.json?" + urlencode(params)
        r = requests.get(u, timeout=15, headers={"User-Agent": "EarlyNewsChecker/1.0"})
        r.raise_for_status()
        data = r.json()
        out: List[Dict[str, Any]] = []
        for ch in (data.get("data", {}).get("children", []) or []):
            d = ch.get("data", {})
            out.append({
                "title": d.get("title",""),
                "url": d.get("url_overridden_by_dest") or d.get("url") or "",
                "permalink": "https://www.reddit.com"+d.get("permalink",""),
                "created_utc": d.get("created_utc",0),
                "num_comments": d.get("num_comments", 0),
                "score": d.get("score", 0),
                "source": "reddit_search",
                "subreddit": d.get("subreddit",""),
            })
        return out
    except Exception:
        return []

# ------------------ Signal logging ------------------

def log_signal_row(sig_id: str, ticker: str, ts_iso: str,
                   feats_vec: List[float], titles: List[str], links: List[str],
                   score_max: float, comments_max: float, hits_in_window: int) -> None:
    ensure_csv_headers(SIGNALS_CSV, [
        "id","ticker","ts","hits_in_window","score_max","comments_max","titles","links","features_json"
    ])
    features_json = json.dumps(feats_vec)
    titles_join   = " || ".join(titles or [])[:1800]
    links_join    = " | ".join(links or [])[:1800]
    append_csv(SIGNALS_CSV, [
        sig_id, ticker, ts_iso, str(hits_in_window), f"{score_max:.4f}", f"{comments_max:.4f}",
        titles_join, links_join, features_json
    ])

# ------------------ News Scan (watchlist) ------------------

def allowed_news_post(title: str, url: str) -> bool:
    """Require that we have a non-blacklisted host OR investor/IR/SEC domains."""
    if not url:
        return True
    if is_blacklisted(url):
        return False
    # If there is a host and it's neither allowed nor investor/ir, still permit
    # because Reddit posts often summarize. We only *block* clear blacklist.
    return True

def scan_news(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For each watchlist ticker:
      - Gather posts from RSS feeds and Reddit search that mention aliases
      - Require also lexicon hit (positive/crisis/earnings)
      - Fire alert if (>= REQUIRED_MATCHES within CLUSTER_WINDOW_MIN minutes) OR
        (single consensus post meets score/comments)
      - Model gate if model present (optional)
      - Log to alerts.csv and schedule follow-ups
    """
    new_alerts: List[Dict[str, Any]] = []

    # Load optional model + threshold
    mdl = None
    learned_thr = None
    if os.path.exists(MODEL_FILE):
        try:
            import pickle
            with open(MODEL_FILE, "rb") as f:
                mdl = pickle.load(f)
        except Exception:
            mdl = None
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON,"r",encoding="utf-8") as f:
                m = json.load(f)
                learned_thr = float(m.get("best_threshold", MODEL_THRESHOLD))
        except Exception:
            learned_thr = None

    # Cooldowns
    cds = state.setdefault("cooldowns_news", {})
    now_ts = int(time.time())

    # Fetch pool
    pool: List[Dict[str, Any]] = []
    for u in RSS_SOURCES:
        pool.extend(fetch_rss(u))

    # Build per-ticker clusters
    window_sec = CLUSTER_WINDOW_MIN * 60
    for ticker, aliases in WATCH_TICKERS.items():
        # collect posts mentioning this ticker/aliases
        posts: List[Dict[str, Any]] = []
        q_parts = [quote_plus(a) for a in aliases]
        # 1) RSS pool
        for it in pool:
            title = it.get("title","")
            url   = it.get("url","") or it.get("permalink","")
            if not allowed_news_post(title, url):
                continue
            if contains_any(title, aliases) and (
                contains_any(title, POSITIVE_WORDS) or
                contains_any(title, CRISIS_WORDS) or
                contains_any(title, EARNINGS_WORDS)
            ):
                posts.append(it)

        # 2) Reddit search JSON (richer engagement metrics)
        q = " OR ".join(aliases)
        posts.extend(fetch_reddit_search(q, limit=25))

        # Time window filter
        recent_posts = []
        now_ts = int(time.time())
        for p in posts:
            ts = int(p.get("created_utc") or now_ts)
            if (now_ts - ts) <= window_sec:
                recent_posts.append(p)

        if not recent_posts:
            continue

        # Cluster rule OR single-post consensus
        titles = [p.get("title","") for p in recent_posts]
        urls   = [p.get("url","") or p.get("permalink","") for p in recent_posts]
        scores = [int(p.get("score",0) or 0) for p in recent_posts]
        comms  = [int(p.get("num_comments",0) or 0) for p in recent_posts]

        # consensus
        consensus_ok = False
        for s, c in zip(scores, comms):
            if s >= SINGLE_POST_MIN_SCORE and c >= SINGLE_POST_MIN_COMMS:
                consensus_ok = True
                break

        # corroboration
        corroboration_ok = len(recent_posts) >= REQUIRED_MATCHES

        if not (consensus_ok or corroboration_ok):
            continue

        # cooldown
        if ticker in cds and (now_ts - int(cds[ticker])) < COOLDOWN_MIN_NEWS * 60:
            continue
        cds[ticker] = now_ts

        # price
        spot = get_last_price(ticker)

        # Message & features
        msg = f"**{ticker}** — {len(recent_posts)} hits in {CLUSTER_WINDOW_MIN}m\n" + \
              ("\n".join(f"- {t}" for t in titles[:5]))
        if urls:
            msg += "\nLinks: " + " | ".join(urls[:5])

        # log training signal
        feats_vec, _map = build_features(titles, urls, scores, comms, hits_in_window=len(recent_posts))
        sig_id = f"SIGNAL-{ticker}-{now_ts}"
        log_signal_row(sig_id, ticker, datetime.now(timezone.utc).isoformat(),
                       feats_vec, titles, urls, max(scores or [0]), max(comms or [0]),
                       len(recent_posts))

        # optional model gate
        if mdl:
            try:
                prob = float(mdl["clf"].predict_proba([feats_vec])[0][1])  # type: ignore
                use_thr = learned_thr if learned_thr is not None else MODEL_THRESHOLD
                msg += f"\nModel score: {prob:.2f} (thr {use_thr:.2f})"
                if prob < use_thr:
                    # still log the training row above, but skip alerting
                    continue
            except Exception:
                pass

        # Send + schedule follow-ups
        send_discord(msg, title="Early Event Scanner")

        aid = f"NEWS-{ticker}-{now_ts}"
        t60  = now_ts + 60*60
        t24h = now_ts + 24*60*60
        state.setdefault("pending", [])
        state["pending"].extend([
            {"id": aid, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t60, "label": "t+60m", "done": False},
            {"id": aid, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t24h, "label": "t+24h", "done": False},
        ])

        new_alerts.append({"id": aid, "ticker": ticker, "ts": datetime.now(timezone.utc).isoformat(),
                           "price": spot, "hits": len(recent_posts)})

    return new_alerts

# --------------------- SILENT price spike logging ---------------------

PRICE_SPIKE = {
    "from_open_abs_pct": 4.0,      # |last/open - 1| >= 4%
    "ten_min_abs_pct":   1.5,      # |last/last_10m - 1| >= 1.5%
    "volume_mult":       1.4,      # 10m avg vol vs today's per-minute avg
    "big_move_abs_pct":  5.0,      # bypass volume if >= 5%
    "cooldown_min":      45,
}

def scan_price_spikes(state: Dict[str, Any]) -> int:
    if not PRICE_SPIKES_ENABLED:
        return 0
    ensure_csv_headers(SPIKES_CSV, ["ts","ticker","open","last","chg_from_open_pct","chg_10m_pct","vol_mult"])
    ensure_csv_headers(LABELS_CSV, ["ts","ticker","label"])

    cds = state.setdefault("cooldowns_spike", {})
    logged = 0

    for ticker in WATCH_TICKERS.keys():
        now_ts = int(time.time())
        if ticker in cds and (now_ts - int(cds[ticker])) < PRICE_SPIKE["cooldown_min"]*60:
            continue

        df = get_intraday_df(ticker)
        if df is None or df.empty or len(df) < 12:
            continue

        last = float(df["Close"].iloc[-1])
        opn  = float(df["Open"].iloc[0])
        chg_from_open_pct = (last / opn - 1.0) * 100.0

        # 10-minute change
        last10 = float(df["Close"].iloc[-11])
        chg_10m_pct = (last / last10 - 1.0) * 100.0

        # volume ratio
        vol10 = float(df["Volume"].iloc[-11:-1].mean() or 0.0)
        volday= float(df["Volume"].mean() or 1.0)
        vol_mult = (vol10 / max(1.0, volday)) if volday else 0.0

        big_move = abs(chg_from_open_pct) >= PRICE_SPIKE["big_move_abs_pct"]
        if (abs(chg_from_open_pct) >= PRICE_SPIKE["from_open_abs_pct"] and
            abs(chg_10m_pct)      >= PRICE_SPIKE["ten_min_abs_pct"] and
            (vol_mult >= PRICE_SPIKE["volume_mult"] or big_move)) or big_move:

            cds[ticker] = now_ts
            append_csv(SPIKES_CSV, [
                datetime.now(timezone.utc).isoformat(), ticker,
                f"{opn:.4f}", f"{last:.4f}",
                f"{chg_from_open_pct:.3f}", f"{chg_10m_pct:.3f}",
                f"{vol_mult:.3f}"
            ])
            # Optional: add a positive label at this timestamp to seed learning
            append_csv(LABELS_CSV, [datetime.now(timezone.utc).isoformat(), ticker, "spike"])
            logged += 1

    return logged

# ---------------------- Market-wide Hype Scanner ----------------------

def scan_hype_market(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Market-wide hype scan:
      - Pull recent items from RSS_SOURCES (Reddit subs)
      - Group by detected entities (tickers/ETFs via cashtags & uppercase tokens)
      - Compute a simple burst score per entity vs. rolling baseline in state["metrics"]
      - Emit alerts when z >= HYPE_MIN_Z and hits >= HYPE_MIN_HITS
      - Schedule +60m and +24h follow-ups (like news alerts)
    Returns a list of alert dicts with id/ticker/ts/price/hits.
    """
    if not HYPE_SCAN_ENABLED or not is_market_open():
        return []

    # Collect recent items from all RSS feeds
    items: List[Dict[str, Any]] = []
    for url in RSS_SOURCES:
        try:
            items.extend(fetch_rss(url))
        except Exception:
            pass
    if not items:
        return []

    # Cluster by symbol/entity
    buckets = cluster_by_entity(items)
    t_now   = datetime.now(timezone.utc)
    alerts: List[Dict[str, Any]] = []

    # rolling metrics for z-score baselines
    metrics = state.setdefault("metrics", {})
    state.setdefault("pending", [])
    cds = state.setdefault("cooldowns_news", {})  # reuse news cooldown
    now_ts = int(t_now.timestamp())

    for ent, cluster in buckets.items():
        feats = build_cluster_features(cluster)
        hits  = int(feats["hits"])
        if hits < HYPE_MIN_HITS:
            continue

        # baseline BEFORE updating
        m          = metrics.setdefault(ent, {"count": 0, "total": 0})
        prev_count = int(m.get("count", 0))
        prev_total = int(m.get("total", 0))
        mu         = max(1.0, prev_count / max(1, prev_total))  # naive baseline
        z          = (hits - mu) / (mu ** 0.5)

        if z < HYPE_MIN_Z:
            continue

        # Update baseline AFTER measuring
        m["count"] = prev_count + hits
        m["total"] = prev_total + 1

        # cooldown
        if ent in cds and (now_ts - int(cds[ent])) < COOLDOWN_MIN_NEWS * 60:
            continue
        cds[ent] = now_ts

        # Build output + (optional) model features for training
        titles = [it.get("title","") for it in cluster]
        links  = [it.get("url","") or it.get("permalink","") for it in cluster]
        scores = [int(it.get("score",0) or 0) for it in cluster]
        comms  = [int(it.get("num_comments",0) or 0) for it in cluster]

        feats_vec, _ = build_features(titles, links, scores, comms, hits_in_window=hits)

        # Price and IDs
        ticker  = ent
        sig_id  = f"HYPE-{ticker}-{now_ts}"
        try:
            spot = float(get_last_price(ticker))
        except Exception:
            spot = 0.0

        # Training log
        log_signal_row(sig_id, ticker, t_now.isoformat(), feats_vec, titles, links,
                       max(scores or [0]), max(comms or [0]), hits)

        # Human-friendly why + hype score
        why = f"burst z={z:.1f}, hits={hits}, subs={feats['subs']}, hosts={feats['hosts']}, pol={feats['polarity']:+.2f}"
        hyp = hype_score(feats, z)

        # Optional Discord ping
        if HYPE_SEND_DISCORD:
            preview = " • ".join(titles[:3])
            msg = (f"**HYPE** {ticker} — {why}\n"
                   f"Price: {spot:.2f}\n"
                   f"{('Links: ' + ' | '.join(links[:3])) if links else ''}\n"
                   f"(hype={hyp:.2f})")
            try:
                send_discord(msg, title="Market-wide Hype")
            except Exception:
                pass

        # Schedule follow-ups
        t60  = now_ts + 60*60
        t24h = now_ts + 24*60*60
        state["pending"].extend([
            {"id": sig_id, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t60,  "label": "t+60m", "done": False},
            {"id": sig_id, "ticker": ticker, "alert_ts": now_ts, "alert_px": spot,
             "due_ts": t24h, "label": "t+24h", "done": False},
        ])

        alerts.append({
            "id": sig_id, "ticker": ticker, "ts": t_now.isoformat(),
            "price": spot, "hits": hits, "hype": round(hyp, 3), "why": why
        })

    return alerts

# ---------------------- Follow-ups ----------------------

def process_followups() -> List[Dict[str, Any]]:
    """Run due follow-ups and write outcomes.csv rows."""
    ensure_csv_headers(OUTCOMES_CSV, ["id","ticker","label","due_ts","alert_price","cur_price","ret_pct"])
    if not os.path.exists(STATE_FILE):
        return []
    try:
        state = json.load(open(STATE_FILE,"r",encoding="utf-8"))
    except Exception:
        return []

    now_ts = int(time.time())
    pending = state.get("pending", [])
    out: List[Dict[str, Any]] = []
    new_pending = []

    for p in pending:
        if p.get("done"):  # already processed
            continue
        due = int(p.get("due_ts", 0))
        if now_ts >= due:
            ticker = p.get("ticker","")
            alert_px = float(p.get("alert_px", 0.0))
            cur_px   = get_last_price(ticker)
            ret_pct  = (cur_px / alert_px - 1.0)*100.0 if alert_px else 0.0
            out.append({
                "id": p.get("id",""), "ticker": ticker, "label": p.get("label",""),
                "due_ts": datetime.fromtimestamp(due, tz=timezone.utc).isoformat(),
                "alert_px": alert_px, "cur_px": cur_px, "ret_pct": ret_pct
            })
        else:
            new_pending.append(p)

    state["pending"] = new_pending
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(state, f)

    return out

# ---------------------- Main ----------------------

def main():
    mode = os.getenv("MODE","").strip()
    ensure_csv_headers(ALERTS_CSV, ["id","ticker","ts","price","hits"])
    ensure_csv_headers(OUTCOMES_CSV, ["id","ticker","label","due_ts","alert_price","cur_price","ret_pct"])

    if mode == "weekly":
        # optional weekly summary could be implemented here
        print("Weekly mode: not implemented in this minimal file.")
        return
    if mode == "test":
        send_discord("**TEST** — webhook is working ✅", title="Test Alert")
        print("Sent test message.")
        return

    # Hard gate: skip scan if market is closed
    if not is_market_open():
        print("Market closed — skipping scan and follow-ups.")
        return

    # Load state
    state: Dict[str, Any] = {}
    if os.path.exists(STATE_FILE):
        try:
            state = json.load(open(STATE_FILE,"r",encoding="utf-8"))
        except Exception:
            state = {}
    state.setdefault("pending", []); state.setdefault("metrics", {})

    # 1) News-based alerts (watchlist)
    news_alerts = scan_news(state)

    # 2) Market-wide hype alerts (dynamic discovery)
    hype_alerts = scan_hype_market(state)

    # 3) Silent price spike logging
    spikes_logged = scan_price_spikes(state) if PRICE_SPIKES_ENABLED else 0

    # Log alerts
    for a in news_alerts + hype_alerts:
        append_csv(ALERTS_CSV, [a["id"], a["ticker"], a["ts"], f"{a['price']:.4f}", a["hits"]])

    # Persist state after scheduling follow-ups
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(state, f)

    # 4) Process follow-ups (for both news & hype alerts, we added them to pending)
    outcomes = process_followups()
    for o in outcomes:
        append_csv(OUTCOMES_CSV, [
            o["id"], o["ticker"], o["label"], o["due_ts"],
            f"{o['alert_px']:.4f}", f"{o['cur_px']:.4f}", f"{o['ret_pct']:.3f}"
        ])

    print("Scan complete.",
          "News alerts:", len(news_alerts),
          "Hype alerts:", len(hype_alerts),
          "Spikes logged:", spikes_logged,
          "Follow-ups processed:", len(outcomes))

if __name__ == "__main__":
    main()
