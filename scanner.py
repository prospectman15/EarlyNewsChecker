#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trade Detector — Multi-Source Early Sentiment Scanner
- Reddit (multiple subs)
- Stocktwits (public JSON)
- Seeking Alpha Market Currents (RSS)
- PRNewswire (RSS)

Features
- Validates tickers against official NASDAQ/NYSE listings (cached)
- Accepts bare tickers like "BA" and "$BA" (both work)
- Accident/recall/catastrophe keyword coverage (plane crash, fire, explosion, etc.)
- Cross-source confidence boost (same ticker seen on multiple platforms in short window)
- Name→ticker backfill (e.g., "Boeing" -> BA) for early chatter without cashtags
- Discord alerts with Positive/Negative/Mixed label and Confidence %

Environment:
- DISCORD_WEBHOOK_URL (required)
- TICKER_CACHE_DIR (default: ./data)
- TICKER_CACHE_TTL_DAYS (default: 1)

Run on a schedule (e.g., cron every 10 minutes).
"""

import os
import io
import re
import html
import json
import math
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ======================================================
# 1) TICKER VALIDATION (NASDAQ/NYSE whitelist, cached)
# ======================================================

CACHE_DIR = os.getenv("TICKER_CACHE_DIR", "data")
CACHE_PATH = os.path.join(CACHE_DIR, "valid_tickers.csv")
META_PATH = os.path.join(CACHE_DIR, "valid_tickers.meta.json")
TTL_DAYS = int(os.getenv("TICKER_CACHE_TTL_DAYS", "1"))

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://ftp.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# Accept bare tickers (BA) and dot/hyphen classes (BRK.B, RDS-A)
TICKER_SHAPE_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")
DOLLAR_TICKER_RE = re.compile(r"\$([A-Za-z][A-Za-z.\-]{0,6})\b")  # $BA
BARE_TICKER_RE   = re.compile(r"\b([A-Z]{1,5}(?:[.-][A-Z]{1,2})?)\b")  # BA

COMMON_CAPS_SKIP = {
    # common finance/casual caps that are not equities
    "A","AN","AND","ARE","AT","BE","BUY","CALL","CEO","CFO","DD","EPS","FUD","FYI",
    "GDP","IMO","IV","IVR","JAN","JUL","JUN","MAR","MAY","MIT","MOON","NAV","OR",
    "OTM","PCE","PE","PFD","PLS","PM","PR","PUT","Q","QQQ","ROFL","RSI","SEC",
    "SPAC","SPY","TA","TBA","TBD","TBF","TIPS","TLT","UK","US","USD","VIX","WSB",
    "WTF","YOLO"
}

def _ensure_cache_dir():
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

def _stale_meta(ttl_days: int) -> bool:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        last = datetime.fromisoformat(meta.get("last_refresh"))
        return datetime.utcnow() - last > timedelta(days=ttl_days)
    except Exception:
        return True

def _write_meta():
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_refresh": datetime.utcnow().isoformat()}, f)
    except Exception:
        pass

def _download_txt(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

def _parse_nasdaq_table(text: str, symbol_col: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if not ln.startswith("File Creation Time")]
    buf = io.StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep="|")
    df.columns = [c.strip() for c in df.columns]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
    if "NextShares" in df.columns:
        df = df[df["NextShares"].astype(str).str.upper().ne("Y")]
    df = df[[symbol_col]].dropna().copy()
    df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
    df.rename(columns={symbol_col: "Symbol"}, inplace=True)
    return df

def refresh_valid_tickers() -> pd.DataFrame:
    logging.info("Refreshing ticker lists from NASDAQ Trader…")
    nas_text = _download_txt(NASDAQ_LISTED_URL)
    other_text = _download_txt(OTHER_LISTED_URL)

    nas = _parse_nasdaq_table(nas_text, "Symbol")
    try:
        oth = _parse_nasdaq_table(other_text, "ACT Symbol")
    except Exception:
        oth = _parse_nasdaq_table(other_text, "Symbol")

    all_syms = pd.concat([nas, oth], ignore_index=True)
    all_syms.drop_duplicates(subset=["Symbol"], inplace=True)
    all_syms = all_syms[all_syms["Symbol"].str.len() > 0]

    _ensure_cache_dir()
    all_syms.to_csv(CACHE_PATH, index=False)
    _write_meta()
    logging.info("Saved %d symbols to %s", len(all_syms), CACHE_PATH)
    return all_syms

def load_or_refresh_valid_tickers() -> pd.Series:
    _ensure_cache_dir()
    needs_refresh = (not os.path.isfile(CACHE_PATH)) or _stale_meta(TTL_DAYS)
    if needs_refresh:
        try:
            df = refresh_valid_tickers()
        except Exception as e:
            logging.warning("Refresh failed: %s", e)
            if os.path.isfile(CACHE_PATH):
                df = pd.read_csv(CACHE_PATH)
            else:
                df = pd.DataFrame({"Symbol": []})
    else:
        df = pd.read_csv(CACHE_PATH)
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df = df[df["Symbol"].str.len() > 0].drop_duplicates("Symbol")
    return df["Symbol"]

def build_valid_set() -> set:
    return set(load_or_refresh_valid_tickers().tolist())

def is_ticker_shaped(token: str) -> bool:
    return bool(TICKER_SHAPE_RE.match(token))

def normalize_token(token: str) -> str:
    t = token.strip().upper()
    t = re.sub(r"^[^\w$]+|[^\w.%-]+$", "", t)
    if t.startswith("$"):
        t = t[1:]
    return t

def extract_candidates(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    cands = []
    for m in DOLLAR_TICKER_RE.finditer(text):
        cands.append(m.group(1))
    for m in BARE_TICKER_RE.finditer(text):
        cands.append(m.group(1))
    out = []
    for raw in cands:
        tok = normalize_token(raw)
        if not tok or tok in COMMON_CAPS_SKIP:
            continue
        if is_ticker_shaped(tok):
            out.append(tok)
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

# Lightweight name→ticker hints for early chatter without cashtags
NAME_TO_TICKER = {
    "boeing": "BA",
    "tesla": "TSLA",
    "apple": "AAPL",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    "ford": "F",
    "generalmotors": "GM",
    "general motors": "GM",
    "united airlines": "UAL",
    "delta": "DAL",
    "american airlines": "AAL",
    "southwest": "LUV",
    "spirit": "SAVE",
    "jetblue": "JBLU",
    "fidelity": "FNF",  # example; adjust per your focus
}

def backfill_name_tickers(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits = []
    for name, sym in NAME_TO_TICKER.items():
        if name in t:
            hits.append(sym)
    # dedupe keep order
    seen = set()
    out = []
    for s in hits:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

class TickerValidator:
    def __init__(self):
        self._valid = None

    @property
    def valid_set(self) -> set:
        if self._valid is None:
            self._valid = build_valid_set()
        return self._valid

    def extract_valid(self, text: str, limit: Optional[int] = None, allow_backfill: bool = True) -> List[str]:
        # candidates from shapes
        cands = extract_candidates(text)
        # optionally add name->ticker hints for early chatter
        if allow_backfill:
            cands.extend(backfill_name_tickers(text))
        # validate against exchange list
        valid = [t for t in cands if t in self.valid_set]
        if limit and limit > 0:
            valid = valid[:limit]
        return valid

# ================================================
# 2) SOURCE FETCHERS — Reddit, Stocktwits, RSSes
# ================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AITradeDetector/1.0)"
}

def fetch_reddit(subreddits: List[str], limit_per_sub: int = 25) -> List[Dict[str, Any]]:
    items = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit={limit_per_sub}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                logging.warning("Reddit %s status %s", sub, r.status_code)
                continue
            data = r.json()
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                items.append({
                    "platform": "reddit",
                    "source": f"Reddit/r/{sub}",
                    "title": d.get("title") or "",
                    "text": d.get("selftext") or "",
                    "url": f"https://www.reddit.com{d.get('permalink')}" if d.get("permalink") else d.get("url"),
                    "ups": d.get("ups") or 0,
                    "num_comments": d.get("num_comments") or 0,
                    "created_ts": datetime.utcfromtimestamp(d.get("created_utc")) if d.get("created_utc") else datetime.utcnow(),
                })
        except Exception as e:
            logging.warning("Reddit fetch error for %s: %s", sub, e)
    return items

def fetch_stocktwits_trending(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Public endpoint for trending streams. If unavailable, this will just skip.
    """
    url = "https://api.stocktwits.com/api/2/streams/trending.json"
    items = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return items
        data = r.json()
        for m in data.get("messages", [])[:limit]:
            body = m.get("body") or ""
            ts_str = m.get("created_at")
            try:
                created = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ") if ts_str else datetime.utcnow()
            except Exception:
                created = datetime.utcnow()
            # Include any cashtags provided by Stocktwits
            tags = [t.get("symbol") for t in (m.get("symbols") or []) if t.get("symbol")]
            items.append({
                "platform": "stocktwits",
                "source": "Stocktwits/trending",
                "title": body[:140],
                "text": body,
                "url": f"https://stocktwits.com/message/{m.get('id')}",
                "ups": int(m.get("likes", {}).get("total", 0)),
                "num_comments": 0,
                "created_ts": created,
                "cashtags_hint": tags,  # extra hint for tickers
            })
    except Exception as e:
        logging.warning("Stocktwits fetch error: %s", e)
    return items

def fetch_rss(feed_url: str, label: str, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Minimal RSS parser: title/description/link. Timestamps if pubDate exists.
    """
    items = []
    try:
        r = requests.get(feed_url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return items
        txt = r.text
        for chunk in re.findall(r"<item\b.*?>.*?</item>", txt, flags=re.DOTALL | re.IGNORECASE)[:limit]:
            title = re.search(r"<title>(.*?)</title>", chunk, flags=re.DOTALL | re.IGNORECASE)
            link = re.search(r"<link>(.*?)</link>", chunk, flags=re.DOTALL | re.IGNORECASE)
            desc = re.search(r"<description>(.*?)</description>", chunk, flags=re.DOTALL | re.IGNORECASE)
            pub = re.search(r"<pubDate>(.*?)</pubDate>", chunk, flags=re.DOTALL | re.IGNORECASE)

            title = html.unescape(title.group(1)).strip() if title else ""
            title = re.sub(r"^<!\[CDATA\[|\]\]>$", "", title)
            link = html.unescape(link.group(1)).strip() if link else ""
            link = re.sub(r"^<!\[CDATA\[|\]\]>$", "", link)
            description = html.unescape(desc.group(1)).strip() if desc else ""
            description = re.sub(r"<.*?>", "", description)
            try:
                created = datetime.strptime(pub.group(1), "%a, %d %b %Y %H:%M:%S %Z") if pub else datetime.utcnow()
            except Exception:
                created = datetime.utcnow()

            items.append({
                "platform": label.lower(),
                "source": label,
                "title": title,
                "text": description,
                "url": link,
                "ups": 0,
                "num_comments": 0,
                "created_ts": created,
            })
    except Exception as e:
        logging.warning("RSS fetch error (%s): %s", label, e)
    return items

def load_sources() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # Reddit subs that surface early chatter
    subs = ["stocks", "wallstreetbets", "StockMarket", "investing", "options", "pennystocks", "shortsqueeze"]
    items += fetch_reddit(subs, limit_per_sub=25)

    # Stocktwits trending (fastest pulse on cashtags)
    items += fetch_stocktwits_trending(limit=60)

    # Seeking Alpha Market Currents (news tickers in title/body often)
    items += fetch_rss("https://seekingalpha.com/market_currents.xml", "SeekingAlpha/MarketCurrents", limit=40)

    # PRNewswire (corporate PRs; look for recalls/reschedules/updates)
    items += fetch_rss("https://www.prnewswire.com/rss/all-news-releases-news.rss", "PRNewswire", limit=40)

    return items

# ============================================
# 3) SENTIMENT, EVENT WORDS & CONFIDENCE MATH
# ============================================

POSITIVE_WORDS = {
    "beat","beats","beating","surpass","surpassed","record","upgrade","upgraded",
    "raise","raised","raises","strong","bull","bullish","moat","profit","profits",
    "profitable","guidance raise","surprise","outperform","outperformance","buy",
    "accumulate","momentum","breakout","catalyst","expansion","growth","contract win",
    "award","awarded","approval","fda approval","certified","secured","partnership",
}

NEGATIVE_WORDS = {
    "miss","misses","missed","downgrade","downgraded","warning","warns","guide down",
    "weak","bear","bearish","loss","losses","unprofitable","fraud","scandal","probe",
    "investigation","lawsuit","recall","delist","halted","layoff","layoffs","bankruptcy",
    "chapter 11","going concern","plunge","collapse","restatement","sec charges",
    # Accident/Catastrophe additions:
    "crash","plane crash","air crash","explosion","blast","fire","factory fire",
    "accident","incident","fatal","fatalities","engine failure","grounded","grounding",
    "emergency landing","mid-air","derailed","oil spill","leak","breach","data breach",
}

def sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, (pos + neg))
    return float(np.clip(score, -1.0, 1.0))

def hype_score(item: Dict[str, Any]) -> float:
    ups = int(item.get("ups") or 0)
    com = int(item.get("num_comments") or 0)
    def squash(x): return 1.0 - math.exp(-x / 200.0)
    return float(np.clip(squash(ups + 1.5 * com), 0.0, 1.0))

def consensus_multiplier(num_sources: int, num_mentions: int) -> float:
    """
    Boost confidence when:
      - same ticker appears across multiple distinct platforms
      - there are multiple mentions in the window
    """
    src_boost = {1: 1.00, 2: 1.08, 3: 1.13}
    src_mult = src_boost.get(min(num_sources, 3), 1.15)  # 4+ sources cap at 1.15
    mention_mult = 1.0 + min(0.10, 0.03 * max(0, num_mentions - 1))
    return src_mult * mention_mult

def make_confidence_percent(sentiment: float, hype: float, num_sources: int, num_mentions: int) -> int:
    base = 0.55 * abs(sentiment) + 0.35 * hype
    base *= consensus_multiplier(num_sources, num_mentions)
    return int(np.clip(round(base * 100), 20, 98))

def label_from_sentiment(s: float) -> str:
    if s > 0.06:
        return "Positive"
    elif s < -0.06:
        return "Negative"
    return "Mixed/Neutral"

def short_reason(text: str, max_len: int = 220) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return (t[: max_len - 1] + "…") if len(t) > max_len else t

# ------------------------------------------------
# 4) CROSS-SOURCE WINDOW AGGREGATION (45 minutes)
# ------------------------------------------------

WINDOW_MINUTES = 45

def aggregate_cross_source(alert_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a per-ticker aggregate over a short window to measure:
      - how many sources referenced it
      - how many total mentions
      - most negative/positive sentiment seen (for direction)
    Returns dict: ticker -> aggregate info
    """
    by_ticker: Dict[str, Dict[str, Any]] = {}
    cutoff = datetime.utcnow() - timedelta(minutes=WINDOW_MINUTES)

    for row in alert_rows:
        ts = row["created_ts"]
        if ts < cutoff:
            continue
        for tkr in row["tickers"]:
            agg = by_ticker.setdefault(tkr, {
                "sources": set(),
                "mentions": 0,
                "sentiments": [],
            })
            agg["sources"].add(row["platform"])
            agg["mentions"] += 1
            agg["sentiments"].append(row["sentiment"])

    # finalize
    for tkr, agg in by_ticker.items():
        agg["num_sources"] = len(agg["sources"])
        agg["num_mentions"] = agg["mentions"]
        # representative sentiment = mean clipped [-1,1]
        if agg["sentiments"]:
            agg["rep_sentiment"] = float(np.clip(np.mean(agg["sentiments"]), -1.0, 1.0))
        else:
            agg["rep_sentiment"] = 0.0

    return by_ticker

# --------------------------
# 5) DISCORD SENDER
# --------------------------

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def discord_send(content: str, embeds: Optional[List[Dict[str, Any]]] = None):
    if not DISCORD_WEBHOOK_URL:
        logging.error("DISCORD_WEBHOOK_URL not set. Skipping Discord send.")
        return
    payload = {"content": content}
    if embeds:
        payload["embeds"] = embeds
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
        if r.status_code >= 300:
            logging.error("Discord error %s: %s", r.status_code, r.text[:1000])
    except Exception as e:
        logging.error("Discord send exception: %s", e)

def build_discord_embed(item: Dict[str, Any], tickers: List[str], label: str, confidence_pct: int) -> Dict[str, Any]:
    color = 0x19A974  # green
    if label.startswith("Negative"):
        color = 0xE7040F  # red
    elif label.startswith("Mixed"):
        color = 0xFFBF00  # amber

    title = item.get("title") or "(no title)"
    url = item.get("url") or ""
    source = item.get("source") or "unknown"
    reason = short_reason(item.get("text") or item.get("title") or "")

    desc_lines = [
        f"**Sentiment:** {label}  |  **Confidence:** {confidence_pct}%",
        f"**Tickers:** {', '.join(tickers)}",
        f"**Source:** {source}",
    ]
    if reason:
        desc_lines.append(f"**Why:** {reason}")

    embed = {
        "title": title[:250],
        "url": url,
        "description": "\n".join(desc_lines)[:3900],
        "color": color,
        "footer": {"text": "AI Trade Detector • Validated tickers only"},
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    return embed

# -------------------------------------------------
# 6) MAIN PIPE: fetch → score → aggregate → alert
# -------------------------------------------------

def process_and_alert(items: List[Dict[str, Any]],
                      tv: TickerValidator,
                      min_abs_sentiment: float = 0.14,
                      min_hype: float = 0.12,
                      max_tickers_per_item: int = 3) -> int:
    """
    - Validates tickers (accepts BA and $BA)
    - Scores sentiment & hype
    - Aggregates cross-source to produce confidence %
    - Sends Discord alert per qualifying item
    """
    # Step 1: compute per-item basics
    candidate_rows: List[Dict[str, Any]] = []
    for it in items:
        try:
            text_join = " \n".join(filter(None, [it.get("title"), it.get("text")]))
            # Accept bare ticker names (BA) and cashtags
            tickers = tv.extract_valid(text_join, limit=max_tickers_per_item, allow_backfill=True)

            # If Stocktwits provided cashtags, include them (will be validated)
            for hint in it.get("cashtags_hint") or []:
                if hint and hint.upper() not in tickers and hint.upper() in tv.valid_set:
                    tickers.append(hint.upper())

            # HARD GATE: must have at least one validated ticker
            if not tickers:
                continue

            sent = sentiment_score(text_join)
            hype = hype_score(it)
            # Basic screen: either sentiment or hype needs to clear a bar
            if abs(sent) < min_abs_sentiment and hype < min_hype:
                continue

            candidate_rows.append({
                "platform": it.get("platform") or "unknown",
                "source": it.get("source"),
                "title": it.get("title"),
                "text": it.get("text"),
                "url": it.get("url"),
                "created_ts": it.get("created_ts") or datetime.utcnow(),
                "tickers": tickers,
                "sentiment": sent,
                "hype": hype,
            })
        except Exception as e:
            logging.error("Item processing error: %s\n%s", e, traceback.format_exc())

    if not candidate_rows:
        return 0

    # Step 2: cross-source window aggregation
    by_ticker = aggregate_cross_source(candidate_rows)

    # Step 3: send an alert per candidate, using aggregated confidence
    alerts = 0
    for row in candidate_rows:
        try:
            # Combine info for this item's best ticker group
            # Use the first ticker (post is already filtered to top few)
            tkr = row["tickers"][0]
            agg = by_ticker.get(tkr, {"num_sources": 1, "num_mentions": 1, "rep_sentiment": row["sentiment"]})

            conf = make_confidence_percent(
                sentiment = row["sentiment"] if abs(row["sentiment"]) >= abs(agg.get("rep_sentiment", 0.0)) else agg.get("rep_sentiment", 0.0),
                hype = row["hype"],
                num_sources = agg.get("num_sources", 1),
                num_mentions = agg.get("num_mentions", 1)
            )

            label = label_from_sentiment(row["sentiment"])
            embed = build_discord_embed(row, row["tickers"], label, conf)
            content = f"**{label}** alert for **{', '.join(row['tickers'])}** — Confidence **{conf}%**"
            discord_send(content, embeds=[embed])
            alerts += 1
        except Exception as e:
            logging.error("Alert send error: %s\n%s", e, traceback.format_exc())

    return alerts

def run_once():
    tv = TickerValidator()
    items = load_sources()
    n = process_and_alert(items, tv)
    logging.info("Run complete. Alerts sent: %d", n)

if __name__ == "__main__":
    run_once()
