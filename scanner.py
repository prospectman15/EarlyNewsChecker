#!/usr/bin/env python3
"""
Early Event Scanner (Discord version)
- Runs in GitHub Actions every 10 minutes
- Scans Reddit (search + subreddit RSS)
- Looks for ticker/company terms near crisis words
- Ignores mainstream media links (tries to catch pre-media chatter)
- Sends concise alerts to Discord via webhook
- Simple cooldown via state.json so you don't get spammed

ENV:
  DISCORD_WEBHOOK_URL   (required in GitHub Actions secrets)

Edit WATCH_TICKERS / CRISIS_WORDS / MEDIA_BLACKLIST / RSS_SOURCES as you like.
"""

import os
import re
import json
import time
import html
import hashlib
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

# ------------------ Settings ------------------

# Tickers and the terms we associate with them (add both company & product words)
WATCH_TICKERS: Dict[str, List[str]] = {
    "BA":  ["boeing", "737", "787", "777", "max 9", "door plug"],
    "UAL": ["united airlines", "ual", "united flight", "united air"],
    "DAL": ["delta", "dal", "delta air lines"],
    "AAL": ["american airlines", "aal"],
    "NVDA":["nvidia", "nvda"],
    "AMD": ["amd", "advanced micro devices"],
    # add more:
    # "TSLA": ["tesla", "elon"],
}

# Crisis / incident keywords (sector-specific + general)
CRISIS_WORDS: List[str] = [
    "crash", "emergency landing", "mayday", "grounded", "evacuate",
    "explosion", "fire", "smoke", "engine failure", "bird strike",
    "faa", "ntsb", "probe", "recall", "lawsuit", "hack",
    "breach", "ransomware", "outage", "strike", "walkout", "ceo resign",
]

# Big media domains to EXCLUDE (we want pre-media hints, not official headlines)
MEDIA_BLACKLIST: List[str] = [
    "bloomberg.com","reuters.com","wsj.com","nytimes.com","cnn.com",
    "cnbc.com","apnews.com","foxnews.com","marketwatch.com",
    "seekingalpha.com","financialtimes.com","ft.com","benzinga.com",
    "yahoo.com","forbes.com","theguardian.com","investing.com",
    "washingtonpost.com","usatoday.com","nbcnews.com","abcnews.go.com",
    "bbc.com","barrons.com","coindesk.com","cointelegraph.com",
]

# Subreddit RSS feeds to skim (community chatter that often surfaces early)
RSS_SOURCES: List[str] = [
    "https://www.reddit.com/r/aviation/.rss",
    "https://www.reddit.com/r/airlines/.rss",
    "https://www.reddit.com/r/stocks/.rss",
    "https://www.reddit.com/r/wallstreetbets/.rss",
    # add niche ones you care about:
    # "https://www.reddit.com/r/technews/.rss",
]

# Scoring / spam control
MULTI_POST_WINDOW_MIN = 10       # look back this many minutes for clustering
REQUIRED_MATCHES = 2             # require at least N independent posts to alert
COOLDOWN_MIN = 90                # don't alert again for the same ticker for N minutes

# File to store last alert times (kept across runs via Actions cache or commit)
STATE_FILE = "state.json"

# Discord webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ------------------ Helpers ------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape((s or "")).strip()).lower()

def contains_any(text: str, terms: List[str]) -> bool:
    t = clean_text(text)
    return any(term in t for term in terms)

def is_blacklisted(url: str) -> bool:
    u = (url or "").lower()
    return any(dom in u for dom in MEDIA_BLACKLIST)

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)

def send_discord(message: str, title: str = "Early Event Scanner") -> None:
    if not DISCORD_WEBHOOK_URL:
        print("[DRY RUN] ALERT:", title, "\n", message)
        return
    payload = {
        "embeds": [{
            "title": title,
            "description": message
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    except Exception as e:
        print("Discord post failed:", e)

def short_url(u: str, max_len: int = 120) -> str:
    if not u:
        return ""
    return (u if len(u) <= max_len else u[: max_len - 1] + "…")

def sig_of_post(p: Dict[str, Any]) -> str:
    """Make a simple signature to de-dupe (title + url)."""
    raw = f"{p.get('title','')}|{p.get('url','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ------------------ Sources ------------------

def fetch_reddit_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """
    Public Reddit search JSON (no API key). Keep it light to avoid rate limits.
    """
    url = f"https://www.reddit.com/search.json?q={requests.utils.quote(query)}&sort=new&limit={limit}&t=hour"
    headers = {"User-Agent": "early-event-scanner/0.1 by github-actions"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return []
        out = []
        for c in r.json().get("data", {}).get("children", []):
            d = c.get("data", {})
            out.append({
                "title": d.get("title", ""),
                "url": d.get("url_overridden_by_dest") or d.get("url") or "",
                "permalink": "https://www.reddit.com" + d.get("permalink", ""),
                "created_utc": d.get("created_utc", 0),
                "source": "reddit_search",
            })
        return out
    except Exception:
        return []

def fetch_rss(url: str) -> List[Dict[str, Any]]:
    """
    Minimal RSS reader (regex-based) to avoid extra dependencies.
    Works fine for Reddit RSS.
    """
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        txt = r.text
        items = []
        for m in re.finditer(r"<item>(.*?)</item>", txt, re.S | re.I):
            chunk = m.group(1)
            tit = re.search(r"<title>(.*?)</title>", chunk, re.S | re.I)
            link = re.search(r"<link>(.*?)</link>", chunk, re.S | re.I)
            title = html.unescape(tit.group(1)) if tit else ""
            linku = html.unescape(link.group(1)) if link else ""
            items.append({
                "title": title,
                "url": linku,
                "created_utc": time.time(),  # Reddit RSS often fine without exact pubDate
                "source": url,
            })
        return items
    except Exception:
        return []

# ------------------ Core scan ------------------

def scan_once() -> None:
    state = load_state()
    t_now = now_utc()
    window_start = t_now - timedelta(minutes=MULTI_POST_WINDOW_MIN)

    # Collect candidate posts
    candidates: List[Dict[str, Any]] = []

    # Reddit search per ticker
    for ticker, terms in WATCH_TICKERS.items():
        # build a naive query string (OR terms)
        q = " OR ".join(list({*terms, ticker.lower()}))
        posts = fetch_reddit_search(q, limit=25)

        for p in posts:
            created = datetime.fromtimestamp(p.get("created_utc", 0) or 0, tz=timezone.utc)
            if created < window_start:
                continue
            title = p.get("title", "")
            url = p.get("url") or p.get("permalink", "")
            if is_blacklisted(url):
                continue  # skip big media links
            # require both: (ticker terms) AND (crisis word)
            if contains_any(title, terms + [ticker.lower()]) and contains_any(title, CRISIS_WORDS):
                p2 = {
                    "ticker": ticker,
                    "title": title,
                    "url": url or p.get("permalink", ""),
                    "created": created,
                    "source": p.get("source", "reddit_search"),
                }
                candidates.append(p2)

    # RSS feeds (treat like community chatter)
    for feed in RSS_SOURCES:
        items = fetch_rss(feed)
        for it in items:
            created = datetime.fromtimestamp(it.get("created_utc", time.time()), tz=timezone.utc)
            if created < window_start:
                continue
            title = it.get("title", "")
            url = it.get("url", "")
            if is_blacklisted(url):
                continue
            for ticker, terms in WATCH_TICKERS.items():
                if contains_any(title, terms + [ticker.lower()]) and contains_any(title, CRISIS_WORDS):
                    p2 = {
                        "ticker": ticker,
                        "title": title,
                        "url": url,
                        "created": created,
                        "source": feed,
                    }
                    candidates.append(p2)

    # De-dup very similar items (by simple signature)
    seen_sigs = set()
    unique_candidates = []
    for c in candidates:
        sig = sig_of_post({"title": c["title"], "url": c["url"]})
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        unique_candidates.append(c)

    # Group by ticker; require multiple hits inside the window
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for c in unique_candidates:
        by_ticker.setdefault(c["ticker"], []).append(c)

    alerts: List[Dict[str, Any]] = []
    for ticker, posts in by_ticker.items():
        posts = [p for p in posts if p["created"] >= window_start]
        posts.sort(key=lambda x: x["created"], reverse=True)
        if len(posts) >= REQUIRED_MATCHES:
            last_alert_ts = state.get(ticker, 0)
            minutes_since = (
                (t_now - datetime.fromtimestamp(last_alert_ts, tz=timezone.utc)).total_seconds() / 60
                if last_alert_ts else 1e9
            )
            if minutes_since >= COOLDOWN_MIN:
                # Build a concise alert
                top_lines = []
                for p in posts[:3]:
                    ts = p["created"].strftime("%H:%M UTC")
                    top_lines.append(f"• {ts} — {p['title']}")
                links = " | ".join(short_url(p["url"]) for p in posts[:3])

                message = (
                    f"**{ticker}** — possible early event "
                    f"({len(posts)} hits / {MULTI_POST_WINDOW_MIN}m)\n"
                    + "\n".join(top_lines) + "\n"
                    + (f"Links: {links}" if links else "")
                )

                alerts.append({"ticker": ticker, "message": message})

    # Send and update state
    for a in alerts:
        send_discord(a["message"], title="Early Event Scanner")
        state[a["ticker"]] = int(t_now.timestamp())

    save_state(state)

# ------------------ Main ------------------

if __name__ == "__main__":
    scan_once()
    print("Scan complete.")
