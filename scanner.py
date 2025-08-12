#!/usr/bin/env python3
"""
Early Event Scanner (Discord + Price Tracking)
- Scans Reddit search + subreddit RSS every 10 min (GitHub Actions)
- Alerts when multiple crisis-keyword posts mention your tickers within window
- Excludes mainstream media links (tries to catch pre-media chatter)
- On alert, records price and schedules follow-ups at +60m and +24h
- Logs to CSVs; maintains rolling "learning" metrics per ticker

ENV:
  DISCORD_WEBHOOK_URL  (Discord webhook URL, set as GitHub Actions secret)

FILES (committed/cached between runs):
  state.json           - cooldowns, pending followups, per-ticker metrics
  alerts.csv           - alert events (append-only)
  outcomes.csv         - follow-up results (append-only)

Requires:
  pip install requests yfinance
"""

import os, re, json, time, html, hashlib, csv
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple

import yfinance as yf

# ------------------ Settings ------------------

WATCH_TICKERS: Dict[str, List[str]] = {
    "BA":  ["boeing", "737", "787", "777", "max 9", "door plug"],
    "UAL": ["united airlines", "ual", "united flight", "united air"],
    "DAL": ["delta", "dal", "delta air lines"],
    "AAL": ["american airlines", "aal"],
    "NVDA":["nvidia", "nvda"],
    "AMD": ["amd", "advanced micro devices"],
}

CRISIS_WORDS: List[str] = [
    "crash","emergency landing","mayday","grounded","evacuate",
    "explosion","fire","smoke","engine failure","bird strike",
    "faa","ntsb","probe","recall","lawsuit","hack",
    "breach","ransomware","outage","strike","walkout","ceo resign",
]

MEDIA_BLACKLIST: List[str] = [
    "bloomberg.com","reuters.com","wsj.com","nytimes.com","cnn.com",
    "cnbc.com","apnews.com","foxnews.com","marketwatch.com",
    "seekingalpha.com","financialtimes.com","ft.com","benzinga.com",
    "yahoo.com","forbes.com","theguardian.com","investing.com",
    "washingtonpost.com","usatoday.com","nbcnews.com","abcnews.go.com",
    "bbc.com","barrons.com","coindesk.com","cointelegraph.com",
]

RSS_SOURCES: List[str] = [
    "https://www.reddit.com/r/aviation/.rss",
    "https://www.reddit.com/r/airlines/.rss",
    "https://www.reddit.com/r/stocks/.rss",
    "https://www.reddit.com/r/wallstreetbets/.rss",
]

MULTI_POST_WINDOW_MIN = 10
REQUIRED_MATCHES = 2
COOLDOWN_MIN = 90

STATE_FILE = "state.json"
ALERTS_CSV = "alerts.csv"
OUTCOMES_CSV = "outcomes.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL","").strip()

# ------------------ Helpers ------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def clean_text(s: str) -> str:
    import re, html as _html
    return re.sub(r"\s+"," ", _html.unescape((s or "")).strip()).lower()

def contains_any(text: str, terms: List[str]) -> bool:
    t = clean_text(text)
    return any(term in t for term in terms)

def is_blacklisted(url: str) -> bool:
    u = (url or "").lower()
    return any(dom in u for dom in MEDIA_BLACKLIST)

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE): return {}
    try:
        return json.load(open(STATE_FILE,"r",encoding="utf-8"))
    except Exception:
        return {}

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(state,f)

def ensure_csv_headers(path: str, headers: List[str]) -> None:
    exists = os.path.exists(path)
    if not exists:
        with open(path,"w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_csv(path: str, row: List[Any]) -> None:
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def send_discord(message: str, title: str = "Early Event Scanner") -> None:
    if not DISCORD_WEBHOOK_URL:
        print("[DRY RUN] ALERT:", title, "\n", message)
        return
    payload = {"embeds":[{"title": title, "description": message}]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    except Exception as e:
        print("Discord post failed:", e)

def short_url(u: str, max_len: int = 120) -> str:
    return (u if (u and len(u)<=max_len) else (u[:max_len-1]+"…")) if u else ""

def sig_of_post(p: Dict[str, Any]) -> str:
    raw = f"{p.get('title','')}|{p.get('url','')}"
    import hashlib
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ------------------ Price utils ------------------

def get_last_price(ticker: str) -> Tuple[float, str]:
    """
    Returns (price, source_note).
    Tries 1m intraday; falls back to last close.
    """
    try:
        df = yf.download(tickers=ticker, period="2d", interval="1m", progress=False, threads=False)
        if df is not None and len(df)>0:
            price = float(df["Close"].dropna().iloc[-1])
            return price, "1m"
    except Exception:
        pass
    try:
        df = yf.download(tickers=ticker, period="5d", interval="1d", progress=False, threads=False)
        if df is not None and len(df)>0:
            price = float(df["Close"].dropna().iloc[-1])
            return price, "1d"
    except Exception:
        pass
    return float("nan"), "na"

# ------------------ Sources ------------------

def fetch_reddit_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/search.json?q={requests.utils.quote(query)}&sort=new&limit={limit}&t=hour"
    headers = {"User-Agent":"early-event-scanner/0.2 by github-actions"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return []
        out=[]
        for c in r.json().get("data",{}).get("children",[]):
            d=c.get("data",{})
            out.append({
                "title": d.get("title",""),
                "url": d.get("url_overridden_by_dest") or d.get("url") or "",
                "permalink": "https://www.reddit.com"+d.get("permalink",""),
                "created_utc": d.get("created_utc",0),
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
        import re, html as _html
        items=[]
        for m in re.finditer(r"<item>(.*?)</item>", txt, re.S|re.I):
            chunk=m.group(1)
            tit = re.search(r"<title>(.*?)</title>", chunk, re.S|re.I)
            link= re.search(r"<link>(.*?)</link>", chunk, re.S|re.I)
            items.append({
                "title": _html.unescape(tit.group(1)) if tit else "",
                "url": _html.unescape(link.group(1)) if link else "",
                "created_utc": time.time(),
                "source": url,
            })
        return items
    except Exception:
        return []

# ------------------ Core scan + tracking ------------------

def scan_and_alert() -> List[Dict[str, Any]]:
    """Returns list of newly-created alerts (for downstream logging)."""
    state = load_state()
    t_now = now_utc()
    window_start = t_now - timedelta(minutes=MULTI_POST_WINDOW_MIN)
    candidates: List[Dict[str, Any]] = []

    # Reddit search
    for ticker, terms in WATCH_TICKERS.items():
        q = " OR ".join(list({*terms, ticker.lower()}))
        posts = fetch_reddit_search(q, limit=25)
        for p in posts:
            created = datetime.fromtimestamp(p.get("created_utc",0) or 0, tz=timezone.utc)
            if created < window_start: continue
            title = p.get("title","")
            url   = p.get("url") or p.get("permalink","")
            if is_blacklisted(url): continue
            if contains_any(title, terms + [ticker.lower()]) and contains_any(title, CRISIS_WORDS):
                candidates.append({
                    "ticker": ticker, "title": title, "url": url or p.get("permalink",""),
                    "created": created, "source": p.get("source","reddit_search")
                })

    # RSS
    for feed in RSS_SOURCES:
        items = fetch_rss(feed)
        for it in items:
            created = datetime.fromtimestamp(it.get("created_utc",time.time()), tz=timezone.utc)
            if created < window_start: continue
            title = it.get("title",""); url = it.get("url","")
            if is_blacklisted(url): continue
            for ticker, terms in WATCH_TICKERS.items():
                if contains_any(title, terms + [ticker.lower()]) and contains_any(title, CRISIS_WORDS):
                    candidates.append({
                        "ticker": ticker, "title": title, "url": url,
                        "created": created, "source": feed
                    })

    # De-dup
    seen=set(); uniq=[]
    for c in candidates:
        sig = sig_of_post({"title": c["title"], "url": c["url"]})
        if sig in seen: continue
        seen.add(sig); uniq.append(c)

    # Group + build alerts
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for c in uniq:
        by_ticker.setdefault(c["ticker"], []).append(c)

    new_alerts=[]
    state.setdefault("cooldowns", {})
    state.setdefault("pending", [])     # list of followups: {id,ticker,alert_ts,alert_px,when,label}
    state.setdefault("metrics", {})     # per-ticker: {count, win_count, avg_abs_move}

    for ticker, posts in by_ticker.items():
        posts = [p for p in posts if p["created"] >= window_start]
        posts.sort(key=lambda x: x["created"], reverse=True)
        if len(posts) < REQUIRED_MATCHES: continue

        last_ts = state["cooldowns"].get(ticker, 0)
        minutes_since = ( (t_now - datetime.fromtimestamp(last_ts, tz=timezone.utc)).total_seconds()/60
                          if last_ts else 1e9 )
        if minutes_since < COOLDOWN_MIN: continue

        # Get price now
        alert_px, px_src = get_last_price(ticker)

        # Build message
        top_lines=[]
        for p in posts[:3]:
            ts = p["created"].strftime("%H:%M UTC")
            top_lines.append(f"• {ts} — {p['title']}")
        links = " | ".join(short_url(p["url"]) for p in posts[:3])
        message = (
            f"**{ticker}** — possible early event ({len(posts)} hits / {MULTI_POST_WINDOW_MIN}m)\n"
            f"Spot price: ${alert_px:.2f} (src: {px_src})\n"
            + "\n".join(top_lines) + ("\nLinks: " + links if links else "")
        )

        # Append quick metrics if exist
        m = state["metrics"].get(ticker)
        if m and m.get("count",0) >= 5:
            win_rate = (m.get("win_count",0)/max(1,m["count"])) * 100
            avg_move = m.get("avg_abs_move",0.0)
            message += f"\nHistory: {win_rate:.0f}% win, avg |Δ|={avg_move:.2f}% (n={m['count']})"

        send_discord(message, title="Early Event Scanner")

        # Register cooldown
        state["cooldowns"][ticker] = int(t_now.timestamp())

        # Create an alert ID
        aid = f"{ticker}-{int(t_now.timestamp())}"

        # Schedule follow-ups
        t60  = int((t_now + timedelta(minutes=60)).timestamp())
        t24h = int((t_now + timedelta(hours=24)).timestamp())
        state["pending"].extend([
            {"id": aid, "ticker": ticker, "alert_ts": int(t_now.timestamp()), "alert_px": alert_px,
             "due_ts": t60,  "label": "t+60m",  "done": False},
            {"id": aid, "ticker": ticker, "alert_ts": int(t_now.timestamp()), "alert_px": alert_px,
             "due_ts": t24h, "label": "t+24h",  "done": False},
        ])

        # Return for CSV logging
        new_alerts.append({"id": aid, "ticker": ticker, "ts": t_now.isoformat(), "price": alert_px, "hits": len(posts)})

    save_state(state)
    return new_alerts

def process_followups() -> List[Dict[str, Any]]:
    """Check pending follow-ups; when due, compute return and log outcome."""
    state = load_state()
    t_now = int(now_utc().timestamp())
    state.setdefault("pending", [])
    state.setdefault("metrics", {})

    results=[]
    new_pending=[]
    for item in state["pending"]:
        if item.get("done"):  # shouldn't happen, but skip
            continue
        if t_now >= int(item["due_ts"]):
            ticker   = item["ticker"]
            alert_px = float(item["alert_px"])
            cur_px, src = get_last_price(ticker)
            if not (cur_px and cur_px==cur_px):  # NaN guard
                # try again next run
                new_pending.append(item)
                continue
            ret_pct = ((cur_px/alert_px)-1.0)*100.0
            label   = item["label"]
            ts_iso  = datetime.fromtimestamp(item["due_ts"], tz=timezone.utc).isoformat()

            # Log result row
            results.append({
                "id": item["id"], "ticker": ticker, "label": label,
                "due_ts": ts_iso, "alert_px": alert_px, "cur_px": cur_px, "ret_pct": ret_pct
            })

            # Update simple "learning" metrics on 24h only
            if label == "t+24h":
                m = state["metrics"].get(ticker, {"count":0,"win_count":0,"avg_abs_move":0.0})
                m["count"] = int(m.get("count",0)) + 1
                # "Win" = absolute move >= 5% (you can tune)
                win = 1 if abs(ret_pct) >= 5.0 else 0
                m["win_count"] = int(m.get("win_count",0)) + win
                # Update running average of absolute move
                prev_n = m["count"] - 1
                prev_avg = float(m.get("avg_abs_move",0.0))
                m["avg_abs_move"] = ((prev_avg*prev_n) + abs(ret_pct)) / m["count"]
                state["metrics"][ticker] = m

                # Optional: post a small summary to Discord
                summary = (
                    f"**{ticker}** follow-up ({label}): {ret_pct:+.2f}% "
                    f"(alert ${alert_px:.2f} → now ${cur_px:.2f})\n"
                    f"Updated: win_rate={ (m['win_count']/m['count']*100):.0f}%  "
                    f"avg |Δ|={m['avg_abs_move']:.2f}%  (n={m['count']})"
                )
                send_discord(summary, title="Price Follow-up")

            elif label == "t+60m":
                # Optional: also post the 60m result
                msg = f"**{ticker}** follow-up ({label}): {ret_pct:+.2f}% (alert ${alert_px:.2f} → now ${cur_px:.2f})"
                send_discord(msg, title="Price Follow-up")

            # mark done (we won't carry it forward)
            item["done"] = True
            # do not append to new_pending
        else:
            # not due yet, keep it
            new_pending.append(item)

    state["pending"] = new_pending
    save_state(state)
    return results

# ------------------ Main ------------------

def main():
    # Ensure CSV headers exist
    ensure_csv_headers(ALERTS_CSV, ["id","ticker","alert_iso","alert_price","hits"])
    ensure_csv_headers(OUTCOMES_CSV, ["id","ticker","label","due_iso","alert_price","cur_price","ret_pct"])

    # 1) Run the scanner and send any new alerts
    new_alerts = scan_and_alert()
    for a in new_alerts:
        append_csv(ALERTS_CSV, [a["id"], a["ticker"], a["ts"], f"{a['price']:.4f}", a["hits"]])

    # 2) Process due follow-ups (60m/24h), send summaries, and log outcomes
    outcomes = process_followups()
    for o in outcomes:
        append_csv(OUTCOMES_CSV, [
            o["id"], o["ticker"], o["label"], o["due_ts"],
            f"{o['alert_px']:.4f}", f"{o['cur_px']:.4f}", f"{o['ret_pct']:.3f}"
        ])

    print("Scan complete. Alerts:", len(new_alerts), "Follow-ups processed:", len(outcomes))

if __name__ == "__main__":
    main()
