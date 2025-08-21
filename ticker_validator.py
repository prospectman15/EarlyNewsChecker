# ticker_validator.py
# Purpose: Extract tickers from free text and validate them against official US exchange listings.
# Deps: requests, pandas (already in your requirements)

import os
import io
import re
import time
import gzip
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

# ---- Config ----
CACHE_DIR = os.getenv("TICKER_CACHE_DIR", "data")
CACHE_PATH = os.path.join(CACHE_DIR, "valid_tickers.csv")
META_PATH = os.path.join(CACHE_DIR, "valid_tickers.meta.json")
TTL_DAYS = int(os.getenv("TICKER_CACHE_TTL_DAYS", "1"))

# NASDAQ Trader listing files (pipe-delimited with a footer row)
NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://ftp.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# Allow BRK.B / BF.B / RDS-A styles. 1–5 letters, optional . or - then 1–2 letters
TICKER_SHAPE_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")

# Fast finder: $TSLA, $nvda, or bare TSLA-like tokens
DOLLAR_TICKER_RE = re.compile(r"\$([A-Za-z][A-Za-z.\-]{0,6})\b")
BARE_TICKER_RE   = re.compile(r"\b([A-Z]{1,5}(?:[.-][A-Z]{1,2})?)\b")

# Common all-caps words we’ll skip early (extra safety before whitelist)
COMMON_CAPS_SKIP = {
    "A", "AN", "AND", "ARE", "AT", "BE", "BUY", "CALL", "DD", "CEO", "CFO",
    "EPS", "FUD", "FYI", "GDP", "IMO", "IV", "IVR", "JAN", "JUL", "JUN", "MAR",
    "MAY", "MIT", "MOON", "NAV", "OR", "OTM", "PCE", "PE", "PFD", "PLS", "PM",
    "PR", "PUT", "Q", "QQQ", "ROFL", "RSI", "SEC", "SPAC", "SPY", "TA", "TBA",
    "TBD", "TBF", "TIPS", "TLT", "UK", "US", "USD", "VIX", "W", "WSB", "WTF",
    "YOLO",
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
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    # NASDAQ trader returns text; ensure string
    return resp.text

def _parse_nasdaq_table(text: str, symbol_col: str = "Symbol") -> pd.DataFrame:
    # Drop footer row like "File Creation Time: ..."
    lines = [ln for ln in text.splitlines() if not ln.startswith("File Creation Time")]
    buf = io.StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep="|")
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    # Remove test issues if present
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
    # Remove NextShares if present (not common equities)
    if "NextShares" in df.columns:
        df = df[df["NextShares"].astype(str).str.upper().ne("Y")]
    # Only keep the symbol column
    df = df[[symbol_col]].dropna()
    df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
    return df

def refresh_valid_tickers(cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Download & build the combined US equities symbol table (NASDAQ + NYSE/NYSE American).
    Returns a DataFrame with a single 'Symbol' column.
    """
    logging.info("Refreshing valid ticker list from NASDAQ Trader…")
    nas = _parse_nasdaq_table(_download_txt(NASDAQ_LISTED_URL), symbol_col="Symbol")
    oth = _parse_nasdaq_table(_download_txt(OTHER_LISTED_URL),  symbol_col="ACT Symbol" if "ACT Symbol" in _download_txt(OTHER_LISTED_URL) else "Symbol")
    # Re-parse OTHER to get proper column (avoid double downloading/text reuse)
    other_text = _download_txt(OTHER_LISTED_URL)
    try:
        oth = _parse_nasdaq_table(other_text, symbol_col="ACT Symbol")
        oth.rename(columns={"ACT Symbol": "Symbol"}, inplace=True)
    except Exception:
        oth = _parse_nasdaq_table(other_text, symbol_col="Symbol")

    all_syms = pd.concat([nas.rename(columns={"Symbol": "Symbol"}), oth.rename(columns={"Symbol": "Symbol"})], ignore_index=True)
    all_syms.drop_duplicates(subset=["Symbol"], inplace=True)
    all_syms = all_syms[all_syms["Symbol"].str.len() > 0]

    _ensure_cache_dir()
    all_syms.to_csv(cache_path, index=False)
    _write_meta()
    logging.info(f"Saved {len(all_syms)} symbols to {cache_path}")
    return all_syms

def load_or_refresh_valid_tickers(cache_path: str = CACHE_PATH, ttl_days: int = TTL_DAYS) -> pd.Series:
    """
    Returns a pandas Series of unique uppercase symbols; refreshes if cache missing/stale.
    """
    _ensure_cache_dir()
    needs_refresh = (not os.path.isfile(cache_path)) or _stale_meta(ttl_days)
    if needs_refresh:
        try:
            df = refresh_valid_tickers(cache_path)
        except Exception as e:
            logging.warning(f"Failed to refresh ticker list: {e}")
            if os.path.isfile(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = pd.DataFrame({"Symbol": []})
    else:
        df = pd.read_csv(cache_path)

    # Normalize & unique
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df = df[df["Symbol"].str.len() > 0].drop_duplicates("Symbol")
    return df["Symbol"]

def build_valid_set() -> set:
    """Return a Python set for O(1) membership checks."""
    syms = load_or_refresh_valid_tickers()
    return set(syms.tolist())

def is_ticker_shaped(token: str) -> bool:
    return bool(TICKER_SHAPE_RE.match(token))

def normalize_token(token: str) -> str:
    t = token.strip().upper()
    # Remove surrounding punctuation commonly attached in text
    t = re.sub(r"^[^\w$]+|[^\w.%-]+$", "", t)
    # Strip leading $
    if t.startswith("$"):
        t = t[1:]
    return t

def extract_candidates(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    candidates = []

    # $TSLA style
    for m in DOLLAR_TICKER_RE.finditer(text):
        candidates.append(m.group(1))

    # bare TSLA style (all-caps words with optional . or -)
    for m in BARE_TICKER_RE.finditer(text):
        candidates.append(m.group(1))

    # Normalize, shape filter, skip obvious caps words
    out = []
    for raw in candidates:
        tok = normalize_token(raw)
        if not tok or tok in COMMON_CAPS_SKIP:
            continue
        if is_ticker_shaped(tok):
            out.append(tok)
    # Keep order but dedupe
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

class TickerValidator:
    """
    Usage:
        tv = TickerValidator()
        valid = tv.extract_valid_tickers(text)
        if valid:  # only alert if non-empty
            ...
    """
    def __init__(self):
        self._valid_set = None

    @property
    def valid_set(self) -> set:
        if self._valid_set is None:
            self._valid_set = build_valid_set()
        return self._valid_set

    def is_valid(self, token: str) -> bool:
        tok = normalize_token(token)
        if not is_ticker_shaped(tok):
            return False
        return tok in self.valid_set

    def extract_valid_tickers(self, text: str, limit: int | None = None) -> list[str]:
        cands = extract_candidates(text)
        valid = [t for t in cands if t in self.valid_set]
        if limit is not None and limit > 0:
            valid = valid[:limit]
        return valid
