# hype_utils.py
# Lightweight, market-wide hype detection helpers.

import re
from collections import defaultdict
from typing import List, Dict, Set

# Avoid false tickers from short ALLCAPS words
STOPWORDS: Set[str] = {
    "A","AN","AND","ARE","AS","AT","BE","BUT","BY","FOR","IF","IN","INTO","IS","IT","NO","NOT",
    "OF","ON","OR","SUCH","THAT","THE","THEIR","THEN","THERE","THESE","THEY","THIS","TO","WAS","WILL","WITH",
    "OTC","CEO","CFO","COO","FDA","FBI","SEC","FTC","AI","IPO","ETF","GDP","EPS","CPI","PPI","YTD"
}

# Extra sentiment cues (your model features already cover positive/crisis/earnings;
# this adds broader market/macro narratives)
POS_EXTRA = {
    "upgrade","raises guidance","beat","beats","record","surge","acquisition","buyback","approved","approval","launch"
}
NEG_EXTRA = {
    "downgrade","cuts guidance","miss","misses","halt","halts","bankruptcy","fires","fraud","bubble","unsustainable",
    "lawsuit","recall","probe","ban","banned","blocked","not successful","fail","failing","collapse"
}

# Track sector/market ETFs, too (they can become the signal)
MARKET_ETFS = {"SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLY","XLI","XLV","XLB","XLU","SMH","SOXL","TQQQ","SQQQ"}

CASHTAG   = re.compile(r'(?<!\w)\$[A-Za-z]{1,5}\b')
UPPERWORD = re.compile(r'\b[A-Z]{1,5}\b')

def extract_entities(text: str) -> Set[str]:
    """Return a set of potential tickers/entities detected in text."""
    if not text:
        return set()
    ents = set()
    # Cashtags are strong signals
    ents |= {t[1:].upper() for t in CASHTAG.findall(text)}
    # Bare uppercase tokens (filter obvious stopwords)
    for m in UPPERWORD.findall(text):
        if m in STOPWORDS or len(m) <= 1:
            continue
        ents.add(m)
    # Keep ETFs
    ents |= (MARKET_ETFS & ents) | {e for e in ents if e in MARKET_ETFS}
    return ents

def lexicon_polarity(text: str) -> float:
    """Simple polarity: positive minus negative hits normalized to [-1, 1]."""
    t = (text or "").lower()
    pos = sum(1 for w in POS_EXTRA if w in t)
    neg = sum(1 for w in NEG_EXTRA if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, pos + neg)
    if score > 1: score = 1
    if score < -1: score = -1
    return score

def cluster_by_entity(items: List[Dict]) -> Dict[str, List[Dict]]:
    """Group feed items by detected entity (ticker/ETF-like symbol)."""
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for it in items:
        text = f"{it.get('title','')} {it.get('selftext','')}"
        ents = extract_entities(text)
        for e in ents:
            buckets[e].append(it)
    return buckets

def build_cluster_features(cluster: List[Dict]) -> Dict[str, float]:
    """Aggregate simple features from a cluster of items."""
    scores   = [it.get("score", 0) or 0 for it in cluster]
    comments = [it.get("num_comments", 0) or 0 for it in cluster]
    titles   = [it.get("title","") for it in cluster]
    urls     = [it.get("url","") or it.get("permalink","") for it in cluster]
    subs     = {it.get("subreddit","") or it.get("source","") for it in cluster}

    # host count is a weak proxy for cross-domain amplification (kept simple)
    hosts = []
    from urllib.parse import urlparse
    for u in urls:
        try:
            hosts.append(urlparse(u).netloc)
        except Exception:
            pass
    host_set = set(h for h in hosts if h)

    text = " || ".join(titles)
    pol  = lexicon_polarity(text)
    hits = len(cluster)
    return {
        "hits": hits,
        "score_max": max(scores+[0]),
        "score_sum": sum(scores),
        "comments_max": max(comments+[0]),
        "comments_sum": sum(comments),
        "subs": len({s for s in subs if s}),
        "hosts": len(host_set),
        "polarity": pol,
    }

def hype_score(feats: Dict[str, float], z: float) -> float:
    """Combine burst z-score + amplification + polarity magnitude."""
    amp    = 1.0 + 0.15 * feats.get("subs", 0) + 0.10 * feats.get("hosts", 0)
    polmag = 1.0 + 0.25 * abs(feats.get("polarity", 0.0))
    return max(0.0, z) * amp * polmag
