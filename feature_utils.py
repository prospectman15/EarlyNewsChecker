# feature_utils.py
import re
import math
from urllib.parse import urlparse

MEDIA_ALLOW = {"sec.gov","businesswire.com","prnewswire.com"}
POSITIVE_WORDS = [
    "upgrade","upgraded","raises guidance","raised guidance",
    "raise price target","price target","initiated coverage",
    "beats","beat estimates","contract win","buyback","approval","fda",
]
CRISIS_WORDS = [
    "crash","emergency landing","mayday","grounded","evacuate",
    "explosion","fire","smoke","engine failure","bird strike",
    "faa","ntsb","probe","recall","lawsuit","hack","breach",
    "ransomware","outage","strike","walkout","ceo resign",
]
EARNINGS_WORDS = [
    r"\bearnings\b", r"\beps\b", "results", "miss", "beat", "guide", "guidance",
    "revenue", "margin", "outlook", "forecast", "loss", "profit",
]

def _has_any(text: str, terms) -> int:
    t = (text or "").lower()
    for term in terms:
        pattern = term if term.startswith(r"\b") else re.escape(term)
        if re.search(pattern, t):
            return 1
    return 0

def _is_primary_source(url: str) -> int:
    try:
        host = urlparse((url or "").lower()).netloc
    except Exception:
        return 0
    if not host: return 0
    if host.startswith("investor.") or host.startswith("ir."): return 1
    return int(any(host == d or host.endswith("." + d) for d in MEDIA_ALLOW))

MODEL_FEATURES = [
    # numeric
    "hits_in_window",
    "score_max","score_sum","comments_max","comments_sum",
    # booleans (as 0/1)
    "has_positive","has_crisis","has_earnings",
    "has_primary_source_link",
]

def build_features(titles, urls, scores, comments, hits_in_window: int):
    titles_text = " || ".join(titles or [])
    has_pos  = _has_any(titles_text, POSITIVE_WORDS)
    has_cri  = _has_any(titles_text, CRISIS_WORDS)
    has_earn = _has_any(titles_text, EARNINGS_WORDS)
    prim = 1 if any(_is_primary_source(u) for u in (urls or [])) else 0

    score_max = max([s for s in (scores or [])] + [0])
    comments_max = max([c for c in (comments or [])] + [0])
    score_sum = sum(scores or [])
    comments_sum = sum(comments or [])

    feats = {
        "hits_in_window": int(hits_in_window),
        "score_max": float(score_max),
        "score_sum": float(score_sum),
        "comments_max": float(comments_max),
        "comments_sum": float(comments_sum),
        "has_positive": int(has_pos),
        "has_crisis": int(has_cri),
        "has_earnings": int(has_earn),
        "has_primary_source_link": int(prim),
    }
    # return in fixed column order:
    return [feats[k] for k in MODEL_FEATURES], feats
