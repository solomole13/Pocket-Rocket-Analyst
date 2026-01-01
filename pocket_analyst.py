#!/usr/bin/env python3
"""
POCKET ANALYST v2 (clean, single-file)

Goals:
- Keep the same "Pocket Analyst" workflow + menu style.
- No pip required. Uses ONLY stdlib for HTTP (urllib).
- CSV caching to ./data
- Outputs to ./reports and optional plots to ./plots
- Fixes: no top-level runtime variables (no csv_path/rows NameErrors)

Run:
  python3 pocket_analyst.py
or
  python3 pocket_analyst.py watchlists/watchlist.txt 10

Folders created if missing:
  data/, reports/, plots/, watchlists/, tmp/
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Paths / defaults
# -----------------------------
APP_NAME = "POCKET ANALYST"
DEFAULT_WATCHLIST = os.path.join("watchlists", "watchlist.txt")
DATA_DIR = "data"
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"
TMP_DIR = "tmp"

MIN_ROWS = 260  # ~1y trading days


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs() -> None:
    for d in [DATA_DIR, REPORTS_DIR, PLOTS_DIR, TMP_DIR, "watchlists"]:
        os.makedirs(d, exist_ok=True)


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat(sep=" ")


def slug(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[^A-Z0-9\.\-\_]+", "_", s)
    return s[:64] if s else "SYMBOL"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(path: str, txt: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def human(n: Optional[float], digits: int = 2) -> str:
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(n)


def pct(n: Optional[float], digits: int = 1) -> str:
    if n is None:
        return "NA"
    return f"{n*100:.{digits}f}%"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Row:
    date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]


def row_to_dict(r: Row) -> Dict[str, Optional[float] | str]:
    return {
        "date": r.date,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
    }


# -----------------------------
# Watchlist
# -----------------------------
def load_watchlist(path: str) -> List[str]:
    if not os.path.exists(path):
        # create a starter file
        starter = "NFLX\nABNB\nKHC\n"
        ensure_dirs()
        write_text(path, starter)
        return ["NFLX", "ABNB", "KHC"]

    out: List[str] = []
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow comma/space separated lines
        parts = re.split(r"[,\s]+", s)
        for p in parts:
            p = p.strip().upper()
            if p and not p.startswith("#"):
                out.append(p)
    # unique preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


# -----------------------------
# Fetching (no requests)
# -----------------------------
def http_get(url: str, timeout: int = 20) -> Tuple[int, bytes]:
    # stdlib only
    from urllib.request import Request, urlopen

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (PocketAnalyst/2.0; +https://github.com/)"
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        data = resp.read()
        return int(status), data


def stooq_symbol(sym: str) -> str:
    """
    Stooq uses:
      - US: aapl.us, nflx.us
      - Some EU/others: varies
    We'll default to ".us" if no suffix provided.
    """
    s = sym.strip().lower()
    if "." in s:
        return s
    return f"{s}.us"


def stooq_url(sym: str) -> str:
    s = stooq_symbol(sym)
    return f"https://stooq.com/q/d/l/?s={s}&i=d"


def cache_csv_path(sym: str) -> str:
    return os.path.join(DATA_DIR, f"{slug(sym)}.csv")


def fetch_or_load_csv(sym: str, max_age_hours: int = 12) -> Optional[str]:
    """
    Returns path to cached CSV (freshened if needed), or None if fetch failed.
    """
    ensure_dirs()
    path = cache_csv_path(sym)

    if os.path.exists(path):
        age_sec = time.time() - os.path.getmtime(path)
        if age_sec < max_age_hours * 3600:
            return path

    url = stooq_url(sym)
    try:
        status, data = http_get(url)
        if status < 200 or status >= 300:
            return path if os.path.exists(path) else None
        text = data.decode("utf-8", errors="replace").strip()
        # stooq returns "Date,Open,High,Low,Close,Volume" lines
        if "Date,Open,High,Low,Close,Volume" not in text:
            return path if os.path.exists(path) else None
        write_text(path, text + "\n")
        return path
    except Exception:
        return path if os.path.exists(path) else None


# -----------------------------
# CSV parsing
# -----------------------------
def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x or x.lower() in ("nan", "null", "none"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_ohlcv(csv_path: str) -> List[Dict[str, Optional[float] | str]]:
    """
    Parses Stooq daily CSV into list[dict] with keys:
      date, open, high, low, close, volume
    """
    rows: List[Dict[str, Optional[float] | str]] = []
    if not csv_path or not os.path.exists(csv_path):
        return rows

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = (r.get("Date") or "").strip()
            if not d:
                continue
            rows.append(
                {
                    "date": d,
                    "open": _to_float(r.get("Open", "")),
                    "high": _to_float(r.get("High", "")),
                    "low": _to_float(r.get("Low", "")),
                    "close": _to_float(r.get("Close", "")),
                    "volume": _to_float(r.get("Volume", "")),
                }
            )

    # Stooq is oldest->newest already; keep as is
    return rows


# -----------------------------
# Metrics / scoring helpers
# -----------------------------
def years_covered(dates: List[str]) -> float:
    if not dates:
        return 0.0
    try:
        d0 = dt.date.fromisoformat(dates[0])
        d1 = dt.date.fromisoformat(dates[-1])
        return max(0.0, (d1 - d0).days / 365.25)
    except Exception:
        return 0.0


def cagr(closes: List[Optional[float]], yrs: float) -> Optional[float]:
    if yrs <= 0.0:
        return None
    first = next((x for x in closes if x and x > 0), None)
    last = next((x for x in reversed(closes) if x and x > 0), None)
    if not first or not last or first <= 0 or last <= 0:
        return None
    try:
        return (last / first) ** (1.0 / yrs) - 1.0
    except Exception:
        return None


def ann_vol_from_daily_rets(daily: List[float]) -> Optional[float]:
    if len(daily) < 30:
        return None
    mu = sum(daily) / len(daily)
    var = sum((x - mu) ** 2 for x in daily) / max(1, len(daily) - 1)
    sd = math.sqrt(var)
    return sd * math.sqrt(252.0)


def max_drawdown(closes: List[Optional[float]]) -> Optional[float]:
    xs = [c for c in closes if c is not None]
    if len(xs) < 30:
        return None
    peak = xs[0]
    mdd = 0.0
    for x in xs:
        if x > peak:
            peak = x
        if peak > 0:
            dd = (x / peak) - 1.0
            mdd = min(mdd, dd)
    return mdd  # negative number


def volume_trend_1y(vols: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vols if v is not None]
    if len(xs) < 260:
        return None
    tail = xs[-252:]
    n = len(tail)
    if n < 50:
        return None
    # simple slope on normalized index (0..1)
    x_mean = 0.5
    y_mean = sum(tail) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(tail):
        x = i / max(1, n - 1)
        num += (x - x_mean) * (y - y_mean)
        den += (x - x_mean) ** 2
    if den == 0:
        return None
    slope = num / den
    # scale slope relative to mean volume
    if y_mean <= 0:
        return None
    return slope / y_mean


def vol_ratio(daily: List[float], short: int, long: int) -> Optional[float]:
    if len(daily) < long + 5:
        return None
    s = daily[-short:]
    l = daily[-long:]
    vs = ann_vol_from_daily_rets(s)
    vl = ann_vol_from_daily_rets(l)
    if vs is None or vl is None or vl == 0:
        return None
    return vs / vl


def shock_effectiveness(
    closes: List[Optional[float]],
    daily: List[float],
    horizon: int = 60,
    shock_sigma: float = 2.0,
) -> Dict[str, Optional[float]]:
    """
    Simple event-study style:
    - Identify 'shocks' where |ret| > shock_sigma * std(ret)
    - Measure average post-shock drift over horizon and persistence (share of positive post windows)
    Returns dict with keys: n, avg_post, persistence
    """
    out = {"n": 0, "avg_post": None, "persistence": None}
    if len(daily) < 260:
        return out
    mu = sum(daily) / len(daily)
    var = sum((x - mu) ** 2 for x in daily) / max(1, len(daily) - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return out

    shock_idx: List[int] = []
    for i, r in enumerate(daily):
        if abs(r) >= shock_sigma * sd:
            shock_idx.append(i)

    if not shock_idx:
        return out

    post_drifts: List[float] = []
    pers: List[int] = []
    for i in shock_idx:
        start = i + 1
        end = min(len(daily), i + 1 + horizon)
        if end - start < max(10, horizon // 4):
            continue
        window = daily[start:end]
        drift = sum(window)  # cumulative simple
        post_drifts.append(drift)
        pers.append(1 if drift > 0 else 0)

    if not post_drifts:
        return out

    out["n"] = len(post_drifts)
    out["avg_post"] = sum(post_drifts) / len(post_drifts)
    out["persistence"] = sum(pers) / len(pers) if pers else None
    return out


def capital_identity(cagr_val: Optional[float], mdd: Optional[float], vol_ann: Optional[float]) -> Tuple[str, float]:
    """
    Returns:
      (identity_label, strength_score)
    strength_score: 0..100
    """
    # fallbacks
    c = cagr_val if cagr_val is not None else 0.0
    v = vol_ann if vol_ann is not None else 0.0
    d = mdd if mdd is not None else 0.0  # negative

    # strength: reward CAGR, penalize vol and drawdown
    # keep it simple, stable
    score = 50.0
    score += 120.0 * max(-0.5, min(0.5, c))  # +/- 60 max
    score -= 35.0 * max(0.0, min(1.5, v))    # up to -52.5
    score += 80.0 * max(-0.9, min(0.0, d))   # d is negative, so add negative -> reduce
    score = max(0.0, min(100.0, score))

    if c > 0.15 and (d is not None and d > -0.35):
        ident = "COMPOUNDER"
    elif v > 0.8:
        ident = "VOLATILE"
    elif c < 0.0 and (d is not None and d < -0.5):
        ident = "DRAWDOWNED"
    else:
        ident = "BALANCED"

    return ident, score


def strength_label(strength: float) -> str:
    if strength >= 75:
        return "STRONG"
    if strength >= 50:
        return "OK"
    return "WEAK"


# -----------------------------
# Optional plotting (guarded)
# -----------------------------
def maybe_plot_price(sym: str, rows: List[Dict[str, Optional[float] | str]]) -> Optional[str]:
    """
    Attempts to create a simple price plot. If matplotlib missing, skip silently.
    """
    try:
        import matplotlib  # noqa
        import matplotlib.pyplot as plt
    except Exception:
        return None

    dates = [r["date"] for r in rows][-252:]
    closes = [r["close"] for r in rows][-252:]
    xs = list(range(len(dates)))
    ys = [c if c is not None else float("nan") for c in closes]

    plt.figure()
    plt.plot(xs, ys)
    plt.title(sym)
    plt.xlabel("Days")
    plt.ylabel("Close")

    out = os.path.join(PLOTS_DIR, f"{slug(sym)}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


# -----------------------------
# Reporting
# -----------------------------
def write_symbol_report(
    sym: str,
    as_of: str,
    score: float,
    label: str,
    verdict: str,
    reasons: List[str],
    metrics: Dict[str, Optional[float] | str],
) -> str:
    sym_dir = os.path.join(REPORTS_DIR, "symbols")
    os.makedirs(sym_dir, exist_ok=True)
    out = os.path.join(sym_dir, f"{slug(sym)}.txt")

    lines = []
    lines.append(f"{APP_NAME} v2")
    lines.append("=" * 60)
    lines.append(f"Symbol:   {sym}")
    lines.append(f"As of:    {as_of}")
    lines.append(f"Score:    {int(round(score))}")
    lines.append(f"Label:    {label}")
    lines.append(f"Verdict:  {verdict}")
    lines.append("")
    lines.append("Reasons:")
    for r in reasons[:20]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Key metrics:")
    for k in [
        "cagr",
        "vol_ann",
        "mdd",
        "vr",
        "tight",
        "avg_post",
        "persistence",
        "vtrend",
    ]:
        v = metrics.get(k)
        if isinstance(v, float):
            if k in ("cagr", "avg_post"):
                vv = pct(v, 1)
            elif k in ("mdd",):
                vv = pct(v, 1)
            else:
                vv = human(v, 3)
        else:
            vv = str(v)
        lines.append(f"- {k:12s}: {vv}")

    write_text(out, "\n".join(lines) + "\n")
    return out


def append_csv(path: str, header: List[str], row: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


# -----------------------------
# Scoring engine (v2)
# -----------------------------
def score_v2(rows: List[Dict[str, Optional[float] | str]]) -> Tuple[float, str, List[str], Dict[str, Optional[float] | str], str]:
    """
    Returns:
      score, label, reasons, metrics, verdict
    """
    reasons: List[str] = []

    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]
    vols = [r["volume"] for r in rows]

    # daily returns
    daily = [
        (closes[i] / closes[i - 1] - 1.0)
        for i in range(1, len(closes))
        if closes[i] and closes[i - 1]
    ]

    yrs = years_covered(dates)
    cagr_val = cagr(closes, yrs)
    vol_ann = ann_vol_from_daily_rets(daily)
    mdd = max_drawdown(closes)
    vtrend = volume_trend_1y(vols)

    shock = shock_effectiveness(closes, daily, horizon=60)
    ident, strength = capital_identity(cagr_val, mdd, vol_ann)
    slabel = strength_label(strength)

    vr = (
        (sum(v for v in vols[-20:] if v) / max(1, len([v for v in vols[-20:] if v])))
        /
        (sum(v for v in vols[-252:] if v) / max(1, len([v for v in vols[-252:] if v])))
        if len(vols) > 260 else None
    )

    tight = vol_ratio(daily, 20, 60)
    breakout20 = 1 if len(closes) > 21 and closes[-1] and max([x for x in closes[-21:-1] if x]) and closes[-1] > max([x for x in closes[-21:-1] if x]) else 0

    # strong close (from your screenshot)
    strong_close = 0
    if rows and rows[-1].get("high") and rows[-1].get("low") and rows[-1].get("close"):
        hi = rows[-1]["high"]
        lo = rows[-1]["low"]
        cl = rows[-1]["close"]
        if hi and lo and cl and (hi - lo) > 0:
            strong_close = 1 if ((cl - lo) / (hi - lo)) >= 0.8 else 0

    # deployment quality label (from your screenshot)
    deploy_quality = "NEUTRAL"
    if shock.get("n", 0) >= 8 and shock.get("avg_post") is not None:
        if (shock["avg_post"] is not None and shock["avg_post"] > 0.05) and (shock.get("persistence") is not None and shock["persistence"] >= 0.55):
            deploy_quality = "POSITIVE"
        elif (shock["avg_post"] is not None and shock["avg_post"] < 0.0) and (shock.get("persistence") is not None and shock["persistence"] < 0.45):
            deploy_quality = "NEGATIVE"

    # regime (from your screenshot)
    regime = "MIXED"
    if tight is not None and tight < 0.8 and breakout20:
        regime = "BUILDING"
    elif tight is not None and tight > 1.2:
        regime = "CHOPPY"

    # Build reasons
    if cagr_val is not None:
        reasons.append(f"CAGR {pct(cagr_val)} over ~{human(yrs, 1)}y")
    if vol_ann is not None:
        reasons.append(f"Annualized vol {human(vol_ann, 2)}")
    if mdd is not None:
        reasons.append(f"Max drawdown {pct(mdd, 1)}")
    reasons.append(f"Identity {ident} | Strength {slabel} ({int(round(strength))})")

    if vr is not None:
        reasons.append(f"Volume ratio (20d/1y) {human(vr, 2)}")
    if vtrend is not None:
        reasons.append(f"Volume trend (1y) {human(vtrend, 4)}")

    if shock.get("n", 0) > 0:
        reasons.append(f"Shock events: {shock['n']} | avg_post {pct(shock.get('avg_post'), 1)} | persistence {human(shock.get('persistence'), 2)}")

    if breakout20:
        reasons.append("20d breakout = YES")
    if strong_close:
        reasons.append("Strong close (upper range) = YES")

    reasons.append(f"Deploy quality = {deploy_quality}")
    reasons.append(f"Regime = {regime}")

    # Score composition (simple + stable)
    score = strength
    if deploy_quality == "POSITIVE":
        score += 8
    elif deploy_quality == "NEGATIVE":
        score -= 8

    if regime == "BUILDING":
        score += 4
    elif regime == "CHOPPY":
        score -= 3

    if breakout20:
        score += 3
    if strong_close:
        score += 2

    if vr is not None and vr > 1.3:
        score += 2
    if mdd is not None and mdd < -0.6:
        score -= 6

    score = max(0.0, min(100.0, score))

    # final verdict (from your screenshot)
    verdict = "AMBIGUOUS"
    if slabel == "STRONG" and deploy_quality in ("POSITIVE",):
        verdict = "LIKELY BUY CONTENDER"
    elif deploy_quality == "NEGATIVE" and slabel == "WEAK":
        verdict = "LIKELY AVOID"

    # label for summary line
    label = "BUY" if verdict == "LIKELY BUY CONTENDER" else ("AVOID" if verdict == "LIKELY AVOID" else "NEUTRAL")

    metrics: Dict[str, Optional[float] | str] = {
        "yrs": yrs,
        "cagr": cagr_val,
        "vol_ann": vol_ann,
        "mdd": mdd,
        "vr": vr,
        "tight": tight,
        "avg_post": shock.get("avg_post"),
        "persistence": shock.get("persistence"),
        "vtrend": vtrend,
        "deploy_quality": deploy_quality,
        "regime": regime,
    }

    return score, label, reasons, metrics, verdict


# -----------------------------
# Run engine
# -----------------------------
def run_engine(watchlist_path: str, limit: int) -> None:
    ensure_dirs()
    tickers = load_watchlist(watchlist_path)
    as_of = dt.date.today().isoformat()

    print(f"Symbols: {min(limit, len(tickers))}")
    print(f"Output:  {os.path.abspath('.')}")
    print("")

    # summary outputs
    signals_csv = os.path.join(REPORTS_DIR, "signals.csv")
    overlay_txt = os.path.join(REPORTS_DIR, "overlay.txt")
    overlay_csv = os.path.join(REPORTS_DIR, "overlay.csv")

    # clear overlay.txt each run
    write_text(overlay_txt, f"{APP_NAME} v2 overlay run @ {now_iso()}\n\n")

    n_ok = 0
    for sym in tickers[:limit]:
        csv_path = fetch_or_load_csv(sym)
        if not csv_path:
            print(f"! {sym}: Fetch failed: No data returned")
            continue

        rows = load_ohlcv(csv_path)
        if len(rows) < MIN_ROWS:
            print(f"! {sym}: Fetch failed: too few rows ({len(rows)})")
            continue

        score, label, reasons, metrics, verdict = score_v2(rows)

        # Write per-symbol report
        report_path = write_symbol_report(
            sym=sym,
            as_of=as_of,
            score=score,
            label=label,
            verdict=verdict,
            reasons=reasons,
            metrics=metrics,
        )

        # Optional plot
        maybe_plot_price(sym, rows)

        # Print summary line
        print(f"+ {sym}: OK score={int(round(score))} label={('BUY' if label=='BUY' else ('AVOID' if label=='AVOID' else 'NEUTRAL'))} as_of={as_of}")

        # Append signals.csv
        append_csv(
            signals_csv,
            header=["as_of", "symbol", "score", "label", "verdict"],
            row=[as_of, sym, str(int(round(score))), label, verdict],
        )

        # Append overlay artifacts
        with open(overlay_txt, "a", encoding="utf-8") as f:
            f.write(f"{sym} | score={int(round(score))} | label={label} | verdict={verdict}\n")
            f.write(f"  deploy_quality={metrics.get('deploy_quality')} | regime={metrics.get('regime')}\n")
            f.write(f"  cagr={pct(metrics.get('cagr'), 1)} | vol={human(metrics.get('vol_ann'), 2)} | mdd={pct(metrics.get('mdd'), 1)}\n")
            f.write("\n")

        append_csv(
            overlay_csv,
            header=[
                "as_of", "symbol", "score", "label", "verdict",
                "deploy_quality", "regime", "cagr", "vol_ann", "mdd",
                "vr", "tight", "avg_post", "persistence", "vtrend"
            ],
            row=[
                as_of, sym, str(int(round(score))), label, verdict,
                str(metrics.get("deploy_quality")),
                str(metrics.get("regime")),
                str(metrics.get("cagr")),
                str(metrics.get("vol_ann")),
                str(metrics.get("mdd")),
                str(metrics.get("vr")),
                str(metrics.get("tight")),
                str(metrics.get("avg_post")),
                str(metrics.get("persistence")),
                str(metrics.get("vtrend")),
            ],
        )

        n_ok += 1

    print("\nSaved:")
    print(f"- {os.path.abspath(signals_csv)}")
    print(f"- {os.path.abspath(overlay_txt)}")
    print(f"- {os.path.abspath(overlay_csv)}")
    print(f"- {os.path.abspath(os.path.join(REPORTS_DIR, 'symbols'))}/*.txt")
    print(f"- {os.path.abspath(PLOTS_DIR)}/*.png")
    print("")

    if n_ok == 0:
        print("No successful symbols. Check tickers or data source.")
    else:
        print(f"Done. OK symbols: {n_ok}")


# -----------------------------
# Simple menu UI
# -----------------------------
def edit_watchlist(path: str) -> None:
    # No nano assumption; use a basic inline editor
    tickers = load_watchlist(path)
    print("\nCurrent watchlist:")
    for t in tickers:
        print(" -", t)
    print("\nPaste tickers separated by space/comma/newlines. Blank line to finish.\n")

    buf: List[str] = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        buf.append(line)

    if not buf:
        print("No changes.\n")
        return

    merged = "\n".join(buf)
    parts = []
    for line in merged.splitlines():
        for p in re.split(r"[,\s]+", line.strip()):
            p = p.strip().upper()
            if p:
                parts.append(p)

    # unique preserve order
    seen = set()
    uniq = []
    for t in parts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    write_text(path, "\n".join(uniq) + "\n")
    print(f"Saved watchlist: {path}\n")


def show_outputs() -> None:
    print("\nOutputs:")
    for p in [
        os.path.join(REPORTS_DIR, "signals.csv"),
        os.path.join(REPORTS_DIR, "overlay.txt"),
        os.path.join(REPORTS_DIR, "overlay.csv"),
    ]:
        if os.path.exists(p):
            print(" -", os.path.abspath(p))
    sym_dir = os.path.join(REPORTS_DIR, "symbols")
    if os.path.isdir(sym_dir):
        print(" -", os.path.abspath(sym_dir), "/*.txt")
    if os.path.isdir(PLOTS_DIR):
        print(" -", os.path.abspath(PLOTS_DIR), "/*.png")
    print("")


def interactive_menu() -> None:
    ensure_dirs()
    watchlist = DEFAULT_WATCHLIST
    tickers = load_watchlist(watchlist)

    while True:
        tickers = load_watchlist(watchlist)
        print(f"\n{APP_NAME}")
        print("=" * 40)
        print(f"Watchlist: {os.path.abspath(watchlist)} (symbols: {len(tickers)})")
        print(f"Output:    {os.path.abspath('.')}\n")
        print("1) Run analysis")
        print("2) Edit watchlist")
        print("3) Show outputs")
        print("4) Exit")

        choice = input("\nSelect [1-4]: ").strip()
        if choice == "1":
            lim = input("How many symbols to run? (e.g. 10): ").strip()
            try:
                limit = int(lim) if lim else min(10, len(tickers))
            except Exception:
                limit = min(10, len(tickers))
            run_engine(watchlist, limit)
        elif choice == "2":
            edit_watchlist(watchlist)
        elif choice == "3":
            show_outputs()
        elif choice == "4":
            print("Bye.")
            return
        else:
            print("Invalid selection.")


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    ensure_dirs()

    # CLI mode
    if len(sys.argv) >= 3:
        watchlist_path = sys.argv[1]
        try:
            limit = int(sys.argv[2])
        except Exception:
            limit = 10
        run_engine(watchlist_path, limit)
        return

    # Interactive mode
    interactive_menu()


if __name__ == "__main__":
    main()
