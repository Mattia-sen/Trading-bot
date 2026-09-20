#!/usr/bin/env python3
"""
bot.py - MARK 1b. Runs via GitHub Actions, catches up any missed hours, exits.

SIMULATION ONLY. No exchange account, no exchange API key, no order function.
Public price data only. Nothing here can spend money. Euros throughout.

WHAT CHANGED IN 1b (from 87 bars of 1a data)
  1. CATCH-UP. GitHub was dropping ~75% of scheduled runs, so the bot was
     getting 5.8 bars/day instead of 24 and phase A would have taken four
     months. It already fetched 67 candles per run but processed only the
     newest. Now it processes every candle since last_ts, up to MAX_CATCHUP.

  2. NEWS WITH NO LOOK-AHEAD. Catching up a 4-hour-old bar while showing
     today's headlines would leak the future into the NEWS condition - the
     exact bias that manufactures a fake news effect. Every headline now
     carries its publication time, and a bar only sees headlines published
     BEFORE that bar closed.

  3. THE NEWS MANIPULATION FAILED IN 1a. NEWS books mentioned a headline in
     only 4 of 16 trades; VOLUME books mentioned volume in 12 of 13. The
     information was there and went unread, so a null result would have been
     uninterpretable. Fix: one extra instruction line, BYTE-IDENTICAL across
     all four conditions, asking every book to name which part of its data
     drove the decision. No condition is told what to look at - they simply
     differ in what there is to look at. Anything else would reintroduce the
     wording confound that Mark 0.1 already disproved.

  Prompt changed at bar 87, so pre/post must be analysed separately. The bar
  number is recorded in prompt_changes.

THE EXPERIMENT
  4 conditions x 5 replicas = 20 books
      PRICE   48 closes                 <- control
      VOLUME  48 closes + volumes
      NEWS    48 closes + headlines
      BOTH    closes + volumes + headlines

  Measured in 1a: within-condition sd EUR 1.0-1.7, SE 0.47-0.74.
  Resolution is about EUR 1.8 - better than the EUR 4.0 originally forecast.

RULES
  One BUY per book per day. Selling unlimited. Nothing forced.
  Long or flat only. No shorting, no leverage, no margin.
  1% risk rule the model cannot override.

CONTROLS
  HODL   - buys once, never sells. Leading again.
  CROSS  - moving-average crossover, no AI.
  RANDOM - coin flip. Currently beating all 20 AI books.
"""

import os, csv, json, time, random, statistics, urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests

# ───────────────────────────── settings ─────────────────────────────

VERSION = "mark1-b"
PROMPT_V = 2                 # bumped whenever the prompt text changes

SYMBOL       = "ETH/EUR"
TIMEFRAME    = "1h"
CAPITAL      = 100.0
RISK_PCT     = 0.01
FEE_RATE     = 0.001
SLIPPAGE     = 0.0005
ATR_LEN      = 14
STOP_MULT    = 5.0
WINDOW       = 48
MIN_NOTIONAL = 5.0
PHASE_A_BARS = 720
PHASE_B_BARS = 720
HISTORY_MAX  = 900
CANDLE_MAX   = 600
TRADES_MAX   = 600
FAILLOG_MAX  = 2000
ERR_CHARS    = 250
NEWS_N       = 8
MAX_CATCHUP  = 5             # bars per run; 5 x 20 books x 10s ~= 17 min

PROVIDER    = "groq"
GROQ_M      = "openai/gpt-oss-20b"
GEMINI_M    = "gemini-2.5-flash"
ANTHROPIC_M = "claude-sonnet-4-6"
CALL_GAP    = 10.0
MAX_TOK     = 700

EXCHANGES = ["kraken", "coinbaseexchange", "bitstamp"]

STATE = Path("state.json")
DATA  = Path("docs/data.json")
FAILS = Path("docs/fails.csv")

CONDITIONS = [
    ("PRICE",  False, False),   # name, show_volume, show_news
    ("VOLUME", True,  False),
    ("NEWS",   False, True),
    ("BOTH",   True,  True),
]
REPLICAS = 5
BOOKS = [f"{n}-{i}" for n, _, _ in CONDITIONS for i in range(1, REPLICAS + 1)]
COND  = {n: (v, w) for n, v, w in CONDITIONS}
CTRLS = ("RANDOM", "HODL", "CROSS")

def cond_of(book):
    return book.rsplit("-", 1)[0]

# ───────────────────────────── state ────────────────────────────────

def blank(name, kind="ai"):
    return dict(name=name, kind=kind, cash=CAPITAL, pos=None, trades=[],
                peak=CAPITAL, last_buy="", phaseA_end=None, confs=[])

def load():
    if STATE.exists():
        s = json.loads(STATE.read_text())
        for k, v in (("candles", []), ("fail_log", []), ("version", ""),
                     ("archived", 0), ("news", {}), ("prompt_changes", []),
                     ("prompt_v", 1), ("caught_up", 0)):
            s.setdefault(k, v)
        s.setdefault("fails_all", s.get("fails", 0))
        for b in BOOKS:
            s["books"].setdefault(b, blank(b))
        for c in CTRLS:
            s["books"].setdefault(c, blank(c, kind="ctrl"))
        return s
    books = {b: blank(b) for b in BOOKS}
    for c in CTRLS:
        books[c] = blank(c, kind="ctrl")
    return dict(bar=0, last_ts=None, phase="A", winner=None, done=False,
                calls=0, calls_today=0, day="", fails=0, fails_all=0,
                version="", archived=0, fail_log=[], news={}, books=books,
                history=[], candles=[], recent=[], prompt_changes=[],
                prompt_v=PROMPT_V, caught_up=0)

# ───────────────────── failure archiving ────────────────────────────

def archive_fails(s):
    rows = s.get("fail_log", [])
    if rows:
        FAILS.parent.mkdir(parents=True, exist_ok=True)
        first = not FAILS.exists()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with FAILS.open("a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["archived_at", "from_version", "bar_time",
                            "book", "action", "detail"])
            for r in rows:
                w.writerow([stamp, r.get("v", s.get("version", "")), r.get("t", ""),
                            r.get("book", ""), r.get("act", ""), r.get("why", "")])
        s["archived"] = s.get("archived", 0) + len(rows)
    s["fail_log"] = []
    s["fails"] = 0
    s["recent"] = [r for r in s.get("recent", []) if r.get("act") not in ("FAIL", "BLOCK")]
    return len(rows)

# ────────────────────────────── news ────────────────────────────────

def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def _epoch(datestr):
    """RSS pubDate -> epoch seconds. 0 if unparseable (treated as very old)."""
    if not datestr:
        return 0
    try:
        return int(parsedate_to_datetime(datestr).timestamp())
    except Exception:
        return 0

def _rss(url, src):
    root = ET.fromstring(_get(url))
    out = []
    for it in root.findall(".//item")[:40]:
        t = (it.findtext("title") or "").strip()
        if t:
            out.append({"t": t[:150], "src": src, "ts": _epoch(it.findtext("pubDate"))})
    return out

def _cryptocompare():
    j = json.loads(_get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN"))
    out = []
    for a in j.get("Data", [])[:40]:
        t = (a.get("title") or "").strip()
        if t:
            out.append({"t": t[:150],
                        "src": a.get("source_info", {}).get("name", "cc"),
                        "ts": int(a.get("published_on") or 0)})
    return out

NEWS_SOURCES = [
    ("cryptocompare", _cryptocompare),
    ("google",  lambda: _rss("https://news.google.com/rss/search?"
                             "q=ethereum+OR+bitcoin+OR+crypto+when:3d"
                             "&hl=en-US&gl=US&ceid=US:en", "google")),
    ("coindesk", lambda: _rss("https://www.coindesk.com/arc/outboundfeeds/rss/", "coindesk")),
    ("cointelegraph", lambda: _rss("https://cointelegraph.com/rss", "cointelegraph")),
]

CORE = ("ethereum", "eth", "bitcoin", "btc", "fed", "sec", "etf", "rate",
        "regulat", "stablecoin", "coinbase", "binance", "treasury")

def fetch_news(s, now_ms):
    """Fetched once per RUN and shared by every NEWS/BOTH book, so replicas
    differ only by model randomness - never by which headlines they saw."""
    cached = s.get("news") or {}
    if cached.get("fetched_ms") and now_ms - cached["fetched_ms"] < 45 * 60 * 1000 \
       and cached.get("items"):
        return cached
    for name, fn in NEWS_SOURCES:
        try:
            items = fn()
            if items:
                # keep the pool broad; ordering only decides which 8 a bar shows
                items.sort(key=lambda i: (-i["ts"],))
                s["news"] = {"fetched_ms": now_ms, "src": name, "items": items[:40],
                             "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
                print(f"news from {name}: {len(items)} headlines")
                return s["news"]
        except Exception as e:
            print(f"news {name} failed: {type(e).__name__}")
    if cached.get("items"):
        print("news: all sources failed, reusing last good set")
        return cached
    s["news"] = {"fetched_ms": now_ms, "src": "none", "items": [], "at": ""}
    return s["news"]

def news_for_bar(news, bar_close_ms):
    """Only headlines PUBLISHED BEFORE this bar closed. Without this, catching
    up an old bar would show it tomorrow's news - a look-ahead bias that would
    fabricate a news effect out of nothing."""
    cutoff = bar_close_ms // 1000
    avail = [i for i in (news or {}).get("items", [])
             if 0 < i.get("ts", 0) <= cutoff]
    # core-topic headlines first, then the rest, newest within each block
    core  = [i for i in avail if any(k in i["t"].lower() for k in CORE)]
    other = [i for i in avail if i not in core]
    return (core + other)[:NEWS_N]

# ──────────────────────────── indicators ────────────────────────────

def atr(c, n=ATR_LEN):
    tr = [max(c[i][2]-c[i][3], abs(c[i][2]-c[i-1][4]), abs(c[i][3]-c[i-1][4]))
          for i in range(1, len(c))]
    return statistics.fmean(tr[-n:]) if tr else c[-1][4]*0.01

def sma(c, n):
    return statistics.fmean([x[4] for x in c[-n:]])

def eq(b, px):
    return b["cash"] + (b["pos"]["qty"]*px if b["pos"] else 0.0)

# ────────────────────────── paper execution ─────────────────────────

def buy(b, px, stop, day, ts, conf=None, why=""):
    if b["last_buy"] == day:
        return False
    e = eq(b, px)
    stop = min(stop, px * 0.99)
    dist = px - stop
    if dist <= 0:
        return False
    afford = b["cash"] / (px * (1 + SLIPPAGE) * (1 + FEE_RATE) * 1.01)
    qty = min((e * RISK_PCT) / dist, afford)
    if qty * px < MIN_NOTIONAL:
        return False
    fill = px * (1 + SLIPPAGE)
    cost = qty * fill
    fee  = cost * FEE_RATE
    if cost + fee > b["cash"]:
        return False
    b["cash"] -= cost + fee
    b["pos"]   = dict(qty=qty, entry=fill, stop=stop, risk=qty*dist,
                      conf=conf, t=ts, why=why, eq_in=round(e, 2))
    b["last_buy"] = day
    return True

def sell(b, px, ts, why=""):
    p = b["pos"]
    if not p:
        return 0.0
    fill  = px * (1 - SLIPPAGE)
    gross = p["qty"] * fill
    fee   = gross * FEE_RATE
    b["cash"] += gross - fee
    entry_fee = p["qty"] * p["entry"] * FEE_RATE
    pnl = (fill - p["entry"]) * p["qty"] - fee - entry_fee
    b["trades"].append(dict(
        tin=p.get("t", ""), tout=ts,
        px_in=round(p["entry"], 2), px_out=round(fill, 2),
        stop=round(p["stop"], 2), qty=round(p["qty"], 6),
        size=round(p["qty"] * p["entry"], 2), risk=round(p["risk"], 2),
        pnl=round(pnl, 2), R=round(pnl / p["risk"], 2) if p["risk"] else 0,
        eq_in=p.get("eq_in", CAPITAL), eq_out=round(b["cash"], 2),
        conf=p.get("conf"), pv=PROMPT_V,
        why_in=p.get("why", "")[:70], why_out=why[:70]))
    if p.get("conf") is not None:
        b["confs"].append(dict(conf=p["conf"], won=pnl > 0))
    b["pos"] = None
    return pnl

# ─────────────────────────── model call ─────────────────────────────

def _groq_body(prompt):
    return {"model": GROQ_M, "max_completion_tokens": MAX_TOK, "temperature": 1.0,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}

def call_model(prompt):
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        hdr = {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip(),
               "content-type": "application/json"}
        r = requests.post(url, headers=hdr, json=_groq_body(prompt), timeout=45)
        if r.status_code == 429:
            time.sleep(25)
            r = requests.post(url, headers=hdr, json=_groq_body(prompt), timeout=45)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = r.json()["choices"][0]["message"]["content"]
    elif PROVIDER == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_M}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip(),
                     "content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": MAX_TOK, "temperature": 1.0,
                                       "responseMimeType": "application/json",
                                       "thinkingConfig": {"thinkingBudget": 0}}},
            timeout=45)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(),
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_M, "max_tokens": MAX_TOK,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=45)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = "".join(x.get("text", "") for x in r.json()["content"])
    txt = txt.replace("```json", "").replace("```", "").strip()
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])

def build_prompt(cond, b, candles, a, can_buy, heads):
    """Identical for every condition EXCEPT which DATA BLOCKS are present.
    Every word of instruction is the same in all four. Only the information
    differs - that is the whole experiment."""
    show_vol, show_news = COND[cond]
    px = candles[-1][4]
    p  = b["pos"]

    blocks = ["Last %d hourly closes, oldest first: %s"
              % (WINDOW, ",".join(f"{c[4]:.2f}" for c in candles[-WINDOW:]))]

    if show_vol:
        vols = [c[5] for c in candles[-WINDOW:]]
        avg  = statistics.fmean(vols) if vols else 0
        blocks.append("Matching hourly volumes: %s" % ",".join(f"{v:.0f}" for v in vols))
        blocks.append("Latest volume is %.2fx the %d-hour average."
                      % ((vols[-1] / avg if avg else 0), WINDOW))

    if show_news:
        if heads:
            blocks.append("Crypto headlines published before this hour, newest first:\n" +
                          "\n".join("- " + h["t"] for h in heads))
        else:
            blocks.append("Crypto headlines published before this hour: none available.")

    if p:
        pos  = (f"HOLDING since {p['entry']:.2f}. Your stop is at {p['stop']:.2f}. "
                f"Price is currently {'above' if px > p['stop'] else 'BELOW'} that stop.")
        opts = '"hold" or "close"'
    else:
        pos  = "FLAT (no position)."
        opts = '"hold" or "buy"' if can_buy else '"hold" (no buy left today)'

    return f"""You trade spot {SYMBOL} in euros. You can only be long or flat. No shorting, no leverage.

{chr(10).join(blocks)}

Current price: {px:.2f}
Typical hourly move (ATR{ATR_LEN}): {a:.2f}
Your position: {pos}
You may buy at most once per day.

Weigh every piece of data given above, and in your reason name the specific
part of it that decided you. Your reason must match that data - do not claim a
stop was hit unless the price is actually below it.

Position size is decided for you by a fixed risk rule. Do not choose size.

Reply with ONLY a json object in exactly this form, nothing else:
{{"action": {opts}, "confidence": <0-100>, "reason": "<max 14 words>"}}"""

# ──────────────────────────── one bar ───────────────────────────────

def step(s, candles, news):
    """candles ends at the bar being decided. Never sees anything later."""
    bar_ms = candles[-1][0]
    ts  = datetime.fromtimestamp(bar_ms / 1000, timezone.utc)
    day = ts.strftime("%Y-%m-%d")
    iso = ts.isoformat(timespec="minutes")
    px, low = candles[-1][4], candles[-1][3]
    a = atr(candles)
    books = s["books"]
    random.seed(20260804 + s["bar"])

    if s["day"] != day:
        s["day"], s["calls_today"] = day, 0

    heads = news_for_bar(news, bar_ms)      # no look-ahead, ever

    def note(book, act, why):
        rec = dict(t=iso[5:], book=book, act=act, why=why, v=VERSION)
        s["recent"].insert(0, rec)
        if act in ("FAIL", "BLOCK"):
            s["fail_log"].append(rec)
            s["fail_log"] = s["fail_log"][-FAILLOG_MAX:]
            s["fails"] += 1
            s["fails_all"] = s.get("fails_all", 0) + 1

    seen = {c[0] for c in s["candles"]}
    for c in candles:
        if c[0] not in seen:
            s["candles"].append([c[0], round(c[1], 2), round(c[2], 2),
                                 round(c[3], 2), round(c[4], 2)])
    s["candles"] = sorted(s["candles"], key=lambda c: c[0])[-CANDLE_MAX:]

    for b in books.values():
        if b["pos"] and b["name"] != "HODL" and low <= b["pos"]["stop"]:
            sp = b["pos"]["stop"]
            pnl = sell(b, sp, iso, "stop hit")
            note(b["name"], "STOP", f"stop {sp:.2f} hit ({pnl:+.2f} EUR)")

    h = books["HODL"]
    if not h["pos"] and h["cash"] > MIN_NOTIONAL:
        fill = px * (1 + SLIPPAGE)
        qty  = h["cash"] / (fill * (1 + FEE_RATE))
        h["cash"] -= qty * fill * (1 + FEE_RATE)
        h["pos"] = dict(qty=qty, entry=fill, stop=0.0, risk=1e9, t=iso,
                        why="bought once, never sells - the do-nothing benchmark",
                        eq_in=CAPITAL)

    cb = books["CROSS"]
    if len(candles) >= 30:
        fast, slow = sma(candles, 10), sma(candles, 30)
        if fast > slow and not cb["pos"]:
            if buy(cb, px, px - STOP_MULT * a, day, iso, why="10 crossed above 30"):
                note("CROSS", "BUY", "10 crossed above 30")
        elif fast < slow and cb["pos"]:
            pnl = sell(cb, px, iso, "crossed back below")
            note("CROSS", "CLOSE", f"crossed back ({pnl:+.2f} EUR)")

    active = [s["winner"]] if (s["phase"] == "B" and s["winner"]) else BOOKS
    entered = False
    for name in active:
        b = books[name]
        can_buy = b["last_buy"] != day
        if not b["pos"] and not can_buy:
            continue
        try:
            d = call_model(build_prompt(cond_of(name), b, candles, a, can_buy, heads))
            s["calls"] += 1
            s["calls_today"] += 1
        except Exception as e:
            note(name, "FAIL", str(e)[:ERR_CHARS])
            time.sleep(CALL_GAP)
            continue
        time.sleep(CALL_GAP)

        act  = str(d.get("action", "hold")).lower().strip()
        why  = str(d.get("reason", ""))[:70]
        conf = d.get("confidence")
        conf = conf if isinstance(conf, (int, float)) else None

        if act == "buy" and not b["pos"]:
            if not can_buy:
                note(name, "BLOCK", "tried to buy twice in one day")
            elif buy(b, px, px - STOP_MULT * a, day, iso, conf, why):
                entered = True
                note(name, "BUY", why)
        elif act == "close" and b["pos"]:
            pnl = sell(b, px, iso, why)
            note(name, "CLOSE", f"{why} ({pnl:+.2f} EUR)")
        elif act not in ("buy", "hold", "close"):
            note(name, "FAIL", f"model returned an unknown action: '{act[:40]}'")

    rb = books["RANDOM"]
    if entered and not rb["pos"] and random.random() < 0.5:
        buy(rb, px, px - STOP_MULT * a, day, iso, why="coin flip: enter")
    elif rb["pos"] and random.random() < 0.04:
        sell(rb, px, iso, "coin flip: exit")

    for b in books.values():
        b["peak"] = max(b["peak"], eq(b, px))

    s["history"].append(dict(t=iso, px=round(px, 2),
                             eq={n: round(eq(b, px), 2) for n, b in books.items()}))
    s["history"] = s["history"][-HISTORY_MAX:]
    s["recent"]  = s["recent"][:40]
    s["bar"] += 1
    s["last_ts"] = bar_ms

    if s["phase"] == "A" and s["bar"] >= PHASE_A_BARS:
        s["winner"] = max(BOOKS, key=lambda n: eq(books[n], px))
        s["phase"]  = "B"
        for b in books.values():
            b["phaseA_end"] = round(eq(b, px), 2)
        note("SYSTEM", "PHASE", f"A over. Leader {s['winner']}. Re-testing it alone.")
    elif s["phase"] == "B" and s["bar"] >= PHASE_A_BARS + PHASE_B_BARS:
        s["done"] = True

# ─────────────────────────── dashboard data ─────────────────────────

def emit(s, px):
    books = s["books"]
    rows, all_trades = [], []
    for n, b in books.items():
        tr = b["trades"]; e = eq(b, px); p = b["pos"]
        rows.append(dict(
            name=n, kind=b["kind"], cond=cond_of(n) if b["kind"] == "ai" else "control",
            eur=round(e, 2), pnl=round(e - CAPITAL, 2),
            ret=round((e / CAPITAL - 1) * 100, 2), cash=round(b["cash"], 2),
            n=len(tr),
            win=round(100 * sum(1 for t in tr if t["pnl"] > 0) / len(tr)) if tr else 0,
            avgR=round(statistics.fmean([t["R"] for t in tr]), 2) if tr else 0,
            best=round(max((t["pnl"] for t in tr), default=0), 2),
            worst=round(min((t["pnl"] for t in tr), default=0), 2),
            dd=round((b["peak"] - e) / b["peak"] * 100, 1) if b["peak"] else 0,
            open=dict(px_in=round(p["entry"], 2), t=p.get("t", ""),
                      why=p.get("why", ""), stop=round(p["stop"], 2),
                      size=round(p["qty"] * p["entry"], 2),
                      upl=round((px - p["entry"]) * p["qty"], 2),
                      uplpc=round((px - p["entry"]) / p["entry"] * 100, 2)) if p else None))
        for t in tr:
            all_trades.append(dict(book=n, **t))
    rows.sort(key=lambda x: -x["eur"])
    all_trades.sort(key=lambda t: t["tout"], reverse=True)
    all_trades = all_trades[:TRADES_MAX]

    groups = []
    hodl = eq(books["HODL"], px)
    for cname, _, _ in CONDITIONS:
        vals = [eq(books[f"{cname}-{i}"], px) for i in range(1, REPLICAS + 1)
                if f"{cname}-{i}" in books]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        se   = sd / (len(vals) ** 0.5)
        nt   = sum(len(books[f"{cname}-{i}"]["trades"]) for i in range(1, REPLICAS + 1)
                   if f"{cname}-{i}" in books)
        # manipulation check: does this condition's data show up in its reasons?
        mine = [t for t in all_trades if cond_of(t["book"]) == cname]
        txt  = [(t["why_in"] + " " + t["why_out"]).lower() for t in mine]
        groups.append(dict(name=cname, n=len(vals), mean=round(mean, 2),
                           se=round(se, 2), sd=round(sd, 2),
                           lo=round(min(vals), 2), hi=round(max(vals), 2),
                           trades=nt, vs_hodl=round(mean - hodl, 2),
                           vals=[round(v, 2) for v in vals],
                           cites_vol=sum(1 for x in txt if "volum" in x),
                           cites_news=sum(1 for x in txt if any(
                               k in x for k in ("news", "headline", "sec ", "etf",
                                                "regulat", "announce", "report"))),
                           cited_of=len(txt)))

    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            diff = g1["mean"] - g2["mean"]
            cse  = (g1["se"] ** 2 + g2["se"] ** 2) ** 0.5
            pairs.append(dict(a=g1["name"], b=g2["name"], diff=round(diff, 2),
                              se=round(cse, 2), sig=bool(cse > 0 and abs(diff) > 2 * cse)))

    calib = []
    allc = [c for n in BOOKS if n in books for c in books[n]["confs"]]
    for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 101)):
        g = [c for c in allc if lo <= c["conf"] < hi]
        if len(g) >= 3:
            calib.append(dict(band=f"{lo}-{hi}", n=len(g),
                              hit=round(100 * sum(1 for c in g if c["won"]) / len(g))))

    nw = s.get("news") or {}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(dict(
        updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        symbol=SYMBOL, timeframe=TIMEFRAME, price=round(px, 2), capital=CAPITAL,
        currency="EUR", version=VERSION, mark="Mark 1b", replicas=REPLICAS,
        prompt_v=PROMPT_V, prompt_changes=s.get("prompt_changes", []),
        caught_up=s.get("caught_up", 0),
        bar=s["bar"], phase=s["phase"], winner=s["winner"], done=s["done"],
        phaseA=PHASE_A_BARS, phaseB=PHASE_B_BARS, provider=PROVIDER,
        model=GROQ_M if PROVIDER == "groq" else (GEMINI_M if PROVIDER == "gemini" else ANTHROPIC_M),
        calls=s["calls"], today=s["calls_today"],
        fails=s["fails"], fails_all=s.get("fails_all", 0), archived=s.get("archived", 0),
        groups=groups, pairs=pairs, books=rows, calib=calib, trades=all_trades,
        candles=s["candles"],
        news=dict(src=nw.get("src", ""), at=nw.get("at", ""),
                  items=[i["t"] for i in nw.get("items", [])[:NEWS_N]]),
        history=[dict(t=h["t"][5:], px=h["px"], eq=h["eq"]) for h in s["history"]],
        recent=s["recent"][:30]), separators=(",", ":")))

# ─────────────────────────── prices ─────────────────────────────────

def fetch_candles(limit):
    last = None
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            c = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
            if c and len(c) > ATR_LEN + 5:
                print(f"prices from {name}: {len(c)} candles")
                return c[:-1]                    # drop the still-forming bar
        except Exception as e:
            last = f"{name}: {type(e).__name__}"
    raise RuntimeError(f"no exchange reachable ({last})")

def main():
    s = load()

    if s.get("version") != VERSION:
        pending = len(s.get("fail_log", []))
        archive_fails(s)
        old = s.get("version") or "(none)"
        s["version"] = VERSION
        print(f"version {old} -> {VERSION}: archived {pending} failures, counter reset")

    if s.get("prompt_v") != PROMPT_V:
        s["prompt_changes"].append(dict(bar=s["bar"], to=PROMPT_V,
                                        at=datetime.now(timezone.utc).isoformat(timespec="minutes")))
        s["prompt_v"] = PROMPT_V
        print(f"prompt version -> {PROMPT_V} at bar {s['bar']}; "
              f"analyse before/after separately")

    if s.get("done"):
        print("experiment complete")
        return

    # enough history for the window plus however many bars we may have missed
    c = fetch_candles(WINDOW + ATR_LEN + MAX_CATCHUP + 8)

    # every closed bar we have not processed yet
    todo = [i for i, k in enumerate(c)
            if (s["last_ts"] is None and i == len(c) - 1) or
               (s["last_ts"] is not None and k[0] > s["last_ts"])]
    todo = [i for i in todo if i >= WINDOW]      # need a full window behind it
    if len(todo) > MAX_CATCHUP:
        skipped = len(todo) - MAX_CATCHUP
        todo = todo[-MAX_CATCHUP:]
        print(f"behind by more than {MAX_CATCHUP} bars, skipping {skipped} oldest")

    if not todo:
        print("no new candle, refreshing dashboard only")
        emit(s, c[-1][4])
        STATE.write_text(json.dumps(s, separators=(",", ":")))
        return

    news = fetch_news(s, c[-1][0])
    print(f"processing {len(todo)} bar(s)")
    for k, i in enumerate(todo):
        if k:
            s["caught_up"] = s.get("caught_up", 0) + 1
        window = c[:i + 1]                       # nothing after bar i is visible
        step(s, window, news)
        STATE.write_text(json.dumps(s, separators=(",", ":")))
        print(f"  bar {s['bar']} @ {datetime.fromtimestamp(c[i][0]/1000, timezone.utc):%m-%d %H:%M} "
              f"px {c[i][4]:.2f} fails {s['fails']}")

    emit(s, c[todo[-1]][4])
    print(f"done. bar {s['bar']} phase {s['phase']} "
          f"calls {s['calls_today']} today, fails {s['fails']}, "
          f"caught up {s.get('caught_up', 0)} total")

if __name__ == "__main__":
    main()
