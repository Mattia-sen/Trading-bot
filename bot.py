#!/usr/bin/env python3
"""
bot.py - MARK 1. Runs ONCE per hour via GitHub Actions, then exits.

SIMULATION ONLY. No exchange account, no exchange API key, no order function.
Public price data only. Nothing here can spend money. Euros throughout.

WHAT CHANGED FROM MARK 0.1
  Mark 0.1 gave ten bots the SAME data and different wording. Result: no bot
  beat buy-and-hold over 388 hours, and the gap between identical twins
  (up to EUR 17) dwarfed the gap between strategies. Wording changes nothing
  because the information is identical.

  Mark 1 changes the INFORMATION instead, and trades breadth for power:

      4 conditions x 5 replicas = 20 books   (was 10 x 2)

      PRICE   48 closes                      <- control, = Mark 0.1's BASE
      VOLUME  48 closes + volume
      NEWS    48 closes + recent headlines
      BOTH    closes + volume + headlines

  Measured per-book sigma from Mark 0.1 is about EUR 3.2, so:
      2 replicas -> can separate a EUR 6.4 difference
      5 replicas -> can separate a EUR 4.0 difference
  If news is worth less than ~EUR 4 over a month, this still cannot see it.
  That is a real limit of the design, stated up front.

RULES (unchanged)
  One BUY per book per day. Selling unlimited. Nothing forced.
  Long or flat only. No shorting, no leverage, no margin.
  1% risk rule the model cannot override.

CONTROLS
  HODL   - buys once, never sells. Won Mark 0.1 outright.
  CROSS  - moving-average crossover, no AI.
  RANDOM - coin flip.
"""

import os, csv, json, time, random, statistics, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests

# ───────────────────────────── settings ─────────────────────────────

VERSION = "mark1-a"          # bump to archive failures and reset the counter

SYMBOL       = "ETH/EUR"
TIMEFRAME    = "1h"
CAPITAL      = 100.0
RISK_PCT     = 0.01
FEE_RATE     = 0.001
SLIPPAGE     = 0.0005
ATR_LEN      = 14
STOP_MULT    = 5.0           # was 2.5: that put fee drag at ~30% of risk
WINDOW       = 48
MIN_NOTIONAL = 5.0
PHASE_A_BARS = 720
PHASE_B_BARS = 720
HISTORY_MAX  = 800
CANDLE_MAX   = 500
TRADES_MAX   = 600
FAILLOG_MAX  = 2000
ERR_CHARS    = 250
NEWS_N       = 8             # headlines shown to NEWS/BOTH books
NEWS_MAX_AGE = 36            # hours; older headlines are dropped

PROVIDER    = "groq"
GROQ_M      = "openai/gpt-oss-20b"
GEMINI_M    = "gemini-2.5-flash"
ANTHROPIC_M = "claude-sonnet-4-6"
CALL_GAP    = 10.0           # Mark 0.1 ran 12% failures at 7s
MAX_TOK     = 700

EXCHANGES = ["kraken", "coinbaseexchange", "bitstamp"]

STATE = Path("state.json")
DATA  = Path("docs/data.json")
FAILS = Path("docs/fails.csv")

# ─── the experiment: what each condition SEES ───
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
        s.setdefault("candles", []); s.setdefault("fail_log", [])
        s.setdefault("fails_all", s.get("fails", 0)); s.setdefault("version", "")
        s.setdefault("archived", 0); s.setdefault("news", {})
        # Mark 0.1 books are gone; add any missing Mark 1 books
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
                history=[], candles=[], recent=[])

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

def _rss(url, src):
    root = ET.fromstring(_get(url))
    out = []
    for it in root.findall(".//item")[:20]:
        t = (it.findtext("title") or "").strip()
        if t:
            out.append({"t": t[:150], "src": src, "d": (it.findtext("pubDate") or "")[:31]})
    return out

def _cryptocompare():
    j = json.loads(_get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN"))
    out = []
    for a in j.get("Data", [])[:20]:
        t = (a.get("title") or "").strip()
        if t:
            ts = a.get("published_on")
            d = datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="minutes") if ts else ""
            out.append({"t": t[:150], "src": a.get("source_info", {}).get("name", "cc"), "d": d})
    return out

# tried in order; first one that answers wins. Same pattern that fixed the
# Binance geo-block: never depend on a single provider.
NEWS_SOURCES = [
    ("cryptocompare", _cryptocompare),
    ("google",  lambda: _rss("https://news.google.com/rss/search?"
                             "q=ethereum+OR+crypto+when:2d&hl=en-US&gl=US&ceid=US:en", "google")),
    ("coindesk", lambda: _rss("https://www.coindesk.com/arc/outboundfeeds/rss/", "coindesk")),
    ("cointelegraph", lambda: _rss("https://cointelegraph.com/rss", "cointelegraph")),
]

def fetch_news(s, bar_ts):
    """Fetched ONCE per bar and shared by every NEWS/BOTH book, so replicas
    differ only by model randomness - never by which headlines they saw."""
    cached = s.get("news") or {}
    if cached.get("ts") == bar_ts and cached.get("items"):
        return cached
    for name, fn in NEWS_SOURCES:
        try:
            items = fn()
            if items:
                s["news"] = {"ts": bar_ts, "src": name, "items": items[:20],
                             "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
                print(f"news from {name}: {len(items)} headlines")
                return s["news"]
        except Exception as e:
            print(f"news {name} failed: {type(e).__name__}")
    if cached.get("items"):
        print("news: all sources failed, reusing last good set")
        return cached
    s["news"] = {"ts": bar_ts, "src": "none", "items": [], "at": ""}
    return s["news"]

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
        conf=p.get("conf"),
        why_in=p.get("why", "")[:60], why_out=why[:60]))
    if p.get("conf") is not None:
        b["confs"].append(dict(conf=p["conf"], won=pnl > 0))
    b["pos"] = None
    return pnl

# ─────────────────────────── model call ─────────────────────────────

def _groq_body(prompt):
    return {"model": GROQ_M, "max_completion_tokens": MAX_TOK, "temperature": 1.0,
            "reasoning_effort": "low",           # reasoning tokens ate the budget
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}

def call_model(prompt):
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        hdr = {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip(),
               "content-type": "application/json"}
        r = requests.post(url, headers=hdr, json=_groq_body(prompt), timeout=45)
        if r.status_code == 429:
            time.sleep(25)                       # wait out the window, retry once
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

def build_prompt(cond, b, candles, a, can_buy, news):
    """Identical for every condition EXCEPT which data blocks are included.
    No condition gets different instructions - only different information."""
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
                      % ((vols[-1]/avg if avg else 0), WINDOW))

    if show_news:
        items = (news or {}).get("items", [])[:NEWS_N]
        if items:
            blocks.append("Recent crypto headlines (newest first):\n" +
                          "\n".join("- " + i["t"] for i in items))
        else:
            blocks.append("Recent crypto headlines: none available this hour.")

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

Your reason must match the data above. Do not claim a stop was hit unless the
price is actually below it.

Position size is decided for you by a fixed risk rule. Do not choose size.

Reply with ONLY a json object in exactly this form, nothing else:
{{"action": {opts}, "confidence": <0-100>, "reason": "<max 12 words>"}}"""

# ──────────────────────────── one bar ───────────────────────────────

def step(s, candles):
    ts  = datetime.fromtimestamp(candles[-1][0] / 1000, timezone.utc)
    day = ts.strftime("%Y-%m-%d")
    iso = ts.isoformat(timespec="minutes")
    px, low = candles[-1][4], candles[-1][3]
    a = atr(candles)
    books = s["books"]
    random.seed(20260804 + s["bar"])

    if s["day"] != day:
        s["day"], s["calls_today"] = day, 0

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

    news = fetch_news(s, candles[-1][0])

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
        c = cond_of(name)
        can_buy = b["last_buy"] != day
        if not b["pos"] and not can_buy:
            continue
        try:
            d = call_model(build_prompt(c, b, candles, a, can_buy, news))
            s["calls"] += 1
            s["calls_today"] += 1
        except Exception as e:
            note(name, "FAIL", str(e)[:ERR_CHARS])
            time.sleep(CALL_GAP)
            continue
        time.sleep(CALL_GAP)

        act  = str(d.get("action", "hold")).lower().strip()
        why  = str(d.get("reason", ""))[:60]
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

    # THE measurement: group mean +/- standard error, per condition
    groups = []
    hodl = eq(books["HODL"], px)
    for cname, _, _ in CONDITIONS:
        vals = [eq(books[f"{cname}-{i}"], px) for i in range(1, REPLICAS + 1)
                if f"{cname}-{i}" in books]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        se   = sd / (len(vals) ** 0.5) if vals else 0.0
        nt   = sum(len(books[f"{cname}-{i}"]["trades"]) for i in range(1, REPLICAS + 1)
                   if f"{cname}-{i}" in books)
        groups.append(dict(name=cname, n=len(vals), mean=round(mean, 2),
                           se=round(se, 2), sd=round(sd, 2),
                           lo=round(min(vals), 2), hi=round(max(vals), 2),
                           trades=nt, vs_hodl=round(mean - hodl, 2),
                           vals=[round(v, 2) for v in vals]))

    # every pairwise contrast, with a combined standard error
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            diff = g1["mean"] - g2["mean"]
            cse  = (g1["se"] ** 2 + g2["se"] ** 2) ** 0.5
            pairs.append(dict(a=g1["name"], b=g2["name"], diff=round(diff, 2),
                              se=round(cse, 2),
                              sig=bool(cse > 0 and abs(diff) > 2 * cse)))

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
        currency="EUR", version=VERSION, mark="Mark 1", replicas=REPLICAS,
        bar=s["bar"], phase=s["phase"], winner=s["winner"], done=s["done"],
        phaseA=PHASE_A_BARS, phaseB=PHASE_B_BARS,
        provider=PROVIDER,
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

def fetch_candles():
    last = None
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            c = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=WINDOW + ATR_LEN + 5)
            if c and len(c) > ATR_LEN + 5:
                print(f"prices from {name}")
                return c[:-1]
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

    if s.get("done"):
        print("experiment complete")
        return

    c = fetch_candles()
    if c[-1][0] == s["last_ts"]:
        print("no new candle, refreshing dashboard only")
        emit(s, c[-1][4])
        STATE.write_text(json.dumps(s, separators=(",", ":")))
        return

    s["last_ts"] = c[-1][0]
    step(s, c)
    STATE.write_text(json.dumps(s, separators=(",", ":")))
    emit(s, c[-1][4])
    print(f"bar {s['bar']} phase {s['phase']} px {c[-1][4]:.2f} EUR "
          f"calls {s['calls_today']} today, fails {s['fails']} this version")

if __name__ == "__main__":
    main()
